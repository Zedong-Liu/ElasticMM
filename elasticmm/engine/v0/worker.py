"""
V0Worker - Ray actor for vLLM v0 engine
Based on EPD-Disaggregation's Worker implementation
"""

import copy
import math
import gc
import socket
import time
from typing import List, Optional, Dict, Any, Sequence

import ray
import torch
import torch.distributed
import ray.util.collective as collective
from elasticmm.engine.v0.utils import (
    EngineStage,
    Request,
    BatchedRequests,
    StepOutput,
    random_digits,
    GB, MB
)


@ray.remote(num_cpus=0, num_gpus=1)
class V0Worker:
    """
    Ray worker for vLLM v0 engine
    Executes stage-specific inference tasks
    """
    
    def __init__(
        self,
        worker_id: int,
        stage: EngineStage,
        model_path: str,
        block_size: int = 16,
        dtype: str = "float16",
        tensor_parallel_size: int = 1,
        seed: int = 1024,
        max_model_len: int = 32768,
        gpu_memory_utilization: float = 0.9,
        limit_mm_per_prompt: Optional[Dict[str, int]] = None,
        # Global NCCL parameters (for P2P KV transfer)
        global_rank: Optional[int] = None,
        world_size: Optional[int] = None,
        nccl_init_method: Optional[str] = None,
    ):
        """
        Initialize worker
        
        Args:
            worker_id: Unique worker ID
            stage: Engine stage (ENCODING, PREFILL, DECODING)
            model_path: Path to model
            block_size: KV cache block size
            dtype: Model dtype
            tensor_parallel_size: Tensor parallel size
            seed: Random seed
            max_model_len: Maximum model sequence length
            gpu_memory_utilization: GPU memory utilization ratio
        """
        self.worker_id = worker_id
        self.stage = stage
        self.original_role = stage  # ✅ Track original role for reference
        self.current_role = stage    # ✅ Track current role (can change via switch_role)
        self.model_path = model_path
        self.block_size = block_size
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        self.seed = seed
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.limit_mm_per_prompt = limit_mm_per_prompt
        
        # Global NCCL for P2P transfer
        self.global_rank = global_rank
        self.nccl_world_size = world_size
        self.nccl_init_method = nccl_init_method
        self.nccl_pg_initialized = False
        
        # Model runner (vLLM v0)
        self.model_runner = None
        
        # KV cache tensors (for prefill and decoding stages)
        self.kv_cache = None
        self.k_cache = None
        self.v_cache = None
        
        # Vision embedding cache (for encoding and prefill stages)
        self.ve_cache = None
        
        # CPU swap
        self.kv_swap = None
        self.ve_swap = None
        
        # GPU info - ✅ 必须先设置device，再创建CUDA streams
        self.gpu_id = ray.get_gpu_ids()[0]
        self.device = torch.device(f"cuda:0")
        torch.cuda.set_device(self.device)
        
        # CUDA streams for swapping - ✅ 在设置device之后创建
        self.swap_in_stream = torch.cuda.Stream()
        self.swap_out_stream = torch.cuda.Stream()
        
        # Swap event tracking
        self.swap_event_table = {}
        self.latest_swap_in_event = None
        self.latest_swap_out_event = None
        
        # MultiModal registry (needed for encoding stage)
        from vllm.multimodal import MULTIMODAL_REGISTRY
        self.mm_registry = MULTIMODAL_REGISTRY
        self.vllm_model_config = None  # Will be set in init_model
        
        # Statistics
        self.execution_time = 0.0
        self.blocked_swapping_time = 0.0
        
    
    def ready(self):
        """Check if worker is ready"""
        return True
    
    def get_stage(self) -> EngineStage:
        """Get current stage"""
        return self.stage
    
    def set_stage(self, stage: EngineStage):
        """Set stage (for role switching)"""
        self.stage = stage
    
    def init_model(self):
        """
        Initialize vLLM v0 ModelRunner
        """
        import os
        import sys
        
        # CRITICAL: Set VLLM_USE_V1=0 to use V0 attention backends
        # vLLM 0.10.1 defaults to V1, but we need V0 for disaggregated serving
        os.environ['VLLM_USE_V1'] = '0'
        
        from vllm.config import (
            ModelConfig, ParallelConfig, SchedulerConfig,
            DeviceConfig, CacheConfig, LoadConfig
        )
        from vllm.worker.worker import init_worker_distributed_environment
        from vllm.worker.model_runner import ModelRunner
        
        # Set random seed
        import random
        import numpy as np
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        
        # Create vLLM configs
        from vllm.config import VllmConfig
        
        model_config = ModelConfig(
            model=self.model_path,
            tokenizer=self.model_path,
            tokenizer_mode="auto",
            trust_remote_code=True,
            dtype=self.dtype,
            seed=self.seed,
            max_model_len=self.max_model_len,
            limit_mm_per_prompt=self.limit_mm_per_prompt or {},
            enforce_eager=True,  # Disable CUDA graphs for disaggregated inference
        )
        
        parallel_config = ParallelConfig(
            pipeline_parallel_size=1,
            tensor_parallel_size=self.tensor_parallel_size,
            worker_use_ray=False,
        )
        
        scheduler_config = SchedulerConfig(
            max_num_batched_tokens=self.max_model_len,
            max_num_seqs=256,
            max_model_len=self.max_model_len,
        )
        
        device_config = DeviceConfig(device="cuda")
        
        cache_config = CacheConfig(
            block_size=self.block_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            swap_space=4,  # 4GB swap space
            cache_dtype="auto",  # Must be 'auto' or fp8 variants, not model dtype
        )
        
        load_config = LoadConfig()
        
        # Create VllmConfig for vLLM v0.10.1+
        vllm_config = VllmConfig(
            model_config=model_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            device_config=device_config,
            cache_config=cache_config,
            load_config=load_config,
        )
        
        # CRITICAL: Initialize global NCCL process group BEFORE vLLM initialization
        # This allows all workers to communicate via P2P while vLLM uses the same group
        if self.global_rank is not None and self.nccl_world_size is not None and self.nccl_init_method is not None:
            if not torch.distributed.is_initialized():
                # Set environment variables for PyTorch distributed
                os.environ['MASTER_ADDR'] = self.nccl_init_method.split('//')[1].split(':')[0]
                os.environ['MASTER_PORT'] = self.nccl_init_method.split(':')[-1]
                os.environ['RANK'] = str(self.global_rank)
                os.environ['WORLD_SIZE'] = str(self.nccl_world_size)
                
                # IMPORTANT: Set NCCL environment variables for better performance
                os.environ['NCCL_DEBUG'] = 'WARN'  # Reduce verbosity but show warnings
                os.environ['NCCL_IB_DISABLE'] = '1'  # Disable InfiniBand (use PCIe)
                os.environ['NCCL_P2P_LEVEL'] = 'PHB'  # Enable PCIe P2P
                
                # Initialize global NCCL process group
                torch.distributed.init_process_group(
                    backend='nccl',
                    init_method=self.nccl_init_method,
                    rank=self.global_rank,
                    world_size=self.nccl_world_size,
                )
                
                # Verify backend
                backend_name = torch.distributed.get_backend()
                self.nccl_pg_initialized = True
            else:
                backend_name = torch.distributed.get_backend()
                self.nccl_pg_initialized = True
        
        # Initialize distributed environment (for tensor parallelism)
        # vLLM will detect torch.distributed is already initialized and skip re-initialization
        init_worker_distributed_environment(
            vllm_config=vllm_config,
            rank=self.global_rank if self.global_rank is not None else 0,
            distributed_init_method=self.nccl_init_method if self.nccl_init_method is not None else f'tcp://localhost:{int(random_digits(4))+int(self.gpu_id)}',
            local_rank=self.global_rank if self.global_rank is not None else 0,
        )
        
        # Create model runner (vLLM v0.10.1+ uses vllm_config)
        self.model_runner = ModelRunner(
            vllm_config=vllm_config,
            kv_cache_dtype=cache_config.cache_dtype,
            is_driver_worker=True,
        )
        
        # Load model
        self.model_runner.load_model()
        
        # Initialize MultiModalRegistry for multimodal processing
        from vllm.multimodal import MultiModalRegistry
        self.mm_registry = MultiModalRegistry()
        
        # Store configs for later use
        self.model_config = model_config
        self.cache_config = cache_config
        self.vllm_model_config = vllm_config.model_config
        self.parallel_config = parallel_config
        self.vllm_model_config = model_config  # For multimodal processing
        
        torch.cuda.synchronize()
    
    def profile_num_available_blocks(self) -> Dict[str, int]:
        """
        Profile GPU memory after model loading to determine available KV cache blocks
        
        This is vLLM's standard approach: load model first, then see how much memory
        is left for KV cache.
        
        Returns:
            Dict with 'num_gpu_blocks' and 'num_cpu_blocks'
        """
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Get current GPU memory usage
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated(0)
        reserved_memory = torch.cuda.memory_reserved(0)
        
        # Calculate available memory for KV cache
        # CRITICAL: Leave headroom for activations, temp buffers, and vision embeddings
        # vLLM approach: use (free_memory - buffer) * utilization_ratio
        free_memory = total_memory - reserved_memory
        
        # Reserve buffer for activations and temp tensors
        # For multimodal workloads with large batches, we need MORE headroom:
        # - Base buffer: 1 GB
        # - Per-request overhead: ~100 MB for activations (assuming batch_size ~32)
        # - Vision processing: 1 GB
        # Total: ~5 GB buffer for safety
        buffer_memory = 5.0 * GB  # Conservative buffer for multimodal + large batches
        
        # Apply utilization ratio to remaining memory
        # Note: gpu_memory_utilization is already conservative (0.9)
        available_memory = max(0, free_memory - buffer_memory)
        kv_cache_memory = available_memory * self.gpu_memory_utilization
        
        # Additional safety: cap KV cache to at most 70% of total GPU memory
        max_kv_cache = total_memory * 0.70
        kv_cache_memory = min(kv_cache_memory, max_kv_cache)
        
        # Calculate size of one KV cache block
        # KV cache shape per layer: [2, num_blocks, block_size, num_heads, head_size]
        num_layers = self.model_config.hf_config.num_hidden_layers
        num_kv_heads = getattr(self.model_config.hf_config, 'num_key_value_heads', 
                               self.model_config.hf_config.num_attention_heads)
        head_size = self.model_config.hf_config.hidden_size // self.model_config.hf_config.num_attention_heads
        
        # Bytes per element
        if self.dtype == "float16":
            bytes_per_element = 2
        elif self.dtype == "bfloat16":
            bytes_per_element = 2
        elif self.dtype == "float32":
            bytes_per_element = 4
        else:
            bytes_per_element = 2  # Default to fp16
        
        # Size of one block for all layers
        # Each block: 2 (K+V) * block_size * num_kv_heads * head_size * bytes_per_element
        block_size_bytes = (
            2 * self.block_size * num_kv_heads * head_size * bytes_per_element * num_layers
        )
        
        # Calculate number of blocks
        num_gpu_blocks = int(kv_cache_memory / block_size_bytes)
        
        # CPU blocks: use a reasonable default (10% of GPU blocks or 1000)
        num_cpu_blocks = max(100, num_gpu_blocks // 10)
        
        print(f"[V0Worker] GPU Memory Profile:")
        print(f"  Total GPU memory: {total_memory / GB:.2f} GB")
        print(f"  Model reserved: {reserved_memory / GB:.2f} GB")
        print(f"  Free after model: {free_memory / GB:.2f} GB")
        print(f"  Buffer reserved (activations/vision/batch): {buffer_memory / GB:.2f} GB")
        print(f"  Available for KV: {available_memory / GB:.2f} GB")
        print(f"  KV cache target (util {self.gpu_memory_utilization*100:.0f}%): {available_memory * self.gpu_memory_utilization / GB:.2f} GB")
        print(f"  KV cache capped (70% total): {max_kv_cache / GB:.2f} GB")
        print(f"  ✓ Final KV cache budget: {kv_cache_memory / GB:.2f} GB")
        print(f"  Block size: {block_size_bytes / MB:.2f} MB")
        print(f"  ✓ Calculated num_gpu_blocks: {num_gpu_blocks}")
        print(f"  ✓ Calculated num_cpu_blocks: {num_cpu_blocks}")
        
        return {
            'num_gpu_blocks': num_gpu_blocks,
            'num_cpu_blocks': num_cpu_blocks
        }
    
    def init_kvcache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> List[int]:
        """
        Initialize KV cache using vLLM's CacheEngine (recommended way)
        
        Args:
            num_gpu_blocks: Number of GPU blocks
            num_cpu_blocks: Number of CPU blocks
            
        Returns:
            CUDA IPC memory handle (as list of integers)
        """
        from vllm.worker.cache_engine import CacheEngine
        from vllm.utils import bind_kv_cache
        from vllm.config import get_layers_from_vllm_config
        from vllm.attention import Attention
        
        # Update cache config with block counts
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks
        
        # Create CacheEngine (vLLM's recommended way)
        device_config = self.model_runner.device_config
        self.cache_engine = CacheEngine(
            self.cache_config,
            self.model_config,
            self.parallel_config,
            device_config,
        )
        
        # Get gpu_cache (list of tensors, one per layer)
        self.gpu_cache = [self.cache_engine.gpu_cache]  # Wrap in list for virtual engine support
        self.cpu_cache = [self.cache_engine.cpu_cache]
        
        # Extract layer-specific KV cache references for compatibility
        # gpu_cache[0] is for virtual_engine=0
        # gpu_cache[0][layer_idx] is the KV cache tensor for that layer
        self.kv_cache = self.gpu_cache[0]  # List of per-layer KV cache tensors
        
        # Bind KV cache to model's Attention layers (CRITICAL!)
        # This allows Attention.forward() to access kv_cache via self.kv_cache[ve]
        
        # Get shared KV cache layers (for models with cross-layer KV sharing)
        shared_kv_cache_layers: dict[str, str] = {}
        attn_layers = get_layers_from_vllm_config(self.model_runner.vllm_config, Attention)
        for layer_name, attn_module in attn_layers.items():
            if (kv_tgt_layer := attn_module.kv_sharing_target_layer_name) is not None:
                shared_kv_cache_layers[layer_name] = kv_tgt_layer
        
        # Bind KV cache to Attention layers via forward context
        # In vLLM 0.10.1, static_forward_context is in vllm_config.compilation_config
        bind_kv_cache(
            self.model_runner.vllm_config.compilation_config.static_forward_context,
            self.gpu_cache,
            shared_kv_cache_layers
        )
        
        
        # Get CUDA IPC memory handle (if available)
        # This would require custom CUDA extension like EPD
        # For now, return a placeholder
        return []  # Placeholder for IPC handle
    
    def init_vecache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> List[int]:
        """
        Initialize vision embedding cache
        
        Args:
            num_gpu_blocks: Number of GPU blocks
            num_cpu_blocks: Number of CPU blocks
            
        Returns:
            CUDA IPC memory handle
        """
        # Use dictionary-based storage for vision embeddings
        # This is more flexible for variable-length vision tokens
        # Key: request_id, Value: vision embedding tensor
        self.ve_cache = {}
        self.ve_cache_capacity = num_gpu_blocks
        
        return []  # Placeholder for IPC handle
    
    def step_encoding(
        self,
        batched_requests: BatchedRequests,
        block_tables: Dict[str, List[int]]
    ) -> Dict[str, torch.Tensor]:
        """Following EPD's implementation"""
        from elasticmm.engine.v0.worker_steps import step_encoding_impl
        return step_encoding_impl(self, batched_requests, block_tables)
    
    def step_prefill(
        self,
        batched_requests: BatchedRequests,
        kv_block_tables: Dict[str, List[int]],
        vision_block_tables: Optional[Dict[str, List[int]]] = None,
    ) -> List[StepOutput]:
        """Following EPD's implementation"""
        from elasticmm.engine.v0.worker_steps import step_prefill_impl
        return step_prefill_impl(self, batched_requests, kv_block_tables, vision_block_tables)
    
    def step_decode(
        self,
        batched_requests: BatchedRequests,
        kv_block_tables: Dict[str, List[int]],
    ) -> List[StepOutput]:
        """Following EPD's implementation"""
        from elasticmm.engine.v0.worker_steps import step_decode_impl
        return step_decode_impl(self, batched_requests, kv_block_tables)
    
    def wait_for_all_swap_out(self):
        """Wait for all swap-out operations to complete"""
        if self.latest_swap_out_event:
            self.latest_swap_out_event.wait()
    
    def wait_for_all_swap_in(self):
        """Wait for all swap-in operations to complete"""
        if self.latest_swap_in_event:
            self.latest_swap_in_event.wait()
    
    # Enhanced: Full swap operations based on EPD's implementation
    def swap_blocks(
        self,
        request_ids: List[str],
        source_block_ids: List[int],
        target_block_ids: List[int],
        is_swap_in: bool,
    ):
        """
        Swap some blocks between CPU and GPU
        If is_swap_in, then move blocks from CPU to GPU, i.e. CPU block
        #source_block_ids[0] will be copied to GPU block #target_block_ids[0]
        and so on. Similar for is_swap_in = False
        
        Based on EPD's swap_blocks implementation
        """
        import torch
        
        # Use appropriate stream
        stream = self.swap_in_stream if is_swap_in else self.swap_out_stream
        
        # Record event
        event = torch.cuda.Event()
        event.record(stream)
        
        # Save that event
        for request_id in request_ids:
            if request_id in self.swap_event_table:
                # If we've issued another swapping operation before, we shall wait it
                # Pay attention to the difference between wait() and synchronize()
                self.swap_event_table[request_id].wait(stream)
            self.swap_event_table[request_id] = event
        
        if is_swap_in:
            self.latest_swap_in_event = event
        else:
            self.latest_swap_out_event = event
        
        # Swap using CUDA operations
        with torch.cuda.stream(stream):
            if is_swap_in:
                # CPU to GPU: copy from kv_swap to kv_cache
                if self.kv_swap is not None and self.kv_cache is not None:
                    for i, (src_idx, tgt_idx) in enumerate(zip(source_block_ids, target_block_ids)):
                        self.kv_cache[:, :, tgt_idx, :, :, :] = self.kv_swap[:, :, src_idx, :, :, :]
            else:
                # GPU to CPU: copy from kv_cache to kv_swap
                if self.kv_cache is not None and self.kv_swap is not None:
                    for i, (src_idx, tgt_idx) in enumerate(zip(source_block_ids, target_block_ids)):
                        self.kv_swap[:, :, tgt_idx, :, :, :] = self.kv_cache[:, :, src_idx, :, :, :]
    
    def clear_request_resource(self, request_id: str):
        """Clear the resources associated with the request"""
        """This is called by engine when a request is finished or aborted"""
        # Clear the swap event table
        self.swap_event_table.pop(request_id, None)
    
    def clear_request_resource_batched(self, request_ids: List[str]):
        """Clear the resources associated with the requests"""
        for request_id in request_ids:
            self.clear_request_resource(request_id)
    
    def destruct(self, options: List[str]):
        """
        Destruct and free resources
        
        Args:
            options: List of resources to free ('model', 'kv', 've')
        """
        if 'model' in options and self.model_runner:
            del self.model_runner
            self.model_runner = None
        
        if 'kv' in options and self.kv_cache is not None:
            del self.kv_cache, self.k_cache, self.v_cache, self.kv_swap
            self.kv_cache = None
            self.k_cache = None
            self.v_cache = None
            self.kv_swap = None
        
        if 've' in options and self.ve_cache is not None:
            del self.ve_cache, self.ve_swap
            self.ve_cache = None
            self.ve_swap = None
        
        # Force garbage collection
        for _ in range(2):
            gc.collect()
            torch.cuda.empty_cache()
        
    
    def get_stats(self) -> Dict[str, float]:
        """Get worker statistics"""
        return {
            "execution_time": self.execution_time,
            "blocked_swapping_time": self.blocked_swapping_time,
            "gpu_id": self.gpu_id,
        }
    
    # ===== NCCL Transfer Methods =====
    
    def extract_kv_blocks(self, block_indices: List[int]) -> torch.Tensor:
        """
        Extract KV cache blocks for transfer
        
        Args:
            block_indices: List of block indices to extract
            
        Returns:
            Tensor containing the KV data for specified blocks
            Shape: [num_layers, 2, num_blocks, block_size, num_heads, head_size]
        """
        if self.kv_cache is None:
            raise RuntimeError("KV cache not initialized")
        
        # OPTIMIZED: Use fancy indexing to extract all blocks at once
        # self.kv_cache is a list of per-layer tensors from CacheEngine
        # Each tensor shape: [2, num_blocks_total, block_size, num_heads, head_size]
        
        # Convert block_indices to tensor for fancy indexing
        block_idx_tensor = torch.tensor(block_indices, dtype=torch.long, device=self.kv_cache[0].device)
        
        # Extract all layers at once using fancy indexing
        layer_kv_list = []
        for layer_kv in self.kv_cache:
            # Fancy indexing: layer_kv[:, block_idx_tensor] extracts all blocks in one operation
            # Result: [2, num_blocks, block_size, num_heads, head_size]
            extracted_blocks = layer_kv[:, block_idx_tensor, :, :, :]
            layer_kv_list.append(extracted_blocks)
        
        # Stack all layers: [num_layers, 2, num_blocks, block_size, num_heads, head_size]
        kv_data = torch.stack(layer_kv_list, dim=0)
        
        return kv_data
    
    def write_kv_blocks(self, block_indices: List[int], kv_data: torch.Tensor):
        """
        Write KV cache data to specified blocks (OPTIMIZED: batch indexing)
        
        Args:
            block_indices: List of block indices to write to
            kv_data: KV data tensor [num_layers, 2, num_blocks, block_size, num_heads, head_size]
        """
        if self.kv_cache is None:
            raise RuntimeError("KV cache not initialized")
        
        assert kv_data.shape[2] == len(block_indices), \
            f"Block count mismatch: {kv_data.shape[2]} vs {len(block_indices)}"
        
        # OPTIMIZED: Use batch indexing instead of nested loops
        # Convert indices to tensor for batch indexing
        block_idx_tensor = torch.tensor(block_indices, dtype=torch.long, device=self.device)
        
        # Write all layers at once using advanced indexing
        for layer_idx, layer_kv in enumerate(self.kv_cache):
            # Batch write: layer_kv[:, block_indices, :, :, :] = kv_data[layer_idx]
            layer_kv[:, block_idx_tensor, :, :, :] = kv_data[layer_idx]
    
    def get_kv_cache_layout(self) -> Dict[str, Any]:
        """Return structural metadata for the KV cache."""
        if not self.kv_cache:
            return {}
        
        sample_layer = self.kv_cache[0]
        return {
            "num_layers": len(self.kv_cache),
            "block_size": sample_layer.shape[2],
            "num_heads": sample_layer.shape[3],
            "head_size": sample_layer.shape[4],
            "dtype": str(sample_layer.dtype).split(".")[-1],
        }
    
    def init_global_nccl(self) -> bool:
        """
        Check if global NCCL is ready for P2P transfer.
        
        NOTE: The actual NCCL initialization happens in init_model() before vLLM initialization.
        This method is just a verification step.
        """
        if self.nccl_pg_initialized and torch.distributed.is_initialized():
            return True
        else:
            return False
    
    def get_global_rank(self) -> int:
        """Get global NCCL rank"""
        return self.global_rank if self.global_rank is not None else -1
    
    def p2p_send_kv(self, dst_rank: int, block_indices: List[int]) -> Dict[str, Any]:
        """Send KV cache blocks via PyTorch P2P (ZERO-COPY: layer-by-layer)"""
        if not self.nccl_pg_initialized:
            return {"error": "Global NCCL not initialized", "bytes": 0, "blocks": 0}
        
        try:
            # ZERO-COPY OPTIMIZATION: Send layer by layer without torch.stack
            # This avoids creating intermediate tensors
            block_idx_tensor = torch.tensor(block_indices, dtype=torch.long, device=self.device)
            
            total_bytes = 0
            for layer_kv in self.kv_cache:
                # Extract blocks for this layer (returns a view, not a copy)
                layer_data = layer_kv[:, block_idx_tensor, :, :, :].contiguous()
                
                # Send this layer directly
                torch.distributed.send(layer_data, dst=dst_rank)
                total_bytes += layer_data.numel() * layer_data.element_size()
            
            torch.cuda.synchronize()
            return {"bytes": total_bytes, "blocks": len(block_indices)}
        except Exception as e:
            # ✅ ERROR HANDLING: Return error info instead of raising
            # This allows caller to check and trigger fallback
            error_msg = f"NCCL send failed: {type(e).__name__}: {str(e)}"
            print(f"[V0Worker] {error_msg}")
            return {"error": error_msg, "bytes": 0, "blocks": 0}
    
    def p2p_recv_kv(self, src_rank: int, block_indices: List[int]) -> Dict[str, Any]:
        """Receive KV cache blocks via PyTorch P2P (ZERO-COPY: layer-by-layer)"""
        if not self.nccl_pg_initialized:
            return {"error": "Global NCCL not initialized", "bytes": 0, "blocks": 0}
        
        try:
            # ZERO-COPY OPTIMIZATION: Receive and write layer by layer
            # This avoids creating large intermediate tensors
            block_idx_tensor = torch.tensor(block_indices, dtype=torch.long, device=self.device)
            
            # Get shape info from KV cache
            num_kv = 2  # key and value
            num_blocks = len(block_indices)
            block_size = self.kv_cache[0].shape[2]
            num_heads = self.kv_cache[0].shape[3]
            head_size = self.kv_cache[0].shape[4]
            kv_dtype = self.kv_cache[0].dtype
            
            layer_shape = (num_kv, num_blocks, block_size, num_heads, head_size)
            total_bytes = 0
            
            # Receive layer by layer and write directly to KV cache
            for layer_idx, layer_kv in enumerate(self.kv_cache):
                # Receive this layer's data
                layer_data = torch.empty(layer_shape, dtype=kv_dtype, device=self.device)
                torch.distributed.recv(layer_data, src=src_rank)
                
                # Write directly to KV cache (batch indexing)
                layer_kv[:, block_idx_tensor, :, :, :] = layer_data
                total_bytes += layer_data.numel() * layer_data.element_size()
            
            torch.cuda.synchronize()
            return {"bytes": total_bytes, "blocks": len(block_indices)}
        except Exception as e:
            # ✅ ERROR HANDLING: Return error info instead of raising
            # This allows caller to check and trigger fallback
            error_msg = f"NCCL recv failed: {type(e).__name__}: {str(e)}"
            print(f"[V0Worker] {error_msg}")
            return {"error": error_msg, "bytes": 0, "blocks": 0}
    
    def p2p_coordinated_transfer_kv(
        self, 
        src_worker_ref: Any,
        src_rank: int, 
        src_blocks: List[int], 
        dst_blocks: List[int],
        timeout: float = 10.0
    ) -> Dict[str, Any]:
        """
        ✅ OPTIMIZED: Coordinated KV transfer with single Ray remote call
        
        This method is called on the DESTINATION worker and coordinates with
        the source worker using NCCL's built-in synchronization, reducing
        Ray remote call overhead by 50%.
        
        Args:
            src_worker_ref: Ray actor handle to source worker
            src_rank: NCCL rank of source worker
            src_blocks: Block indices to send from source
            dst_blocks: Block indices to write to in destination
            timeout: Transfer timeout in seconds (default: 10s)
            
        Returns:
            Dict with transfer statistics
            
        Raises:
            RuntimeError: If NCCL not initialized or transfer fails
            ValueError: If block indices are invalid
            TimeoutError: If transfer exceeds timeout
        """
        import time
        
        # ✅ VALIDATION: Pre-flight checks
        if not self.nccl_pg_initialized:
            raise RuntimeError("Destination worker: Global NCCL not initialized")
        
        if len(src_blocks) != len(dst_blocks):
            raise ValueError(f"Block count mismatch: src={len(src_blocks)}, dst={len(dst_blocks)}")
        
        if len(src_blocks) == 0:
            return {"bytes": 0, "blocks": 0, "error": "No blocks to transfer"}
        
        if src_rank < 0 or src_rank >= torch.distributed.get_world_size():
            raise ValueError(f"Invalid src_rank: {src_rank} (world_size={torch.distributed.get_world_size()})")
        
        start_time = time.time()
        
        try:
            # ✅ STEP 1: Trigger send on source worker (fire and forget)
            # Using Ray's async task to avoid blocking
            send_future = src_worker_ref.p2p_send_kv.remote(
                torch.distributed.get_rank(),  # dst_rank = my rank
                src_blocks
            )
            
            # ✅ STEP 2: Immediately start receiving (NCCL will sync)
            # This is the same as p2p_recv_kv but inline for atomicity
            block_idx_tensor = torch.tensor(dst_blocks, dtype=torch.long, device=self.device)
            
            # Get KV cache shape
            num_kv = 2
            num_blocks = len(dst_blocks)
            block_size = self.kv_cache[0].shape[2]
            num_heads = self.kv_cache[0].shape[3]
            head_size = self.kv_cache[0].shape[4]
            kv_dtype = self.kv_cache[0].dtype
            
            layer_shape = (num_kv, num_blocks, block_size, num_heads, head_size)
            total_bytes = 0
            
            # ✅ RECEIVE: Layer by layer with timeout check
            for layer_idx, layer_kv in enumerate(self.kv_cache):
                # Check timeout
                if time.time() - start_time > timeout:
                    return {
                        "error": f"Transfer timeout after {timeout}s at layer {layer_idx}/{len(self.kv_cache)}",
                        "bytes": 0,
                        "blocks": 0,
                        "elapsed": time.time() - start_time
                    }
                
                # Receive this layer's data
                layer_data = torch.empty(layer_shape, dtype=kv_dtype, device=self.device)
                torch.distributed.recv(layer_data, src=src_rank)
                
                # Write directly to KV cache
                layer_kv[:, block_idx_tensor, :, :, :] = layer_data
                total_bytes += layer_data.numel() * layer_data.element_size()
            
            torch.cuda.synchronize()
            
            # ✅ STEP 3: Wait for send to complete (verify no errors)
            # This is non-blocking since NCCL already finished
            try:
                send_result = ray.get(send_future, timeout=1.0)
            except Exception as send_err:
                # Send failed, return error
                elapsed = time.time() - start_time
                return {
                    "error": f"Send failed: {type(send_err).__name__}: {str(send_err)}",
                    "bytes": 0,
                    "blocks": 0,
                    "elapsed": elapsed
                }
            
            elapsed = time.time() - start_time
            
            # ✅ Check if send returned an error
            if "error" in send_result:
                return {
                    "error": f"Send error: {send_result['error']}",
                    "bytes": 0,
                    "blocks": 0,
                    "elapsed": elapsed
                }
            
            # ✅ VERIFICATION: Ensure send and recv agree on bytes
            if send_result["bytes"] != total_bytes:
                print(f"[V0Worker] Warning: Transfer size mismatch: expected {total_bytes}, got {send_result['bytes']}")
            
            return {
                "bytes": total_bytes,
                "blocks": len(dst_blocks),
                "elapsed": elapsed,
                "method": "coordinated_single_call"
            }
            
        except Exception as e:
            # ✅ ERROR HANDLING: Return error info instead of raising
            # This allows caller to check and trigger fallback
            elapsed = time.time() - start_time
            error_msg = f"NCCL coordinated transfer failed after {elapsed:.2f}s: {type(e).__name__}: {str(e)}"
            print(f"[V0Worker] {error_msg}")
            return {
                "error": error_msg,
                "bytes": 0,
                "blocks": 0,
                "elapsed": elapsed
            }
    
    def p2p_pull_kv(self, src_rank: int, src_blocks: List[int], dst_blocks: List[int]) -> Dict[str, Any]:
        """
        Pull KV cache from source worker (SINGLE REMOTE CALL)
        
        This method combines send and recv in a single remote call to reduce overhead.
        The receiver initiates the transfer by sending a request, then receives the data.
        """
        if not self.nccl_pg_initialized:
            raise RuntimeError("Global NCCL not initialized")
        
        # Step 1: Send transfer request (block indices) to source
        request_tensor = torch.tensor(src_blocks, dtype=torch.long, device=self.device)
        torch.distributed.send(request_tensor, dst=src_rank, tag=999)  # tag 999 = request
        
        # Step 2: Receive KV data (same as p2p_recv_kv)
        block_idx_tensor = torch.tensor(dst_blocks, dtype=torch.long, device=self.device)
        
        num_kv = 2
        num_blocks = len(dst_blocks)
        block_size = self.kv_cache[0].shape[2]
        num_heads = self.kv_cache[0].shape[3]
        head_size = self.kv_cache[0].shape[4]
        kv_dtype = self.kv_cache[0].dtype
        
        layer_shape = (num_kv, num_blocks, block_size, num_heads, head_size)
        total_bytes = 0
        
        for layer_idx, layer_kv in enumerate(self.kv_cache):
            layer_data = torch.empty(layer_shape, dtype=kv_dtype, device=self.device)
            torch.distributed.recv(layer_data, src=src_rank, tag=layer_idx)
            layer_kv[:, block_idx_tensor, :, :, :] = layer_data
            total_bytes += layer_data.numel() * layer_data.element_size()
        
        torch.cuda.synchronize()
        return {"bytes": total_bytes, "blocks": len(dst_blocks)}
    
    def p2p_serve_kv(self, requester_rank: int) -> Dict[str, Any]:
        """
        Serve KV cache to requester (responds to pull request)
        
        This is called by the source worker to respond to a pull request.
        """
        if not self.nccl_pg_initialized:
            raise RuntimeError("Global NCCL not initialized")
        
        # Step 1: Receive request (block indices)
        # We don't know the size, so receive a fixed-size request first
        max_blocks = 1024  # Maximum blocks per transfer
        request_tensor = torch.empty(max_blocks, dtype=torch.long, device=self.device)
        torch.distributed.recv(request_tensor, src=requester_rank, tag=999)
        
        # Extract actual block indices (assuming -1 padding for unused slots)
        block_indices = request_tensor[request_tensor >= 0].cpu().tolist()
        
        # Step 2: Send KV data (same as p2p_send_kv)
        block_idx_tensor = torch.tensor(block_indices, dtype=torch.long, device=self.device)
        
        total_bytes = 0
        for layer_idx, layer_kv in enumerate(self.kv_cache):
            layer_data = layer_kv[:, block_idx_tensor, :, :, :].contiguous()
            torch.distributed.send(layer_data, dst=requester_rank, tag=layer_idx)
            total_bytes += layer_data.numel() * layer_data.element_size()
        
        torch.cuda.synchronize()
        return {"bytes": total_bytes, "blocks": len(block_indices)}
    
    def collective_transfer_kv(
        self,
        group_name: str,
        root: int,
        block_indices: List[int],
        tensor_shape: Sequence[int],
        dtype: str,
        mode: str,
    ) -> Dict[str, Any]:
        """
        Participate in a NCCL broadcast to transfer KV cache blocks.
        """
        import math
        
        import torch
        
        normalized_dtype = dtype.replace("torch.", "").lower()
        dtype_mapping = {
            "float16": torch.float16,
            "half": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if normalized_dtype not in dtype_mapping:
            raise ValueError(f"[V0Worker] Unsupported dtype for collective transfer: {dtype}")
        
        torch_dtype = dtype_mapping[normalized_dtype]
        tensor_shape = tuple(int(dim) for dim in tensor_shape)
        bytes_transferred = 0
        
        if mode == "send":
            kv_tensor = self.extract_kv_blocks(block_indices).contiguous()
            flat_tensor = kv_tensor.view(-1).to(dtype=torch_dtype)
            collective.broadcast(
                flat_tensor,
                src_rank=root,
                group_name=group_name,
            )
            bytes_transferred = flat_tensor.numel() * flat_tensor.element_size()
        
        elif mode == "recv":
            numel = math.prod(tensor_shape)
            flat_tensor = torch.empty(
                numel,
                dtype=torch_dtype,
                device=self.device,
            )
            collective.broadcast(
                flat_tensor,
                src_rank=root,
                group_name=group_name,
            )
            kv_tensor = flat_tensor.view(tensor_shape)
            self.write_kv_blocks(block_indices, kv_tensor)
            bytes_transferred = flat_tensor.numel() * flat_tensor.element_size()
        
        else:
            raise ValueError(f"[V0Worker] Unknown collective transfer mode: {mode}")
        
        torch.cuda.synchronize()
        return {
            "bytes": bytes_transferred,
            "mode": mode,
        }
    
    def extract_vision_blocks(self, request_id: str) -> torch.Tensor:
        """
        Extract vision embeddings for transfer (by request_id)
        
        Args:
            request_id: Request ID to extract vision embeddings for
            
        Returns:
            Tensor containing vision embeddings [num_tokens, hidden_dim]
        """
        if self.ve_cache is None:
            raise RuntimeError("Vision embedding cache not initialized")
        
        if request_id not in self.ve_cache:
            raise KeyError(f"Request {request_id} not found in vision cache")
        
        return self.ve_cache[request_id]
    
    def write_vision_blocks(self, request_id: str, ve_data: torch.Tensor):
        """
        Write vision embedding data for a request
        
        Args:
            request_id: Request ID to write vision embeddings for
            ve_data: Vision embedding data [num_tokens, hidden_dim]
        """
        if self.ve_cache is None:
            raise RuntimeError("Vision embedding cache not initialized")
        
        self.ve_cache[request_id] = ve_data
    
    def get_kv_cache_shape(self) -> tuple:
        """Get KV cache tensor shape"""
        if self.kv_cache is None:
            return ()
        return self.kv_cache.shape
    
    def get_ve_cache_shape(self) -> tuple:
        """Get vision embedding cache tensor shape"""
        if self.ve_cache is None:
            return ()
        return self.ve_cache.shape
    
    def get_ve_cache_keys(self):
        """Get all request IDs that have vision embeddings"""
        if self.ve_cache is None:
            return []
        return list(self.ve_cache.keys())
    
    def get_ve_shape(self, request_id: str):
        """Get the shape of vision embeddings for a request"""
        if self.ve_cache and request_id in self.ve_cache:
            return tuple(self.ve_cache[request_id].shape)
        return None
    
    # ========================================================================
    # Role Switching Support
    # ========================================================================
    
    def switch_role(self, new_role: str) -> bool:
        """
        Switch worker's logical role (for elastic scheduling).
        
        This method allows a worker to change its role (e.g., from 'decoding' to 'prefill')
        without destroying the worker or reinitializing NCCL.
        
        Args:
            new_role: New role string ('encoding', 'prefill', 'decoding')
            
        Returns:
            True if role switch successful
            
        Note:
            - The worker's NCCL rank and GPU assignment remain unchanged
            - Model and KV cache structures are already initialized and remain valid
            - Only the logical role and associated queue membership change
        """
        # EngineStage is already imported at the top of this file
        
        # Validate new role
        valid_roles = {'encoding', 'prefill', 'decoding'}
        if new_role not in valid_roles:
            return False
        
        # Convert string to EngineStage enum
        role_mapping = {
            'encoding': EngineStage.ENCODING,
            'prefill': EngineStage.PREFILL,
            'decoding': EngineStage.DECODING
        }
        new_stage = role_mapping[new_role]
        
        # Check if already in this role
        if self.current_role == new_stage:
            return True
        
        # Step 1: Clear all active requests (should be migrated by backend before calling this)
        self.clear_all_requests()
        
        # Step 2: Update role
        old_role = self.current_role
        self.current_role = new_stage
        self.stage = new_stage  # Update stage attribute as well
        
        
        return True
    
    def clear_all_requests(self):
        """
        Clear all active requests from worker state.
        
        This is called before role switching to ensure clean state.
        Note: KV cache blocks should be freed by the block manager before calling this.
        """
        
        # Clear vision embedding cache
        if self.ve_cache is not None and isinstance(self.ve_cache, dict):
            num_ve = len(self.ve_cache)
            self.ve_cache.clear()
            if num_ve > 0:
                print(f"[V0Worker] Cleared {num_ve} vision embedding entries")
        
        # Note: KV cache tensors themselves (self.kv_cache) are pre-allocated and persistent
        # Block manager handles the logical allocation/deallocation
        # We don't need to zero out the tensors here
        
    
    def get_role_info(self) -> dict:
        """
        Get current role information.
        
        Returns:
            Dictionary with role information
        """
        return {
            'worker_id': self.worker_id,
            'original_role': self.original_role.value if isinstance(self.original_role, EngineStage) else str(self.original_role),
            'current_role': self.current_role.value if isinstance(self.current_role, EngineStage) else str(self.current_role),
            'gpu_id': self.gpu_id,
            'global_rank': self.global_rank,
            'nccl_world_size': self.nccl_world_size
        }


        # Note: KV cache tensors themselves (self.kv_cache) are pre-allocated and persistent
        # Block manager handles the logical allocation/deallocation
        # We don't need to zero out the tensors here
        
    
    def get_role_info(self) -> dict:
        """
        Get current role information.
        
        Returns:
            Dictionary with role information
        """
        return {
            'worker_id': self.worker_id,
            'original_role': self.original_role.value if isinstance(self.original_role, EngineStage) else str(self.original_role),
            'current_role': self.current_role.value if isinstance(self.current_role, EngineStage) else str(self.current_role),
            'gpu_id': self.gpu_id,
            'global_rank': self.global_rank,
            'nccl_world_size': self.nccl_world_size
        }

