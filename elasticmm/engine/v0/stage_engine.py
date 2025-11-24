"""
Stage Engines for v0 backend
Implements Encoding, Prefill, and Decoding stages
Based on EPD-Disaggregation's stage engine architecture
"""

import asyncio
import time
from typing import List, Dict, Optional, Callable, Any
from collections import deque

import ray

from elasticmm.engine.v0.utils import (
    EngineStage,
    EngineStatus,
    Request,
    BatchedRequests,
    MigratingRequest,
    StepOutput,
)
from elasticmm.engine.v0.block_predictor import BlockPredictor
from elasticmm.engine.v0.block_manager import V0BlockManager, V0VisionBlockManager, BlockLocation
from elasticmm.engine.v0.worker import V0Worker
from elasticmm.engine.v0.nccl_transfer import V0NCCLTransferManager
from elasticmm.engine.v0.kv_transfer import V0KVTransferManager


class V0StageScheduler:
    """
    Base scheduler for stage engines with memory-aware scheduling
    """
    
    def __init__(self, stage: EngineStage, block_manager=None, max_tokens_per_batch: int = 32768):
        self.stage = stage
        self.waiting_queue: deque[Request] = deque()
        self.running_requests: Dict[str, Request] = {}
        self.block_manager = block_manager  # For memory-aware scheduling
        
        # ✅ Token budget control (vLLM-inspired)
        # Limit total tokens in a batch to prevent OOM
        # Default: 32768 tokens (~2048 tokens/req * 16 reqs, or more short reqs)
        self.max_tokens_per_batch = max_tokens_per_batch
        
        print(f"[{self.stage.value}Scheduler] Initialized with max_tokens_per_batch={max_tokens_per_batch}")
    
    def add_request(self, request: Request):
        """Add request to waiting queue"""
        self.waiting_queue.append(request)
    
    def can_allocate_request(self, request: Request) -> bool:
        """
        Check if we have enough GPU blocks for this request
        
        Args:
            request: The request to check
            
        Returns:
            True if we can allocate, False otherwise
        """
        if not self.block_manager:
            return True  # No memory constraint
        
        # Estimate blocks needed
        if self.stage == EngineStage.DECODING:
            # For decode: current tokens + some buffer for generation
            total_tokens = len(request.prompt_token_ids) + len(request.output_token_ids) + 10
        else:
            # For encoding/prefill: just prompt tokens
            total_tokens = len(request.prompt_token_ids)
        
        blocks_needed = (total_tokens + self.block_manager.block_size - 1) // self.block_manager.block_size
        avail_blocks = self.block_manager.get_num_avail_gpu_blocks()
        
        return avail_blocks >= blocks_needed
    
    def _group_requests_by_modality(self, requests):
        """
        将请求按模态分组
        
        Returns:
            (text_only_requests, multimodal_requests): 两个列表
        """
        text_only = []
        multimodal = []
        
        for req in requests:
            if req.multi_modal_data:
                multimodal.append(req)
            else:
                text_only.append(req)
        
        return text_only, multimodal
    
    def schedule(self, max_batch_size: int = 32) -> Optional[BatchedRequests]:
        """
        Schedule a batch of requests (MEMORY-AWARE + MODALITY-AWARE)
        
        ✅ NEW: Modality-aware scheduling - group requests by modality to enable
        routing to specialized workers (text-only vs multimodal)
        
        For ENCODING/PREFILL: schedule from waiting_queue (one-shot processing)
        For DECODING: schedule running_requests (continuous generation) + new from waiting_queue
        
        Args:
            max_batch_size: Maximum batch size
            
        Returns:
            BatchedRequests or None (homogeneous by modality)
        """
        batch_requests = []
        
        # ✅ MODALITY-AWARE: Group running and waiting requests by modality
        running_text, running_mm = [], []
        waiting_text, waiting_mm = [], []
        
        # For decode stage: group running requests by modality
        if self.stage == EngineStage.DECODING:
            running_text, running_mm = self._group_requests_by_modality(
                list(self.running_requests.values())
            )
        
        # Group waiting queue by modality
        waiting_text, waiting_mm = self._group_requests_by_modality(
            list(self.waiting_queue)
        )
        
        # ✅ STRATEGY: Prioritize multimodal (usually more complex), then text-only
        # This ensures multimodal requests get processed first to reduce latency
        # 优先级：running_mm > running_text > waiting_mm > waiting_text
        
        # Add running multimodal first
        if self.stage == EngineStage.DECODING and running_mm:
            for request in running_mm[:max_batch_size]:
                batch_requests.append(request)
        
        # If batch not full and we have running text-only, add them
        # NOTE: This creates a mixed batch - simplified approach for now
        # TODO: Future optimization - separate batches for each modality
        if self.stage == EngineStage.DECODING and running_text and len(batch_requests) < max_batch_size:
            remaining_slots = max_batch_size - len(batch_requests)
            for request in running_text[:remaining_slots]:
                batch_requests.append(request)
        
        # Add new requests from waiting queue (MEMORY-AWARE + MODALITY-AWARE)
        # Prioritize same modality as existing batch (if any)
        if len(batch_requests) < max_batch_size:
            # Determine batch modality from first request (if any)
            batch_is_multimodal = None
            if batch_requests:
                batch_is_multimodal = bool(batch_requests[0].multi_modal_data)
            
            # Choose which waiting queue to process
            if batch_is_multimodal is None:
                # Empty batch, prioritize multimodal
                primary_queue = waiting_mm
                secondary_queue = waiting_text
            elif batch_is_multimodal:
                # Batch is multimodal, continue with multimodal
                primary_queue = waiting_mm
                secondary_queue = waiting_text
            else:
                # Batch is text-only, continue with text-only
                primary_queue = waiting_text
                secondary_queue = waiting_mm
            
            # Add requests from primary queue
            for request in primary_queue:
                if len(batch_requests) >= max_batch_size:
                    break
                
                # ✅ MEMORY-AWARE: Check if we have enough blocks
                if not self.can_allocate_request(request):
                    print(f"[{self.stage.value}Scheduler] ⚠️  Not enough GPU blocks for {request.request_id}, "
                          f"keeping in waiting queue (avail={self.block_manager.get_num_avail_gpu_blocks() if self.block_manager else 'N/A'} blocks)")
                    break  # Stop adding new requests
                
                # Can allocate, remove from waiting queue and add to batch
                self.waiting_queue.remove(request)
                batch_requests.append(request)
                self.running_requests[request.request_id] = request
            
            # If batch still not full and primary queue exhausted, try secondary
            if len(batch_requests) < max_batch_size and not primary_queue:
                for request in secondary_queue:
                    if len(batch_requests) >= max_batch_size:
                        break
                    
                    if not self.can_allocate_request(request):
                        break
                    
                    self.waiting_queue.remove(request)
                    batch_requests.append(request)
                    self.running_requests[request.request_id] = request
        
        if batch_requests:
            # Log batch modality composition for debugging
            text_count = sum(1 for r in batch_requests if not r.multi_modal_data)
            mm_count = len(batch_requests) - text_count
            if text_count > 0 and mm_count > 0:
                print(f"[{self.stage.value}Scheduler] ⚠️  Mixed modality batch: {text_count} text + {mm_count} multimodal")
            return BatchedRequests(requests=batch_requests)
        
        return None
    
    def finish_request(self, request_id: str):
        """Mark request as finished"""
        if request_id in self.running_requests:
            del self.running_requests[request_id]
    
    def num_waiting_requests(self) -> int:
        """Get number of waiting requests"""
        return len(self.waiting_queue)
    
    def num_running_requests(self) -> int:
        """Get number of running requests"""
        return len(self.running_requests)


class V0BaseStageEngine:
    """
    Base class for stage engines
    """
    
    def __init__(
        self,
        stage: EngineStage,
        model_path: str,
        num_workers: int = 1,
        block_size: int = 16,
        max_num_gpu_blocks: int = 5000,
        max_num_cpu_blocks: int = 1000,
        dtype: str = "float16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        limit_mm_per_prompt: Optional[Dict[str, int]] = None,
        kv_transfer_manager: Optional[V0KVTransferManager] = None,
        nccl_coordinator: Any = None,
        backend_ref: Any = None,  # Reference to V0EngineBackend for migration
    ):
        """
        Initialize stage engine
        
        Args:
            stage: Engine stage
            model_path: Path to model
            num_workers: Number of workers
            block_size: KV cache block size
            max_num_gpu_blocks: Maximum GPU blocks
            max_num_cpu_blocks: Maximum CPU blocks
            dtype: Model dtype
            tensor_parallel_size: Tensor parallel size
            gpu_memory_utilization: GPU memory utilization
            kv_transfer_manager: KV transfer manager
            nccl_coordinator: NCCL coordinator
            backend_ref: Reference to backend for instance migration
        """
        self.stage = stage
        self.model_path = model_path
        self.num_workers = num_workers
        self.block_size = block_size
        self.max_num_gpu_blocks = max_num_gpu_blocks
        self.max_num_cpu_blocks = max_num_cpu_blocks
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.limit_mm_per_prompt = limit_mm_per_prompt
        self.kv_transfer_manager = kv_transfer_manager
        self.nccl_coordinator = nccl_coordinator
        self._backend_ref = backend_ref  # For migration support
        
        # Workers
        self.workers: List[ray.ObjectRef] = []
        
        # Scheduler (will be properly initialized in subclass after block_manager is created)
        self.scheduler = None
        
        # Status
        self.status = EngineStatus.INACTIVE
        
        # Event loop
        self.pls_stop_loop = asyncio.Event()
        self.is_loop_stopped = asyncio.Event()
        
        # Load balancing
        self.current_worker_index = 0
        
        # ✅ NEW: Modality-aware worker grouping
        # 模态组worker映射（基于ElasticMM设计：text-only和multimodal分离）
        self.text_only_workers: List[int] = []      # worker_id列表（TEXT_ONLY组）
        self.multimodal_workers: List[int] = []     # worker_id列表（MULTIMODAL组）
        # Note: _init_modality_groups() will be called in initialize() after workers are created
        
        # Metrics collection reference (will be set by backend)
        self._metrics_ref = None
    
    def _record_request_metrics(self, request: Request):
        """Record metrics for a completed request"""
        if self._metrics_ref is None:
            return
        
        # Calculate latencies
        encoding_time_ms = 0
        prefill_time_ms = 0
        decode_time_ms = 0
        
        if request.encoding_start_time and request.encoding_end_time:
            encoding_time_ms = (request.encoding_end_time - request.encoding_start_time) * 1000
        
        if request.prefill_start_time and request.prefill_end_time:
            prefill_time_ms = (request.prefill_end_time - request.prefill_start_time) * 1000
        
        # ✅ FIX: Use cumulative compute time for decode, not start-end difference
        if request.total_decode_compute_time > 0:
            decode_time_ms = request.total_decode_compute_time * 1000  # Convert to ms
        
        # Get token counts
        num_input_tokens = request.get_input_len()
        num_output_tokens = request.get_output_len()
        
        # Record to appropriate stage metrics
        if encoding_time_ms > 0 and num_input_tokens > 0:
            self._metrics_ref.encoding_metrics.record_encoding(num_input_tokens, encoding_time_ms)
        
        if prefill_time_ms > 0 and num_input_tokens > 0:
            self._metrics_ref.prefill_metrics.record_prefill(num_input_tokens, prefill_time_ms)
        
        if decode_time_ms > 0 and num_output_tokens > 0:
            self._metrics_ref.decoding_metrics.record_decode(num_output_tokens, decode_time_ms)
            self._metrics_ref.decoding_metrics.requests_completed += 1
    
    def _init_modality_groups(self):
        """
        初始化模态组分配（基于worker_id和stage）
        
        根据ElasticMM设计：
        - Text-Only组：prefill_0, decoding_0 (2 GPUs)
        - Multimodal组：encoding_0-1, prefill_1-2, decoding_1-2 (6 GPUs)
        
        特殊情况处理：
        - 如果只有1个worker，它同时处理text-only和multimodal请求
        
        ⚠️  NOTE: 此方法会在worker role切换后被重新调用以更新分组
        """
        # 计算实际可用的worker数量（非None的worker）
        actual_num_workers = sum(1 for w in self.workers if w is not None)
        
        if self.stage == EngineStage.ENCODING:
            # 所有encoding workers都是multimodal（text-only不需要encoding）
            self.multimodal_workers = [i for i in range(len(self.workers)) if self.workers[i] is not None]
            print(f"[{self.stage.value}Engine] Modality groups: multimodal_workers={self.multimodal_workers} (actual: {actual_num_workers})")
        elif self.stage == EngineStage.PREFILL:
            if actual_num_workers == 1:
                # 只有1个worker时，它同时处理两种模态（测试场景）
                active_worker_id = next(i for i in range(len(self.workers)) if self.workers[i] is not None)
                self.text_only_workers = [active_worker_id]
                self.multimodal_workers = [active_worker_id]
                print(f"[{self.stage.value}Engine] Single worker mode: worker_{active_worker_id} handles BOTH text-only and multimodal")
            else:
                # 找出所有active workers
                active_workers = [i for i in range(len(self.workers)) if self.workers[i] is not None]
                # 第一个是text-only，其余是multimodal
                self.text_only_workers = [active_workers[0]] if active_workers else []
                self.multimodal_workers = active_workers[1:] if len(active_workers) > 1 else []
                print(f"[{self.stage.value}Engine] Modality groups: text_only={self.text_only_workers}, multimodal={self.multimodal_workers} (actual: {actual_num_workers})")
        elif self.stage == EngineStage.DECODING:
            if actual_num_workers == 1:
                # 只有1个worker时，它同时处理两种模态（测试场景）
                active_worker_id = next(i for i in range(len(self.workers)) if self.workers[i] is not None)
                self.text_only_workers = [active_worker_id]
                self.multimodal_workers = [active_worker_id]
                print(f"[{self.stage.value}Engine] Single worker mode: worker_{active_worker_id} handles BOTH text-only and multimodal")
            else:
                # 找出所有active workers
                active_workers = [i for i in range(len(self.workers)) if self.workers[i] is not None]
                # 第一个是text-only，其余是multimodal
                self.text_only_workers = [active_workers[0]] if active_workers else []
                self.multimodal_workers = active_workers[1:] if len(active_workers) > 1 else []
                print(f"[{self.stage.value}Engine] Modality groups: text_only={self.text_only_workers}, multimodal={self.multimodal_workers} (actual: {actual_num_workers})")
    
    def get_worker_by_modality(self, request: Request):
        """
        根据请求的模态类型选择合适的worker（模态感知调度）
        
        这是ElasticMM核心特性：将text-only和multimodal请求隔离到不同的workers
        
        Args:
            request: 请求对象（包含multi_modal_data）
            
        Returns:
            worker: 对应模态组的worker，如果没有可用worker返回None
        """
        is_multimodal = bool(request.multi_modal_data)
        
        if is_multimodal:
            # 多模态请求：从multimodal_workers中round-robin选择
            if not self.multimodal_workers:
                print(f"[{self.stage.value}] ⚠️  No multimodal workers available for request {request.request_id}")
                return None
            
            worker_id = self.multimodal_workers[
                self.current_worker_index % len(self.multimodal_workers)
            ]
            self.current_worker_index += 1
            print(f"[{self.stage.value}] 🎨 Routing MULTIMODAL request {request.request_id} to worker_{worker_id}")
            return self.workers[worker_id]
        else:
            # 纯文本请求：从text_only_workers中round-robin选择
            if not self.text_only_workers:
                # 如果没有text-only workers（如encoding阶段），这不应该发生
                print(f"[{self.stage.value}] ⚠️  No text-only workers available (shouldn't reach here for encoding)")
                return None
            
            worker_id = self.text_only_workers[
                self.current_worker_index % len(self.text_only_workers)
            ]
            self.current_worker_index += 1
            print(f"[{self.stage.value}] 📝 Routing TEXT-ONLY request {request.request_id} to worker_{worker_id}")
            return self.workers[worker_id]
    
    def get_next_worker(self):
        """
        Get next worker using round-robin load balancing (legacy method)
        
        ⚠️  WARNING: This method does not consider modality.
        Use get_worker_by_modality() for modality-aware scheduling.
        """
        if not self.workers:
            return None
        worker = self.workers[self.current_worker_index]
        self.current_worker_index = (self.current_worker_index + 1) % len(self.workers)
        return worker
    
    async def add_worker(self, worker_id: int):
        """
        Add a new worker dynamically with NCCL group setup
        
        Args:
            worker_id: Worker ID to add
        """
        print(f"[{self.stage.value}Engine] ⏳ Adding worker {worker_id}...")
        
        # Check if worker already exists
        if worker_id < len(self.workers) and self.workers[worker_id] is not None:
            print(f"[{self.stage.value}Engine] ⚠️  Worker {worker_id} already exists")
            return
        
        # Get NCCL parameters if coordinator available
        global_rank = None
        world_size = None
        nccl_init_method = None
        if self.nccl_coordinator:
            stage_name = self.stage.value
            # Register worker in NCCL coordinator first
            self.nccl_coordinator.register_worker(stage_name, worker_id)
            global_rank = self.nccl_coordinator.get_rank(stage_name, worker_id)
            world_size = self.nccl_coordinator.get_world_size()
            nccl_init_method = self.nccl_coordinator.get_init_method()
            print(f"[{self.stage.value}Engine] ✓ Registered with NCCL: rank={global_rank}, world_size={world_size}")
        
        # Create new worker
        worker = V0Worker.remote(
            worker_id=worker_id,
            stage=self.stage,
            model_path=self.model_path,
            block_size=self.block_size,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            limit_mm_per_prompt=self.limit_mm_per_prompt,
            # NCCL P2P parameters
            global_rank=global_rank,
            world_size=world_size,
            nccl_init_method=nccl_init_method,
        )
        
        # Add to workers list
        if worker_id >= len(self.workers):
            # Extend list if needed
            self.workers.extend([None] * (worker_id - len(self.workers) + 1))
        
        self.workers[worker_id] = worker
        self.num_workers = max(self.num_workers, worker_id + 1)
        
        # Initialize worker (model + NCCL)
        await worker.ready.remote()
        await worker.init_model.remote()
        
        # Initialize KV cache with profiled block count
        try:
            profile_result = await worker.profile_num_available_blocks.remote()
            num_gpu_blocks = profile_result['num_gpu_blocks']
            num_cpu_blocks = profile_result['num_cpu_blocks']
            await worker.init_kvcache.remote(num_gpu_blocks, num_cpu_blocks)
            print(f"[{self.stage.value}Engine] ✓ KV cache initialized: {num_gpu_blocks} GPU blocks, {num_cpu_blocks} CPU blocks")
        except Exception as e:
            print(f"[{self.stage.value}Engine] ⚠️  KV cache init warning: {e}, using defaults")
            await worker.init_kvcache.remote()
        
        # Update KV transfer manager's worker registry
        if hasattr(self, 'kv_transfer_manager') and self.kv_transfer_manager:
            self.kv_transfer_manager.register_worker(self.stage.value, worker_id, worker)
            print(f"[{self.stage.value}Engine] ✓ Registered with KV transfer manager")
        
        # Reinitialize modality groups after adding worker
        # ⚠️  CRITICAL: Must update modality groups after worker list changes
        self._init_modality_groups()
        
        print(f"[{self.stage.value}Engine] ✅ Worker {worker_id} added and ready")
    
    async def remove_worker(self, worker_id: int, migrate_requests: bool = True):
        """
        Remove a worker dynamically with KV migration and graceful exit
        
        Args:
            worker_id: Worker ID to remove
            migrate_requests: Whether to migrate active requests to other workers
        """
        print(f"[{self.stage.value}Engine] ⏳ Removing worker {worker_id}...")
        
        # Check if worker exists
        if worker_id >= len(self.workers) or self.workers[worker_id] is None:
            print(f"[{self.stage.value}Engine] ⚠️  Worker {worker_id} does not exist")
            return
        
        # Get worker reference
        worker = self.workers[worker_id]
        
        # Step 1: Get all active requests on this worker
        active_request_ids = []
        if migrate_requests:
            # For running requests in scheduler
            for request_id, req in self.scheduler.running_requests.items():
                # Check if this request has blocks allocated on this worker
                blocks = self.block_manager.get_block_table(request_id)
                if blocks:
                    active_request_ids.append(request_id)
            
            if active_request_ids:
                print(f"[{self.stage.value}Engine] Found {len(active_request_ids)} active requests to migrate")
                
                # Step 2: Select target worker (smart selection - based on KV slots)
                # Calculate total blocks needed for migration
                total_blocks_needed = sum(
                    len(self.block_manager.get_block_table(rid))
                    for rid in active_request_ids
                )
                
                target_worker_id = self._select_target_by_kv_slots(
                    exclude_worker_id=worker_id,
                    required_blocks=total_blocks_needed
                )
                
                if target_worker_id is not None:
                    print(f"[{self.stage.value}Engine] Selected target worker: {target_worker_id}")
                    
                    # Step 3: Migrate requests
                    requests_to_migrate = [
                        self.scheduler.running_requests[rid]
                        for rid in active_request_ids
                        if rid in self.scheduler.running_requests
                    ]
                    
                    if requests_to_migrate:
                        # Use backend's migrate_instance method
                        if hasattr(self, '_backend_ref') and self._backend_ref:
                            src_instance_id = f"{self.stage.value}_{worker_id}"
                            dst_instance_id = f"{self.stage.value}_{target_worker_id}"
                            
                            migration_success = await self._backend_ref.migrate_instance(
                                src_instance_id=src_instance_id,
                                dst_instance_id=dst_instance_id,
                                requests=requests_to_migrate
                            )
                            
                            if not migration_success:
                                print(f"[{self.stage.value}Engine] ❌ Migration failed, aborting worker removal")
                                return
                            
                            print(f"[{self.stage.value}Engine] ✓ Migrated {len(requests_to_migrate)} requests")
                        else:
                            print(f"[{self.stage.value}Engine] ⚠️  No backend reference, cannot migrate requests")
                            # Free blocks on source worker
                            for request_id in active_request_ids:
                                self.block_manager.free_blocks(request_id)
                else:
                    print(f"[{self.stage.value}Engine] ⚠️  No target worker available for migration")
                    # Free blocks on source worker
                    for request_id in active_request_ids:
                        self.block_manager.free_blocks(request_id)
            else:
                print(f"[{self.stage.value}Engine] No active requests on worker {worker_id}")
        
        # Step 4: Unregister from NCCL coordinator
        if self.nccl_coordinator:
            try:
                self.nccl_coordinator.unregister_worker(self.stage.value, worker_id)
                print(f"[{self.stage.value}Engine] ✓ Unregistered from NCCL coordinator")
            except Exception as e:
                print(f"[{self.stage.value}Engine] ⚠️  NCCL unregister warning: {e}")
        
        # Step 5: Unregister from KV transfer manager
        if hasattr(self, 'kv_transfer_manager') and self.kv_transfer_manager:
            try:
                self.kv_transfer_manager.unregister_worker(self.stage.value, worker_id)
                print(f"[{self.stage.value}Engine] ✓ Unregistered from KV transfer manager")
            except Exception as e:
                print(f"[{self.stage.value}Engine] ⚠️  KV transfer unregister warning: {e}")
        
        # Step 6: Stop worker gracefully
        try:
            await worker.stop.remote()
            print(f"[{self.stage.value}Engine] ✓ Worker stopped")
        except Exception as e:
            print(f"[{self.stage.value}Engine] ⚠️  Worker stop warning: {e}")
        
        # Step 7: Remove from workers list
        self.workers[worker_id] = None
        
        # Update num_workers if this was the last worker
        if worker_id == self.num_workers - 1:
            # Find the last non-None worker
            for i in range(len(self.workers) - 1, -1, -1):
                if self.workers[i] is not None:
                    self.num_workers = i + 1
                    break
            else:
                self.num_workers = 0
        
        # Step 8: Reinitialize modality groups after worker removal
        # ⚠️  CRITICAL: Must update modality groups after worker list changes
        self._init_modality_groups()
        
        print(f"[{self.stage.value}Engine] ✅ Worker {worker_id} removed successfully")
    
    def _select_target_by_kv_slots(self, exclude_worker_id: int, 
                                   required_blocks: int = 0) -> Optional[int]:
        """
        ✅ Smart target selection based on KV cache slots availability
        
        Args:
            exclude_worker_id: Worker ID to exclude from selection
            required_blocks: Minimum number of free blocks required
            
        Returns:
            Target worker ID with most free KV slots, or None if insufficient
        """
        candidates = []
        
        # Get block manager stats
        free_blocks = self.block_manager.get_num_avail_gpu_blocks()
        total_blocks = self.block_manager.max_num_gpu_blocks
        
        # Note: Our block_manager is global across all workers in this stage
        # For per-worker block tracking, we'd need to query each worker individually
        # For now, we assume uniform distribution and select based on current scheduler state
        
        for i in range(len(self.workers)):
            if i == exclude_worker_id or self.workers[i] is None:
                continue
            
            # Count blocks used by requests on this worker (approximation)
            worker_used_blocks = 0
            for request_id in self.scheduler.running_requests:
                blocks = self.block_manager.get_block_table(request_id)
                if blocks:
                    # Simplified: assume requests are distributed across workers
                    worker_used_blocks += len(blocks) // max(1, len([w for w in self.workers if w]))
            
            # Estimate free blocks for this worker
            estimated_free = total_blocks // max(1, len([w for w in self.workers if w])) - worker_used_blocks
            
            # Check if sufficient space
            if estimated_free >= required_blocks:
                candidates.append((i, estimated_free))
        
        if not candidates:
            print(f"[{self.stage.value}Engine] ⚠️  No worker has >= {required_blocks} free blocks")
            return None
        
        # Select worker with most free blocks
        target_worker_id, max_free = max(candidates, key=lambda x: x[1])
        print(f"[{self.stage.value}Engine] 🎯 Selected worker {target_worker_id} (est. free blocks: {max_free})")
        
        return target_worker_id
    
    async def initialize(self):
        """Initialize workers"""
        print(f"[{self.stage.value}Engine] Initializing {self.num_workers} workers...")
        
        # Create workers
        for worker_id in range(self.num_workers):
            # Get NCCL parameters if coordinator available
            global_rank = None
            world_size = None
            nccl_init_method = None
            if self.nccl_coordinator:
                stage_name = self.stage.value  # 'encoding', 'prefill', 'decoding'
                global_rank = self.nccl_coordinator.get_rank(stage_name, worker_id)
                world_size = self.nccl_coordinator.get_world_size()
                nccl_init_method = self.nccl_coordinator.get_init_method()
            
            worker = V0Worker.remote(
                worker_id=worker_id,
                stage=self.stage,
                model_path=self.model_path,
                block_size=self.block_size,
                dtype=self.dtype,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                limit_mm_per_prompt=self.limit_mm_per_prompt,
                # NCCL P2P parameters
                global_rank=global_rank,
                world_size=world_size,
                nccl_init_method=nccl_init_method,
            )
            self.workers.append(worker)
        
        # Wait for workers to be ready
        await asyncio.gather(*[worker.ready.remote() for worker in self.workers])
        
        # Initialize models (NCCL will be initialized inside init_model before vLLM init)
        await asyncio.gather(*[worker.init_model.remote() for worker in self.workers])
        
        # ========================================================================
        # ✅ DYNAMIC KV CACHE SIZING (vLLM standard approach)
        # ========================================================================
        # Profile GPU memory and calculate optimal block count
        print(f"[{self.stage.value}Engine] Profiling GPU memory for KV cache sizing...")
        block_profiles = await asyncio.gather(*[worker.profile_num_available_blocks.remote() for worker in self.workers])
        
        # Use the minimum across all workers for safety
        min_gpu_blocks = min(profile['num_gpu_blocks'] for profile in block_profiles)
        min_cpu_blocks = min(profile['num_cpu_blocks'] for profile in block_profiles)
        
        print(f"[{self.stage.value}Engine] Dynamic KV cache sizing:")
        print(f"  ✓ num_gpu_blocks: {min_gpu_blocks} (was {self.max_num_gpu_blocks})")
        print(f"  ✓ num_cpu_blocks: {min_cpu_blocks} (was {self.max_num_cpu_blocks})")
        
        # Update block manager with profiled values
        if hasattr(self, 'block_manager'):
            self.block_manager.max_num_gpu_blocks = min_gpu_blocks
            self.block_manager.max_num_cpu_blocks = min_cpu_blocks
            self.block_manager._reset_free_blocks()  # Reinitialize free block lists
            print(f"[{self.stage.value}Engine] ✓ Block manager updated with dynamic sizing")
        
        if hasattr(self, 'vision_block_manager'):
            self.vision_block_manager.max_num_gpu_blocks = min_gpu_blocks
            self.vision_block_manager.max_num_cpu_blocks = min_cpu_blocks
            self.vision_block_manager._reset_free_blocks()
            print(f"[{self.stage.value}Engine] ✓ Vision block manager updated with dynamic sizing")
        
        # Update config for consistency
        self.max_num_gpu_blocks = min_gpu_blocks
        self.max_num_cpu_blocks = min_cpu_blocks
        
        # Verify NCCL P2P is ready
        if self.nccl_coordinator:
            print(f"[{self.stage.value}Engine] Verifying global NCCL for P2P...")
            results = await asyncio.gather(*[worker.init_global_nccl.remote() for worker in self.workers])
            if all(results):
                print(f"[{self.stage.value}Engine] ✓ All workers ready for NCCL P2P transfer")
            else:
                print(f"[{self.stage.value}Engine] ✗ Warning: Some workers failed NCCL P2P verification")
        
        # Initialize modality groups after workers are created
        # ⚠️  CRITICAL: Must be called after workers list is populated
        self._init_modality_groups()
        
        print(f"[{self.stage.value}Engine] Initialized successfully")
        self.status = EngineStatus.ACTIVE
    
    def add_request(self, request: Request):
        """Add request to scheduler"""
        self.scheduler.add_request(request)
    
    async def start_event_loop(self):
        """Start the event loop"""
        print(f"[{self.stage.value}Engine] Starting event loop...")
        self.status = EngineStatus.ACTIVE
        
        while not self.pls_stop_loop.is_set():
            await self.step()
            await asyncio.sleep(0.001)  # Small sleep to yield control
        
        self.is_loop_stopped.set()
        print(f"[{self.stage.value}Engine] Event loop stopped")
    
    async def stop_event_loop(self):
        """Stop the event loop"""
        self.pls_stop_loop.set()
        await self.is_loop_stopped.wait()
    
    async def step(self):
        """Execute one step (to be implemented by subclasses)"""
        raise NotImplementedError


class V0EncodingEngine(V0BaseStageEngine):
    """
    Encoding stage engine
    Processes multimodal inputs and generates vision embeddings
    """
    
    def __init__(
        self,
        encode_prefill_bridge_queue: asyncio.Queue,
        model_path: str,
        num_workers: int = 2,
        **kwargs
    ):
        super().__init__(
            stage=EngineStage.ENCODING,
            model_path=model_path,
            num_workers=num_workers,
            **kwargs
        )
        
        # Bridge queue to prefill stage
        self.encode_prefill_bridge_queue = encode_prefill_bridge_queue
        
        # Vision block manager
        self.vision_block_manager = V0VisionBlockManager(
            stage="encoding",
            max_num_gpu_blocks=3000,  # Separate pool for vision embeddings
            max_num_cpu_blocks=100,
            block_size=self.block_size,
        )
        
        # Initialize scheduler (after block_manager is created)
        self.scheduler = V0StageScheduler(self.stage, block_manager=self.vision_block_manager)
    
    async def initialize(self):
        """Initialize encoding engine"""
        await super().initialize()
        
        # Initialize vision embedding cache on workers
        await asyncio.gather(*[
            worker.init_vecache.remote(
                num_gpu_blocks=3000,
                num_cpu_blocks=100
            ) for worker in self.workers
        ])
    
    async def step(self):
        """Execute one encoding step"""
        import time
        
        # Schedule a batch
        batched_requests = self.scheduler.schedule(max_batch_size=16)
        
        if not batched_requests:
            await asyncio.sleep(0.01)  # No work, sleep longer
            return
        
        # Record encoding start time
        start_time = time.time()
        for req in batched_requests:
            if req.encoding_start_time is None:
                req.encoding_start_time = start_time
        
        # NOTE: We'll allocate vision blocks AFTER we know the actual size of vision embeddings
        # For now, create empty block tables
        block_tables = {req.request_id: [] for req in batched_requests}
        
        # Execute encoding with modality-aware load balancing
        # Use first request in batch to determine worker (all requests in batch should have same modality)
        first_request = batched_requests[0] if isinstance(batched_requests, list) else batched_requests.requests[0]
        worker = self.get_worker_by_modality(first_request)
        if not worker:
            print(f"[Encoding] ❌ No worker available for batch, skipping")
            return
        
        vision_embeddings = await worker.step_encoding.remote(
            batched_requests,
            block_tables
        )
        
        # Record encoding end time
        end_time = time.time()
        for req in batched_requests:
            req.encoding_end_time = end_time
        
        # Allocate vision blocks based on actual vision embeddings size and get block tables
        for request in batched_requests:
            if request.request_id in vision_embeddings:
                result = vision_embeddings[request.request_id]
                # Check if result contains actual vision embeddings tensor
                ve_tensor = None
                if isinstance(result, dict) and 'embeddings' in result:
                    # Result is a dict with 'embeddings', 'mm_kwargs', 'mm_placeholders'
                    ve_tensor = result['embeddings']
                elif hasattr(result, 'shape'):
                    # result is a tensor (vision embeddings)
                    ve_tensor = result
                
                if ve_tensor is not None:
                    # Calculate number of blocks needed for vision embeddings
                    # Each block stores (block_size, hidden_dim) embeddings
                    num_vision_tokens = ve_tensor.shape[0]
                    num_blocks_needed = (num_vision_tokens + self.vision_block_manager.block_size - 1) // self.vision_block_manager.block_size
                    print(f"[Encoding] {request.request_id}: vision_embeddings shape={ve_tensor.shape}, num_tokens={num_vision_tokens}, blocks_needed={num_blocks_needed}")
                    
                    # Allocate blocks
                    self.vision_block_manager.allocate_blocks(request, num_blocks=num_blocks_needed)
                    block_tables[request.request_id] = self.vision_block_manager.get_block_table(request.request_id)
                    print(f"[Encoding] {request.request_id}: allocated vision blocks={block_tables[request.request_id]}")
        
        # Send to prefill stage
        for request in batched_requests:
            # Extract multi_modal_kwargs, multi_modal_placeholders AND vision embeddings tensor
            mm_kwargs = None
            mm_placeholders = None
            ve_tensor = None
            
            if request.request_id in vision_embeddings:
                result = vision_embeddings[request.request_id]
                if isinstance(result, dict):
                    # Extract from dict structure: {'embeddings': tensor, 'multi_modal_kwargs': ..., 'multi_modal_placeholders': ...}
                    if 'embeddings' in result:
                        ve_tensor = result['embeddings']
                    if 'multi_modal_kwargs' in result:
                        mm_kwargs = result['multi_modal_kwargs']
                    if 'multi_modal_placeholders' in result:
                        mm_placeholders = result['multi_modal_placeholders']
                elif hasattr(result, 'shape'):
                    # result is a tensor (vision embeddings)
                    ve_tensor = result
            
            migrating_req = MigratingRequest(
                req=request,
                vision_block_indexes=block_tables[request.request_id],
                vision_embeddings=ve_tensor,  # CRITICAL: Pass vision embeddings directly!
                multi_modal_kwargs=mm_kwargs,  # CRITICAL: Pass extracted mm_kwargs!
                multi_modal_placeholders=mm_placeholders,  # CRITICAL: Pass extracted mm_placeholders!
                source_stage=EngineStage.ENCODING,
                target_stage=EngineStage.PREFILL,
            )
            await self.encode_prefill_bridge_queue.put(migrating_req)
            
            # ✅ FIX: Free vision blocks after sending to prefill
            # Vision embeddings are now in the migrating request, blocks no longer needed
            if request.request_id in block_tables:
                self.vision_block_manager.free_blocks(request.request_id)
            
            # Finish request in scheduler
            self.scheduler.finish_request(request.request_id)


class V0PrefillEngine(V0BaseStageEngine):
    """
    Prefill stage engine
    Processes full sequences and generates KV cache
    """
    
    def __init__(
        self,
        encode_prefill_bridge_queue: asyncio.Queue,
        prefill_decode_bridge_queue: asyncio.Queue,
        model_path: str,
        num_workers: int = 4,
        nccl_transfer_manager: Optional[V0NCCLTransferManager] = None,
        kv_transfer_manager: Optional['V0KVTransferManager'] = None,
        memory_coordinator: Optional['GlobalMemoryCoordinator'] = None,
        **kwargs
    ):
        super().__init__(
            stage=EngineStage.PREFILL,
            model_path=model_path,
            num_workers=num_workers,
            kv_transfer_manager=kv_transfer_manager,
            **kwargs
        )
        
        # Bridge queues
        self.encode_prefill_bridge_queue = encode_prefill_bridge_queue
        self.prefill_decode_bridge_queue = prefill_decode_bridge_queue
        
        # NCCL transfer manager
        self.nccl_transfer_manager = nccl_transfer_manager
        self.kv_transfer_manager = kv_transfer_manager
        
        # ✅ Global memory coordinator
        self.memory_coordinator = memory_coordinator
        
        # Block managers
        self.block_manager = V0BlockManager(
            stage="prefill",
            max_num_gpu_blocks=self.max_num_gpu_blocks,
            max_num_cpu_blocks=self.max_num_cpu_blocks,
            block_size=self.block_size,
        )
        
        # Initialize scheduler (after block_manager is created)
        self.scheduler = V0StageScheduler(self.stage, block_manager=self.block_manager)
        
        self.vision_block_manager = V0VisionBlockManager(
            stage="prefill",
            max_num_gpu_blocks=3000,
            max_num_cpu_blocks=100,
            block_size=self.block_size,
        )
    
    async def initialize(self):
        """Initialize prefill engine"""
        await super().initialize()
        
        # Initialize KV cache and vision embedding cache on workers
        await asyncio.gather(*[
            worker.init_kvcache.remote(
                num_gpu_blocks=self.max_num_gpu_blocks,
                num_cpu_blocks=self.max_num_cpu_blocks
            ) for worker in self.workers
        ])
        
        await asyncio.gather(*[
            worker.init_vecache.remote(
                num_gpu_blocks=3000,
                num_cpu_blocks=100
            ) for worker in self.workers
        ])
    
    async def _receive_from_encoding(self):
        """Receive requests from encoding stage"""
        while not self.encode_prefill_bridge_queue.empty():
            try:
                migrating_req = await asyncio.wait_for(
                    self.encode_prefill_bridge_queue.get(),
                    timeout=0.001
                )
                
                # CRITICAL: Restore multi_modal_kwargs AND multi_modal_placeholders from MigratingRequest
                # (Ray serialization breaks torch tensors)
                if hasattr(migrating_req, 'multi_modal_kwargs') and migrating_req.multi_modal_kwargs:
                    migrating_req.req.multi_modal_kwargs = migrating_req.multi_modal_kwargs
                
                if hasattr(migrating_req, 'multi_modal_placeholders') and migrating_req.multi_modal_placeholders:
                    migrating_req.req.multi_modal_placeholders = migrating_req.multi_modal_placeholders
                else:
                    migrating_req.req.multi_modal_placeholders = None
                
                # CRITICAL: Store vision embeddings in worker's ve_cache
                if hasattr(migrating_req, 'vision_embeddings') and migrating_req.vision_embeddings is not None:
                    ve_tensor = migrating_req.vision_embeddings
                    print(f"[Prefill] Received vision embeddings for {migrating_req.req.request_id}: shape={ve_tensor.shape}")
                    # Store in first worker's ve_cache
                    worker = self.workers[0]
                    await worker.write_vision_blocks.remote(migrating_req.req.request_id, ve_tensor)
                    print(f"[Prefill] Stored vision embeddings in worker ve_cache for {migrating_req.req.request_id}")
                
                # Add to scheduler
                self.scheduler.add_request(migrating_req.req)
                print(f"[Prefill] Received request {migrating_req.req.request_id} from encoding, added to scheduler")
                
                # Migrate vision embeddings
                # CRITICAL: Currently we just allocate new blocks and assume data is there
                # In reality, we need to copy vision embedding data from encoding workers
                # This requires either:
                # 1. CUDA IPC zero-copy (best, needs CUDA extension)
                # 2. Direct GPU-to-GPU copy via PyTorch
                # 3. Ray object store (slower but works)
                
                # Allocate blocks and transfer vision embeddings
                if migrating_req.vision_block_indexes:
                    # Allocate blocks for vision embeddings in prefill stage
                    self.vision_block_manager.allocate_blocks(migrating_req.req)
                    
                    # Transfer vision embeddings from encoding workers
                    if hasattr(self, 'encoding_workers') and self.encoding_workers:
                        await self._transfer_vision_embeddings_nccl(migrating_req, self.encoding_workers)
                    
            except asyncio.TimeoutError:
                break
    
    async def _transfer_vision_embeddings_nccl(self, migrating_req, encoding_workers):
        """
        Transfer vision embeddings via NCCL from encoding to prefill workers
        
        Args:
            migrating_req: Migration request containing source blocks
            encoding_workers: List of encoding worker references
        """
        if not self.nccl_transfer_manager:
            print("[Warning] NCCL transfer manager not available, skipping vision embedding transfer")
            return
        
        # Get source and target workers
        src_worker_id = migrating_req.source_worker_id if migrating_req.source_worker_id is not None else 0
        dst_worker_id = 0  # Use first prefill worker
        
        request_id = migrating_req.req.request_id
        
        try:
            # Get workers
            src_worker = encoding_workers[src_worker_id]
            dst_worker = self.workers[dst_worker_id]
            
            # Extract from source (by request_id)
            ve_data = await src_worker.extract_vision_blocks.remote(request_id)
            
            # Write to destination (by request_id)
            # TODO: Use actual NCCL transfer for better performance
            await dst_worker.write_vision_blocks.remote(request_id, ve_data)
            
            print(f"[Transfer] Transferred vision embeddings for request {request_id} (shape: {ve_data.shape}) from encoding to prefill")
            
        except Exception as e:
            print(f"[Error] Vision embedding transfer failed: {e}")
            raise
    
    async def step(self):
        """Execute one prefill step"""
        # ========================================================================
        # ✅ BACKPRESSURE: Pause when Decode is overloaded
        # ========================================================================
        # KEY INSIGHT: When Decode memory is high (>92%), Prefill should slow down
        # This prevents:
        # 1. GPU memory bus contention (Prefill compute vs NCCL transfer)
        # 2. Decode getting overwhelmed with new requests
        # 3. KV transfer slowdown (325ms → 5ms when Prefill is idle)
        should_pause_new = False
        if self.memory_coordinator:
            status = self.memory_coordinator.get_memory_status()
            if status and status.usage_ratio > 0.92:
                # Decode memory is high
                running_count = self.scheduler.num_running_requests()
                waiting_count = self.scheduler.num_waiting_requests()
                
                if running_count == 0 and waiting_count > 0:
                    # We have waiting but no running → Decode is blocking us
                    # Pause to let Decode catch up
                    print(f"[Prefill] ⏸️  Paused: Decode memory high ({status.usage_ratio:.1%}), "
                          f"waiting for Decode to free up (we have {waiting_count} waiting)")
                    await asyncio.sleep(0.05)
                    return
                elif running_count > 0:
                    # We have running requests, let them finish but don't add new ones
                    # This reduces GPU contention for KV transfer
                    should_pause_new = True
                    if not hasattr(self, '_backpressure_logged') or not self._backpressure_logged:
                        print(f"[Prefill] 🐢 Slowed: Decode memory high ({status.usage_ratio:.1%}), "
                              f"completing {running_count} running, not starting new")
                        self._backpressure_logged = True
            else:
                self._backpressure_logged = False
        
        # Receive from encoding stage
        await self._receive_from_encoding()
        
        # Schedule a batch
        # ✅ BACKPRESSURE: If Decode is overloaded, only schedule running requests (no new)
        if should_pause_new:
            # Only process currently running requests, don't add new ones from waiting queue
            running_reqs = list(self.scheduler.running_requests.values())
            if running_reqs:
                batched_requests = BatchedRequests(requests=running_reqs[:32])
            else:
                batched_requests = None
        else:
            # Normal: schedule as usual (running + new from waiting)
            batched_requests = self.scheduler.schedule(max_batch_size=32)
        
        if not batched_requests:
            # Check queue status
            if self.scheduler.num_waiting_requests() > 0 or self.scheduler.num_running_requests() > 0:
                print(f"[Prefill] Scheduler has {self.scheduler.num_waiting_requests()}w/{self.scheduler.num_running_requests()}r but no batch scheduled")
            await asyncio.sleep(0.01)
            return
        
        print(f"[Prefill] Processing batch of {len(batched_requests.requests)} requests")
        
        # ✅ MEMORY-AWARE: Try to allocate KV cache blocks
        try:
            self.block_manager.allocate_blocks_batched(batched_requests)
        except AssertionError as e:
            # Not enough memory for initial allocation
            print(f"[Prefill] ⚠️  Not enough memory for batch allocation: {e}")
            print(f"[Prefill]    Putting requests back to waiting queue and retrying later")
            # Put all requests back to waiting queue
            for req in batched_requests.requests:
                if req.request_id in self.scheduler.running_requests:
                    self.scheduler.running_requests.pop(req.request_id)
                    self.scheduler.waiting_queue.append(req)
            await asyncio.sleep(0.1)  # Wait for memory to free up
            return
        
        # ========================================================================
        # CRITICAL: Expand blocks to account for vision tokens (MEMORY-AWARE)
        # ========================================================================
        worker = self.workers[0]
        ve_cache_keys = await worker.get_ve_cache_keys.remote()
        
        requests_to_skip = []
        for req in batched_requests.requests:
            if req.request_id in ve_cache_keys:
                ve_shape = await worker.get_ve_shape.remote(req.request_id)
                if ve_shape:
                    num_vision_tokens = ve_shape[0]  # e.g., 216
                    original_len = len(req.prompt_token_ids)  # e.g., 19
                    # Expanded length: remove 1 placeholder, add vision_tokens
                    expanded_len = original_len - 1 + num_vision_tokens
                    
                    # Calculate blocks needed for expansion
                    current_blocks = len(self.block_manager.block_table.get(req.request_id, []))
                    blocks_needed = (expanded_len + self.block_manager.block_size - 1) // self.block_manager.block_size
                    additional_blocks = blocks_needed - current_blocks
                    
                    # ✅ MEMORY-AWARE: Check if we have enough blocks
                    if additional_blocks > 0:
                        avail_blocks = self.block_manager.get_num_avail_gpu_blocks()
                        if avail_blocks < additional_blocks:
                            print(f"[Prefill] ⚠️  Not enough blocks to expand {req.request_id} for vision tokens "
                                  f"(need {additional_blocks}, avail {avail_blocks}), skipping this request")
                            requests_to_skip.append(req.request_id)
                            continue
                    
                    # Safe to expand
                    self.block_manager.expand_blocks_for_seq_len(req.request_id, expanded_len)
                    print(f"[Prefill-Engine] Expanded blocks for {req.request_id}: seq_len {original_len} -> {expanded_len}")
        
        # Remove requests that couldn't be expanded
        if requests_to_skip:
            batched_requests.requests = [
                req for req in batched_requests.requests 
                if req.request_id not in requests_to_skip
            ]
            # Put skipped requests back to waiting queue
            for req_id in requests_to_skip:
                for req in self.scheduler.running_requests.values():
                    if req.request_id == req_id:
                        self.scheduler.running_requests.pop(req_id, None)
                        self.scheduler.waiting_queue.append(req)
                        break
            print(f"[Prefill] 🔄 Skipped {len(requests_to_skip)} requests due to memory, will retry: {requests_to_skip[:3]}{'...' if len(requests_to_skip) > 3 else ''}")
        
        # If all requests were skipped, return early
        if not batched_requests.requests:
            print("[Prefill] ⏸️  All requests skipped due to memory pressure, waiting...")
            await asyncio.sleep(0.05)
            return
        
        # Get block tables (after expansion)
        kv_block_tables = {
            req.request_id: self.block_manager.get_block_table(req.request_id)
            for req in batched_requests
        }
        
        vision_block_tables = {
            req.request_id: self.vision_block_manager.get_block_table(req.request_id)
            for req in batched_requests
        }
        
        # Record prefill start time
        import time
        start_time = time.time()
        for req in batched_requests:
            if req.prefill_start_time is None:
                req.prefill_start_time = start_time
        
        # ✅ MODALITY-AWARE: Select worker based on request modality
        # Use the first request's modality (batch should be homogeneous)
        first_request = batched_requests.requests[0]
        worker = self.get_worker_by_modality(first_request)
        
        if not worker:
            print(f"[Prefill] ❌ No worker available for modality, batch size={len(batched_requests.requests)}")
            # Put requests back to waiting queue
            for req in batched_requests.requests:
                if req.request_id in self.scheduler.running_requests:
                    self.scheduler.running_requests.pop(req.request_id)
                    self.scheduler.waiting_queue.append(req)
            await asyncio.sleep(0.1)
            return
        
        # Execute prefill with modality-aware worker selection
        result = await worker.step_prefill.remote(
            batched_requests,
            kv_block_tables,
            vision_block_tables
        )
        
        # Record prefill end time
        end_time = time.time()
        for req in batched_requests:
            req.prefill_end_time = end_time
        
        # Unpack result (outputs and expanded_tokens_map)
        if isinstance(result, tuple) and len(result) == 2:
            outputs, expanded_tokens_map = result
        else:
            # Fallback for old API
            outputs = result
            expanded_tokens_map = {}
        
        # CRITICAL: Update request with expanded prompt_token_ids from worker
        for request in batched_requests.requests:
            if request.request_id in expanded_tokens_map:
                request.prompt_token_ids = expanded_tokens_map[request.request_id]
                # Prompt tokens updated (debug removed)
        
        # CRITICAL: Update request with generated tokens before sending to decode
        output_dict = {output.request_id: output for output in outputs}
        
        # ========================================================================
        # ✅ BACKPRESSURE MECHANISM (Advisory only, Decode will reject if needed)
        # ========================================================================
        # NOTE: We don't block here! We still send requests, and Decode will
        # reject them if it can't accept. This prevents deadlock where Prefill
        # holds memory but can't free it because it can't send to Decode.
        if self.memory_coordinator:
            memory_status = self.memory_coordinator.get_memory_status()
            if memory_status and not memory_status.can_accept_new_requests:
                print(f"[Prefill] ⚠️  Decode memory pressure detected ({memory_status.usage_ratio:.1%}), "
                      f"but continuing to send (Decode will reject if needed)")
        
        # Send to decode stage via bridge queue
        for request in batched_requests.requests:
            output = output_dict.get(request.request_id)
            
            if output and not output.finished:
                # Send to decode with KV cache blocks
                migrating_req = MigratingRequest(
                    req=request,
                    kv_block_indexes=kv_block_tables[request.request_id],
                    output_token_ids=output.output_token_ids,
                    expanded_prompt_token_ids=request.prompt_token_ids,  # CRITICAL: Save expanded tokens!
                    multi_modal_kwargs=request.multi_modal_kwargs,
                    multi_modal_placeholders=getattr(request, 'multi_modal_placeholders', None),
                    source_stage=EngineStage.PREFILL,
                    target_stage=EngineStage.DECODING,
                )
                await self.prefill_decode_bridge_queue.put(migrating_req)
                
                # ✅ FIX: Free KV blocks after sending to decode
                # KV cache data is transferred to decode stage, prefill no longer needs these blocks
                self.block_manager.free_blocks(request.request_id)
                
                # CRITICAL: Finish request in prefill scheduler (it will be scheduled in decode)
                self.scheduler.finish_request(request.request_id)
            else:
                # Request finished in prefill (EOS or error)
                self.scheduler.finish_request(request.request_id)
                print(f"[Prefill] Request {request.request_id} finished in prefill")
        
        print(f"[Prefill] Successfully completed prefill for {len(batched_requests.requests)} requests")


class V0DecodingEngine(V0BaseStageEngine):
    """
    Decoding stage engine
    Autoregressive generation
    """
    
    def __init__(
        self,
        prefill_decode_bridge_queue: asyncio.Queue,
        model_path: str,
        num_workers: int = 2,
        output_callback: Optional[Callable[[StepOutput], None]] = None,
        nccl_transfer_manager: Optional[V0NCCLTransferManager] = None,
        kv_transfer_manager: Optional[V0KVTransferManager] = None,
        memory_coordinator: Optional['GlobalMemoryCoordinator'] = None,
        **kwargs
    ):
        super().__init__(
            stage=EngineStage.DECODING,
            model_path=model_path,
            num_workers=num_workers,
            kv_transfer_manager=kv_transfer_manager,
            **kwargs
        )
        
        # Bridge queue from prefill stage
        self.prefill_decode_bridge_queue = prefill_decode_bridge_queue
        
        # Output callback
        self.output_callback = output_callback
        
        # NCCL transfer manager
        self.nccl_transfer_manager = nccl_transfer_manager
        
        # ✅ Global memory coordinator
        self.memory_coordinator = memory_coordinator
        
        # Block manager
        self.block_manager = V0BlockManager(
            stage="decoding",
            max_num_gpu_blocks=self.max_num_gpu_blocks,
            max_num_cpu_blocks=self.max_num_cpu_blocks,
            block_size=self.block_size,
        )
        
        # ✅ Block predictor for intelligent pre-allocation
        self.block_predictor = BlockPredictor(block_size=self.block_size)
        
        # Initialize scheduler (after block_manager is created)
        self.scheduler = V0StageScheduler(self.stage, block_manager=self.block_manager)
    
    async def initialize(self):
        """Initialize decoding engine"""
        await super().initialize()
        
        # Initialize KV cache on workers
        await asyncio.gather(*[
            worker.init_kvcache.remote(
                num_gpu_blocks=self.max_num_gpu_blocks,
                num_cpu_blocks=self.max_num_cpu_blocks
            ) for worker in self.workers
        ])
    
    async def _receive_from_prefill(self):
        """Receive requests from prefill stage"""
        while not self.prefill_decode_bridge_queue.empty():
            try:
                migrating_req = await asyncio.wait_for(
                    self.prefill_decode_bridge_queue.get(),
                    timeout=0.001
                )
                
                # CRITICAL: Restore expanded prompt_token_ids from MigratingRequest
                # (Ray serialization breaks request object updates)
                if migrating_req.expanded_prompt_token_ids:
                    migrating_req.req.prompt_token_ids = migrating_req.expanded_prompt_token_ids
                
                # CRITICAL: Restore output_token_ids from MigratingRequest
                # (request object updates don't survive Ray serialization)
                if migrating_req.output_token_ids:
                    migrating_req.req.output_token_ids = migrating_req.output_token_ids
                
                # CRITICAL: Restore multi_modal_kwargs AND multi_modal_placeholders from MigratingRequest
                # (Ray serialization breaks torch tensors)
                if hasattr(migrating_req, 'multi_modal_kwargs') and migrating_req.multi_modal_kwargs:
                    migrating_req.req.multi_modal_kwargs = migrating_req.multi_modal_kwargs
                
                if hasattr(migrating_req, 'multi_modal_placeholders') and migrating_req.multi_modal_placeholders:
                    migrating_req.req.multi_modal_placeholders = migrating_req.multi_modal_placeholders
                else:
                    migrating_req.req.multi_modal_placeholders = None
                
                # Migrate KV cache from prefill to decode stage
                # CRITICAL: This is the most important migration!
                # The KV cache contains all the computed key-value pairs from prefill
                # We MUST transfer this data, otherwise decode will fail
                
                if migrating_req.kv_block_indexes:
                    # ✅ INTELLIGENT INITIAL ALLOCATION
                    # Instead of only allocating enough for current tokens (prefill's blocks),
                    # use BlockPredictor to pre-allocate for expected output tokens
                    
                    # Minimum: match prefill's KV cache blocks
                    min_blocks_needed = len(migrating_req.kv_block_indexes)
                    
                    # Predict total blocks needed (including future output)
                    prompt_tokens = len(migrating_req.req.prompt_token_ids)
                    output_tokens = len(migrating_req.req.output_token_ids) if migrating_req.req.output_token_ids else 1
                    
                    predicted_blocks = self.block_predictor.predict_initial_blocks(
                        prompt_tokens=prompt_tokens,
                        max_tokens=migrating_req.req.max_tokens,
                        has_vision=bool(migrating_req.req.multi_modal_data)
                    )
                    
                    # Use the larger of min_blocks and predicted_blocks
                    num_blocks_to_allocate = max(min_blocks_needed, predicted_blocks)
                    
                    avail_blocks = self.block_manager.get_num_avail_gpu_blocks()
                    
                    # Try to allocate predicted amount, fall back to minimum if needed
                    if avail_blocks < num_blocks_to_allocate:
                        if avail_blocks >= min_blocks_needed:
                            # Can allocate minimum, use that
                            num_blocks_to_allocate = min_blocks_needed
                            print(f"[Decode-BlockPredictor] {migrating_req.req.request_id}: "
                                  f"wanted {predicted_blocks} blocks, only {avail_blocks} available, "
                                  f"allocating minimum {min_blocks_needed}")
                        else:
                            # Not enough even for minimum
                            print(f"[Decode] ⚠️  Not enough GPU blocks for {migrating_req.req.request_id} "
                                  f"(need {min_blocks_needed}, avail {avail_blocks}), putting back to queue")
                            await self.prefill_decode_bridge_queue.put(migrating_req)
                            break
                    else:
                        # Success! Pre-allocated extra blocks
                        if num_blocks_to_allocate > min_blocks_needed:
                            print(f"[Decode-BlockPredictor] {migrating_req.req.request_id}: "
                                  f"pre-allocated {num_blocks_to_allocate} blocks "
                                  f"(min={min_blocks_needed}, extra={num_blocks_to_allocate-min_blocks_needed}) "
                                  f"for {prompt_tokens} tokens + {migrating_req.req.max_tokens} max_output")
                    
                    # Allocate the blocks
                    self.block_manager.allocate_blocks(
                        migrating_req.req,
                        num_blocks=num_blocks_to_allocate
                    )
                    
                    # Transfer KV cache data from prefill workers
                    if hasattr(self, 'prefill_workers') and self.prefill_workers:
                        await self._transfer_kv_cache_nccl(migrating_req, self.prefill_workers)
                
                # Add to scheduler (after successful allocation)
                self.scheduler.add_request(migrating_req.req)
                
            except asyncio.TimeoutError:
                break
    
    async def _transfer_kv_cache_nccl(self, migrating_req, prefill_workers):
        """
        Transfer KV cache via Ray-based transfer from prefill to decoding workers
        CRITICAL: This ensures decode has the correct KV cache from prefill!
        
        Args:
            migrating_req: Migration request containing source KV blocks
            prefill_workers: List of prefill worker references
        """
        # Get source and target workers
        src_worker_id = migrating_req.source_worker_id if migrating_req.source_worker_id is not None else 0
        dst_worker_id = 0  # Use first decode worker
        
        # Get block mappings
        src_blocks = migrating_req.kv_block_indexes
        dst_blocks_full = self.block_manager.get_block_table(migrating_req.req.request_id)
        
        # ✅ CRITICAL FIX: Only use the first len(src_blocks) destination blocks
        # We may have pre-allocated more blocks than src has data for!
        # Extra blocks will be used for future decode tokens.
        dst_blocks = dst_blocks_full[:len(src_blocks)]
        
        if not src_blocks or not dst_blocks:
            print("[Warning] No KV blocks to transfer")
            return
        
        if self.kv_transfer_manager:
            # ✅ NEW: Use global ranks instead of stage/worker_id
            src_rank = self.nccl_coordinator.get_rank("prefill", src_worker_id)
            dst_rank = self.nccl_coordinator.get_rank("decoding", dst_worker_id)
            
            if src_rank >= 0 and dst_rank >= 0:
                success = await self.kv_transfer_manager.transfer_kv_cache(
                    request_id=migrating_req.req.request_id,
                    src_rank=src_rank,
                    dst_rank=dst_rank,
                    src_blocks=src_blocks,
                    dst_blocks=dst_blocks,
                )
                if success:
                    print(f"[Transfer] ✓ NCCL KV transfer complete for {migrating_req.req.request_id}")
                    return
                else:
                    print(f"[Transfer] ⚠ NCCL transfer failed for {migrating_req.req.request_id}, falling back to Ray object transfer")
            else:
                print(f"[Transfer] ⚠ Invalid ranks: src_rank={src_rank}, dst_rank={dst_rank}, falling back")
        
        try:
            src_worker = prefill_workers[src_worker_id]
            dst_worker = self.workers[dst_worker_id]
            
            kv_data = await src_worker.extract_kv_blocks.remote(src_blocks)
            await dst_worker.write_kv_blocks.remote(dst_blocks, kv_data)
            
            print(f"[Transfer] ✓ (fallback) Transferred {len(src_blocks)} KV blocks from prefill to decode")
            print(f"           Request {migrating_req.req.request_id} now has valid KV cache in decode stage")
            
        except Exception as e:
            print(f"[Error] Fallback KV cache transfer FAILED: {e}")
            print(f"        This will cause incorrect decoding!")
            raise
    
    async def step(self):
        """Execute one decode step"""
        # ========================================================================
        # ✅ SMART MEMORY MANAGEMENT (improved strategy)
        # ========================================================================
        if self.memory_coordinator:
            total_blocks = self.block_manager.max_num_gpu_blocks
            free_blocks = self.block_manager.get_num_avail_gpu_blocks()
            memory_status = await self.memory_coordinator.update_memory_status(total_blocks, free_blocks)
            
            # 🎯 KEY INSIGHT: Only preempt if there are NO waiting requests
            # If we have waiting requests, let running requests finish naturally
            # to free up memory, then schedule new ones
            waiting_count = self.scheduler.num_waiting_requests()
            running_count = self.scheduler.num_running_requests()
            
            # ✅ SMART PREEMPTION LOGIC:
            # Case 1: High memory usage + waiting queue → DO NOT PREEMPT, let them finish
            # Case 2: High memory usage + no waiting + all stuck → PREEMPT to break deadlock
            if memory_status.needs_preemption and running_count > 0:
                if waiting_count > 0:
                    # Have waiting requests, let running ones finish naturally
                    print(f"[Decode] ⏳ High memory pressure ({memory_status.usage_ratio:.1%}) but "
                          f"{waiting_count} requests waiting - will let {running_count} running complete naturally")
                else:
                    # No waiting requests, might be stuck, consider preemption
                    # But ONLY if memory is EXTREMELY critical (99.5%+)
                    if memory_status.usage_ratio > 0.995:
                        running_requests_info = [
                            {
                                'request_id': req.request_id,
                                'output_token_count': len(req.output_token_ids)
                            }
                            for req in self.scheduler.running_requests.values()
                        ]
                        
                        # Conservative preemption: only 10-20% of running requests
                        num_to_preempt = max(1, len(running_requests_info) // 10)
                        preempt_ids = await self.memory_coordinator.select_requests_to_preempt(
                            running_requests_info, num_to_preempt
                        )
                        
                        # Force-finish preempted requests
                        for request_id in preempt_ids:
                            if request_id in self.scheduler.running_requests:
                                request = self.scheduler.running_requests[request_id]
                                request.is_finished = True
                                
                                # ✅ Call output callback to mark request as completed
                                if self.output_callback and len(request.output_token_ids) > 0:
                                    output = StepOutput(
                                        request_id=request_id,
                                        output_token_ids=request.output_token_ids,
                                        finished=True
                                    )
                                    self.output_callback(output)
                                
                                self.block_manager.free_blocks(request_id)
                                self.scheduler.finish_request(request_id)
                                print(f"[Decode] 🔪 PREEMPTED: {request_id} with {len(request.output_token_ids)} tokens (freed memory)")
                    else:
                        print(f"[Decode] ℹ️  Memory at {memory_status.usage_ratio:.1%}, but not critical enough for preemption")
        
        # Receive from prefill stage
        await self._receive_from_prefill()
        
        # Debug: check scheduler status
        waiting = self.scheduler.num_waiting_requests()
        running = self.scheduler.num_running_requests()
        if waiting > 0 or running > 0:
            print(f"[Decode] Scheduler before schedule(): {waiting}w/{running}r")
        
        # Schedule a batch
        # ✅ CRITICAL: Limit batch size for multimodal workloads
        # Each multimodal request needs ~50-60 blocks, so 32 requests need ~2000 blocks
        # With 11k blocks available, we can support ~180 requests theoretically,
        # but in practice need to limit due to activation memory
        batched_requests = self.scheduler.schedule(max_batch_size=32)  # Conservative for multimodal
        
        if not batched_requests:
            if waiting > 0 or running > 0:
                print(f"[Decode] ⚠ Scheduler has requests but schedule() returned None!")
            await asyncio.sleep(0.01)
            return
        
        print(f"[Decode] Processing batch of {len(batched_requests.requests)} requests")
        
        
        # ========================================================================
        # MEMORY-AWARE BLOCK EXPANSION (inspired by vLLM's approach)
        # ========================================================================
        # vLLM philosophy:
        # 1. Prioritize completing running requests (avoid starvation)
        # 2. When memory is tight, reduce batch size dynamically
        # 3. Keep removed requests in running state for next iteration
        # ========================================================================
        
        requests_that_can_run = []
        requests_delayed = []
        
        # Block expansion check (debug prints removed for performance)
        
        for idx, request in enumerate(batched_requests.requests):
            if request.request_id not in self.block_manager.block_table:
                # First time seeing this request in decode, should have been allocated in _receive_from_prefill
                print(f"[Decode] WARNING: {request.request_id} has no blocks allocated!")
                continue
            
            # ✅ INTELLIGENT BLOCK PREDICTION (vLLM-inspired)
            # Use predictor to estimate total blocks needed, reducing frequent re-allocation
            actual_prompt_tokens = len(request.prompt_token_ids)
            current_output_tokens = len(request.output_token_ids)
            
            # Predict total blocks needed (includes safety margin)
            predicted_blocks = self.block_predictor.predict_total_blocks(
                request_id=request.request_id,
                current_prompt_tokens=actual_prompt_tokens,
                current_output_tokens=current_output_tokens,
                max_tokens=request.max_tokens,
                has_vision=bool(request.multi_modal_data)
            )
            
            # Use prediction for allocation
            blocks_needed = predicted_blocks
            current_blocks = len(self.block_manager.block_table[request.request_id])
            
            # Check if expansion is needed
            if current_blocks < blocks_needed:
                additional_blocks_needed = blocks_needed - current_blocks
                avail_blocks = self.block_manager.get_num_avail_gpu_blocks()
                
                # Log block expansion (only every 10 requests to reduce noise)
                if not hasattr(self, '_block_expansion_count'):
                    self._block_expansion_count = 0
                self._block_expansion_count += 1
                if self._block_expansion_count <= 5:  # Log first 5 expansions
                    print(f"[Decode-BlockPredictor] {request.request_id}: predicted={blocks_needed}, "
                          f"current={current_blocks}, expand_by={additional_blocks_needed}, "
                          f"tokens={actual_prompt_tokens}+{current_output_tokens}, max={request.max_tokens}")
                
                # ✅ MEMORY-AWARE: Check if we have enough blocks for expansion
                if avail_blocks < additional_blocks_needed:
                    # Not enough blocks, delay this request
                    requests_delayed.append(request.request_id)
                    continue
                
                # Allocate additional blocks
                new_blocks = self.block_manager._get_free_blocks(additional_blocks_needed, BlockLocation.GPU)
                self.block_manager.block_table[request.request_id].extend(new_blocks)
            
            # This request can run in current iteration
            requests_that_can_run.append(request)
        
        # Update batch to only include requests that can run
        if requests_delayed:
            print(f"[Decode] 🔄 Delayed {len(requests_delayed)} requests due to memory, "
                  f"will retry next iteration: {requests_delayed[:3]}{'...' if len(requests_delayed) > 3 else ''}")
            batched_requests.requests = requests_that_can_run
        
        # If all requests were delayed, skip this step but don't crash
        if not batched_requests.requests:
            print("[Decode] ⏸️  All requests delayed due to memory pressure, waiting for blocks to free...")
            await asyncio.sleep(0.05)  # Brief pause to let other stages finish and free blocks
            return
        
        # Get block tables
        kv_block_tables = {
            req.request_id: self.block_manager.get_block_table(req.request_id)
            for req in batched_requests
        }
        
        # Record decoding start time (only for first iteration)
        import time
        for req in batched_requests:
            if req.decoding_start_time is None:
                req.decoding_start_time = time.time()
        
        # ✅ MODALITY-AWARE: Select worker based on request modality
        # For decoding, batch may have mixed modalities, so we group them
        # Simplified version: use first request's modality (TODO: split batch by modality for finer control)
        first_request = batched_requests.requests[0]
        worker = self.get_worker_by_modality(first_request)
        
        if not worker:
            print(f"[Decode] ❌ No worker available for modality, batch size={len(batched_requests.requests)}")
            # Keep requests in running queue and try again next iteration
            await asyncio.sleep(0.01)
            return
        
        # Execute decode with modality-aware worker selection - measure ONLY GPU compute time
        compute_start = time.time()
        outputs = await worker.step_decode.remote(
            batched_requests,
            kv_block_tables
        )
        compute_end = time.time()
        compute_time = compute_end - compute_start
        
        # ✅ FIX: 平摊计算时间到batch中的每个request
        # 因为decode是batch processing，所有requests是并行处理的
        if len(batched_requests) > 0:
            compute_time_per_request = compute_time / len(batched_requests)
            for req in batched_requests:
                req.total_decode_compute_time += compute_time_per_request
        
        # Process outputs
        finished_count = 0
        request_dict = {req.request_id: req for req in batched_requests}
        
        for output in outputs:
            # CRITICAL: Update request with new generated tokens
            if output.request_id in request_dict:
                request = request_dict[output.request_id]
                request.output_token_ids = output.output_token_ids
            
            # Call output callback if provided
            if self.output_callback:
                self.output_callback(output)
            
            # Check if output is finished
            if output.finished:
                # Record metrics for this completed request
                if output.request_id in request_dict:
                    request = request_dict[output.request_id]
                    self._record_request_metrics(request)
                
                # Record statistics for block predictor
                self.block_predictor.record_request_finish(
                    output.request_id,
                    len(output.output_token_ids)
                )
                
                # Free blocks
                self.block_manager.free_blocks(output.request_id)
                self.scheduler.finish_request(output.request_id)
                finished_count += 1
        
        if finished_count > 0:
            # Log predictor statistics periodically
            if not hasattr(self, '_finished_total'):
                self._finished_total = 0
            self._finished_total += finished_count
            
            # Every 10 finished requests, show predictor stats
            if self._finished_total % 10 == 0:
                stats = self.block_predictor.get_statistics()
                print(f"[Decode-BlockPredictor] Stats after {self._finished_total} requests: "
                      f"avg_output={stats['avg_output_tokens']:.1f} tokens, "
                      f"samples={stats['samples_collected']}")


# Export
__all__ = [
    'EngineStage',
    'V0EncodingEngine',
    'V0PrefillEngine',
    'V0DecodingEngine',
]

