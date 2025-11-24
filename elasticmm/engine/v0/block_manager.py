"""
Block Manager for v0 engine
Based on EPD-Disaggregation's BlockManager implementation
Manages KV cache and vision embeddings at block level
"""

import asyncio
from typing import List, Dict, Callable, Optional, Any
from enum import Enum
import torch

from elasticmm.engine.v0.utils import Request, BatchedRequests


class BlockLocation(Enum):
    """The location of a block"""
    GPU = "gpu"
    CPU = "cpu"


class V0BlockManager:
    """
    Block Manager for KV cache
    Maintains key-value cache at block level with GPU/CPU swapping support
    
    Based on EPD's BlockManager implementation with adaptations for ElasticMM
    Enhanced with full swap operations and CUDA event synchronization
    """
    
    def __init__(
        self,
        stage: str,  # 'encoding', 'prefill', 'decoding'
        max_num_gpu_blocks: int,
        max_num_cpu_blocks: int,
        block_size: int,
        engine_remote_call_all_workers_async: Optional[Callable] = None,
    ):
        """
        Args:
            stage: Engine stage name
            max_num_gpu_blocks: Maximum number of GPU blocks
            max_num_cpu_blocks: Maximum number of CPU blocks
            block_size: Number of tokens per block
            engine_remote_call_all_workers_async: Function to call workers asynchronously
        """
        self.stage = stage
        self.max_num_gpu_blocks = max_num_gpu_blocks
        self.max_num_cpu_blocks = max_num_cpu_blocks
        self.block_size = block_size
        self.engine_remote_call_all_workers_async = engine_remote_call_all_workers_async
        
        # Free block pools
        self.free_gpu_blocks_list = list(range(max_num_gpu_blocks))
        self.free_cpu_blocks_list = list(range(max_num_cpu_blocks))
        
        # Blocks currently being swapped (EPD's approach)
        self.swapping_gpu_blocks_list = []
        self.swapping_cpu_blocks_list = []
        
        # Block table: request_id => [block0_id, block1_id, ...]
        self.block_table: Dict[str, List[int]] = {}
        
        # Request location: request_id => BlockLocation
        self.request_location: Dict[str, BlockLocation] = {}
        
        # Enhanced: Swap event tracking (from EPD)
        self.swap_event_table: Dict[str, Any] = {}  # request_id => CUDA event
        self.latest_swap_in_event: Optional[Any] = None
        self.latest_swap_out_event: Optional[Any] = None
        
        # Enhanced: Statistics tracking
        self.swap_count = 0
        self.total_swap_time = 0.0
    
    def _reset_free_blocks(self):
        """
        Reset free block lists after dynamic resizing
        Called after profiling GPU memory to update block counts
        """
        # Clear and reinitialize free block lists with new sizes
        self.free_gpu_blocks_list = list(range(self.max_num_gpu_blocks))
        self.free_cpu_blocks_list = list(range(self.max_num_cpu_blocks))
        
        # Clear swapping lists (should be empty at init anyway)
        self.swapping_gpu_blocks_list = []
        self.swapping_cpu_blocks_list = []
        
        # Note: block_table and request_location are NOT cleared
        # as they may contain allocated blocks from before resizing
        # This method should only be called during initialization
        
        print(f"[BlockManager-{self.stage}] Reset free blocks: {self.max_num_gpu_blocks} GPU, {self.max_num_cpu_blocks} CPU")
    
    def get_num_avail_gpu_blocks(self) -> int:
        """Get the number of available GPU blocks"""
        return len(self.free_gpu_blocks_list) + len(self.swapping_gpu_blocks_list)
    
    def get_num_avail_cpu_blocks(self) -> int:
        """Get the number of available CPU blocks"""
        return len(self.free_cpu_blocks_list) + len(self.swapping_cpu_blocks_list)
    
    def _get_free_blocks(self, num_blocks: int, location: BlockLocation) -> List[int]:
        """
        Get free blocks from the free block pool indicated by `location`
        
        Args:
            num_blocks: Number of blocks to allocate
            location: GPU or CPU
            
        Returns:
            List of block IDs
        """
        if location == BlockLocation.GPU:
            num_avail_blocks = self.get_num_avail_gpu_blocks()
            assert num_avail_blocks >= num_blocks, \
                f"Not enough free blocks on GPU, requested {num_blocks}, available {num_avail_blocks}"
            
            if len(self.free_gpu_blocks_list) < num_blocks:
                # Wait for swap-out operations to complete
                if self.engine_remote_call_all_workers_async:
                    self.engine_remote_call_all_workers_async("wait_for_all_swap_out")
                self.free_gpu_blocks_list += self.swapping_gpu_blocks_list
                self.swapping_gpu_blocks_list = []
            
            blocks = self.free_gpu_blocks_list[:num_blocks]
            self.free_gpu_blocks_list = self.free_gpu_blocks_list[num_blocks:]
        else:
            num_avail_blocks = self.get_num_avail_cpu_blocks()
            assert num_avail_blocks >= num_blocks, \
                f"Not enough free blocks on CPU, requested {num_blocks}, available {num_avail_blocks}"
            
            if len(self.free_cpu_blocks_list) < num_blocks:
                # Wait for swap-in operations to complete
                if self.engine_remote_call_all_workers_async:
                    self.engine_remote_call_all_workers_async("wait_for_all_swap_in")
                self.free_cpu_blocks_list += self.swapping_cpu_blocks_list
                self.swapping_cpu_blocks_list = []
            
            blocks = self.free_cpu_blocks_list[:num_blocks]
            self.free_cpu_blocks_list = self.free_cpu_blocks_list[num_blocks:]
        
        return blocks
    
    def get_allocated_num_blocks(self, request_id: str) -> int:
        """Get the number of allocated blocks for a request"""
        return len(self.block_table.get(request_id, []))
    
    def get_location(self, request_id: str) -> Optional[BlockLocation]:
        """Get the KV cache blocks location of a request"""
        return self.request_location.get(request_id, None)
    
    def get_num_blocks_needed(self, request: Request) -> int:
        """
        Calculate the number of blocks needed for a request
        
        Args:
            request: Request object
            
        Returns:
            Number of blocks needed
        """
        total_len = request.get_total_len()
        return (total_len + self.block_size - 1) // self.block_size
    
    def get_num_append_blocks_needed(self, request: Request) -> int:
        """Get the number of additional blocks needed for a request already on GPU"""
        assert self.request_location.get(request.request_id) == BlockLocation.GPU, \
            f"Request {request.request_id} is not on GPU"
        
        num_blocks_cur = len(self.block_table[request.request_id])
        num_blocks_needed = self.get_num_blocks_needed(request)
        return max(0, num_blocks_needed - num_blocks_cur)
    
    def allocate_blocks(self, request: Request, num_blocks: Optional[int] = None):
        """
        Allocate blocks for a request
        
        Args:
            request: Request to allocate blocks for
            num_blocks: Optional explicit number of blocks to allocate. If None, calculated from request.
        """
        # Ensure request is not on CPU
        assert (request.request_id not in self.block_table
                or self.request_location.get(request.request_id) == BlockLocation.GPU), \
            f"Request {request.request_id} is on CPU. Migrate to GPU first."
        
        # Use explicit num_blocks if provided, otherwise calculate
        if num_blocks is None:
            num_blocks_needed = self.get_num_blocks_needed(request)
        else:
            num_blocks_needed = num_blocks
        
        print(f"[{self.stage}] Allocating {num_blocks_needed} blocks for {request.request_id}")
        
        if request.request_id not in self.block_table:
            # First time allocation
            self.block_table[request.request_id] = self._get_free_blocks(
                num_blocks_needed, BlockLocation.GPU
            )
            self.request_location[request.request_id] = BlockLocation.GPU
        else:
            # Request already has blocks, append if needed
            assert self.request_location[request.request_id] == BlockLocation.GPU
            num_blocks_cur = len(self.block_table[request.request_id])
            if num_blocks_cur < num_blocks_needed:
                additional_blocks = self._get_free_blocks(
                    num_blocks_needed - num_blocks_cur, BlockLocation.GPU
                )
                self.block_table[request.request_id] += additional_blocks
    
    def allocate_blocks_batched(self, batched_requests: BatchedRequests):
        """Allocate blocks for a batch of requests"""
        for request in batched_requests.requests:
            self.allocate_blocks(request)
    
    def expand_blocks_for_seq_len(self, request_id: str, expanded_seq_len: int) -> List[int]:
        """
        Expand blocks for a request based on expanded sequence length.
        Used when vision tokens are added to the sequence.
        
        Args:
            request_id: Request ID
            expanded_seq_len: Total sequence length after expansion (including vision tokens)
        
        Returns:
            Updated block table for the request
        """
        if request_id not in self.block_table:
            raise ValueError(f"Request {request_id} not allocated")
        
        # Calculate blocks needed for expanded sequence
        blocks_needed = (expanded_seq_len + self.block_size - 1) // self.block_size
        current_blocks = len(self.block_table[request_id])
        
        if current_blocks >= blocks_needed:
            return self.block_table[request_id]
        
        # Allocate additional blocks
        additional_blocks_needed = blocks_needed - current_blocks
        additional_blocks = self._get_free_blocks(additional_blocks_needed, BlockLocation.GPU)
        self.block_table[request_id] += additional_blocks
        
        print(f"[{self.stage}] Expanded blocks for {request_id}: {current_blocks} -> {blocks_needed} (seq_len={expanded_seq_len}), added={additional_blocks}")
        
        return self.block_table[request_id]
    
    def free_blocks(self, request_id: str):
        """Free blocks for a request"""
        assert request_id in self.block_table, f"Request {request_id} not allocated"
        
        if self.request_location[request_id] == BlockLocation.GPU:
            self.free_gpu_blocks_list += self.block_table.pop(request_id)
        else:
            self.free_cpu_blocks_list += self.block_table.pop(request_id)
        
        self.request_location.pop(request_id)
    
    def free_blocks_batched(self, requests: List[Request]):
        """Free blocks for a batch of requests"""
        for request in requests:
            if request.request_id in self.block_table:
                self.free_blocks(request.request_id)
    
    def get_block_table(self, request_id: str) -> List[int]:
        """Get the block table for a request"""
        return self.block_table.get(request_id, [])
    
    def can_allocate(self, request: Request) -> bool:
        """Check if we can allocate blocks for a request"""
        num_blocks_needed = self.get_num_blocks_needed(request)
        return self.get_num_avail_gpu_blocks() >= num_blocks_needed
    
    def can_append(self, request: Request) -> bool:
        """Check if we can append blocks for a request"""
        if request.request_id not in self.block_table:
            return self.can_allocate(request)
        
        num_append_blocks = self.get_num_append_blocks_needed(request)
        return self.get_num_avail_gpu_blocks() >= num_append_blocks
    
    def append_blocks(self, request: Request):
        """
        Append additional blocks to an existing request (for autoregressive decode)
        
        Args:
            request: Request to append blocks for
        """
        assert request.request_id in self.block_table, \
            f"Request {request.request_id} not allocated yet"
        assert self.request_location.get(request.request_id) == BlockLocation.GPU, \
            f"Request {request.request_id} is not on GPU"
        
        num_append_blocks = self.get_num_append_blocks_needed(request)
        
        if num_append_blocks <= 0:
            return  # No need to append
        
        # Allocate new blocks
        new_blocks = self._get_free_blocks(num_append_blocks, BlockLocation.GPU)
        self.block_table[request.request_id].extend(new_blocks)
        
        print(f"[{self.stage}] Appended {num_append_blocks} blocks to {request.request_id}, "
              f"total blocks: {len(self.block_table[request.request_id])}")
    
    def swap_out(self, request_id: str):
        """Swap blocks from GPU to CPU"""
        assert request_id in self.block_table
        assert self.request_location[request_id] == BlockLocation.GPU
        
        num_blocks = len(self.block_table[request_id])
        cpu_blocks = self._get_free_blocks(num_blocks, BlockLocation.CPU)
        
        # Mark GPU blocks as swapping
        gpu_blocks = self.block_table[request_id]
        self.swapping_gpu_blocks_list += gpu_blocks
        
        # Update block table and location
        self.block_table[request_id] = cpu_blocks
        self.request_location[request_id] = BlockLocation.CPU
        
        return gpu_blocks, cpu_blocks
    
    def swap_in(self, request_id: str):
        """Swap blocks from CPU to GPU"""
        assert request_id in self.block_table
        assert self.request_location[request_id] == BlockLocation.CPU
        
        num_blocks = len(self.block_table[request_id])
        gpu_blocks = self._get_free_blocks(num_blocks, BlockLocation.GPU)
        
        # Mark CPU blocks as swapping
        cpu_blocks = self.block_table[request_id]
        self.swapping_cpu_blocks_list += cpu_blocks
        
        # Update block table and location
        self.block_table[request_id] = gpu_blocks
        self.request_location[request_id] = BlockLocation.GPU
        
        return cpu_blocks, gpu_blocks
    
    # Enhanced: Full swap operations based on EPD's implementation
    def swap_requests(self, request_ids: List[str], is_swap_in: bool):
        """
        Swap blocks for a batch of requests
        If `is_swap_in` is True, then swap in blocks from CPU to GPU, and vice versa
        
        Based on EPD's swap_requests implementation
        """
        cur_location = BlockLocation.CPU if is_swap_in else BlockLocation.GPU
        target_location = BlockLocation.GPU if is_swap_in else BlockLocation.CPU
        source_block_ids = []  # block ids on cur_location
        target_block_ids = []  # block ids on target_location
        
        for request_id in request_ids:
            assert request_id in self.block_table, f"request {request_id} not allocated"
            assert self.request_location[request_id] == cur_location, \
                f"request {request_id} is on {target_location} now"
            
            old_block_ids = self.block_table[request_id]
            new_block_ids = self._get_free_blocks(len(old_block_ids), target_location)
            source_block_ids += old_block_ids
            target_block_ids += new_block_ids
            self.block_table[request_id] = new_block_ids
            self.request_location[request_id] = target_location
            
            if cur_location == BlockLocation.CPU:
                self.swapping_cpu_blocks_list += old_block_ids
            else:
                self.swapping_gpu_blocks_list += old_block_ids
        
        # Call worker swap operation
        if self.engine_remote_call_all_workers_async:
            self.engine_remote_call_all_workers_async(
                "swap_blocks", request_ids, source_block_ids, target_block_ids, is_swap_in
            )
        
        self.swap_count += 1
    
    def swap_in_requests(self, request_ids: List[str]):
        """Swap in blocks for a batch of requests"""
        self.swap_requests(request_ids, is_swap_in=True)
    
    def swap_out_requests(self, request_ids: List[str]):
        """Swap out blocks for a batch of requests"""
        self.swap_requests(request_ids, is_swap_in=False)
    
    def is_all_requests_on_gpu(self, request_ids: List[str]) -> bool:
        """Check if all requests in a batch are on GPU"""
        for request_id in request_ids:
            if self.request_location.get(request_id) == BlockLocation.CPU:
                return False
        return True
    
    def clear_request_resource(self, request_id: str):
        """Clear the resources associated with the request"""
        """This is called when a request is finished or aborted"""
        # Clear the swap event table
        self.swap_event_table.pop(request_id, None)
    
    def clear_request_resource_batched(self, request_ids: List[str]):
        """Clear the resources associated with the requests"""
        for request_id in request_ids:
            self.clear_request_resource(request_id)
    
    def wait_for_all_swap_in(self):
        """Wait for all swap in to finish"""
        if self.latest_swap_in_event is not None:
            self.latest_swap_in_event.synchronize()
            self.latest_swap_in_event = None
    
    def wait_for_all_swap_out(self):
        """Wait for all swap out to finish"""
        if self.latest_swap_out_event is not None:
            self.latest_swap_out_event.synchronize()
            self.latest_swap_out_event = None
    
    def get_block_usage(self) -> Dict[str, str]:
        """Get block usage statistics (from EPD)"""
        num_cpu_blocks_used = (
            self.max_num_cpu_blocks - len(self.free_cpu_blocks_list) - len(self.swapping_cpu_blocks_list)
        )
        num_gpu_blocks_used = (
            self.max_num_gpu_blocks - len(self.free_gpu_blocks_list) - len(self.swapping_gpu_blocks_list)
        )
        
        safe_div = lambda n, d: n / d if d else 0
        
        return {
            'gpu': f'{round(safe_div(num_gpu_blocks_used, self.max_num_gpu_blocks)*100)}% ({num_gpu_blocks_used}/{self.max_num_gpu_blocks})',
            'cpu': f'{round(safe_div(num_cpu_blocks_used, self.max_num_cpu_blocks)*100)}% ({num_cpu_blocks_used}/{self.max_num_cpu_blocks})',
            'swap': f'{len(self.swapping_gpu_blocks_list)} -> {len(self.swapping_cpu_blocks_list)}',
            '#req': f'{len(self.block_table)}'
        }
    
    def print_block_usage(self):
        """Print block usage statistics"""
        usage = self.get_block_usage()
        print(f"({self.stage}) Block usage: GPU={usage['gpu']}, CPU={usage['cpu']}, "
              f"Swapping={usage['swap']}, Requests={usage['#req']}")
    
    # ===== KV Cache Tensor Management =====
    
    def create_kv_cache_tensors(
        self, 
        batched_requests: BatchedRequests, 
        num_layers: int, 
        num_heads: int, 
        head_size: int, 
        dtype: torch.dtype, 
        device: str = "cuda"
    ) -> List[torch.Tensor]:
        """
        Create KV cache tensors for a batch of requests
        
        Args:
            batched_requests: Batch of requests to process
            num_layers: Number of model layers
            num_heads: Number of attention heads
            head_size: Size of each attention head
            dtype: Data type for the tensors
            device: Device to create tensors on
            
        Returns:
            List of KV cache tensors, one per layer
        """
        # Calculate maximum blocks needed across all requests
        max_blocks_needed = 0
        for request in batched_requests.requests:
            blocks_needed = self.get_num_blocks_needed(request)
            max_blocks_needed = max(max_blocks_needed, blocks_needed)
        
        print(f"[{self.stage}] Creating KV cache tensors: max_blocks={max_blocks_needed}, "
              f"layers={num_layers}, heads={num_heads}, head_size={head_size}")
        
        # Create KV cache tensor for each layer
        kv_caches = []
        for layer_idx in range(num_layers):
            kv_cache = torch.empty(
                (2, max_blocks_needed, self.block_size, num_heads, head_size),
                dtype=dtype,
                device=device
            )
            kv_caches.append(kv_cache)
        
        return kv_caches
    
    def get_max_blocks_needed(self, batched_requests: BatchedRequests) -> int:
        """
        Get the maximum number of blocks needed for a batch of requests
        
        Args:
            batched_requests: Batch of requests
            
        Returns:
            Maximum number of blocks needed
        """
        max_blocks = 0
        for request in batched_requests.requests:
            blocks_needed = self.get_num_blocks_needed(request)
            max_blocks = max(max_blocks, blocks_needed)
        return max_blocks


class V0VisionBlockManager(V0BlockManager):
    """
    Block Manager for vision embeddings
    Similar to KV cache manager but for multimodal embeddings
    """
    
    def __init__(
        self,
        stage: str,
        max_num_gpu_blocks: int,
        max_num_cpu_blocks: int,
        block_size: int,
        engine_remote_call_all_workers_async: Optional[Callable] = None,
    ):
        super().__init__(
            stage=stage,
            max_num_gpu_blocks=max_num_gpu_blocks,
            max_num_cpu_blocks=max_num_cpu_blocks,
            block_size=block_size,
            engine_remote_call_all_workers_async=engine_remote_call_all_workers_async,
        )
    
    def get_num_blocks_needed(self, request: Request) -> int:
        """
        Calculate blocks needed for vision embeddings
        Each image typically needs a fixed number of blocks
        """
        if not request.images:
            return 0
        
        # Assuming each image needs certain blocks (model-dependent)
        # This is a simplified calculation
        num_images = len(request.images)
        # Typical vision embeddings: 576 tokens for LLaVA, 256 for InternVL2, etc.
        # We'll use a conservative estimate
        embedding_tokens_per_image = 576  # Can be configured
        blocks_per_image = (embedding_tokens_per_image + self.block_size - 1) // self.block_size
        
        return num_images * blocks_per_image

