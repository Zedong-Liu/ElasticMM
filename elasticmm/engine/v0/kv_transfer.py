"""KV Transfer Manager for v0 engine."""

import asyncio
import math
import time
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

import ray
import ray.util.collective as collective


class TransferMethod(Enum):
    """KV transfer method"""
    CUDA_IPC = "cuda_ipc"  # Zero-copy via CUDA IPC (requires custom CUDA extension)
    NCCL = "nccl"  # NCCL collective communication
    P2P_COPY = "p2p_copy"  # Direct GPU-to-GPU copy
    STAGED_COPY = "staged_copy"  # Copy via CPU (fallback)


class V0KVTransferManager:
    """
    Manages KV cache transfers between stages
    """
    
    def __init__(self, transfer_method: TransferMethod = TransferMethod.P2P_COPY, nccl_coordinator=None, 
                 enable_batching: bool = False, batch_size: int = 3, batch_timeout: float = 0.005, metrics_callback=None):
        """
        Initialize KV transfer manager
        
        Args:
            transfer_method: Method for transferring KV cache
            nccl_coordinator: NCCL coordinator for P2P transfer
            enable_batching: Whether to enable batch transfer (合并多个小传输)
            batch_size: Minimum number of transfers to trigger batch (default: 3)
            batch_timeout: Maximum wait time before flushing batch in seconds (default: 5ms)
            metrics_callback: Callback function to record transfer metrics (transfer_time, num_blocks)
        """
        self.transfer_method = transfer_method
        self.nccl_coordinator = nccl_coordinator
        self.enable_batching = enable_batching
        self.metrics_callback = metrics_callback
        
        # IPC memory handles registry
        # Structure: {stage_name: {worker_id: {cache_type: ipc_handle}}}
        self.ipc_mem_handles: Dict[str, Dict[int, Dict[str, Any]]] = {}

        # ✅ NEW: Worker registry using global_rank as key (decoupled from stage)
        # Structure: {global_rank: worker_ref}
        self.worker_registry: Dict[int, Any] = {}
        
        # ✅ NEW: Track current stage of each rank (for debugging/logging)
        # Structure: {global_rank: stage_name}
        self.rank_to_stage: Dict[int, str] = {}
        # Precomputed collective group mapping
        # Key: (src_stage, src_worker_id, dst_stage, dst_worker_id)
        self.collective_pairs: Dict[tuple, Dict[str, Any]] = {}
        # Store group metadata for cleanup
        self.collective_groups: Dict[str, Dict[str, Any]] = {}
        
        # Track active NCCL transfers per group to manage concurrency
        # Key: group_name, Value: asyncio.Lock()
        self._nccl_group_locks: Dict[str, asyncio.Lock] = {}
        
        # Counter for generating unique group names for concurrent transfers
        self._transfer_counter = 0
        self._counter_lock = asyncio.Lock()
        
        # OPTIMIZATION: Cache worker ranks to avoid repeated remote calls
        # Key: (stage, worker_id) -> global_rank
        self._worker_rank_cache: Dict[tuple, int] = {}
        
        # OPTIMIZATION: Batch transfer queue
        # Key: (src_stage, src_worker_id, dst_stage, dst_worker_id)
        # Value: List of pending transfers [(request_id, src_blocks, dst_blocks), ...]
        self._transfer_queue: Dict[tuple, List[tuple]] = {}
        self._batch_flush_threshold = batch_size
        self._batch_flush_timeout = batch_timeout
        self._last_flush_time: Dict[tuple, float] = {}
        
        # Transfer statistics (excluding cold start)
        self.transfer_count = 0
        self.total_transfer_time = 0.0
        self.total_bytes_transferred = 0
        self.first_transfer_done = False  # Track cold start
        

    def register_ipc_handle(
        self,
        stage: str,
        worker_id: int,
        cache_type: str,  # 'kv' or 've' (vision embedding)
        ipc_handle: List[int]
    ):
        """
        Register CUDA IPC memory handle
        
        Args:
            stage: Stage name ('encoding', 'prefill', 'decoding')
            worker_id: Worker ID
            cache_type: Cache type ('kv' or 've')
            ipc_handle: CUDA IPC memory handle
        """
        if stage not in self.ipc_mem_handles:
            self.ipc_mem_handles[stage] = {}
        
        if worker_id not in self.ipc_mem_handles[stage]:
            self.ipc_mem_handles[stage][worker_id] = {}
        
        self.ipc_mem_handles[stage][worker_id][cache_type] = ipc_handle
        
    
    def register_workers(self, stage: str, workers: Sequence[Any], global_ranks: Sequence[int]) -> None:
        """
        ✅ NEW: Register workers using their global_ranks as keys.
        Decouples NCCL layer (ranks) from logical layer (stages).
        
        Args:
            stage: Stage name (for tracking only)
            workers: List of worker references
            global_ranks: List of corresponding global ranks
        """
        normalized_stage = self._normalize_stage(stage)
        count = 0
        
        for worker, rank in zip(workers, global_ranks):
            if worker is not None and rank >= 0:
                self.worker_registry[rank] = worker
                self.rank_to_stage[rank] = normalized_stage
                count += 1
        
    
    def register_worker(self, global_rank: int, worker: Any, stage: str) -> None:
        """
        ✅ NEW: Register a single worker using global_rank as key.
        
        Args:
            global_rank: Worker's global NCCL rank
            worker: Worker reference
            stage: Current stage (for tracking)
        """
        self.worker_registry[global_rank] = worker
        self.rank_to_stage[global_rank] = self._normalize_stage(stage)
    
    def unregister_worker(self, global_rank: int) -> None:
        """
        ✅ NEW: Unregister a worker by global_rank.
        
        Args:
            global_rank: Worker's global NCCL rank
        """
        if global_rank in self.worker_registry:
            stage = self.rank_to_stage.get(global_rank, "unknown")
            del self.worker_registry[global_rank]
            if global_rank in self.rank_to_stage:
                del self.rank_to_stage[global_rank]
    
    def update_worker_stage(self, global_rank: int, new_stage: str) -> None:
        """
        ✅ NEW: Update worker's stage mapping (for role switching).
        This ONLY updates the stage tracking - NCCL rank stays the same!
        
        Args:
            global_rank: Worker's global NCCL rank (unchangeable)
            new_stage: New stage name
        """
        if global_rank not in self.worker_registry:
            print(f"[V0KVTransferManager] Warning: Rank {global_rank} not registered")
            return
        
        old_stage = self.rank_to_stage.get(global_rank, "unknown")
        new_normalized = self._normalize_stage(new_stage)
        self.rank_to_stage[global_rank] = new_normalized
        
    
    def register_collective_group(
        self,
        group_name: str,
        participants: Sequence[Dict[str, Any]],
    ) -> None:
        """
        Register workers for P2P transfer (NO Ray Collective needed).
        
        Args:
            group_name: Unique group identifier (for tracking only).
            participants: Sequence of dicts with keys:
                - stage: Stage name
                - worker_id: Worker index
                - worker: Ray actor handle
                - rank: NCCL rank within the group (unused with P2P)
        """
        # With PyTorch P2P, we don't need to create Ray Collective groups
        # Just store worker references for later P2P communication
        if group_name in self.collective_groups:
            return

        self.collective_groups[group_name] = {
            "participants": participants,
            "world_size": len(participants),
        }
        
        # Store worker references for P2P lookup
        for src in participants:
            for dst in participants:
                if src is dst:
                    continue
                key = (
                    self._normalize_stage(src["stage"]),
                    src["worker_id"],
                    self._normalize_stage(dst["stage"]),
                    dst["worker_id"],
                )
                self.collective_pairs[key] = {
                    "base_group_name": group_name,
                    "src_rank": src["rank"],
                    "dst_rank": dst["rank"],
                    "src_worker": src["worker"],
                    "dst_worker": dst["worker"],
                    "world_size": len(participants),
                }
        
    
    async def transfer_kv_cache(
        self,
        request_id: str,
        src_rank: int,
        dst_rank: int,
        src_blocks: List[int],
        dst_blocks: List[int],
    ) -> bool:
        """
        ✅ NEW: Transfer KV cache using global ranks (decoupled from stages).
        
        Args:
            request_id: Request ID
            src_rank: Source worker's global NCCL rank
            dst_rank: Destination worker's global NCCL rank
            src_blocks: Source block IDs
            dst_blocks: Destination block IDs
            
        Returns:
            True if transfer successful
        """
        # ✅ NEW: All transfer methods now use ranks directly
        if self.transfer_method == TransferMethod.NCCL:
            # Try P2P first (direct PyTorch P2P with optional batching)
            success = await self._transfer_via_nccl_p2p(
                'kv', request_id,
                src_rank, src_blocks,
                dst_rank, dst_blocks,
                use_batching=self.enable_batching
            )
            return success
        elif self.transfer_method == TransferMethod.P2P_COPY:
            return await self._transfer_via_p2p_copy(
                'kv', request_id,
                src_rank, src_blocks,
                dst_rank, dst_blocks
            )
        else:
            return False
    
    async def transfer_vision_embeddings(
        self,
        request_id: str,
        src_stage: str,
        src_worker_id: int,
        src_blocks: List[int],
        dst_stage: str,
        dst_worker_id: int,
        dst_blocks: List[int],
    ) -> bool:
        """
        Transfer vision embeddings from source to destination
        
        Args:
            request_id: Request ID
            src_stage: Source stage name
            src_worker_id: Source worker ID
            src_blocks: Source block IDs
            dst_stage: Destination stage name
            dst_worker_id: Destination worker ID
            dst_blocks: Destination block IDs
            
        Returns:
            True if transfer successful
        """
        if self.transfer_method == TransferMethod.CUDA_IPC:
            return await self._transfer_via_cuda_ipc(
                've', request_id,
                src_stage, src_worker_id, src_blocks,
                dst_stage, dst_worker_id, dst_blocks
            )
        else:
            # Use P2P copy for vision embeddings (simpler)
            return await self._transfer_via_p2p_copy(
                've', request_id,
                src_stage, src_worker_id, src_blocks,
                dst_stage, dst_worker_id, dst_blocks
            )
    
    async def _transfer_via_cuda_ipc(
        self,
        cache_type: str,
        request_id: str,
        src_stage: str,
        src_worker_id: int,
        src_blocks: List[int],
        dst_stage: str,
        dst_worker_id: int,
        dst_blocks: List[int],
    ) -> bool:
        """
        Transfer via CUDA IPC (zero-copy)
        
        This requires custom CUDA extension (like EPD's block_migration_ops)
        For now, this is a placeholder
        """
        # TODO: Implement CUDA IPC transfer
        # This would call:
        # torch.ops.block_migration_ops.migrate_blocks(...)
        
        print(f"[V0KVTransferManager] CUDA IPC transfer: {request_id} "
              f"from {src_stage}/worker{src_worker_id} to {dst_stage}/worker{dst_worker_id}")
        
        # Placeholder: simulate transfer
        await asyncio.sleep(0.001)
        
        self.transfer_count += 1
        return True
    
    async def _get_worker_rank(self, stage: str, worker_id: int, worker_ref: Any) -> int:
        """Get worker rank with caching to avoid repeated remote calls"""
        cache_key = (stage, worker_id)
        if cache_key not in self._worker_rank_cache:
            rank = await worker_ref.get_global_rank.remote()
            self._worker_rank_cache[cache_key] = rank
        return self._worker_rank_cache[cache_key]
    
    async def _flush_transfer_batch(
        self,
        src_stage: str,
        src_worker_id: int,
        dst_stage: str,
        dst_worker_id: int,
    ) -> bool:
        """Flush batched transfers for a worker pair"""
        batch_key = (src_stage, src_worker_id, dst_stage, dst_worker_id)
        
        if batch_key not in self._transfer_queue or not self._transfer_queue[batch_key]:
            return True
        
        transfers = self._transfer_queue[batch_key]
        self._transfer_queue[batch_key] = []
        
        # Merge all blocks
        all_src_blocks = []
        all_dst_blocks = []
        request_ids = []
        
        for request_id, src_blocks, dst_blocks in transfers:
            all_src_blocks.extend(src_blocks)
            all_dst_blocks.extend(dst_blocks)
            request_ids.append(request_id)
        
        # Execute batched transfer
        src_worker = self._get_worker_ref(src_stage, src_worker_id)
        dst_worker = self._get_worker_ref(dst_stage, dst_worker_id)
        
        if not src_worker or not dst_worker:
            return False
        
        try:
            start_time = time.perf_counter()
            
            src_rank, dst_rank = await asyncio.gather(
                self._get_worker_rank(src_stage, src_worker_id, src_worker),
                self._get_worker_rank(dst_stage, dst_worker_id, dst_worker),
            )
            
            if src_rank < 0 or dst_rank < 0:
                return False
            
            # Batched P2P transfer
            send_result, recv_result = await asyncio.gather(
                src_worker.p2p_send_kv.remote(dst_rank, all_src_blocks),
                dst_worker.p2p_recv_kv.remote(src_rank, all_dst_blocks),
            )
            
            # ✅ Check for errors
            if "error" in send_result:
                print(f"[V0KVTransferManager] ⚠ Batched transfer send failed: {send_result['error']}")
                return False
            if "error" in recv_result:
                print(f"[V0KVTransferManager] ⚠ Batched transfer recv failed: {recv_result['error']}")
                return False
            
            elapsed = time.perf_counter() - start_time
            bytes_transferred = recv_result.get("bytes", 0)
            
            if bytes_transferred == 0:
                print(f"[V0KVTransferManager] ⚠ Batched transfer returned 0 bytes")
                return False
            
            self.transfer_count += len(transfers)
            self.total_transfer_time += elapsed
            self.total_bytes_transferred += bytes_transferred
            
            bandwidth_gbps = (bytes_transferred / 1e9) / elapsed if elapsed > 0 else 0
            print(f"[V0KVTransferManager] ✓ BATCHED transfer: {len(transfers)} requests, "
                  f"{len(all_src_blocks)} blocks, {bytes_transferred/1e6:.2f} MB, "
                  f"{elapsed*1000:.2f} ms, {bandwidth_gbps:.2f} GB/s - {request_ids}")
            return True
            
        except Exception as e:
            print(f"[V0KVTransferManager] Batched transfer failed: {e}")
            return False
    
    async def _transfer_via_nccl_p2p(
        self,
        cache_type: str,
        request_id: str,
        src_rank: int,
        src_blocks: List[int],
        dst_rank: int,
        dst_blocks: List[int],
        use_batching: bool = False,
    ) -> bool:
        """
        ✅ NEW: Transfer via PyTorch P2P using global ranks.
        Completely decoupled from stage logic.
        """
        if cache_type != 'kv':
            return False
        
        # OPTIMIZATION: Batch transfer (optional)
        if use_batching:
            batch_key = (src_stage, src_worker_id, dst_stage, dst_worker_id)
            
            # Add to queue
            if batch_key not in self._transfer_queue:
                self._transfer_queue[batch_key] = []
                self._last_flush_time[batch_key] = time.perf_counter()
            
            self._transfer_queue[batch_key].append((request_id, src_blocks, dst_blocks))
            
            # Check if we should flush
            current_time = time.perf_counter()
            queue_size = len(self._transfer_queue[batch_key])
            time_since_last_flush = current_time - self._last_flush_time[batch_key]
            
            should_flush = (
                queue_size >= self._batch_flush_threshold or
                time_since_last_flush >= self._batch_flush_timeout
            )
            
            if should_flush:
                self._last_flush_time[batch_key] = current_time
                return await self._flush_transfer_batch(src_stage, src_worker_id, dst_stage, dst_worker_id)
            else:
                # Queued, will be flushed later
                return True
        
        # ✅ NEW: Direct transfer using ranks (no batching)
        src_worker = self.worker_registry.get(src_rank)
        dst_worker = self.worker_registry.get(dst_rank)
        
        if not src_worker or not dst_worker:
            print(f"[V0KVTransferManager] Workers not found: src_rank={src_rank}, dst_rank={dst_rank}")
            return False
        
        try:
            start_time = time.perf_counter()
            
            # ✅ Ranks are already provided - no need to query!
            # Log for debugging
            src_stage = self.rank_to_stage.get(src_rank, "unknown")
            dst_stage = self.rank_to_stage.get(dst_rank, "unknown")
            
            # ✅ Single Ray remote call (coordinated transfer)
            recv_result = await dst_worker.p2p_coordinated_transfer_kv.remote(
                src_worker,    # Pass src worker reference
                src_rank,      # NCCL rank of source
                src_blocks,    # Blocks to send from source
                dst_blocks,    # Blocks to write in destination
                timeout=10.0   # 10s timeout for safety
            )
            
            elapsed = time.perf_counter() - start_time
            
            # ✅ Check for errors in the result
            if "error" in recv_result:
                error_msg = recv_result.get("error", "Unknown error")
                print(f"[V0KVTransferManager] ⚠ NCCL P2P transfer failed for {request_id}: {error_msg}")
                return False
            
            # Update statistics (exclude first transfer - cold start)
            bytes_transferred = recv_result.get("bytes", 0)
            if bytes_transferred == 0:
                print(f"[V0KVTransferManager] ⚠ NCCL P2P transfer returned 0 bytes for {request_id}")
                return False
            
            bandwidth_gbps = (bytes_transferred / 1e9) / elapsed if elapsed > 0 else 0
            
            if not self.first_transfer_done:
                self.first_transfer_done = True
                print(f"[V0KVTransferManager] ✓ Coordinated P2P (COLD START): {len(src_blocks)} blocks, "
                      f"{bytes_transferred/1e6:.2f} MB, {elapsed*1000:.2f} ms, {bandwidth_gbps:.2f} GB/s - {request_id}")
            else:
                self.transfer_count += 1
                self.total_transfer_time += elapsed
                self.total_bytes_transferred += bytes_transferred
                print(f"[V0KVTransferManager] ✓ Coordinated P2P: {len(src_blocks)} blocks, "
                      f"{bytes_transferred/1e6:.2f} MB, {elapsed*1000:.2f} ms, {bandwidth_gbps:.2f} GB/s - {request_id}")
            
            # ✅ Record to metrics callback (including COLD START)
            if self.metrics_callback:
                self.metrics_callback(transfer_time_sec=elapsed, num_blocks=len(src_blocks))
            return True
            
        except Exception as e:
            print(f"[V0KVTransferManager] P2P transfer failed: {e}")
            return False
    
    async def _transfer_via_nccl(
        self,
        cache_type: str,
        request_id: str,
        src_stage: str,
        src_worker_id: int,
        src_blocks: List[int],
        dst_stage: str,
        dst_worker_id: int,
        dst_blocks: List[int],
    ) -> bool:
        """
        Transfer via NCCL collective communication using Ray collective APIs.
        """
        if cache_type != 'kv':
            print(f"[V0KVTransferManager] NCCL transfer currently supports KV cache only (requested {cache_type})")
            return False
        
        key = (
            self._normalize_stage(src_stage),
            src_worker_id,
            self._normalize_stage(dst_stage),
            dst_worker_id,
        )
        group_info = self.collective_pairs.get(key)
        if not group_info:
            print(f"[V0KVTransferManager] No NCCL group registered for {key}")
            return False
        
        src_worker = self._get_worker_ref(src_stage, src_worker_id)
        dst_worker = self._get_worker_ref(dst_stage, dst_worker_id)
        
        if src_worker is None or dst_worker is None:
            print(f"[V0KVTransferManager] Worker not registered for NCCL transfer: "
                  f"src={src_stage}/{src_worker_id}, dst={dst_stage}/{dst_worker_id}")
            return False
        
        if len(src_blocks) != len(dst_blocks):
            print(f"[V0KVTransferManager] Block count mismatch for request {request_id}: "
                  f"src_blocks={len(src_blocks)}, dst_blocks={len(dst_blocks)}")
            return False
        
        try:
            layout = await src_worker.get_kv_cache_layout.remote()
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[V0KVTransferManager] Failed to obtain KV layout from worker {src_stage}/{src_worker_id}: {exc}")
            return False
        
        if not layout:
            print(f"[V0KVTransferManager] Empty KV layout from worker {src_stage}/{src_worker_id}")
            return False
        
        num_layers = layout.get("num_layers")
        block_size = layout.get("block_size")
        num_heads = layout.get("num_heads")
        head_size = layout.get("head_size")
        dtype_repr = layout.get("dtype")
        
        if None in (num_layers, block_size, num_heads, head_size, dtype_repr):
            print(f"[V0KVTransferManager] Incomplete KV layout for request {request_id}: {layout}")
            return False
        
        num_blocks = len(src_blocks)
        tensor_shape = [
            num_layers,
            2,  # K and V
            num_blocks,
            block_size,
            num_heads,
            head_size,
        ]
        
        dtype_str = self._dtype_to_string(dtype_repr)
        element_size = self._dtype_size(dtype_repr)
        bytes_transferred = math.prod(tensor_shape) * element_size
        
        base_group_name = group_info["base_group_name"]
        src_rank = group_info["src_rank"]
        world_size = group_info["world_size"]
        start_time = time.perf_counter()
        
        # Create unique group name for this transfer to avoid Ray collective conflicts
        async with self._counter_lock:
            transfer_id = self._transfer_counter
            self._transfer_counter += 1
        
        unique_group_name = f"{base_group_name}_transfer_{transfer_id}"
        
        try:
            # Create temporary NCCL group for this specific transfer
            # This avoids the "name already taken" error from concurrent operations
            try:
                collective.create_collective_group(
                    [src_worker, dst_worker],
                    world_size=world_size,
                    ranks=[src_rank, group_info["dst_rank"]],
                    backend="nccl",
                    group_name=unique_group_name,
                )
            except (RuntimeError, ValueError) as create_exc:
                # Group might already exist (race condition), try to use it
                print(f"[V0KVTransferManager] Group creation warning (proceeding): {create_exc}")
            
            # Initialize group on workers
            await asyncio.gather(
                src_worker.init_collective_group.remote(unique_group_name, world_size, src_rank),
                dst_worker.init_collective_group.remote(unique_group_name, world_size, group_info["dst_rank"]),
            )
            
            # Perform transfer
            send_task = src_worker.collective_transfer_kv.remote(
                group_name=unique_group_name,
                root=src_rank,
                block_indices=src_blocks,
                tensor_shape=tensor_shape,
                dtype=dtype_str,
                mode="send",
            )
            recv_task = dst_worker.collective_transfer_kv.remote(
                group_name=unique_group_name,
                root=src_rank,
                block_indices=dst_blocks,
                tensor_shape=tensor_shape,
                dtype=dtype_str,
                mode="recv",
            )
            
            await asyncio.gather(send_task, recv_task)
            
            # Clean up temporary group
            try:
                await asyncio.gather(
                    src_worker.destroy_collective_group.remote(unique_group_name),
                    dst_worker.destroy_collective_group.remote(unique_group_name),
                    return_exceptions=True,
                )
                collective.destroy_collective_group(unique_group_name)
            except Exception:
                pass  # Cleanup errors are non-fatal
        
        except Exception as exc:
            print(f"[V0KVTransferManager] NCCL transfer failed for request {request_id}: {exc}")
            # Try to clean up on error
            try:
                await asyncio.gather(
                    src_worker.destroy_collective_group.remote(unique_group_name),
                    dst_worker.destroy_collective_group.remote(unique_group_name),
                    return_exceptions=True,
                )
            except Exception:
                pass
            return False
        
        elapsed = time.perf_counter() - start_time
        self.transfer_count += 1
        self.total_transfer_time += elapsed
        self.total_bytes_transferred += bytes_transferred
        
        print(f"[V0KVTransferManager] Transfer completed: "
              f"blocks={num_blocks}, bytes={bytes_transferred}, time={elapsed:.4f}s")
        
        # ✅ Record to metrics callback
        if self.metrics_callback:
            self.metrics_callback(transfer_time_sec=elapsed, num_blocks=num_blocks)
        
        return True
    
    async def _transfer_via_p2p_copy(
        self,
        cache_type: str,
        request_id: str,
        src_rank: int,
        src_blocks: List[int],
        dst_rank: int,
        dst_blocks: List[int],
    ) -> bool:
        """
        ✅ NEW: Transfer via direct P2P GPU copy using global ranks.
        Completely decoupled from stage logic.
        """
        try:
            # ✅ Get workers by rank
            src_worker = self.worker_registry.get(src_rank)
            dst_worker = self.worker_registry.get(dst_rank)
            
            if src_worker is None or dst_worker is None:
                src_stage = self.rank_to_stage.get(src_rank, "unknown")
                dst_stage = self.rank_to_stage.get(dst_rank, "unknown")
                print(f"[V0KVTransferManager] Worker not found: src=rank{src_rank}({src_stage}), dst=rank{dst_rank}({dst_stage})")
                return False
            
            # ✅ Measure transfer time
            start_time = time.perf_counter()
            
            # Extract data from source worker
            if cache_type == 'kv':
                kv_data = await src_worker.extract_kv_blocks.remote(src_blocks)
                # Write data to destination worker
                await dst_worker.write_kv_blocks.remote(dst_blocks, kv_data)
            elif cache_type == 've':
                ve_data = await src_worker.extract_vision_blocks.remote(request_id)
                # Write data to destination worker
                await dst_worker.write_vision_blocks.remote(request_id, ve_data)
            
            elapsed = time.perf_counter() - start_time
            
            # ✅ Get stage names for logging
            src_stage = self.rank_to_stage.get(src_rank, "unknown")
            dst_stage = self.rank_to_stage.get(dst_rank, "unknown")
            
            print(f"[V0KVTransferManager] P2P copy completed: {request_id} "
                  f"from rank{src_rank}({src_stage}) to rank{dst_rank}({dst_stage}), "
                  f"blocks={len(src_blocks)}, time={elapsed:.4f}s")
            
            self.transfer_count += 1
            
            # ✅ Record to metrics callback
            if self.metrics_callback and cache_type == 'kv':
                self.metrics_callback(transfer_time_sec=elapsed, num_blocks=len(src_blocks))
            
            return True
            
        except Exception as e:
            print(f"[V0KVTransferManager] P2P copy failed: {e}")
            return False
    
    def _normalize_stage(self, stage: str) -> str:
        """Normalize stage name to registry key."""
        return stage.lower()
    
    def _parse_dtype(self, dtype: Any):
        """Convert dtype representations to torch.dtype."""
        import torch
        
        if isinstance(dtype, torch.dtype):
            return dtype
        
        if isinstance(dtype, str):
            normalized = dtype.replace("torch.", "").lower()
        else:
            normalized = str(dtype).replace("torch.", "").lower()
        
        mapping = {
            "float16": torch.float16,
            "half": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if normalized not in mapping:
            raise ValueError(f"Unsupported dtype for KV transfer: {dtype}")
        return mapping[normalized]
    
    def _dtype_to_string(self, dtype: Any) -> str:
        """Convert dtype to canonical string form (e.g., 'float16')."""
        import torch
        torch_dtype = self._parse_dtype(dtype)
        return str(torch_dtype).split(".")[-1]
    
    def _dtype_size(self, dtype: Any) -> int:
        """Return element size in bytes for dtype."""
        import torch
        torch_dtype = self._parse_dtype(dtype)
        return torch.tensor([], dtype=torch_dtype).element_size()
    
    def _get_worker_ref(self, stage: str, worker_id: int):
        """Get worker reference by stage and worker_id."""
        normalized_stage = self._normalize_stage(stage)
        stage_registry = self.worker_registry.get(normalized_stage, {})
        worker = stage_registry.get(worker_id)
        if worker is None:
            print(f"[V0KVTransferManager] Worker not registered: stage={normalized_stage} worker_id={worker_id}")
        return worker
    
    async def destroy_all_collective_groups(self) -> None:
        """Destroy all registered collective groups."""
        # Destroy the driver-side collective group
        collective.destroy_collective_group()
        self.collective_groups.clear()
        self.collective_pairs.clear()
        tasks = []
        for group_name, info in list(self.collective_groups.items()):
            for participant in info["participants"]:
                tasks.append(participant["worker"].destroy_collective_group.remote(group_name))
            try:
                collective.destroy_collective_group(group_name)
            except Exception as exc:
                print(f"[V0KVTransferManager] destroy_collective_group warning: {exc}")
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.collective_groups.clear()
        self.collective_pairs.clear()
    
    async def _transfer_via_staged_copy(
        self,
        cache_type: str,
        request_id: str,
        src_stage: str,
        src_worker_id: int,
        src_blocks: List[int],
        dst_stage: str,
        dst_worker_id: int,
        dst_blocks: List[int],
    ) -> bool:
        """
        Transfer via CPU staging (fallback method)
        """
        # TODO: Implement staged copy via CPU
        
        
        await asyncio.sleep(0.002)  # Slower than other methods
        
        self.transfer_count += 1
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get transfer statistics"""
        return {
            "transfer_count": self.transfer_count,
            "total_transfer_time": self.total_transfer_time,
            "total_bytes_transferred": self.total_bytes_transferred,
            "avg_transfer_time": (
                self.total_transfer_time / self.transfer_count
                if self.transfer_count > 0 else 0.0
            ),
        }
    
    def reset_stats(self):
        """Reset transfer statistics"""
        self.transfer_count = 0
        self.total_transfer_time = 0.0
        self.total_bytes_transferred = 0


class V0KVMigrationManager:
    """
    KV Migration Manager for cross-stage transfers
    Based on EPD's migration implementation
    """
    
    def __init__(self):
        self.migration_count = 0
        self.total_migration_time = 0.0
        
    async def migrate_kv_blocks(
        self,
        request_id: str,
        src_stage: str,
        src_worker_id: int,
        src_blocks: List[int],
        dst_stage: str,
        dst_worker_id: int,
        dst_blocks: List[int],
    ) -> bool:
        """
        Migrate KV blocks from source stage to destination stage
        
        Based on EPD's migrate_blocks_context implementation
        """
        import time
        start_time = time.time()
        
        try:
            # Get source and destination workers
            src_worker = self._get_worker_ref(src_stage, src_worker_id)
            dst_worker = self._get_worker_ref(dst_stage, dst_worker_id)
            
            if src_worker is None or dst_worker is None:
                print(f"[V0KVMigrationManager] Worker not found: src={src_stage}/{src_worker_id}, dst={dst_stage}/{dst_worker_id}")
                return False
            
            # Extract KV data from source
            kv_data = await src_worker.extract_kv_blocks.remote(src_blocks)
            
            # Write KV data to destination
            await dst_worker.write_kv_blocks.remote(dst_blocks, kv_data)
            
            migration_time = time.time() - start_time
            self.migration_count += 1
            self.total_migration_time += migration_time
            
            
            return True
            
        except Exception as e:
            print(f"[V0KVMigrationManager] KV migration failed: {e}")
            return False
    
    async def migrate_vision_embeddings(
        self,
        request_id: str,
        src_stage: str,
        src_worker_id: int,
        dst_stage: str,
        dst_worker_id: int,
    ) -> bool:
        """
        Migrate vision embeddings from source stage to destination stage
        
        Based on EPD's migrate_blocks_encoding implementation
        """
        import time
        start_time = time.time()
        
        try:
            # Get source and destination workers
            src_worker = self._get_worker_ref(src_stage, src_worker_id)
            dst_worker = self._get_worker_ref(dst_stage, dst_worker_id)
            
            if src_worker is None or dst_worker is None:
                print(f"[V0KVMigrationManager] Worker not found: src={src_stage}/{src_worker_id}, dst={dst_stage}/{dst_worker_id}")
                return False
            
            # Extract vision embeddings from source
            ve_data = await src_worker.extract_vision_blocks.remote(request_id)
            
            # Write vision embeddings to destination
            await dst_worker.write_vision_blocks.remote(request_id, ve_data)
            
            migration_time = time.time() - start_time
            self.migration_count += 1
            self.total_migration_time += migration_time
            
            print(f"[V0KVMigrationManager] Vision embedding migration completed: {request_id} "
                  f"from {src_stage}/worker{src_worker_id} to {dst_stage}/worker{dst_worker_id} "
                  f"in {migration_time:.3f}s")
            
            return True
            
        except Exception as e:
            print(f"[V0KVMigrationManager] Vision embedding migration failed: {e}")
            return False
    
    def _get_worker_ref(self, stage: str, worker_id: int):
        """
        Get worker reference by stage and worker_id
        This is a placeholder - in real implementation, this would maintain
        a registry of worker references
        """
        # TODO: Implement worker reference registry
        # For now, return None to indicate not implemented
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get migration statistics"""
        return {
            "migration_count": self.migration_count,
            "total_migration_time": self.total_migration_time,
            "avg_migration_time": (
                self.total_migration_time / self.migration_count
                if self.migration_count > 0 else 0.0
            ),
        }


# Export
__all__ = [
    'TransferMethod',
    'V0KVTransferManager',
    'V0KVMigrationManager',
]


__all__ = [
    'TransferMethod',
    'V0KVTransferManager',
    'V0KVMigrationManager',
]

