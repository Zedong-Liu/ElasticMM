"""
V0 Engine Backend implementation
Coordinates encoding, prefill, and decoding stages
"""

import asyncio
import time
import threading
from typing import List, Dict, Any, Optional
from collections import deque

from elasticmm.engine.backend_interface import EngineBackend
from elasticmm.engine.v0.utils import Request, StepOutput, StageMetrics, MigratingRequest
from elasticmm.engine.v0.stage_engine import (
    V0EncodingEngine,
    V0PrefillEngine,
    V0DecodingEngine,
)
from elasticmm.engine.v0.kv_transfer import V0KVTransferManager, TransferMethod
from elasticmm.engine.v0.nccl_transfer import V0NCCLTransferManager
from elasticmm.engine.v0.nccl_coordinator import NCCLCoordinator
from elasticmm.engine.v0.memory_coordinator import GlobalMemoryCoordinator
from elasticmm.core.balancer import ModalityType
from elasticmm.core.allocator import InferenceStage


class V0EngineBackend(EngineBackend):
    """
    V0 Engine Backend
    
    Implements disaggregated architecture with separate encoding, prefill, and decoding stages
    """
    
    def __init__(
        self,
        model_path: str,
        num_encoding_workers: int = 2,
        num_prefill_workers: int = 4,
        num_decoding_workers: int = 2,
        block_size: int = 16,
        max_num_gpu_blocks: int = 5000,
        max_num_cpu_blocks: int = 1000,
        dtype: str = "float16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        kv_transfer_method: str = "p2p_copy",
        limit_mm_per_prompt: Optional[Dict[str, int]] = None,
        sd_host: str = "0.0.0.0",
        sd_port: int = 30002,
        proxy_port: int = 10001,
    ):
        """
        Initialize V0 Engine Backend
        
        Args:
            model_path: Path to model
            num_encoding_workers: Number of encoding workers
            num_prefill_workers: Number of prefill workers
            num_decoding_workers: Number of decoding workers
            block_size: KV cache block size
            max_num_gpu_blocks: Maximum GPU blocks
            max_num_cpu_blocks: Maximum CPU blocks
            dtype: Model dtype
            tensor_parallel_size: Tensor parallel size
            gpu_memory_utilization: GPU memory utilization
            kv_transfer_method: KV transfer method ('cuda_ipc', 'nccl', 'p2p_copy')
        """
        self.model_path = model_path
        self.num_encoding_workers = num_encoding_workers
        self.num_prefill_workers = num_prefill_workers
        self.num_decoding_workers = num_decoding_workers
        
        # Service discovery configuration
        self.sd_host = sd_host
        self.sd_port = sd_port
        self.proxy_port = proxy_port
        
        # Engine configuration
        self.engine_config = {
            'model_path': model_path,
            'block_size': block_size,
            'max_num_gpu_blocks': max_num_gpu_blocks,
            'max_num_cpu_blocks': max_num_cpu_blocks,
            'dtype': dtype,
            'tensor_parallel_size': tensor_parallel_size,
            'gpu_memory_utilization': gpu_memory_utilization,
            'limit_mm_per_prompt': limit_mm_per_prompt,
        }
        
        # Bridge queues for stage communication
        self.encode_prefill_bridge_queue = asyncio.Queue()
        self.prefill_decode_bridge_queue = asyncio.Queue()
        
        # Output queue (global queue for backward compatibility)
        self.output_queue: deque[StepOutput] = deque()
        
        # Per-request output tracking (for HTTP streaming with multiple concurrent requests)
        self._request_outputs: Dict[str, List[StepOutput]] = {}
        self._request_output_events: Dict[str, asyncio.Event] = {}
        
        # NCCL coordinator for global P2P
        self.nccl_coordinator = NCCLCoordinator(master_addr="127.0.0.1", master_port=29600)
        
        # KV transfer manager (updated to use NCCL coordinator)
        transfer_method_enum = TransferMethod[kv_transfer_method.upper()]
        self.kv_transfer_manager = V0KVTransferManager(
            transfer_method=transfer_method_enum,
            nccl_coordinator=self.nccl_coordinator,
            enable_batching=False,  # ❌ DISABLED: Batching causes incomplete transfers (requests queued but never flushed)
            batch_size=3,           # Not used when batching disabled
            batch_timeout=0.005,    # Not used when batching disabled
            metrics_callback=self._record_kv_transfer_metrics  # ✅ Real-time metrics recording
        )
        
        # NCCL transfer manager (legacy, kept for compatibility)
        self.nccl_transfer_manager = V0NCCLTransferManager(base_port=29500)
        
        # ✅ Global Memory Coordinator (vLLM-style)
        self.memory_coordinator = GlobalMemoryCoordinator(
            high_watermark=0.90,  # Stop accepting at 90% usage
            low_watermark=0.50,   # Resume at 50% usage
        )
        
        # Stage engines
        self.encoding_engine: Optional[V0EncodingEngine] = None
        self.prefill_engine: Optional[V0PrefillEngine] = None
        self.decoding_engine: Optional[V0DecodingEngine] = None
        
        # Event loop tasks
        self.engine_tasks: List[asyncio.Task] = []
        
        # Statistics
        self.total_requests_received = 0
        self.total_requests_completed = 0
        
        # Performance metrics (for profiling and Gain-Cost calibration)
        self.encoding_metrics = StageMetrics(stage_name="encoding", sample_interval=10)
        self.prefill_metrics = StageMetrics(stage_name="prefill", sample_interval=10)
        self.decoding_metrics = StageMetrics(stage_name="decoding", sample_interval=10)
        
        # Heartbeat
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_heartbeat = False
        
        # Tokenizer (initialized during init())
        self.tokenizer = None
        
        print(f"[V0EngineBackend] Initialized with {num_encoding_workers}E + "
              f"{num_prefill_workers}P + {num_decoding_workers}D workers")
    
    async def initialize(self):
        """Initialize all stage engines"""
        print("[V0EngineBackend] Initializing stage engines...")
        
        # Register all workers with NCCL coordinator
        for stage_name, num_workers in [
            ("encoding", self.num_encoding_workers),
            ("prefill", self.num_prefill_workers),
            ("decoding", self.num_decoding_workers)
        ]:
            for worker_id in range(num_workers):
                self.nccl_coordinator.register_worker(stage_name, worker_id)
        
        
        # Initialize tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        # Create encoding engine
        self.encoding_engine = V0EncodingEngine(
            encode_prefill_bridge_queue=self.encode_prefill_bridge_queue,
            num_workers=self.num_encoding_workers,
            nccl_coordinator=self.nccl_coordinator,
            backend_ref=self,  # ✅ Pass backend reference for migration
            **self.engine_config
        )
        
        # Create prefill engine
        self.prefill_engine = V0PrefillEngine(
            encode_prefill_bridge_queue=self.encode_prefill_bridge_queue,
            prefill_decode_bridge_queue=self.prefill_decode_bridge_queue,
            num_workers=self.num_prefill_workers,
            nccl_transfer_manager=self.nccl_transfer_manager,
            kv_transfer_manager=self.kv_transfer_manager,
            nccl_coordinator=self.nccl_coordinator,
            memory_coordinator=self.memory_coordinator,  # ✅ Pass coordinator
            backend_ref=self,  # ✅ Pass backend reference for migration
            **self.engine_config
        )
        
        # Store encoding workers reference for prefill to access during vision embedding transfer
        self.prefill_engine.encoding_workers = self.encoding_engine.workers
        
        # Create decoding engine with output callback
        self.decoding_engine = V0DecodingEngine(
            prefill_decode_bridge_queue=self.prefill_decode_bridge_queue,
            num_workers=self.num_decoding_workers,
            output_callback=self._on_decode_output,
            nccl_transfer_manager=self.nccl_transfer_manager,
            kv_transfer_manager=self.kv_transfer_manager,
            nccl_coordinator=self.nccl_coordinator,
            memory_coordinator=self.memory_coordinator,  # ✅ Pass coordinator
            backend_ref=self,  # ✅ Pass backend reference for migration
            **self.engine_config
        )
        
        # Store prefill workers reference for decode to access during KV cache transfer
        self.decoding_engine.prefill_workers = self.prefill_engine.workers
        
        # Set metrics references for all engines
        self.encoding_engine._metrics_ref = self
        self.prefill_engine._metrics_ref = self
        self.decoding_engine._metrics_ref = self
        
        # Initialize engines in parallel
        await asyncio.gather(
            self.encoding_engine.initialize(),
            self.prefill_engine.initialize(),
            self.decoding_engine.initialize(),
        )
        

        # ✅ NEW: Register all workers with KV transfer manager using their global ranks
        self._register_workers_with_kv_manager()
    
    def _register_workers_with_kv_manager(self):
        """
        ✅ NEW: Register all workers with KV transfer manager using global ranks.
        This decouples NCCL layer from logical layer.
        """
        if not self.kv_transfer_manager:
            return
        
        # Register encoding workers
        for worker_id, worker in enumerate(self.encoding_engine.workers):
            if worker is not None:
                global_rank = self.nccl_coordinator.get_rank("encoding", worker_id)
                if global_rank >= 0:
                    self.kv_transfer_manager.register_worker(global_rank, worker, "encoding")
        
        # Register prefill workers
        for worker_id, worker in enumerate(self.prefill_engine.workers):
            if worker is not None:
                global_rank = self.nccl_coordinator.get_rank("prefill", worker_id)
                if global_rank >= 0:
                    self.kv_transfer_manager.register_worker(global_rank, worker, "prefill")
        
        # Register decoding workers
        for worker_id, worker in enumerate(self.decoding_engine.workers):
            if worker is not None:
                global_rank = self.nccl_coordinator.get_rank("decoding", worker_id)
                if global_rank >= 0:
                    self.kv_transfer_manager.register_worker(global_rank, worker, "decoding")
        
    
    async def start(self):
        """Start all stage engines"""
        
        if not all([self.encoding_engine, self.prefill_engine, self.decoding_engine]):
            raise RuntimeError("Engines not initialized. Call initialize() first.")
        
        print("[V0Backend] Starting event loops for all stage engines...")
        
        # Start event loops for all stages
        encoding_task = asyncio.create_task(self.encoding_engine.start_event_loop())
        prefill_task = asyncio.create_task(self.prefill_engine.start_event_loop())
        decoding_task = asyncio.create_task(self.decoding_engine.start_event_loop())
        
        self.engine_tasks = [encoding_task, prefill_task, decoding_task]
        
        print(f"[V0Backend] Event loop tasks created: encoding={not encoding_task.done()}, prefill={not prefill_task.done()}, decoding={not decoding_task.done()}")
        
        # Give tasks a moment to start
        await asyncio.sleep(0.1)
        
        # Check if tasks are running
        print(f"[V0Backend] After 0.1s: encoding={not encoding_task.done()}, prefill={not prefill_task.done()}, decoding={not decoding_task.done()}")
        
        # Start heartbeat
        self._start_heartbeat()
    
    def _start_heartbeat(self):
        """Start heartbeat thread to register with scheduler"""
        def heartbeat_worker():
            import msgpack
            import zmq
            context = zmq.Context()
            
            # Use configured ZMQ service discovery port
            zmq_port = self.sd_port
            zmq_host = self.sd_host if self.sd_host != "0.0.0.0" else "127.0.0.1"
            print(f"🔗 V0 backend connecting to scheduler ZMQ at {zmq_host}:{zmq_port}")
            time.sleep(2)  # Wait for scheduler to be ready
            
            # Heartbeat success counter for less verbose output
            heartbeat_success_count = 0
            
            while not self._stop_heartbeat:
                try:
                    # Send heartbeat for each instance (create new socket for each)
                    success_count = 0
                    failed_instances = []
                    
                    for instance_id in self.get_all_instances():
                        instance_info = self.get_instance_info(instance_id)
                        stage = instance_info['stage']
                        
                        # Only send heartbeat for prefill and decode stages (proxy only handles P and D)
                        if stage not in ["prefill", "decode"]:
                            continue
                            
                        # Determine role based on stage
                        role = "P" if stage == "prefill" else "D"
                        
                        heartbeat_data = {
                            "http_address": f"127.0.0.1:{self.proxy_port}",  # Proxy HTTP address
                            "zmq_address": f"127.0.0.1:{self.sd_port}",   # ZMQ service discovery address
                            "type": role,
                            "instance_id": instance_id
                        }
                        
                        # Create new socket for each heartbeat to avoid state issues
                        socket = context.socket(zmq.REQ)
                        socket.setsockopt(zmq.LINGER, 0)
                        socket.setsockopt(zmq.RCVTIMEO, 2000)  # 2 second timeout for recv
                        socket.connect(f"tcp://{zmq_host}:{zmq_port}")
                        
                        try:
                            # Send heartbeat
                            socket.send(msgpack.packb(heartbeat_data))
                            
                            # Wait for response with shorter timeout
                            try:
                                response = socket.recv()
                                if response == b"OK":
                                    success_count += 1
                                else:
                                    failed_instances.append((instance_id, f"Unexpected response: {response}"))
                            except zmq.Again:
                                failed_instances.append((instance_id, "Timeout"))
                        except Exception as e:
                            failed_instances.append((instance_id, str(e)))
                        finally:
                            socket.close()
                    
                    # Only print summary, not every heartbeat
                    if failed_instances:
                        for inst_id, reason in failed_instances:
                            print(f"  ✗ {inst_id}: {reason}")
                    else:
                        heartbeat_success_count += 1
                        # Only print success every 6 cycles (60 seconds)
                        if heartbeat_success_count % 6 == 1:
                            print(f"[V0Backend] Heartbeat OK (cycle {heartbeat_success_count})")
                    
                    time.sleep(10)  # Send heartbeat every 10 seconds
                    
                except Exception as e:
                    print(f"[V0Backend] Heartbeat error: {e}")
                    time.sleep(10)
            
            context.term()
        
        self._heartbeat_thread = threading.Thread(target=heartbeat_worker, daemon=True)
        self._heartbeat_thread.start()
    
    async def stop(self):
        """Stop all stage engines"""
        
        # Stop heartbeat first
        self._stop_heartbeat = True
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=2)  # Reduced timeout
        
        # Stop event loops with timeout
        stop_tasks = []
        if self.encoding_engine:
            stop_tasks.append(self.encoding_engine.stop_event_loop())
        if self.prefill_engine:
            stop_tasks.append(self.prefill_engine.stop_event_loop())
        if self.decoding_engine:
            stop_tasks.append(self.decoding_engine.stop_event_loop())
        
        # Wait for event loops to stop with timeout
        if stop_tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*stop_tasks, return_exceptions=True), timeout=10.0)
            except asyncio.TimeoutError:
                print("[V0Backend] Event loop stop timed out, continuing...")
        
        # Cancel engine tasks if still running
        if self.engine_tasks:
            for task in self.engine_tasks:
                if not task.done():
                    task.cancel()
            # Wait with timeout
            try:
                await asyncio.wait_for(asyncio.gather(*self.engine_tasks, return_exceptions=True), timeout=5.0)
            except asyncio.TimeoutError:
                print("[V0Backend] Engine tasks cancellation timed out, continuing...")
        

        # Note: destroy_all_collective_groups() can only be called from within a Ray actor
        # Since we're using PyTorch P2P (not Ray Collective), we don't need to destroy groups
        # The workers will be destroyed automatically when they go out of scope
    
    async def add_request(self, request: Request):
        """
        Add request to appropriate stage based on modality
        
        Args:
            request: Request to add
        """
        # CRITICAL: Tokenize prompt if not already done
        if request.prompt_token_ids is None and request.prompt:
            if self.tokenizer is None:
                raise RuntimeError("Tokenizer not initialized. Call initialize() first.")
            request.prompt_token_ids = self.tokenizer.encode(request.prompt)
        
        # Initialize per-request output tracking for this request
        if request.request_id not in self._request_outputs:
            self._request_outputs[request.request_id] = []
            self._request_output_events[request.request_id] = asyncio.Event()
        
        # Determine if this is a multimodal request
        is_multimodal = bool(request.multi_modal_data)
        
        print(f"[V0Backend] Adding request {request.request_id}")
        print(f"[V0Backend] Request prompt length: {len(request.prompt) if request.prompt else 0}")
        print(f"[V0Backend] Request is_multimodal: {is_multimodal}")
        
        if is_multimodal:
            # Multimodal requests: go through encoding -> prefill -> decode
            # Encoding stage handles vision processing
            if not self.encoding_engine:
                raise RuntimeError("Backend not initialized: encoding_engine is None")
            print(f"[V0Backend] Routing multimodal request {request.request_id} to encoding engine")
            self.encoding_engine.add_request(request)
        else:
            # Text-only requests: skip encoding, go directly to prefill
            # Use prefill_0 worker (TEXT_ONLY modality)
            if not self.prefill_engine:
                raise RuntimeError("Backend not initialized: prefill_engine is None")
            print(f"[V0Backend] Routing text-only request {request.request_id} directly to prefill engine (skipping encoding)")
            # For text-only requests, we need to create a "migrated" request that bypasses encoding
            # The prefill engine expects requests from encoding, so we need to add it directly
            # Create a MigratingRequest wrapper for compatibility
            migrating_req = MigratingRequest(
                req=request,
                vision_embeddings=None,  # No vision embeddings for text-only
                vision_block_indexes=None,  # No vision blocks for text-only
                multi_modal_kwargs=None,
                multi_modal_placeholders=None,
                source_stage=None,  # Text-only requests don't come from encoding
                target_stage=None
            )
            # Add directly to prefill engine's bridge queue (simulating encoding->prefill migration)
            await self.encode_prefill_bridge_queue.put(migrating_req)
            print(f"[V0Backend] Text-only request {request.request_id} added to prefill bridge queue")
        
        self.total_requests_received += 1
        print(f"[V0Backend] Request {request.request_id} added successfully, total_requests_received={self.total_requests_received}")
    
    async def get_outputs(self, request_id: Optional[str] = None) -> List[StepOutput]:
        """
        Get outputs from output queue
        
        Args:
            request_id: Optional request ID to filter outputs. If None, returns all outputs and clears queue.
        
        Returns:
            List of step outputs
        """
        if request_id is not None:
            # Return outputs for specific request (non-destructive)
            outputs = self._request_outputs.get(request_id, [])
            if outputs:
                # Clear the outputs for this request (already consumed)
                self._request_outputs[request_id] = []
            return outputs
        
        # Return all outputs and clear queue (backward compatibility)
        outputs = []
        while self.output_queue:
            outputs.append(self.output_queue.popleft())
        return outputs
    
    def _on_decode_output(self, output: StepOutput):
        """Callback for decode outputs"""
        # Add to global queue (backward compatibility)
        self.output_queue.append(output)
        
        # Also store per-request for HTTP streaming
        request_id = output.request_id
        if request_id not in self._request_outputs:
            self._request_outputs[request_id] = []
        self._request_outputs[request_id].append(output)
        
        # Signal event if waiting for this request
        if request_id in self._request_output_events:
            self._request_output_events[request_id].set()
        
        if output.finished:
            self.total_requests_completed += 1
            # Clean up per-request tracking after finished
            # Keep outputs for a short time in case of late reads, but remove event
            if request_id in self._request_output_events:
                del self._request_output_events[request_id]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get backend statistics"""
        encoding_stats = {
            'num_workers': self.encoding_engine.num_workers if self.encoding_engine else 0,
            'num_waiting': self.encoding_engine.scheduler.num_waiting_requests() if self.encoding_engine else 0,
            'num_running': self.encoding_engine.scheduler.num_running_requests() if self.encoding_engine else 0,
        }
        
        prefill_stats = {
            'num_workers': self.prefill_engine.num_workers if self.prefill_engine else 0,
            'num_waiting': self.prefill_engine.scheduler.num_waiting_requests() if self.prefill_engine else 0,
            'num_running': self.prefill_engine.scheduler.num_running_requests() if self.prefill_engine else 0,
        }
        
        decoding_stats = {
            'num_workers': self.decoding_engine.num_workers if self.decoding_engine else 0,
            'num_waiting': self.decoding_engine.scheduler.num_waiting_requests() if self.decoding_engine else 0,
            'num_running': self.decoding_engine.scheduler.num_running_requests() if self.decoding_engine else 0,
        }
        
        return {
            'backend_type': 'v0',
            'total_requests_received': self.total_requests_received,
            'total_requests_completed': self.total_requests_completed,
            'encoding': encoding_stats,
            'prefill': prefill_stats,
            'decoding': decoding_stats,
            'kv_transfer': self.kv_transfer_manager.get_stats(),
        }
    
    def get_decode_token_count(self) -> int:
        """
        Get total token count in decode stage (for memory pressure detection)
        
        Returns:
            Total number of tokens currently in decode batch
        """
        if not self.decoding_engine:
            return 0
        
        total_tokens = 0
        
        # Count tokens in running requests (set of request_ids)
        for request_id in self.decoding_engine.scheduler.running_requests:
            # Get block table for this request
            blocks = self.decoding_engine.block_manager.get_block_table(request_id)
            if blocks:
                # Estimate tokens from blocks (blocks * block_size)
                total_tokens += len(blocks) * self.decoding_engine.block_manager.block_size
        
        # Count tokens in waiting requests (list of Request objects)
        for req in self.decoding_engine.scheduler.waiting_queue:
            prompt_len = len(req.prompt_token_ids) if req.prompt_token_ids else 0
            output_len = len(req.output_token_ids) if req.output_token_ids else 0
            total_tokens += (prompt_len + output_len)
        
        return total_tokens
    
    def get_worker_allocation(self) -> Dict[int, str]:
        """
        获取当前worker分配映射
        
        Returns:
            Dict[int, str]: {worker_id: stage_name}
            例如: {0: 'encoding', 1: 'prefill', 2: 'prefill', 3: 'decoding', 4: 'decoding'}
        """
        allocation = {}
        worker_id = 0
        
        # Encoding workers
        if self.encoding_engine:
            for i in range(self.encoding_engine.num_workers):
                allocation[worker_id] = 'encoding'
                worker_id += 1
        
        # Prefill workers
        if self.prefill_engine:
            for i in range(self.prefill_engine.num_workers):
                allocation[worker_id] = 'prefill'
                worker_id += 1
        
        # Decoding workers
        if self.decoding_engine:
            for i in range(self.decoding_engine.num_workers):
                allocation[worker_id] = 'decoding'
                worker_id += 1
        
        return allocation
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics for profiling"""
        return {
            'encoding': self.encoding_metrics.get_stats(),
            'prefill': self.prefill_metrics.get_stats(),
            'decoding': self.decoding_metrics.get_stats(),
        }
    
    def export_metrics_to_json(self, filepath: str):
        """Export performance metrics to JSON file"""
        import json
        metrics = self.get_performance_metrics()
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def _record_kv_transfer_metrics(self, transfer_time_sec: float, num_blocks: int):
        """Callback to record KV transfer metrics (called by KV transfer manager)"""
        # Record to appropriate stage metrics (decode stage receives KV from prefill)
        self.decoding_metrics.record_kv_transfer(transfer_time_sec, num_blocks)
    
    async def migrate_instance(
        self,
        src_instance_id: str,
        dst_instance_id: str,
        requests: List[Request]
    ) -> bool:
        """
        Migrate requests between instances
        
        For v0 backend, this involves migrating KV cache blocks
        
        Args:
            src_instance_id: Source instance ID (e.g., "decoding_0")
            dst_instance_id: Destination instance ID (e.g., "decoding_1")
            requests: Requests to migrate (can be None to migrate all)
            
        Returns:
            True if migration successful
        """
        if not requests or len(requests) == 0:
            return True
        
        
        try:
            # Step 1: Parse instance IDs and get workers
            src_stage_name, src_worker_id = self._parse_instance_id(src_instance_id)
            dst_stage_name, dst_worker_id = self._parse_instance_id(dst_instance_id)
            
            # Step 2: Get stage engines and workers
            src_engine = self._get_stage_engine(src_stage_name)
            dst_engine = self._get_stage_engine(dst_stage_name)
            
            if not src_engine or not dst_engine:
                print(f"[V0EngineBackend] ❌ Engine not found: src={src_stage_name}, dst={dst_stage_name}")
                return False
            
            # Validate worker IDs
            if src_worker_id >= len(src_engine.workers) or src_engine.workers[src_worker_id] is None:
                print(f"[V0EngineBackend] ❌ Source worker {src_instance_id} not found")
                return False
            
            if dst_worker_id >= len(dst_engine.workers) or dst_engine.workers[dst_worker_id] is None:
                print(f"[V0EngineBackend] ❌ Destination worker {dst_instance_id} not found")
                return False
            
            src_worker = src_engine.workers[src_worker_id]
            dst_worker = dst_engine.workers[dst_worker_id]
            
            # Step 3: Migrate each request's KV cache
            migration_start = time.time()
            successful_migrations = 0
            
            for req in requests:
                request_id = req.request_id
                
                # Get source blocks
                src_blocks = src_engine.block_manager.get_block_table(request_id)
                if not src_blocks:
                    print(f"[V0EngineBackend] ⚠️  No blocks found for request {request_id}, skipping")
                    continue
                
                # Allocate destination blocks
                num_blocks_needed = len(src_blocks)
                try:
                    dst_engine.block_manager.allocate_blocks(request_id, num_blocks_needed)
                except Exception as e:
                    print(f"[V0EngineBackend] ❌ Failed to allocate {num_blocks_needed} blocks for {request_id}: {e}")
                    # Clean up: free any already allocated blocks
                    for prev_req in requests[:successful_migrations]:
                        dst_engine.block_manager.free_blocks(prev_req.request_id)
                    return False
                
                dst_blocks = dst_engine.block_manager.get_block_table(request_id)
                
                # ✅ NEW: Transfer KV cache using ranks (decoupled from stages)
                if self.kv_transfer_manager:
                    # Convert instance IDs to global ranks
                    src_rank = self._instance_id_to_rank(src_instance_id)
                    dst_rank = self._instance_id_to_rank(dst_instance_id)
                    
                    transfer_success = await self.kv_transfer_manager.transfer_kv_cache(
                        request_id=request_id,
                        src_rank=src_rank,
                        dst_rank=dst_rank,
                        src_blocks=src_blocks,
                        dst_blocks=dst_blocks
                    )
                    
                    if not transfer_success:
                        print(f"[V0EngineBackend] ❌ KV transfer failed for {request_id}")
                        # Clean up
                        dst_engine.block_manager.free_blocks(request_id)
                        for prev_req in requests[:successful_migrations]:
                            dst_engine.block_manager.free_blocks(prev_req.request_id)
                        return False
                
                successful_migrations += 1
            
            # Step 4: Update scheduler states
            # Move requests from source to destination in the scheduler
            for req in requests:
                # Remove from source scheduler
                if req.request_id in src_engine.scheduler.running_requests:
                    src_engine.scheduler.running_requests.pop(req.request_id)
                
                # Add to destination scheduler
                dst_engine.scheduler.running_requests[req.request_id] = req
            
            # Step 5: Free source blocks
            for req in requests:
                src_engine.block_manager.free_blocks(req.request_id)
            
            migration_time = time.time() - migration_start
            
            return successful_migrations == len(requests)
            
        except Exception as e:
            print(f"[V0EngineBackend] ❌ Migration failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def switch_worker_role(
        self,
        worker_id: int,
        from_stage: str,
        to_stage: str,
        migrate_kv: bool = True
    ) -> bool:
        """
        Switch a worker's logical role (e.g., Decode → Prefill).
        
        This is a core method for elastic scheduling, enabling workers to change
        their roles without destroying/recreating the worker or reinitializing NCCL.
        
        Steps:
        1. Get worker reference from source stage
        2. Migrate active requests if requested (to other workers in same stage)
        3. Remove worker from source stage engine's worker list
        4. Call worker.switch_role() to update internal state
        5. Add worker to target stage engine's worker list
        6. Update NCCL coordinator and KV transfer manager mappings
        
        Args:
            worker_id: Worker ID (also GPU ID / NCCL rank)
            from_stage: Source stage name ('encoding', 'prefill', 'decoding')
            to_stage: Target stage name ('encoding', 'prefill', 'decoding')
            migrate_kv: Whether to migrate active requests before switching
            
        Returns:
            True if role switch successful
            
        Example:
            # Switch decode worker 3 to prefill
            await backend.switch_worker_role(
                worker_id=3,
                from_stage='decoding',
                to_stage='prefill'
            )
        """
        
        # Step 1: Validate stages
        src_engine = self._get_stage_engine(from_stage)
        dst_engine = self._get_stage_engine(to_stage)
        
        if not src_engine:
            print(f"[V0EngineBackend] ❌ Source stage '{from_stage}' not found")
            return False
        
        if not dst_engine:
            print(f"[V0EngineBackend] ❌ Target stage '{to_stage}' not found")
            return False
        
        # Step 2: Get worker reference
        if worker_id >= len(src_engine.workers) or src_engine.workers[worker_id] is None:
            print(f"[V0EngineBackend] ❌ Worker {worker_id} not found in {from_stage}")
            return False
        
        worker_ref = src_engine.workers[worker_id]
        
        # Step 3: Get active requests on this worker
        active_requests = []
        for request_id, req in src_engine.scheduler.running_requests.items():
            blocks = src_engine.block_manager.get_block_table(request_id)
            if blocks:  # This request has blocks allocated (on this worker)
                active_requests.append(req)
        
        
        # Step 4: Migrate KV cache if requested
        if migrate_kv and active_requests:
            
            # Calculate total blocks needed
            total_blocks_needed = sum(
                len(src_engine.block_manager.get_block_table(req.request_id))
                for req in active_requests
            )
            
            # Select target worker with most KV slots
            target_worker_id = src_engine._select_target_by_kv_slots(
                exclude_worker_id=worker_id,
                required_blocks=total_blocks_needed
            )
            
            if target_worker_id is None:
                print(f"[V0EngineBackend] ❌ No target worker with sufficient KV slots ({total_blocks_needed} blocks needed)")
                return False
            
            # Migrate using existing migrate_instance method
            src_instance_id = f"{from_stage}_{worker_id}"
            dst_instance_id = f"{from_stage}_{target_worker_id}"
            
            success = await self.migrate_instance(
                src_instance_id=src_instance_id,
                dst_instance_id=dst_instance_id,
                requests=active_requests
            )
            
            if not success:
                print(f"[V0EngineBackend] ❌ KV migration failed")
                return False
            
        
        # Step 5: Remove worker from source stage (logical removal, don't destroy)
        src_engine.workers[worker_id] = None
        # Don't decrement num_workers here as we'll add to target
        
        # Step 6: Call worker.switch_role() to update internal state
        switch_success = await worker_ref.switch_role.remote(to_stage)
        
        if not switch_success:
            print(f"[V0EngineBackend] ❌ Worker failed to switch role")
            # Rollback: add worker back to source stage
            src_engine.workers[worker_id] = worker_ref
            return False
        
        # Step 6.5: Initialize ve_cache if switching to encoding or prefill
        # ⚠️  CRITICAL: Prefill and encoding stages need ve_cache for vision embeddings
        if to_stage in ['encoding', 'prefill']:
            try:
                await worker_ref.init_vecache.remote(
                    num_gpu_blocks=3000,
                    num_cpu_blocks=100
                )
                print(f"[V0EngineBackend] ✅ Initialized ve_cache for worker {worker_id} in {to_stage} stage")
            except Exception as e:
                print(f"[V0EngineBackend] ⚠️  Warning: Failed to initialize ve_cache: {e}")
        
        # Step 7: Add worker to target stage
        
        # Ensure target stage's worker list is large enough
        if worker_id >= len(dst_engine.workers):
            dst_engine.workers.extend([None] * (worker_id - len(dst_engine.workers) + 1))
        
        dst_engine.workers[worker_id] = worker_ref
        dst_engine.num_workers = len([w for w in dst_engine.workers if w is not None])
        src_engine.num_workers = len([w for w in src_engine.workers if w is not None])
        
        # Step 8: Reinitialize modality groups after worker changes
        # ⚠️  CRITICAL: Must update modality groups after worker list changes
        src_engine._init_modality_groups()
        dst_engine._init_modality_groups()
        print(f"[V0EngineBackend] ✅ Modality groups updated for both {from_stage} and {to_stage} engines")
        
        # Step 9: Update NCCL coordinator mapping
        self.nccl_coordinator.update_worker_role(worker_id, from_stage, to_stage)
        
        # Step 10: Update KV transfer manager stage tracking
        # ✅ NEW: Only update stage mapping, rank stays the same!
        global_rank = self.nccl_coordinator.get_rank(to_stage, worker_id)
        if global_rank >= 0:
            self.kv_transfer_manager.update_worker_stage(global_rank, to_stage)
        
        
        return True
    
    async def cascading_role_switch(
        self,
        worker_id: int,
        from_stage: str,
        to_stage: str
    ) -> bool:
        """
        Cascading role switch with automatic rebalancing.
        
        This performs a role switch and then triggers load rebalancing in the
        target stage to distribute workload across all workers.
        
        Steps:
        1. Switch worker role (includes source KV migration)
        2. Trigger rebalancing in target stage
        
        Args:
            worker_id: Worker ID (GPU ID)
            from_stage: Source stage name
            to_stage: Target stage name
            
        Returns:
            True if successful
            
        Example:
            # Decode worker 3 → Prefill (with auto-rebalance)
            await backend.cascading_role_switch(
                worker_id=3,
                from_stage='decoding',
                to_stage='prefill'
            )
        """
        
        # Step 1: Perform role switch
        success = await self.switch_worker_role(
            worker_id=worker_id,
            from_stage=from_stage,
            to_stage=to_stage,
            migrate_kv=True
        )
        
        if not success:
            print(f"[V0EngineBackend] ❌ Role switch failed")
            return False
        
        # Step 2: Trigger rebalancing in target stage
        dst_engine = self._get_stage_engine(to_stage)
        if dst_engine:
            await self._rebalance_stage(dst_engine)
        
        
        return True
    
    async def _rebalance_stage(self, stage_engine):
        """
        Rebalance workload within a stage.
        
        Checks KV cache usage across all workers and migrates requests from
        heavily-loaded workers to lightly-loaded workers if imbalance detected.
        
        Args:
            stage_engine: Stage engine to rebalance
        """
        stage_name = stage_engine.stage.value
        
        # Step 1: Collect worker loads
        worker_loads = []
        for i in range(len(stage_engine.workers)):
            if stage_engine.workers[i] is None:
                continue
            
            # Count active requests on this worker
            worker_requests = []
            for request_id, req in stage_engine.scheduler.running_requests.items():
                blocks = stage_engine.block_manager.get_block_table(request_id)
                if blocks:
                    worker_requests.append(req)
            
            # Calculate blocks used
            total_blocks = stage_engine.block_manager.max_num_gpu_blocks or 1000
            free_blocks = stage_engine.block_manager.get_num_avail_gpu_blocks()
            used_blocks = total_blocks - free_blocks
            usage_ratio = used_blocks / total_blocks if total_blocks > 0 else 0
            
            worker_loads.append({
                'worker_id': i,
                'usage_ratio': usage_ratio,
                'free_blocks': free_blocks,
                'num_requests': len(worker_requests),
                'requests': worker_requests
            })
            
            print(f"  Worker {i}: {usage_ratio:.1%} usage, {len(worker_requests)} requests, {free_blocks} free blocks")
        
        if len(worker_loads) < 2:
            return
        
        # Step 2: Sort by usage (highest first)
        worker_loads.sort(key=lambda x: x['usage_ratio'], reverse=True)
        
        highest = worker_loads[0]
        lowest = worker_loads[-1]
        usage_diff = highest['usage_ratio'] - lowest['usage_ratio']
        
        
        # Step 3: Rebalance if imbalance exceeds threshold
        REBALANCE_THRESHOLD = 0.3  # 30% difference
        
        if usage_diff > REBALANCE_THRESHOLD and lowest['free_blocks'] > 0:
            print(f"\n[V0EngineBackend] ⚠️  Imbalance detected (>{REBALANCE_THRESHOLD:.0%}), rebalancing...")
            
            # Calculate how many requests to migrate
            # Target: move 1-2 requests from high to low
            num_to_migrate = min(2, len(highest['requests']), lowest['free_blocks'] // 10)
            
            if num_to_migrate > 0:
                requests_to_migrate = highest['requests'][:num_to_migrate]
                
                src_instance_id = f"{stage_name}_{highest['worker_id']}"
                dst_instance_id = f"{stage_name}_{lowest['worker_id']}"
                
                
                success = await self.migrate_instance(
                    src_instance_id=src_instance_id,
                    dst_instance_id=dst_instance_id,
                    requests=requests_to_migrate
                )
                
                if success:
                    print(f"[V0EngineBackend] ✅ Rebalanced: migrated {num_to_migrate} request(s)")
                else:
                    print(f"[V0EngineBackend] ❌ Rebalancing migration failed")
            else:
                print(f"[V0EngineBackend] ℹ️  No requests to migrate (low worker at capacity)")
        else:
            print(f"[V0EngineBackend] ✅ Load is balanced (within {REBALANCE_THRESHOLD:.0%} threshold)")
    
    def _parse_instance_id(self, instance_id: str) -> tuple:
        """
        Parse instance_id to get stage name and worker ID
        
        Args:
            instance_id: Instance ID (e.g., "decoding_0", "prefill_1")
            
        Returns:
            Tuple of (stage_name, worker_id)
        """
        parts = instance_id.split('_')
        if len(parts) != 2:
            raise ValueError(f"Invalid instance_id format: {instance_id}")
        
        stage_name, worker_id_str = parts
        worker_id = int(worker_id_str)
        
        return stage_name, worker_id
    
    def _instance_id_to_rank(self, instance_id: str) -> int:
        """
        ✅ NEW: Convert instance_id to global NCCL rank.
        
        Args:
            instance_id: Format "stage_worker_id" (e.g., "decoding_2")
            
        Returns:
            global_rank: Worker's NCCL rank
        """
        parts = instance_id.split('_')
        if len(parts) != 2:
            raise ValueError(f"Invalid instance_id format: {instance_id}. Expected 'stage_workerid'")
        
        stage_name = parts[0]
        worker_id = int(parts[1])
        
        # Get rank from NCCL coordinator
        rank = self.nccl_coordinator.get_rank(stage_name, worker_id)
        if rank < 0:
            raise ValueError(f"Worker not found in NCCL coordinator: {instance_id}")
        
        return rank
    
    def _get_stage_engine(self, stage_name: str):
        """
        Get stage engine by name
        
        Args:
            stage_name: Stage name ("encoding", "prefill", "decoding")
            
        Returns:
            Stage engine instance or None
        """
        if stage_name == "encoding":
            return self.encoding_engine
        elif stage_name == "prefill":
            return self.prefill_engine
        elif stage_name == "decoding":
            return self.decoding_engine
        else:
            return None
    
    def get_num_instances(self) -> int:
        """Get total number of instances"""
        return (
            self.num_encoding_workers +
            self.num_prefill_workers +
            self.num_decoding_workers
        )
    
    def get_instance_info(self, instance_id: str) -> Dict[str, Any]:
        """Get information about a specific instance"""
        # Parse instance_id to determine stage and worker
        # Format: "encoding_0", "prefill_2", "decoding_1", etc.
        
        parts = instance_id.split('_')
        if len(parts) != 2:
            return {}
        
        stage, worker_id = parts[0], int(parts[1])
        
        # Determine modality and stage enum values
        # V0 backend configuration:
        # - Text group: prefill_0, decoding_0 (1 prefill + 1 decode)
        # - Multimodal group: encoding_0, encoding_1, prefill_1, prefill_2, decoding_1, decoding_2
        #   (2 encode + 2 prefill + 2 decode)
        modality = ModalityType.MULTIMODAL  # Default
        stage_enum = InferenceStage.PREFILL  # Default
        
        if stage == "encoding":
            # All encoding workers are MULTIMODAL (text-only doesn't need encoding)
            stage_enum = InferenceStage.ENCODE
            modality = ModalityType.MULTIMODAL
        elif stage == "prefill":
            stage_enum = InferenceStage.PREFILL
            # prefill_0 is TEXT_ONLY, others are MULTIMODAL
            if worker_id == 0:
                modality = ModalityType.TEXT_ONLY
            else:
                modality = ModalityType.MULTIMODAL
        elif stage == "decoding":
            stage_enum = InferenceStage.DECODE
            # decoding_0 is TEXT_ONLY, others are MULTIMODAL
            if worker_id == 0:
                modality = ModalityType.TEXT_ONLY
            else:
                modality = ModalityType.MULTIMODAL
        
        return {
            'instance_id': instance_id,
            'modality': modality.value,
            'stage': stage_enum.value,
            'stage_name': stage,
            'worker_id': worker_id,
            'status': 'active',  # TODO: Get actual status
        }
    
    def can_add_request(self, request: Request) -> bool:
        """Check if backend can accept a new request"""
        if not self.encoding_engine:
            return False
        
        # Check if encoding stage has capacity
        # Simple heuristic: check queue length
        waiting = self.encoding_engine.scheduler.num_waiting_requests()
        running = self.encoding_engine.scheduler.num_running_requests()
        
        # Allow if total < 100 (configurable)
        return (waiting + running) < 100
    
    # Dynamic switching methods
    
    async def add_instance(self, instance_id: str, modality: ModalityType, stage: InferenceStage):
        """
        Add a new instance dynamically
        
        Args:
            instance_id: Instance identifier
            modality: Modality type (TEXT_ONLY or MULTIMODAL)
            stage: Inference stage (ENCODE, PREFILL, or DECODE)
        """
        
        # Parse instance_id to determine stage and worker
        parts = instance_id.split('_')
        if len(parts) != 2:
            raise ValueError(f"Invalid instance_id format: {instance_id}")
        
        stage_name, worker_id = parts[0], int(parts[1])
        
        if stage_name == "encoding":
            # Add encoding worker
            if self.encoding_engine:
                await self.encoding_engine.add_worker(worker_id)
        elif stage_name == "prefill":
            # Add prefill worker
            if self.prefill_engine:
                await self.prefill_engine.add_worker(worker_id)
        elif stage_name == "decoding":
            # Add decoding worker
            if self.decoding_engine:
                await self.decoding_engine.add_worker(worker_id)
        else:
            raise ValueError(f"Unknown stage: {stage_name}")
    
    async def remove_instance(self, instance_id: str):
        """
        Remove an instance dynamically
        
        Args:
            instance_id: Instance identifier
        """
        print(f"[V0EngineBackend] Removing instance: {instance_id}")
        
        # Parse instance_id to determine stage and worker
        parts = instance_id.split('_')
        if len(parts) != 2:
            raise ValueError(f"Invalid instance_id format: {instance_id}")
        
        stage_name, worker_id = parts[0], int(parts[1])
        
        if stage_name == "encoding":
            # Remove encoding worker
            if self.encoding_engine:
                await self.encoding_engine.remove_worker(worker_id)
        elif stage_name == "prefill":
            # Remove prefill worker
            if self.prefill_engine:
                await self.prefill_engine.remove_worker(worker_id)
        elif stage_name == "decoding":
            # Remove decoding worker
            if self.decoding_engine:
                await self.decoding_engine.remove_worker(worker_id)
        else:
            raise ValueError(f"Unknown stage: {stage_name}")
    
    def get_all_instances(self) -> List[str]:
        """Get all instance IDs"""
        instances = []
        
        # Add encoding instances
        for i in range(self.num_encoding_workers):
            instances.append(f"encoding_{i}")
        
        # Add prefill instances
        for i in range(self.num_prefill_workers):
            instances.append(f"prefill_{i}")
        
        # Add decoding instances
        for i in range(self.num_decoding_workers):
            instances.append(f"decoding_{i}")
        
        return instances
    
    def get_instances_by_modality(self, modality: ModalityType) -> List[str]:
        """Get instances by modality type"""
        # For v0 backend, all instances can handle both modalities
        # This is a simplified implementation
        return self.get_all_instances()
    
    def get_instances_by_stage(self, stage: InferenceStage) -> List[str]:
        """Get instances by inference stage"""
        instances = []
        
        if stage == InferenceStage.ENCODE:
            for i in range(self.num_encoding_workers):
                instances.append(f"encoding_{i}")
        elif stage == InferenceStage.PREFILL:
            for i in range(self.num_prefill_workers):
                instances.append(f"prefill_{i}")
        elif stage == InferenceStage.DECODE:
            for i in range(self.num_decoding_workers):
                instances.append(f"decoding_{i}")
        
        return instances
    
    def get_instance_modality(self, instance_id: str) -> ModalityType:
        """Get modality type for an instance"""
        # For v0 backend, all instances can handle both modalities
        return ModalityType.MULTIMODAL
    
    def get_instance_stage(self, instance_id: str) -> InferenceStage:
        """Get inference stage for an instance"""
        parts = instance_id.split('_')
        if len(parts) != 2:
            return InferenceStage.PREFILL
        
        stage_name = parts[0]
        if stage_name == "encoding":
            return InferenceStage.ENCODE
        elif stage_name == "prefill":
            return InferenceStage.PREFILL
        elif stage_name == "decoding":
            return InferenceStage.DECODE
        else:
            return InferenceStage.PREFILL

    def _register_nccl_groups(self) -> None:
        if not (self.prefill_engine and self.decoding_engine):
            return

        for pre_id, pre_worker in enumerate(self.prefill_engine.workers):
            if pre_worker is None:
                continue
            for dec_id, dec_worker in enumerate(self.decoding_engine.workers):
                if dec_worker is None:
                    continue
                group_name = f"prefill_decode_{pre_id}_{dec_id}"
                participants = [
                    {
                        "stage": "prefill",
                        "worker_id": pre_id,
                        "worker": pre_worker,
                        "rank": 0,
                    },
                    {
                        "stage": "decoding",
                        "worker_id": dec_id,
                        "worker": dec_worker,
                        "rank": 1,
                    },
                ]
                self.kv_transfer_manager.register_collective_group(group_name, participants)


__all__ = [
    'V0EngineBackend',
]

