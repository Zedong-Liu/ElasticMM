"""
ElasticMM System Manager
Integrates all components and provides complete system management functionality
"""

import asyncio
import time
import subprocess
import multiprocessing as mp
import os
import sys
import requests
import signal
import atexit
import weakref
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from .core.balancer import ModalityAwareLoadBalancer, ModalityType, WorkloadStats
from .core.allocator import StageLevelResourceAllocator, InferenceStage, Request
from .core.scheduler import EMPScheduler
from .engine.vllm_instance import VLLMEngineManager, VLLMEngineConfig
from .engine.pipeline import create_disagg_proxy_app
from .engine.backend_interface import EngineBackendFactory
from enum import Enum

# Global registry for system instances to enable automatic cleanup
_system_instances = weakref.WeakSet()

def _cleanup_all_systems():
    """Cleanup all registered system instances"""
    print("ElasticMM LOG: Cleaning up all system instances...")
    for system in list(_system_instances):
        try:
            if system.is_running:
                print(f"ElasticMM LOG: Force stopping system instance...")
                asyncio.run(system._force_cleanup())
        except Exception as e:
            print(f"ElasticMM LOG: Error during cleanup: {e}")

def _signal_handler(signum, frame):
    """Signal handler for graceful shutdown"""
    print(f"ElasticMM LOG: Received signal {signum}, initiating cleanup...")
    _cleanup_all_systems()
    sys.exit(0)

# Register signal handlers and cleanup function
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)
atexit.register(_cleanup_all_systems)


@dataclass
class GPUInfo:
    """GPU information"""
    gpu_id: int
    memory_total: int  # MB
    memory_free: int   # MB


@dataclass
class NCCLProfilingData:
    """NCCL profiling data"""
    timestamp: float
    bandwidth_gbps: float
    latency_ms: float


@dataclass
class PerformanceMetrics:
    """Performance metrics"""
    timestamp: float
    decode_time_ms: float
    prefill_time_ms: float
    gpu_utilization: float


class ElasticMMSystemError(Exception):
    """ElasticMM system error"""
    pass


class ConfigType(Enum):
    """Configuration type"""
    DEFAULT = "default"
    CUSTOM = "custom"


@dataclass
class SystemConfig:
    """System configuration"""
    total_gpus: int
    text_gpus: int
    multimodal_gpus: int
    model_path: str
    backend_type: str = "v1"  # "v0" or "v1"
    proxy_host: str = "0.0.0.0"
    proxy_port: int = 10001
    sd_host: str = "0.0.0.0"
    sd_port: int = 30002
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 8192

    def __post_init__(self):
        if self.total_gpus < 2:
            raise ValueError("System requires at least 2 GPUs")
        if self.backend_type not in ["v0", "v1"]:
            raise ValueError("backend_type must be 'v0' or 'v1'")


class ElasticMMSystem:
    """
    ElasticMM System Manager
    
    Core responsibilities:
    1. Start/stop proxy and engines (delegate to their respective classes)
    2. Start/stop scheduler management
    3. System monitoring and health checks
    4. Route requests to proxy (not handle engine details)
    """

    def __init__(self, config: Optional[SystemConfig] = None, config_type: ConfigType = ConfigType.DEFAULT):
        """
        Initialize ElasticMM system
        
        Args:
            config: Custom system configuration (required if config_type is CUSTOM)
            config_type: Configuration type (DEFAULT or CUSTOM)
        """
        # Set configuration
        if config_type == ConfigType.DEFAULT:
            self.config = self._get_default_config()
        elif config_type == ConfigType.CUSTOM:
            if config is None:
                raise ValueError("Custom configuration must be provided when config_type is CUSTOM")
            self.config = config
        else:
            raise ValueError(f"Invalid config_type: {config_type}")
        
        # System configuration from config
        self.model_path = self.config.model_path
        self.backend_type = self.config.backend_type
        self.proxy_host = self.config.proxy_host
        self.proxy_port = self.config.proxy_port
        self.sd_host = self.config.sd_host
        self.sd_port = self.config.sd_port
        self.required_gpus = self.config.total_gpus

        # Create backend based on backend_type
        self.backend = self._create_backend()
        
        # Components - delegate to backend
        self.engine_manager = self.backend.get_engine_manager() if hasattr(self.backend, 'get_engine_manager') else None
        self.scheduler = EMPScheduler(backend=self.backend)
        self.proxy_proc: Optional[mp.Process] = None
        self.proxy_task: Optional[asyncio.Task] = None  # For v0 backend (runs in main process)
        
        # System state
        self.is_running = False
        self.start_time = 0.0
        
        # Task management for monitoring
        self.monitoring_tasks = []
        
        # Engine configurations (for v1 backend compatibility)
        self.engine_configs: List[VLLMEngineConfig] = []
        self.engine_names: List[str] = []
        
        # Generate engine configurations (for v1 backend)
        if self.backend_type == "v1":
            self._generate_engine_configs()
        
        # Register this instance for automatic cleanup
        _system_instances.add(self)
        
    def _get_default_config(self) -> SystemConfig:
        """Get default system configuration (8 GPUs: 2 text + 6 multimodal)"""
        return SystemConfig(
            total_gpus=8,
            text_gpus=2,
            multimodal_gpus=6,
            model_path="/root/lzd/model/qwen2.5-VL",
            backend_type="v0"  # Default to v0 backend
        )
    
    def _create_backend(self):
        """Create backend based on backend_type"""
        if self.backend_type == "v0":
            # V0 backend specific configuration
            # Text group: 2 workers (1 prefill + 1 decode) = 2 GPUs
            # Multimodal group: 6 workers (2 encode + 2 prefill + 2 decode) = 6 GPUs
            # Total: 8 GPUs
            backend_config = {
                'model_path': self.config.model_path,
                'num_encoding_workers': 2,  # Multimodal encode workers only
                'num_prefill_workers': 3,   # Text + Multimodal prefill workers (1+2)
                'num_decoding_workers': 3,  # Text + Multimodal decode workers (1+2)
                'block_size': 16,
                'max_num_gpu_blocks': 5000,
                'max_num_cpu_blocks': 1000,
                'dtype': "float16",
                'tensor_parallel_size': 1,
                'gpu_memory_utilization': self.config.gpu_memory_utilization,
                'kv_transfer_method': 'p2p_copy',
                'limit_mm_per_prompt': None,
                'sd_host': self.config.sd_host,
                'sd_port': self.config.sd_port,
                'proxy_port': self.config.proxy_port,
            }
        else:
            # V1 backend specific configuration
            backend_config = {
                'model_path': self.config.model_path,
                'total_gpus': self.config.total_gpus,
                'text_gpus': self.config.text_gpus,
                'multimodal_gpus': self.config.multimodal_gpus,
                'proxy_host': self.config.proxy_host,
                'proxy_port': self.config.proxy_port,
                'sd_host': self.config.sd_host,
                'sd_port': self.config.sd_port,
                'gpu_memory_utilization': self.config.gpu_memory_utilization,
                'max_model_len': self.config.max_model_len,
            }
        
        backend = EngineBackendFactory.create(self.backend_type, **backend_config)
        print(f"[ElasticMMSystem] Created {self.backend_type} backend")
        return backend
    
    def _generate_engine_configs(self):
        """Generate engine configurations based on system config"""
        self.engine_configs = []
        self.engine_names = []
        
        gpu_id = 0
        kv_rank = 0
        
        # Generate text group configurations
        text_gpus = self.config.text_gpus
        for i in range(text_gpus):
            is_producer = i < text_gpus // 2
            role = "kv_producer" if is_producer else "kv_consumer"
            name = f"text_{role}" if is_producer else f"text_{role}"
            
            config = VLLMEngineConfig(
                model_path=self.config.model_path,
                http_host=self.config.proxy_host,
                http_port=8100 + i,
                kv_role=role,
                kv_rank=kv_rank,
                kv_parallel_size=self.config.total_gpus,
                kv_port=21000 + i,
                proxy_ip=self.config.sd_host,
                proxy_port=self.config.sd_port,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_model_len=self.config.max_model_len,
                gpu_id=gpu_id
            )
            
            self.engine_configs.append(config)
            self.engine_names.append(name)
            gpu_id += 1
            kv_rank += 1
        
        # Generate multimodal group configurations
        multimodal_gpus = self.config.multimodal_gpus
        for i in range(multimodal_gpus):
            is_producer = i < multimodal_gpus // 2
            role = "kv_producer" if is_producer else "kv_consumer"
            name = f"multimodal_{role}_{i+1}"
            
            config = VLLMEngineConfig(
                model_path=self.config.model_path,
                http_host=self.config.proxy_host,
                http_port=8200 + i,
                kv_role=role,
                kv_rank=kv_rank,
                kv_parallel_size=self.config.total_gpus,
                kv_port=22000 + i,
                proxy_ip=self.config.sd_host,
                proxy_port=self.config.sd_port,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_model_len=self.config.max_model_len,
                gpu_id=gpu_id
            )
            
            self.engine_configs.append(config)
            self.engine_names.append(name)
            gpu_id += 1
            kv_rank += 1
        
        print(f"ElasticMM LOG: Generated {len(self.engine_configs)} engine configurations")
        print(f"ElasticMM LOG: Text group: {self.config.text_gpus} GPUs, Multimodal group: {self.config.multimodal_gpus} GPUs")
        
    async def start(self):
        """
        Start ElasticMM system:
        1. Start proxy service
        2. Start all engines (delegate to engine_manager)
        3. Start scheduler management
        4. Start system monitoring
        """
        if self.is_running:
            print("ElasticMM LOG: System already running")
            return

        print("ElasticMM LOG: Starting ElasticMM system...")
        
        # Basic initialization
        self.is_running = True
        self.start_time = time.time()
        
        # For v0 backend: initialize backend first (before proxy) so backend is available in proxy
        # For v1 backend: can start proxy first
        if self.backend_type == "v0":
            # 1. Initialize backend first (v0 backend needs to be ready before proxy starts)
            print("ElasticMM LOG: Initializing v0 backend before starting proxy...")
            await self.backend.initialize()
            
            # 2. Start proxy service (now backend is initialized and available)
            await self._start_proxy()
            
            # 3. Start backend engines
            await self.backend.start()
            
            # 4. Wait for engines to be ready
            await self._wait_for_engines_ready()
        else:
            # V1 backend: original order
            # 1. Start proxy service
            await self._start_proxy()
            
            # 2. Start all engines (delegate to backend)
            await self._start_all_engines()
            
            # 3. Wait for engines to be ready
            await self._wait_for_engines_ready()
        
        # 4. Start scheduler management
        await self._start_scheduler_management()
        
        # 5. Start system monitoring
        await self._start_system_monitoring()
        
        print("ElasticMM LOG: System started successfully with all components running")
    
    async def _start_proxy(self):
        """Start the proxy service"""
        print("ElasticMM LOG: Starting proxy service...")
        
        # Check if ZMQ port is available before starting
        import socket as sock
        zmq_check_socket = sock.socket(sock.AF_INET, sock.SOCK_STREAM)
        zmq_check_socket.setsockopt(sock.SOL_SOCKET, sock.SO_REUSEADDR, 1)
        try:
            zmq_check_socket.bind(('0.0.0.0', self.config.sd_port))
            zmq_check_socket.close()
            print(f"✓ ZMQ port {self.config.sd_port} is available")
        except OSError:
            zmq_check_socket.close()
            print(f"⚠️  ZMQ port {self.config.sd_port} appears to be in use.")
            print(f"   Waiting 3 seconds for port to be released...")
            await asyncio.sleep(3)
        
        # For v0 backend, run proxy in main process (backend object not serializable)
        # For v1 backend, run in separate process
        if self.backend_type == "v0":
            # Run proxy in background task (same process)
            print("ElasticMM LOG: Running proxy in main process for v0 backend...")
            async def run_proxy_task():
                try:
                    app = self._create_proxy_app()
                    # Use hypercorn to run Quart app asynchronously
                    from hypercorn.asyncio import serve
                    from hypercorn.config import Config
                    config = Config()
                    config.bind = [f"{self.config.proxy_host}:{self.config.proxy_port}"]
                    await serve(app, config)
                except Exception as e:
                    print(f"❌ Proxy task error: {e}")
                    import traceback
                    traceback.print_exc()
            
            self.proxy_task = asyncio.create_task(run_proxy_task())
            await asyncio.sleep(2)  # Give it time to start
            print("ElasticMM LOG: Proxy service started (main process)")
        else:
            # V1 backend: run in separate process
            def run_proxy_process():
                try:
                    # Create proxy app with scheduler integration
                    app = self._create_proxy_app()
                    app.run(host=self.config.proxy_host, port=self.config.proxy_port)
                except Exception as e:
                    print(f"❌ Proxy process error: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
            
            self.proxy_proc = mp.Process(target=run_proxy_process, daemon=True)
            self.proxy_proc.start()
            
            # Wait for proxy to start and check if it's still alive
            await asyncio.sleep(3)
            
            # Check if process is still running
            if not self.proxy_proc.is_alive():
                print("❌ Proxy process died immediately after start!")
                print("   Check the error messages above for details.")
                raise RuntimeError("Proxy service failed to start")
            
            print("ElasticMM LOG: Proxy service started (separate process)")
    
    def _create_proxy_app(self):
        """Create proxy app with integrated scheduler"""
        from .engine.pipeline import create_disagg_proxy_app
        
        # Create proxy app with scheduler
        app = create_disagg_proxy_app(
            service_discovery_host=self.config.sd_host,
            service_discovery_port=self.config.sd_port,
            api_host=self.config.proxy_host,
            api_port=self.config.proxy_port,
            scheduler=self.scheduler  # Pass scheduler to proxy
        )
        return app
    
    async def _start_all_engines(self):
        """Start all engines (delegate to backend)"""
        print("ElasticMM LOG: Starting all engines...")
        
        # Initialize and start backend
        await self.backend.initialize()
        await self.backend.start()
        
        print("ElasticMM LOG: All engines started")
    
    async def _wait_for_engines_ready(self):
        """Wait for all engines to be ready"""
        print("ElasticMM LOG: Waiting for all engines to be ready...")
        
        # Wait for model loading
        await asyncio.sleep(60)
        
        # For v1 backend, check HTTP health endpoints
        if self.backend_type == "v1" and self.engine_configs:
            # Check all engines health status
            for attempt in range(60):  # Wait up to 2 minutes
                all_ready = True
                for config, name in zip(self.engine_configs, self.engine_names):
                    try:
                        response = requests.get(f"http://127.0.0.1:{config.http_port}/health", timeout=5)
                        if response.status_code != 200:
                            all_ready = False
                            break
                    except Exception:
                        all_ready = False
                        break
                
                if all_ready:
                    print("ElasticMM LOG: All engines health check passed, system ready!")
                    return True
                
                if attempt % 10 == 0 and attempt > 0:
                    print(f"ElasticMM LOG: Waiting... ({attempt*2}s / 120s)")
                await asyncio.sleep(2)
            
            print("ElasticMM LOG: Some engines failed to start in time")
            return False
        else:
            # For v0 backend, just wait a bit for initialization
            await asyncio.sleep(10)
            print("ElasticMM LOG: V0 backend ready")
            return True
    
    async def _start_scheduler_management(self):
        """Start scheduler management tasks"""
        print("ElasticMM LOG: Starting scheduler management...")
        
        # Initialize scheduler with engine configurations
        await self._initialize_scheduler()
        
        # Start elastic scheduling loop (100-step interval gain-cost reallocation)
        if hasattr(self.scheduler, 'start_elastic_scheduling'):
            self.scheduler.start_elastic_scheduling()
            print("ElasticMM LOG: Elastic scheduling loop started (100-step interval)")
        
        # Start scheduler monitoring tasks
        scheduler_task = asyncio.create_task(self._scheduler_monitoring_loop())
        self.monitoring_tasks.append(scheduler_task)
        
        print("ElasticMM LOG: Scheduler management started")
    
    async def _initialize_scheduler(self):
        """Initialize scheduler with engine instances"""
        print("ElasticMM LOG: Initializing scheduler with engine instances...")
        
        # Add instances to scheduler based on backend
        if self.backend_type == "v1":
            # For v1 backend, use engine configurations
            for config, name in zip(self.engine_configs, self.engine_names):
                # Determine modality type based on engine name
                if "text" in name:
                    modality = ModalityType.TEXT_ONLY
                else:
                    modality = ModalityType.MULTIMODAL
                
                # Determine stage based on role
                if "producer" in name:
                    stage = InferenceStage.PREFILL
                else:
                    stage = InferenceStage.DECODE
                
                # Add instance to scheduler
                await self.scheduler.add_instance(
                    instance_id=name,
                    modality=modality,
                    stage=stage
                )
        else:
            # For v0 backend, use backend's instance information
            all_instances = self.backend.get_all_instances()
            print(f"ElasticMM LOG: V0 backend instances: {all_instances}")
            for instance_id in all_instances:
                instance_info = self.backend.get_instance_info(instance_id)
                print(f"ElasticMM LOG: Instance {instance_id} info: {instance_info}")
                modality = ModalityType(instance_info['modality'])
                stage = InferenceStage(instance_info['stage'])
                
                await self.scheduler.add_instance(
                    instance_id=instance_id,
                    modality=modality,
                    stage=stage
                )
                print(f"ElasticMM LOG: Added instance {instance_id} to scheduler")
        
        print("ElasticMM LOG: Scheduler initialized with all engine instances")
    
    async def _start_system_monitoring(self):
        """Start system monitoring tasks"""
        print("ElasticMM LOG: Starting system monitoring...")
        
        # Start health monitoring
        health_task = asyncio.create_task(self._health_monitoring_loop())
        self.monitoring_tasks.append(health_task)
        
        # Start performance monitoring
        perf_task = asyncio.create_task(self._performance_monitoring_loop())
        self.monitoring_tasks.append(perf_task)
        
        # Start 5-minute periodic modality group rebalancing
        if hasattr(self.scheduler, 'modality_balancer'):
            rebalance_task = asyncio.create_task(self._modality_rebalance_loop())
            self.monitoring_tasks.append(rebalance_task)
            print("ElasticMM LOG: Modality group rebalancing task started (5-minute interval)")
        
        print("ElasticMM LOG: System monitoring started")
    
    async def _scheduler_monitoring_loop(self):
        """Scheduler monitoring loop"""
        while self.is_running:
            try:
                # Monitor scheduler status
                if hasattr(self.scheduler, 'get_system_status'):
                    status = self.scheduler.get_system_status()
                    # Process scheduler status updates
                    pass
                
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                print(f"ElasticMM LOG: Error in scheduler monitoring: {e}")
                await asyncio.sleep(10)
    
    async def _health_monitoring_loop(self):
        """Health monitoring loop"""
        while self.is_running:
            try:
                # Check proxy health
                if self.proxy_proc and self.proxy_proc.is_alive():
                    try:
                        response = requests.get(f"http://127.0.0.1:{self.proxy_port}/health", timeout=5)
                        if response.status_code != 200:
                            print("ElasticMM LOG: Proxy health check failed")
                    except Exception:
                        print("ElasticMM LOG: Proxy health check error")
                
                # Check engine health
                for config, name in zip(self.engine_configs, self.engine_names):
                    try:
                        response = requests.get(f"http://127.0.0.1:{config.http_port}/health", timeout=5)
                        if response.status_code != 200:
                            print(f"ElasticMM LOG: Engine {name} health check failed")
                    except Exception:
                        print(f"ElasticMM LOG: Engine {name} health check error")
                
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                print(f"ElasticMM LOG: Error in health monitoring: {e}")
                await asyncio.sleep(10)
    
    async def _performance_monitoring_loop(self):
        """Performance monitoring loop"""
        while self.is_running:
            try:
                # Collect performance metrics
                # This can be expanded based on requirements
                await asyncio.sleep(120)  # Check every 2 minutes
            except Exception as e:
                print(f"ElasticMM LOG: Error in performance monitoring: {e}")
                await asyncio.sleep(10)
    
    async def _modality_rebalance_loop(self):
        """Modality group rebalancing loop (5-minute interval)"""
        while self.is_running:
            try:
                # Check if rebalancing is needed
                if hasattr(self.scheduler, 'modality_balancer'):
                    balancer = self.scheduler.modality_balancer
                    
                    # Perform periodic rebalancing based on historical data
                    new_allocation = await balancer.periodic_rebalance()
                    
                    if new_allocation:
                        print(f"[ModalityRebalance] New allocation suggested: {new_allocation}")
                        # TODO: Apply the new allocation by migrating instances
                        # This would involve calling backend APIs to reassign workers
                        pass
                    else:
                        print("[ModalityRebalance] No rebalancing needed at this time")
                
                await asyncio.sleep(300)  # Check every 5 minutes
            except Exception as e:
                print(f"ElasticMM LOG: Error in modality rebalancing: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(60)
    
    async def stop(self):
        """Stop system and cleanup all resources"""
        if not self.is_running and self.proxy_proc is None and (not hasattr(self, 'engine_manager') or not self.engine_manager._engines):
            print("ElasticMM LOG: System not running, no need to stop")
            return
        
        print("ElasticMM LOG: Stopping ElasticMM system...")
        self.is_running = False
        
        # Set a flag to allow early exit from blocking operations
        if hasattr(self, 'backend') and self.backend:
            if hasattr(self.backend, '_stop_heartbeat'):
                self.backend._stop_heartbeat = True
        
        try:
            # Stop monitoring tasks
            for task in self.monitoring_tasks:
                if not task.done():
                    task.cancel()
            if self.monitoring_tasks:
                await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
            self.monitoring_tasks.clear()
            
            # Stop engines (delegate to backend)
            if self.backend is not None:
                try:
                    print("ElasticMM LOG: Stopping engines...")
                    await self.backend.stop()
                    print("ElasticMM LOG: Engines stopped")
                except Exception as e:
                    print(f"ElasticMM LOG: Error stopping engines: {e}")
            
            # Stop proxy (process or task depending on backend type)
            if self.backend_type == "v0" and self.proxy_task is not None:
                try:
                    print("ElasticMM LOG: Stopping proxy task...")
                    # Clean up ZMQ resources
                    try:
                        from elasticmm.engine.pipeline import cleanup_zmq_resources
                        cleanup_zmq_resources()
                    except Exception as zmq_e:
                        print(f"ElasticMM LOG: Error cleaning up ZMQ: {zmq_e}")
                    
                    self.proxy_task.cancel()
                    try:
                        await self.proxy_task
                    except asyncio.CancelledError:
                        pass
                    print("ElasticMM LOG: Proxy task stopped")
                except Exception as e:
                    print(f"ElasticMM LOG: Error stopping proxy task: {e}")
            elif self.proxy_proc is not None and self.proxy_proc.is_alive():
                try:
                    print("ElasticMM LOG: Stopping proxy process...")
                    # Clean up ZMQ resources before terminating
                    try:
                        from elasticmm.engine.pipeline import cleanup_zmq_resources
                        cleanup_zmq_resources()
                    except Exception as zmq_e:
                        print(f"ElasticMM LOG: Error cleaning up ZMQ: {zmq_e}")
                    
                    self.proxy_proc.terminate()
                    self.proxy_proc.join(timeout=5)
                    if self.proxy_proc.is_alive():
                        print("ElasticMM LOG: Force killing proxy process...")
                        self.proxy_proc.kill()
                        self.proxy_proc.join()
                    print("ElasticMM LOG: Proxy process stopped")
                except Exception as e:
                    print(f"ElasticMM LOG: Error stopping proxy: {e}")
            
            print("ElasticMM LOG: System stopped successfully")
            
        except Exception as e:
            print(f"ElasticMM LOG: Error during graceful stop: {e}")
            # Fallback to force cleanup
            await self._force_cleanup()
    
    async def _force_cleanup(self):
        """Force cleanup all resources without waiting for graceful shutdown"""
        print("ElasticMM LOG: Force cleaning up system resources...")
        
        # Stop monitoring tasks
        for task in self.monitoring_tasks:
            if not task.done():
                task.cancel()
        
        # Stop engines (delegate to backend)
        if self.backend is not None:
            try:
                print("ElasticMM LOG: Stopping engines...")
                await self.backend.stop()
                print("ElasticMM LOG: Engines stopped")
            except Exception as e:
                print(f"ElasticMM LOG: Error stopping engines: {e}")
        
        # Stop proxy (process or task depending on backend type)
        if hasattr(self, 'backend_type') and self.backend_type == "v0" and self.proxy_task is not None:
            try:
                print("ElasticMM LOG: Stopping proxy task...")
                # Clean up ZMQ resources
                try:
                    from elasticmm.engine.pipeline import cleanup_zmq_resources
                    cleanup_zmq_resources()
                except Exception as zmq_e:
                    print(f"ElasticMM LOG: Error cleaning up ZMQ: {zmq_e}")
                
                self.proxy_task.cancel()
                try:
                    await self.proxy_task
                except asyncio.CancelledError:
                    pass
                print("ElasticMM LOG: Proxy task stopped")
            except Exception as e:
                print(f"ElasticMM LOG: Error stopping proxy task: {e}")
        elif self.proxy_proc is not None and self.proxy_proc.is_alive():
            try:
                print("ElasticMM LOG: Stopping proxy process...")
                # Clean up ZMQ resources before terminating
                try:
                    from elasticmm.engine.pipeline import cleanup_zmq_resources
                    cleanup_zmq_resources()
                except Exception as zmq_e:
                    print(f"ElasticMM LOG: Error cleaning up ZMQ: {zmq_e}")
                
                self.proxy_proc.terminate()
                self.proxy_proc.join(timeout=2)
                if self.proxy_proc.is_alive():
                    print("ElasticMM LOG: Force killing proxy process...")
                    self.proxy_proc.kill()
                    self.proxy_proc.join()
                print("ElasticMM LOG: Proxy process stopped")
            except Exception as e:
                print(f"ElasticMM LOG: Error stopping proxy: {e}")
        
        # Reset system state
        self.is_running = False
        self.monitoring_tasks.clear()
        
        print("ElasticMM LOG: Force cleanup completed")
    
    def __del__(self):
        """Destructor to ensure cleanup when object is garbage collected"""
        if hasattr(self, 'is_running') and self.is_running:
            print("ElasticMM LOG: System instance being destroyed, performing cleanup...")
            try:
                # Try to cleanup synchronously
                asyncio.run(self._force_cleanup())
            except Exception as e:
                print(f"ElasticMM LOG: Error in destructor cleanup: {e}")

    async def submit_request(self, payload: dict, timeout: int = 60) -> dict:
        """
        Submit an inference request to the system.
        This routes the request directly to the proxy.
        """
        if not self.is_running:
            raise ElasticMMSystemError("System not running, cannot process requests")
        
        print("ElasticMM LOG: Submitting request...")
        try:
            response = requests.post(
                f"http://127.0.0.1:{self.proxy_port}/v1/chat/completions",
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                print(f"ElasticMM LOG: Request successful: {content[:80]}...")
            else:
                 print("ElasticMM LOG: Request successful but no content returned")
            return result
        except requests.exceptions.RequestException as e:
            print(f"ElasticMM LOG: Request failed: {e}")
            raise ElasticMMSystemError(f"Request failed: {e}")

    def get_system_info(self) -> dict:
        """Get system information"""
        return {
            "is_running": self.is_running,
            "uptime": time.time() - self.start_time if self.is_running else 0,
            "backend_type": self.backend_type,
            "total_gpus": self.config.total_gpus,
            "text_gpus": self.config.text_gpus,
            "multimodal_gpus": self.config.multimodal_gpus,
            "proxy_port": self.proxy_port,
            "active_engines": self.backend.get_num_instances() if self.backend else 0,
            "monitoring_tasks": len(self.monitoring_tasks),
            "backend_stats": self.backend.get_stats() if self.backend else {}
        }


# Factory functions for easy system creation
def create_default_system() -> ElasticMMSystem:
    """
    Factory function to create an ElasticMMSystem with default configuration.
    
    Returns:
        ElasticMMSystem: System instance with default 8-GPU configuration (2 text + 6 multimodal)
    """
    return ElasticMMSystem(config_type=ConfigType.DEFAULT)


def create_custom_system(
    total_gpus: int,
    text_gpus: int,
    multimodal_gpus: int,
    model_path: str,
    backend_type: str = "v1",
    proxy_host: str = "0.0.0.0",
    proxy_port: int = 10001,
    sd_host: str = "0.0.0.0",
    sd_port: int = 30002,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 8192
) -> ElasticMMSystem:
    """
    Factory function to create an ElasticMMSystem with custom configuration.
        
        Args:
        total_gpus: Total number of GPUs
        text_gpus: Number of GPUs for text-only group
        multimodal_gpus: Number of GPUs for multimodal group
        model_path: Path to the model
        backend_type: Backend type ('v0' or 'v1')
        proxy_host: Proxy host address
        proxy_port: Proxy port number
        sd_host: Service discovery host address
        sd_port: Service discovery port number
        gpu_memory_utilization: GPU memory utilization ratio
        max_model_len: Maximum model length
        
        Returns:
        ElasticMMSystem: System instance with custom configuration
    """
    config = SystemConfig(
        total_gpus=total_gpus,
        text_gpus=text_gpus,
        multimodal_gpus=multimodal_gpus,
        model_path=model_path,
        backend_type=backend_type,
        proxy_host=proxy_host,
        proxy_port=proxy_port,
        sd_host=sd_host,
        sd_port=sd_port,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len
    )
    return ElasticMMSystem(config=config, config_type=ConfigType.CUSTOM)


    def __del__(self):
        """Destructor to ensure cleanup when object is garbage collected"""
        if hasattr(self, 'is_running') and self.is_running:
            print("ElasticMM LOG: System instance being destroyed, performing cleanup...")
            try:
                # Try to cleanup synchronously
                asyncio.run(self._force_cleanup())
            except Exception as e:
                print(f"ElasticMM LOG: Error in destructor cleanup: {e}")

    async def submit_request(self, payload: dict, timeout: int = 60) -> dict:
        """
        Submit an inference request to the system.
        This routes the request directly to the proxy.
        """
        if not self.is_running:
            raise ElasticMMSystemError("System not running, cannot process requests")
        
        print("ElasticMM LOG: Submitting request...")
        try:
            response = requests.post(
                f"http://127.0.0.1:{self.proxy_port}/v1/chat/completions",
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                print(f"ElasticMM LOG: Request successful: {content[:80]}...")
            else:
                 print("ElasticMM LOG: Request successful but no content returned")
            return result
        except requests.exceptions.RequestException as e:
            print(f"ElasticMM LOG: Request failed: {e}")
            raise ElasticMMSystemError(f"Request failed: {e}")

    def get_system_info(self) -> dict:
        """Get system information"""
        return {
            "is_running": self.is_running,
            "uptime": time.time() - self.start_time if self.is_running else 0,
            "backend_type": self.backend_type,
            "total_gpus": self.config.total_gpus,
            "text_gpus": self.config.text_gpus,
            "multimodal_gpus": self.config.multimodal_gpus,
            "proxy_port": self.proxy_port,
            "active_engines": self.backend.get_num_instances() if self.backend else 0,
            "monitoring_tasks": len(self.monitoring_tasks),
            "backend_stats": self.backend.get_stats() if self.backend else {}
        }


# Factory functions for easy system creation
def create_default_system() -> ElasticMMSystem:
    """
    Factory function to create an ElasticMMSystem with default configuration.
    
    Returns:
        ElasticMMSystem: System instance with default 8-GPU configuration (2 text + 6 multimodal)
    """
    return ElasticMMSystem(config_type=ConfigType.DEFAULT)


def create_custom_system(
    total_gpus: int,
    text_gpus: int,
    multimodal_gpus: int,
    model_path: str,
    backend_type: str = "v1",
    proxy_host: str = "0.0.0.0",
    proxy_port: int = 10001,
    sd_host: str = "0.0.0.0",
    sd_port: int = 30002,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 8192
) -> ElasticMMSystem:
    """
    Factory function to create an ElasticMMSystem with custom configuration.
        
        Args:
        total_gpus: Total number of GPUs
        text_gpus: Number of GPUs for text-only group
        multimodal_gpus: Number of GPUs for multimodal group
        model_path: Path to the model
        backend_type: Backend type ('v0' or 'v1')
        proxy_host: Proxy host address
        proxy_port: Proxy port number
        sd_host: Service discovery host address
        sd_port: Service discovery port number
        gpu_memory_utilization: GPU memory utilization ratio
        max_model_len: Maximum model length
        
        Returns:
        ElasticMMSystem: System instance with custom configuration
    """
    config = SystemConfig(
        total_gpus=total_gpus,
        text_gpus=text_gpus,
        multimodal_gpus=multimodal_gpus,
        model_path=model_path,
        backend_type=backend_type,
        proxy_host=proxy_host,
        proxy_port=proxy_port,
        sd_host=sd_host,
        sd_port=sd_port,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len
    )
    return ElasticMMSystem(config=config, config_type=ConfigType.CUSTOM)