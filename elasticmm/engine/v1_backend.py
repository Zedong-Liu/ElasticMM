"""
V1 Engine Backend implementation
Wraps VLLMEngineManager to implement EngineBackend interface
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from elasticmm.engine.backend_interface import EngineBackend
from elasticmm.engine.v0.utils import Request, StepOutput
from elasticmm.engine.vllm_instance import VLLMEngineManager, VLLMEngineConfig
from elasticmm.core.balancer import ModalityType
from elasticmm.core.allocator import InferenceStage


@dataclass
class V1EngineBackendConfig:
    """Configuration for V1 Engine Backend"""
    model_path: str
    total_gpus: int
    text_gpus: int
    multimodal_gpus: int
    proxy_host: str = "0.0.0.0"
    proxy_port: int = 10001
    sd_host: str = "0.0.0.0"
    sd_port: int = 30002
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 8192


class V1EngineBackend(EngineBackend):
    """
    V1 Engine Backend
    
    Wraps the existing VLLMEngineManager to implement the EngineBackend interface.
    This allows the scheduler to work with v1 engines through the same interface as v0 engines.
    """
    
    def __init__(self, config: V1EngineBackendConfig):
        """
        Initialize V1 Engine Backend
        
        Args:
            config: V1 engine backend configuration
        """
        self.config = config
        
        # Create engine manager
        self.engine_manager = VLLMEngineManager(env={"VLLM_USE_V1": "1", "HOST": "127.0.0.1"})
        
        # Engine configurations
        self.engine_configs: List[VLLMEngineConfig] = []
        self.engine_names: List[str] = []
        
        # Instance mapping: instance_id -> (config, name)
        self.instance_mapping: Dict[str, tuple] = {}
        
        # Statistics
        self.total_requests_received = 0
        self.total_requests_completed = 0
        
        # Generate engine configurations
        self._generate_engine_configs()
        
        print(f"[V1EngineBackend] Initialized with {config.text_gpus} text + {config.multimodal_gpus} multimodal GPUs")
    
    def _generate_engine_configs(self):
        """Generate engine configurations based on backend config"""
        self.engine_configs = []
        self.engine_names = []
        
        gpu_id = 0
        kv_rank = 0
        
        # Generate text group configurations
        text_gpus = self.config.text_gpus
        for i in range(text_gpus):
            is_producer = i < text_gpus // 2
            role = "kv_producer" if is_producer else "kv_consumer"
            name = f"text_{role}_{i+1}"
            
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
            self.instance_mapping[name] = (config, name)
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
            self.instance_mapping[name] = (config, name)
            gpu_id += 1
            kv_rank += 1
        
        print(f"[V1EngineBackend] Generated {len(self.engine_configs)} engine configurations")
    
    async def initialize(self):
        """Initialize the backend"""
        print("[V1EngineBackend] Initializing...")
        # V1 backend initialization is handled in start()
        print("[V1EngineBackend] Initialized")
    
    async def start(self):
        """Start all vLLM engines"""
        print("[V1EngineBackend] Starting all vLLM engines...")
        
        for config, name in zip(self.engine_configs, self.engine_names):
            print(f"[V1EngineBackend] Starting {name} on GPU {config.gpu_id} (Port: {config.http_port})")
            self.engine_manager.start_engine(config, name)
        
        print("[V1EngineBackend] All engine start commands sent")
    
    async def stop(self):
        """Stop all vLLM engines"""
        print("[V1EngineBackend] Stopping all vLLM engines...")
        
        if self.engine_manager is not None:
            try:
                self.engine_manager.stop_all()
                print("[V1EngineBackend] All engines stopped")
            except Exception as e:
                print(f"[V1EngineBackend] Error stopping engines: {e}")
    
    async def add_request(self, request: Request):
        """
        Add request to the backend
        
        For v1 backend, requests are handled by the proxy service,
        so this is mainly for interface compatibility.
        """
        self.total_requests_received += 1
        print(f"[V1EngineBackend] Request added: {request.request_id}")
    
    async def get_outputs(self) -> List[StepOutput]:
        """
        Get outputs from completed requests
        
        For v1 backend, outputs are handled by the proxy service,
        so this returns empty list for interface compatibility.
        """
        # V1 backend handles outputs through HTTP API
        return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get backend statistics"""
        return {
            'backend_type': 'v1',
            'total_requests_received': self.total_requests_received,
            'total_requests_completed': self.total_requests_completed,
            'active_engines': len(self.engine_manager._engines) if self.engine_manager else 0,
            'engine_configs': len(self.engine_configs),
            'instance_mapping': len(self.instance_mapping),
        }
    
    async def migrate_instance(
        self,
        src_instance_id: str,
        dst_instance_id: str,
        requests: List[Request]
    ) -> bool:
        """
        Migrate requests between instances
        
        For v1 backend, this uses the existing KV migration functionality.
        """
        if not self.engine_manager:
            print(f"[V1EngineBackend] No engine manager available for migration")
            return False
        
        print(f"[V1EngineBackend] Migrating {len(requests)} requests from {src_instance_id} to {dst_instance_id}")
        
        try:
            # Use existing KV migration functionality
            migration_success = self.engine_manager.migrate_node_kv_cache(
                src_instance_id, [dst_instance_id], "round_robin"
            )
            
            if migration_success:
                print(f"[V1EngineBackend] Migration successful: {src_instance_id} -> {dst_instance_id}")
            else:
                print(f"[V1EngineBackend] Migration failed: {src_instance_id} -> {dst_instance_id}")
            
            return migration_success
            
        except Exception as e:
            print(f"[V1EngineBackend] Migration error: {e}")
            return False
    
    def get_num_instances(self) -> int:
        """Get total number of instances"""
        return len(self.instance_mapping)
    
    def get_instance_info(self, instance_id: str) -> Dict[str, Any]:
        """Get information about a specific instance"""
        if instance_id not in self.instance_mapping:
            return {}
        
        config, name = self.instance_mapping[instance_id]
        
        # Determine modality and stage based on instance name
        if "text" in instance_id:
            modality = ModalityType.TEXT_ONLY
        else:
            modality = ModalityType.MULTIMODAL
        
        if "producer" in instance_id:
            stage = InferenceStage.PREFILL
        else:
            stage = InferenceStage.DECODE
        
        return {
            'instance_id': instance_id,
            'modality': modality.value,
            'stage': stage.value,
            'http_port': config.http_port,
            'kv_role': config.kv_role,
            'gpu_id': config.gpu_id,
            'status': 'active',
        }
    
    def can_add_request(self, request: Request) -> bool:
        """Check if backend can accept a new request"""
        # V1 backend can always accept requests (handled by proxy)
        return True
    
    # Additional methods for V1-specific functionality
    
    def get_engine_manager(self) -> VLLMEngineManager:
        """Get the underlying engine manager"""
        return self.engine_manager
    
    def get_instance_modality(self, instance_id: str) -> ModalityType:
        """Get modality type for an instance"""
        if "text" in instance_id:
            return ModalityType.TEXT_ONLY
        else:
            return ModalityType.MULTIMODAL
    
    def get_instance_stage(self, instance_id: str) -> InferenceStage:
        """Get inference stage for an instance"""
        if "producer" in instance_id:
            return InferenceStage.PREFILL
        else:
            return InferenceStage.DECODE
    
    def get_all_instances(self) -> List[str]:
        """Get all instance IDs"""
        return list(self.instance_mapping.keys())
    
    def get_instances_by_modality(self, modality: ModalityType) -> List[str]:
        """Get instances by modality type"""
        instances = []
        for instance_id in self.instance_mapping.keys():
            if self.get_instance_modality(instance_id) == modality:
                instances.append(instance_id)
        return instances
    
    def get_instances_by_stage(self, stage: InferenceStage) -> List[str]:
        """Get instances by inference stage"""
        instances = []
        for instance_id in self.instance_mapping.keys():
            if self.get_instance_stage(instance_id) == stage:
                instances.append(instance_id)
        return instances


__all__ = [
    'V1EngineBackend',
    'V1EngineBackendConfig',
]

