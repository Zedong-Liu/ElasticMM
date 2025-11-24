"""
Backend interface for ElasticMM
Abstracts different engine implementations (v0, v1)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import asyncio


class EngineBackend(ABC):
    """
    Abstract base class for engine backends
    
    This interface allows ElasticMM to support different engine implementations:
    - V0EngineBackend: vLLM v0 with manual stage disaggregation
    - V1EngineBackend: vLLM v1 with built-in KV transfer
    """
    
    @abstractmethod
    async def initialize(self):
        """Initialize the backend"""
        pass
    
    @abstractmethod
    async def start(self):
        """Start the backend engines"""
        pass
    
    @abstractmethod
    async def stop(self):
        """Stop the backend engines"""
        pass
    
    @abstractmethod
    async def add_request(self, request):
        """
        Add a new request to the backend
        
        Args:
            request: Request to add
        """
        pass
    
    @abstractmethod
    async def get_outputs(self):
        """
        Get outputs from completed requests
        
        Returns:
            List of step outputs
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get backend statistics
        
        Returns:
            Dictionary of statistics
        """
        pass
    
    @abstractmethod
    async def migrate_instance(
        self,
        src_instance_id: str,
        dst_instance_id: str,
        requests
    ) -> bool:
        """
        Migrate requests from source instance to destination instance
        
        Args:
            src_instance_id: Source instance ID
            dst_instance_id: Destination instance ID
            requests: Requests to migrate
            
        Returns:
            True if migration successful
        """
        pass
    
    @abstractmethod
    def get_num_instances(self) -> int:
        """Get total number of instances"""
        pass
    
    @abstractmethod
    def get_instance_info(self, instance_id: str) -> Dict[str, Any]:
        """Get information about a specific instance"""
        pass
    
    @abstractmethod
    def can_add_request(self, request) -> bool:
        """Check if backend can accept a new request"""
        pass


class EngineBackendFactory:
    """Factory for creating engine backends"""
    
    @staticmethod
    def create(backend_type: str, **kwargs) -> EngineBackend:
        """
        Create an engine backend
        
        Args:
            backend_type: Type of backend ('v0' or 'v1')
            **kwargs: Backend-specific configuration
            
        Returns:
            EngineBackend instance
        """
        if backend_type == 'v0':
            from elasticmm.engine.v0.backend import V0EngineBackend
            return V0EngineBackend(**kwargs)
        elif backend_type == 'v1':
            from elasticmm.engine.v1_backend import V1EngineBackend, V1EngineBackendConfig
            # Convert kwargs to V1EngineBackendConfig
            config = V1EngineBackendConfig(**kwargs)
            return V1EngineBackend(config)
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")


__all__ = [
    'EngineBackend',
    'EngineBackendFactory',
]

