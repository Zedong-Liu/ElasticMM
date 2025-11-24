"""
Configuration for v0 engine backend
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class V0EngineConfig:
    """Configuration for v0 engine backend"""
    
    # Model configuration
    model_path: str
    dtype: str = "float16"
    max_model_len: int = 32768
    tensor_parallel_size: int = 1
    trust_remote_code: bool = True
    
    # Worker configuration
    num_encoding_workers: int = 2
    num_prefill_workers: int = 4
    num_decoding_workers: int = 2
    
    # Block manager configuration
    block_size: int = 16
    max_num_gpu_blocks: int = 5000
    max_num_cpu_blocks: int = 1000
    
    # Resource configuration
    gpu_memory_utilization: float = 0.9
    seed: int = 1024
    
    # KV transfer configuration
    kv_transfer_method: str = "p2p_copy"  # 'cuda_ipc', 'nccl', 'p2p_copy', 'staged_copy'
    
    # Scheduling configuration
    max_batch_size_encoding: int = 16
    max_batch_size_prefill: int = 32
    max_batch_size_decoding: int = 64
    
    # Performance tuning
    enable_chunked_prefill: bool = False
    enable_prefix_caching: bool = False
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'model_path': self.model_path,
            'num_encoding_workers': self.num_encoding_workers,
            'num_prefill_workers': self.num_prefill_workers,
            'num_decoding_workers': self.num_decoding_workers,
            'block_size': self.block_size,
            'max_num_gpu_blocks': self.max_num_gpu_blocks,
            'max_num_cpu_blocks': self.max_num_cpu_blocks,
            'dtype': self.dtype,
            'tensor_parallel_size': self.tensor_parallel_size,
            'gpu_memory_utilization': self.gpu_memory_utilization,
            'kv_transfer_method': self.kv_transfer_method,
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create from dictionary"""
        return cls(**config_dict)
    
    def validate(self):
        """Validate configuration"""
        assert self.num_encoding_workers > 0, "At least one encoding worker required"
        assert self.num_prefill_workers > 0, "At least one prefill worker required"
        assert self.num_decoding_workers > 0, "At least one decoding worker required"
        assert self.block_size > 0, "Block size must be positive"
        assert 0 < self.gpu_memory_utilization <= 1, "GPU memory utilization must be in (0, 1]"
        assert self.kv_transfer_method in ['cuda_ipc', 'nccl', 'p2p_copy', 'staged_copy'], \
            f"Invalid KV transfer method: {self.kv_transfer_method}"


# Default configuration
DEFAULT_V0_CONFIG = V0EngineConfig(
    model_path="meta-llama/Llama-2-7b-hf",  # Placeholder
    num_encoding_workers=2,
    num_prefill_workers=4,
    num_decoding_workers=2,
    block_size=16,
    max_num_gpu_blocks=5000,
    max_num_cpu_blocks=1000,
    dtype="float16",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    kv_transfer_method="p2p_copy",
)


__all__ = [
    'V0EngineConfig',
    'DEFAULT_V0_CONFIG',
]

