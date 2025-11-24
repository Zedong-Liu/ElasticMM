"""
Utility classes and functions for v0 engine
"""

from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class EngineStage(Enum):
    """Engine stage enumeration"""
    ENCODING = "encoding"
    PREFILL = "prefill"
    DECODING = "decoding"


class EngineStatus(Enum):
    """Engine status enumeration"""
    INACTIVE = "inactive"
    ACTIVE = "active"
    MIGRATING = "migrating"
    ERROR = "error"


@dataclass
class Request:
    """Request representation for v0 engine"""
    request_id: str
    prompt: str
    max_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    
    # Multimodal data
    images: Optional[List[Any]] = None
    multi_modal_data: Optional[Dict[str, Any]] = None
    multi_modal_kwargs: Optional[Dict[str, Any]] = None  # Processed mm_kwargs (image_grid_thw, etc.)
    
    # Sequence management
    input_ids: Optional[List[int]] = None
    prompt_token_ids: Optional[List[int]] = None
    output_token_ids: List[int] = None
    
    # Status tracking
    is_finished: bool = False
    finish_reason: Optional[str] = None
    
    # Timing information (for profiling)
    arrival_time: Optional[float] = None  # When request arrived at system
    encoding_start_time: Optional[float] = None
    encoding_end_time: Optional[float] = None
    prefill_start_time: Optional[float] = None
    prefill_end_time: Optional[float] = None
    decoding_start_time: Optional[float] = None  # First time entering decode
    decoding_end_time: Optional[float] = None  # Final completion time
    total_decode_compute_time: float = 0.0  # Cumulative GPU compute time (seconds)
    kv_transfer_time: Optional[float] = None  # Time spent in KV transfer
    
    def __post_init__(self):
        if self.output_token_ids is None:
            self.output_token_ids = []
        import time
        if self.arrival_time is None:
            self.arrival_time = time.time()
    
    def get_input_len(self) -> int:
        """Get input sequence length"""
        if self.prompt_token_ids:
            return len(self.prompt_token_ids)
        return 0
    
    def get_output_len(self) -> int:
        """Get output sequence length"""
        return len(self.output_token_ids)
    
    def get_total_len(self) -> int:
        """Get total sequence length"""
        return self.get_input_len() + self.get_output_len()
    
    def append_token(self, token_id: int):
        """Append a new token to output"""
        self.output_token_ids.append(token_id)
    
    def check_finished(self, eos_token_id: int) -> bool:
        """Check if request is finished"""
        if self.get_output_len() >= self.max_tokens:
            self.is_finished = True
            self.finish_reason = "length"
            return True
        
        if self.output_token_ids and self.output_token_ids[-1] == eos_token_id:
            self.is_finished = True
            self.finish_reason = "stop"
            return True
        
        return False


@dataclass
class BatchedRequests:
    """Batched requests for execution"""
    requests: List[Request]
    
    def __len__(self):
        return len(self.requests)
    
    def __iter__(self):
        return iter(self.requests)
    
    def is_empty(self) -> bool:
        return len(self.requests) == 0


@dataclass
class MigratingRequest:
    """Request being migrated between stages"""
    req: Request
    
    # Block indexes for KV cache
    kv_block_indexes: Optional[List[int]] = None
    
    # Block indexes for vision embeddings
    vision_block_indexes: Optional[List[int]] = None
    
    # Generated tokens from previous stage (for cross-process transfer)
    output_token_ids: Optional[List[int]] = None
    
    # CRITICAL: Expanded prompt tokens (for cross-process transfer)
    # Ray serialization breaks request object updates, so we need to save this explicitly
    expanded_prompt_token_ids: Optional[List[int]] = None
    
    # Multi-modal kwargs from previous stage (for cross-process transfer)
    multi_modal_kwargs: Optional[dict] = None
    
    # Multi-modal placeholders from previous stage (for cross-process transfer)
    multi_modal_placeholders: Optional[dict] = None
    
    # Vision embeddings tensor (for direct transfer, bypassing block manager)
    vision_embeddings: Optional[Any] = None  # torch.Tensor, but avoid import here
    
    # Source stage information
    source_stage: Optional[EngineStage] = None
    source_worker_id: Optional[int] = None
    
    # Target stage information
    target_stage: Optional[EngineStage] = None
    target_worker_id: Optional[int] = None


@dataclass
class StepOutput:
    """Output from a single execution step"""
    request_id: str
    output_token_ids: List[int]
    finished: bool
    finish_reason: Optional[str] = None
    
    # Logprobs (optional)
    logprobs: Optional[Dict[int, float]] = None
    
    # Profiling information
    step_start_time: Optional[float] = None
    step_end_time: Optional[float] = None
    num_input_tokens: Optional[int] = None  # For prefill
    num_output_tokens: Optional[int] = None  # For decode iterations


@dataclass
class StageMetrics:
    """Metrics collection for a stage (encoding/prefill/decoding)"""
    stage_name: str
    
    # Normalized latencies (ms/token)
    prefill_latencies: List[float] = None  # ms per input token
    decode_latencies: List[float] = None   # ms per output token
    encoding_latencies: List[float] = None  # ms per input token
    
    # KV transfer metrics
    kv_transfer_times: List[float] = None  # seconds per transfer
    kv_transfer_sizes: List[int] = None    # number of blocks transferred
    
    # Request completion times
    e2e_latencies: List[float] = None  # end-to-end latency (seconds)
    
    # Throughput metrics
    requests_completed: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    
    # Sampling configuration (to avoid storing every single request)
    sample_interval: int = 10  # Record every Nth request
    sample_counter: int = 0
    
    def __post_init__(self):
        if self.prefill_latencies is None:
            self.prefill_latencies = []
        if self.decode_latencies is None:
            self.decode_latencies = []
        if self.encoding_latencies is None:
            self.encoding_latencies = []
        if self.kv_transfer_times is None:
            self.kv_transfer_times = []
        if self.kv_transfer_sizes is None:
            self.kv_transfer_sizes = []
        if self.e2e_latencies is None:
            self.e2e_latencies = []
    
    def record_prefill(self, num_input_tokens: int, latency_ms: float, should_sample: bool = True):
        """Record prefill latency (normalized by input tokens)"""
        if not should_sample:
            self.sample_counter += 1
            if self.sample_counter % self.sample_interval != 0:
                return
        
        if num_input_tokens > 0:
            normalized_latency = latency_ms / num_input_tokens
            self.prefill_latencies.append(normalized_latency)
            self.total_input_tokens += num_input_tokens
    
    def record_decode(self, num_output_tokens: int, latency_ms: float, should_sample: bool = True):
        """Record decode latency (normalized by output tokens)"""
        if not should_sample:
            self.sample_counter += 1
            if self.sample_counter % self.sample_interval != 0:
                return
        
        if num_output_tokens > 0:
            normalized_latency = latency_ms / num_output_tokens
            self.decode_latencies.append(normalized_latency)
            self.total_output_tokens += num_output_tokens
    
    def record_encoding(self, num_input_tokens: int, latency_ms: float, should_sample: bool = True):
        """Record encoding latency (normalized by input tokens)"""
        if not should_sample:
            self.sample_counter += 1
            if self.sample_counter % self.sample_interval != 0:
                return
        
        if num_input_tokens > 0:
            normalized_latency = latency_ms / num_input_tokens
            self.encoding_latencies.append(normalized_latency)
            self.total_input_tokens += num_input_tokens
    
    def record_kv_transfer(self, transfer_time_sec: float, num_blocks: int):
        """Record KV transfer time and size"""
        self.kv_transfer_times.append(transfer_time_sec)
        self.kv_transfer_sizes.append(num_blocks)
    
    def record_request_completion(self, e2e_latency_sec: float):
        """Record end-to-end request latency"""
        self.requests_completed += 1
        self.e2e_latencies.append(e2e_latency_sec)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics"""
        import numpy as np
        
        stats = {
            'stage': self.stage_name,
            'requests_completed': self.requests_completed,
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
        }
        
        if self.prefill_latencies:
            stats['prefill_latency_ms_per_token'] = {
                'mean': float(np.mean(self.prefill_latencies)),
                'median': float(np.median(self.prefill_latencies)),
                'p50': float(np.percentile(self.prefill_latencies, 50)),
                'p90': float(np.percentile(self.prefill_latencies, 90)),
                'p99': float(np.percentile(self.prefill_latencies, 99)),
            }
        
        if self.decode_latencies:
            stats['decode_latency_ms_per_token'] = {
                'mean': float(np.mean(self.decode_latencies)),
                'median': float(np.median(self.decode_latencies)),
                'p50': float(np.percentile(self.decode_latencies, 50)),
                'p90': float(np.percentile(self.decode_latencies, 90)),
                'p99': float(np.percentile(self.decode_latencies, 99)),
            }
        
        if self.encoding_latencies:
            stats['encoding_latency_ms_per_token'] = {
                'mean': float(np.mean(self.encoding_latencies)),
                'median': float(np.median(self.encoding_latencies)),
                'p50': float(np.percentile(self.encoding_latencies, 50)),
                'p90': float(np.percentile(self.encoding_latencies, 90)),
                'p99': float(np.percentile(self.encoding_latencies, 99)),
            }
        
        if self.kv_transfer_times:
            stats['kv_transfer'] = {
                'avg_time_sec': float(np.mean(self.kv_transfer_times)),
                'avg_blocks': float(np.mean(self.kv_transfer_sizes)),
                'bandwidth_blocks_per_sec': float(np.sum(self.kv_transfer_sizes) / np.sum(self.kv_transfer_times)) if np.sum(self.kv_transfer_times) > 0 else 0,
            }
        
        if self.e2e_latencies:
            stats['e2e_latency_sec'] = {
                'mean': float(np.mean(self.e2e_latencies)),
                'median': float(np.median(self.e2e_latencies)),
                'p50': float(np.percentile(self.e2e_latencies, 50)),
                'p90': float(np.percentile(self.e2e_latencies, 90)),
                'p99': float(np.percentile(self.e2e_latencies, 99)),
            }
        
        return stats


def get_gpu_memory() -> int:
    """Get available GPU memory in bytes"""
    import torch
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory
    return 0


def random_digits(n: int) -> str:
    """Generate random n-digit string"""
    import random
    return ''.join([str(random.randint(0, 9)) for _ in range(n)])


# Constants
GB = 1024 ** 3
MB = 1024 ** 2
KB = 1024

