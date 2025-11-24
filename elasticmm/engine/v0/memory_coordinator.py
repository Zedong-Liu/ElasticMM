"""
Global Memory Coordinator for V0 Engine
========================================

Implements vLLM-style memory coordination between Prefill and Decode stages:

1. **Watermark-based Admission Control**:
   - High watermark (0.9): Stop accepting new requests from Prefill
   - Low watermark (0.5): Resume accepting requests
   
2. **Backpressure Mechanism**:
   - Decode notifies Prefill when memory is tight
   - Prefill pauses KV transfer until memory frees up
   
3. **Preemption/Early Termination**:
   - When critically low on memory, force-finish longest-running requests
   - Priority: finish requests closest to completion first
   
4. **Dynamic Batch Size**:
   - Adjust batch size based on available memory
   - Ensure at least some requests can make progress
"""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, List


class MemoryPressureLevel(Enum):
    """Memory pressure levels for coordination"""
    NORMAL = "normal"           # > 50% blocks available
    WARNING = "warning"         # 20-50% blocks available  
    CRITICAL = "critical"       # 5-20% blocks available
    EMERGENCY = "emergency"     # < 5% blocks available


@dataclass
class MemoryStatus:
    """Memory status snapshot"""
    total_blocks: int
    free_blocks: int
    used_blocks: int
    usage_ratio: float
    pressure_level: MemoryPressureLevel
    timestamp: float
    
    @property
    def can_accept_new_requests(self) -> bool:
        """Check if we can accept new requests from Prefill"""
        # High watermark: stop at 95% usage (more aggressive)
        return self.usage_ratio < 0.95
    
    @property
    def should_reduce_batch(self) -> bool:
        """Check if we should reduce batch size"""
        return self.pressure_level in [MemoryPressureLevel.CRITICAL, MemoryPressureLevel.EMERGENCY]
    
    @property
    def needs_preemption(self) -> bool:
        """Check if we need to preempt requests"""
        return self.pressure_level == MemoryPressureLevel.EMERGENCY


class GlobalMemoryCoordinator:
    """
    Global memory coordinator between stages
    
    Implements vLLM-style memory management:
    - Centralized memory monitoring
    - Backpressure signaling
    - Admission control
    - Preemption decisions
    """
    
    def __init__(self, high_watermark: float = 0.95, low_watermark: float = 0.80):
        """
        Args:
            high_watermark: Stop accepting new requests above this usage (default 95%)
            low_watermark: Resume accepting requests below this usage (default 80%)
        """
        self.high_watermark = high_watermark
        self.low_watermark = low_watermark
        
        # Current memory status
        self._memory_status: Optional[MemoryStatus] = None
        self._lock = asyncio.Lock()
        
        # Backpressure signaling
        self._prefill_paused = False
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Initially not paused
        
        # Statistics
        self.stats = {
            "total_pauses": 0,
            "total_preemptions": 0,
            "total_admission_rejects": 0,
            "total_memory_checks": 0,
        }
    
    async def update_memory_status(self, total_blocks: int, free_blocks: int) -> MemoryStatus:
        """
        Update current memory status (called by Decode stage)
        
        Args:
            total_blocks: Total KV cache blocks
            free_blocks: Available free blocks
            
        Returns:
            Current memory status
        """
        async with self._lock:
            used_blocks = total_blocks - free_blocks
            usage_ratio = used_blocks / total_blocks if total_blocks > 0 else 1.0
            
            # Determine pressure level (more conservative thresholds)
            # EMERGENCY: 99.5%+ (only ~56 blocks left on 11140 total)
            # CRITICAL: 98%+    (only ~223 blocks left)
            # WARNING: 90%+     (~1114 blocks left)
            if usage_ratio >= 0.995:
                pressure_level = MemoryPressureLevel.EMERGENCY
            elif usage_ratio >= 0.98:
                pressure_level = MemoryPressureLevel.CRITICAL
            elif usage_ratio >= 0.90:
                pressure_level = MemoryPressureLevel.WARNING
            else:
                pressure_level = MemoryPressureLevel.NORMAL
            
            self._memory_status = MemoryStatus(
                total_blocks=total_blocks,
                free_blocks=free_blocks,
                used_blocks=used_blocks,
                usage_ratio=usage_ratio,
                pressure_level=pressure_level,
                timestamp=time.time()
            )
            
            self.stats["total_memory_checks"] += 1
            
            # Update backpressure state
            await self._update_backpressure()
            
            return self._memory_status
    
    async def _update_backpressure(self):
        """Update backpressure state based on memory status"""
        if self._memory_status is None:
            return
        
        # Apply hysteresis to avoid flapping
        if self._memory_status.usage_ratio >= self.high_watermark:
            if not self._prefill_paused:
                print(f"[MemoryCoordinator] 🛑 BACKPRESSURE ACTIVATED: {self._memory_status.usage_ratio:.1%} usage")
                print(f"[MemoryCoordinator]    Signaling Prefill to PAUSE KV transfers")
                self._prefill_paused = True
                self._pause_event.clear()
                self.stats["total_pauses"] += 1
        elif self._memory_status.usage_ratio <= self.low_watermark:
            if self._prefill_paused:
                print(f"[MemoryCoordinator] ✅ BACKPRESSURE RELEASED: {self._memory_status.usage_ratio:.1%} usage")
                print(f"[MemoryCoordinator]    Signaling Prefill to RESUME KV transfers")
                self._prefill_paused = False
                self._pause_event.set()
    
    async def wait_for_memory_available(self, timeout: float = 5.0) -> bool:
        """
        Wait until memory pressure is relieved (called by Prefill)
        
        Args:
            timeout: Maximum time to wait
            
        Returns:
            True if memory is available, False if timeout
        """
        try:
            await asyncio.wait_for(self._pause_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False
    
    async def can_admit_new_request(self) -> bool:
        """
        Check if we can admit a new request from Prefill to Decode
        
        Returns:
            True if admission allowed, False otherwise
        """
        async with self._lock:
            if self._memory_status is None:
                return True  # No status yet, allow
            
            can_admit = self._memory_status.can_accept_new_requests
            
            if not can_admit:
                self.stats["total_admission_rejects"] += 1
            
            return can_admit
    
    def get_memory_status(self) -> Optional[MemoryStatus]:
        """Get current memory status (non-async, for quick checks)"""
        return self._memory_status
    
    def is_prefill_paused(self) -> bool:
        """Check if Prefill should pause (non-async)"""
        return self._prefill_paused
    
    async def suggest_batch_size(self, current_running: int, free_blocks: int, blocks_per_request: int = 1) -> int:
        """
        Suggest how many requests can run based on available memory
        
        Args:
            current_running: Number of currently running requests
            free_blocks: Available free blocks
            blocks_per_request: Estimated blocks needed per request iteration
            
        Returns:
            Suggested number of requests that can run
        """
        if self._memory_status is None:
            return current_running
        
        # Calculate how many requests we can support
        if free_blocks <= 0:
            # Critical: can't expand any requests
            # But don't return 0, try to let at least some finish
            return min(current_running, max(1, current_running // 2))
        
        # Calculate max requests we can support
        max_supportable = current_running + (free_blocks // blocks_per_request)
        
        # Apply conservative factor based on pressure level
        if self._memory_status.pressure_level == MemoryPressureLevel.EMERGENCY:
            # Very aggressive reduction
            suggested = min(current_running, max(1, current_running // 4))
        elif self._memory_status.pressure_level == MemoryPressureLevel.CRITICAL:
            # Moderate reduction
            suggested = min(current_running, max(1, current_running // 2))
        elif self._memory_status.pressure_level == MemoryPressureLevel.WARNING:
            # Slight reduction
            suggested = min(current_running, max(1, int(current_running * 0.75)))
        else:
            # Normal: allow all
            suggested = current_running
        
        return suggested
    
    async def select_requests_to_preempt(
        self, 
        running_requests: List[Dict],
        num_to_preempt: int
    ) -> List[str]:
        """
        Select which requests to preempt (force-finish) to free memory
        
        Strategy: Preempt requests that have generated the MOST tokens
        (closest to finishing anyway)
        
        Args:
            running_requests: List of dicts with 'request_id' and 'output_token_count'
            num_to_preempt: Number of requests to preempt
            
        Returns:
            List of request IDs to preempt
        """
        if not running_requests or num_to_preempt <= 0:
            return []
        
        # Sort by output token count (descending)
        sorted_requests = sorted(
            running_requests,
            key=lambda r: r.get('output_token_count', 0),
            reverse=True
        )
        
        # Take the top N
        to_preempt = sorted_requests[:num_to_preempt]
        preempt_ids = [r['request_id'] for r in to_preempt]
        
        self.stats["total_preemptions"] += len(preempt_ids)
        
        print(f"[MemoryCoordinator] 🔪 PREEMPTION: Force-finishing {len(preempt_ids)} requests")
        for r in to_preempt:
            print(f"[MemoryCoordinator]    - {r['request_id']}: {r.get('output_token_count', 0)} tokens generated")
        
        return preempt_ids
    
    def get_stats(self) -> Dict:
        """Get coordinator statistics"""
        stats = self.stats.copy()
        if self._memory_status:
            stats["current_memory_status"] = {
                "usage_ratio": f"{self._memory_status.usage_ratio:.1%}",
                "free_blocks": self._memory_status.free_blocks,
                "pressure_level": self._memory_status.pressure_level.value,
                "prefill_paused": self._prefill_paused,
            }
        return stats
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            "total_pauses": 0,
            "total_preemptions": 0,
            "total_admission_rejects": 0,
            "total_memory_checks": 0,
        }



