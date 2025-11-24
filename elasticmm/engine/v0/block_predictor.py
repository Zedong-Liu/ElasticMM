"""
Block Predictor for intelligent block pre-allocation
Inspired by vLLM's dynamic memory management
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import time


@dataclass
class RequestStats:
    """Statistics for a single request"""
    request_id: str
    initial_prompt_tokens: int
    actual_prompt_tokens: int  # After vision token expansion
    has_vision: bool
    tokens_generated: int = 0
    blocks_used: int = 0
    start_time: float = 0.0
    
    def __post_init__(self):
        if self.start_time == 0.0:
            self.start_time = time.time()


class BlockPredictor:
    """
    Intelligent block predictor based on vLLM's approach
    
    Strategy:
    1. Track historical token expansion ratios for multimodal requests
    2. Predict total blocks needed = prompt_blocks + expected_output_blocks
    3. Pre-allocate with safety margin to reduce frequent expansions
    """
    
    def __init__(self, block_size: int = 16, safety_margin: float = 1.2):
        self.block_size = block_size
        self.safety_margin = safety_margin  # Allocate 20% extra to reduce re-allocation
        
        # Historical statistics
        self.vision_expansion_ratios: List[float] = []  # prompt expansion ratio for vision
        self.avg_output_tokens: List[int] = []  # average output tokens per request
        
        # Current request tracking
        self.active_requests: Dict[str, RequestStats] = {}
        
        # Constants (can be tuned)
        self.default_vision_expansion = 3.0  # Vision tokens typically expand 2-5x
        self.default_output_estimate = 30  # Default expected output tokens
        self.min_safety_blocks = 3  # Minimum extra blocks to allocate
        
    def predict_total_blocks(
        self, 
        request_id: str,
        current_prompt_tokens: int,
        current_output_tokens: int,
        max_tokens: int,
        has_vision: bool = False
    ) -> int:
        """
        Predict total blocks needed for a request
        
        Args:
            request_id: Request identifier
            current_prompt_tokens: Current prompt length (may be expanded)
            current_output_tokens: Tokens generated so far
            max_tokens: Maximum tokens to generate
            has_vision: Whether request has vision input
            
        Returns:
            Total blocks needed (with safety margin)
        """
        # Calculate tokens already used
        tokens_used = current_prompt_tokens + current_output_tokens
        
        # Estimate remaining output tokens
        remaining_output = max_tokens - current_output_tokens
        if remaining_output <= 0:
            # Request is about to finish
            blocks_needed = (tokens_used + self.block_size - 1) // self.block_size
            return blocks_needed
        
        # Use historical data to estimate
        if self.avg_output_tokens:
            avg_output = sum(self.avg_output_tokens) / len(self.avg_output_tokens)
            estimated_output = min(remaining_output, avg_output)
        else:
            estimated_output = min(remaining_output, self.default_output_estimate)
        
        # Total tokens estimate
        estimated_total = tokens_used + estimated_output
        
        # Calculate blocks with safety margin
        blocks_needed = (estimated_total + self.block_size - 1) // self.block_size
        blocks_with_margin = int(blocks_needed * self.safety_margin) + self.min_safety_blocks
        
        return blocks_with_margin
    
    def predict_initial_blocks(
        self,
        prompt_tokens: int,
        max_tokens: int,
        has_vision: bool = False
    ) -> int:
        """
        Predict blocks needed when first allocating for a request
        (called in _receive_from_prefill)
        
        Args:
            prompt_tokens: Prompt length (already expanded if has vision)
            max_tokens: Maximum tokens to generate
            has_vision: Whether request has vision input
            
        Returns:
            Initial blocks to allocate
        """
        # Estimate total sequence length
        if has_vision and self.avg_output_tokens:
            # Vision requests might generate different amounts
            avg_output = sum(self.avg_output_tokens[-10:]) / min(10, len(self.avg_output_tokens))
            estimated_output = min(max_tokens, avg_output * 1.2)  # 20% buffer for vision
        else:
            estimated_output = min(max_tokens, self.default_output_estimate)
        
        estimated_total = prompt_tokens + estimated_output
        
        # Calculate blocks
        blocks_needed = (estimated_total + self.block_size - 1) // self.block_size
        
        # Add safety margin for initial allocation
        blocks_with_margin = int(blocks_needed * self.safety_margin) + self.min_safety_blocks
        
        return blocks_with_margin
    
    def record_request_start(
        self,
        request_id: str,
        initial_prompt_tokens: int,
        actual_prompt_tokens: int,
        has_vision: bool
    ):
        """Record a new request starting decode phase"""
        stats = RequestStats(
            request_id=request_id,
            initial_prompt_tokens=initial_prompt_tokens,
            actual_prompt_tokens=actual_prompt_tokens,
            has_vision=has_vision
        )
        self.active_requests[request_id] = stats
        
        # Update vision expansion ratio
        if has_vision and initial_prompt_tokens > 0:
            ratio = actual_prompt_tokens / initial_prompt_tokens
            self.vision_expansion_ratios.append(ratio)
            # Keep only recent 100 samples
            if len(self.vision_expansion_ratios) > 100:
                self.vision_expansion_ratios.pop(0)
    
    def record_request_finish(self, request_id: str, final_output_tokens: int):
        """Record a request finishing"""
        if request_id in self.active_requests:
            stats = self.active_requests.pop(request_id)
            
            # Update average output tokens
            self.avg_output_tokens.append(final_output_tokens)
            # Keep only recent 100 samples
            if len(self.avg_output_tokens) > 100:
                self.avg_output_tokens.pop(0)
    
    def get_statistics(self) -> Dict:
        """Get predictor statistics"""
        avg_vision_expansion = (
            sum(self.vision_expansion_ratios) / len(self.vision_expansion_ratios)
            if self.vision_expansion_ratios else self.default_vision_expansion
        )
        avg_output = (
            sum(self.avg_output_tokens) / len(self.avg_output_tokens)
            if self.avg_output_tokens else self.default_output_estimate
        )
        
        return {
            'avg_vision_expansion': avg_vision_expansion,
            'avg_output_tokens': avg_output,
            'active_requests': len(self.active_requests),
            'samples_collected': len(self.avg_output_tokens)
        }






