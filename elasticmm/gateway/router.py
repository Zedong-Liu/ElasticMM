class RequestRouter:
    """Request router: implements modality-aware request routing"""
    
    def __init__(self):
        self.routing_stats = {
            "text_requests": 0,
            "multimodal_requests": 0
        }
    
    def route_request(self, request: Request) -> str:
        """
        Route requests to corresponding modality groups
        Corresponds to modality-level management in section 3.1 of the paper
        """
        if request.request_type == RequestType.TEXT_ONLY:
            self.routing_stats["text_requests"] += 1
            return "text_group"
        else:
            self.routing_stats["multimodal_requests"] += 1  
            return "multimodal_group"
    
    def get_routing_distribution(self) -> Dict[str, float]:
        """Get routing distribution statistics for load balancing decisions"""
        total = sum(self.routing_stats.values())
        if total == 0:
            return {"text_ratio": 0.5, "multimodal_ratio": 0.5}
        
        return {
            "text_ratio": self.routing_stats["text_requests"] / total,
            "multimodal_ratio": self.routing_stats["multimodal_requests"] / total
        }