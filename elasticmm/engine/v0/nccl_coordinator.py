"""
NCCL Coordinator for Global Process Group Management
Manages global NCCL ranks and initialization across all V0 workers
"""

from typing import Dict, Tuple


class NCCLCoordinator:
    """
    Coordinates global NCCL process group initialization across all workers.
    
    Key responsibilities:
    1. Assign unique global ranks to all workers
    2. Provide unified init_method for NCCL process group
    3. Track worker-to-rank mapping for P2P communication
    4. Pre-allocate ranks for dynamic scaling
    """
    
    def __init__(self, master_addr: str = "127.0.0.1", master_port: int = 29600):
        """
        Initialize NCCL coordinator.
        
        Args:
            master_addr: Master node address for NCCL rendezvous
            master_port: Master port for NCCL rendezvous (should not conflict with other services)
        """
        self.master_addr = master_addr
        self.master_port = master_port
        
        # Mapping: (stage_name, worker_id) -> global_rank
        self.worker_ranks: Dict[Tuple[str, int], int] = {}
        
        # Reverse mapping: global_rank -> (stage_name, worker_id)
        self.rank_to_worker: Dict[int, Tuple[str, int]] = {}
        
        # Counter for assigning ranks
        self._next_rank = 0
        
        print(f"[NCCLCoordinator] Initialized with master={master_addr}:{master_port}")
    
    def register_worker(self, stage: str, worker_id: int) -> int:
        """
        Register a worker and assign it a global rank.
        
        Args:
            stage: Stage name ('encoding', 'prefill', 'decoding')
            worker_id: Worker ID within the stage
            
        Returns:
            global_rank: Assigned global rank for NCCL process group
        """
        key = (stage, worker_id)
        
        if key in self.worker_ranks:
            # Already registered
            return self.worker_ranks[key]
        
        # Assign new rank
        global_rank = self._next_rank
        self._next_rank += 1
        
        self.worker_ranks[key] = global_rank
        self.rank_to_worker[global_rank] = key
        
        print(f"[NCCLCoordinator] Registered {stage}/worker{worker_id} as global rank {global_rank}")
        
        return global_rank
    
    def unregister_worker(self, stage: str, worker_id: int):
        """
        Unregister a worker (remove from mappings).
        Note: In role-switching scenario, use update_worker_role() instead.
        
        Args:
            stage: Stage name
            worker_id: Worker ID
        """
        key = (stage, worker_id)
        
        if key not in self.worker_ranks:
            print(f"[NCCLCoordinator] Warning: {stage}/worker{worker_id} not registered")
            return
        
        global_rank = self.worker_ranks[key]
        del self.worker_ranks[key]
        del self.rank_to_worker[global_rank]
        
        print(f"[NCCLCoordinator] Unregistered {stage}/worker{worker_id} (rank {global_rank})")
    
    def update_worker_role(self, worker_id: int, old_stage: str, new_stage: str):
        """
        Update worker's stage (for role switching).
        Keeps the same global rank, just updates the stage mapping.
        
        Args:
            worker_id: Worker ID (also GPU ID)
            old_stage: Old stage name
            new_stage: New stage name
        """
        old_key = (old_stage, worker_id)
        
        if old_key not in self.worker_ranks:
            print(f"[NCCLCoordinator] Warning: {old_stage}/worker{worker_id} not registered")
            return
        
        # Get the global rank
        global_rank = self.worker_ranks[old_key]
        
        # Remove old mapping
        del self.worker_ranks[old_key]
        
        # Add new mapping (same rank, new stage)
        new_key = (new_stage, worker_id)
        self.worker_ranks[new_key] = global_rank
        self.rank_to_worker[global_rank] = new_key
        
        print(f"[NCCLCoordinator] Updated worker{worker_id} role: {old_stage} → {new_stage} (rank {global_rank} unchanged)")
    
    def get_rank(self, stage: str, worker_id: int) -> int:
        """
        Get the global rank for a worker.
        
        Args:
            stage: Stage name
            worker_id: Worker ID
            
        Returns:
            global_rank: Global rank of the worker
            
        Raises:
            KeyError: If worker not registered
        """
        key = (stage, worker_id)
        if key not in self.worker_ranks:
            raise KeyError(f"Worker {stage}/worker{worker_id} not registered")
        return self.worker_ranks[key]
    
    def get_worker(self, global_rank: int) -> Tuple[str, int]:
        """
        Get worker info from global rank.
        
        Args:
            global_rank: Global rank
            
        Returns:
            (stage_name, worker_id): Worker identification
            
        Raises:
            KeyError: If rank not found
        """
        if global_rank not in self.rank_to_worker:
            raise KeyError(f"Global rank {global_rank} not found")
        return self.rank_to_worker[global_rank]
    
    def get_rank(self, stage: str, worker_id: int) -> int:
        """
        Get global rank for a worker.
        
        Args:
            stage: Stage name
            worker_id: Worker ID
            
        Returns:
            global_rank, or -1 if not found
        """
        key = (stage, worker_id)
        return self.worker_ranks.get(key, -1)
    
    def get_world_size(self) -> int:
        """
        Get total world size (number of registered workers).
        
        Returns:
            world_size: Total number of registered workers
        """
        return len(self.worker_ranks)
    
    def get_init_method(self) -> str:
        """
        Get the init_method string for PyTorch distributed.
        
        Returns:
            init_method: e.g., "tcp://127.0.0.1:29600"
        """
        return f"tcp://{self.master_addr}:{self.master_port}"
    
    def get_all_workers(self) -> Dict[Tuple[str, int], int]:
        """
        Get all registered workers and their ranks.
        
        Returns:
            worker_ranks: Dictionary mapping (stage, worker_id) to global_rank
        """
        return self.worker_ranks.copy()
    
    def reset(self):
        """Reset the coordinator (useful for testing)."""
        self.worker_ranks.clear()
        self.rank_to_worker.clear()
        self._next_rank = 0
        print("[NCCLCoordinator] Reset completed")




