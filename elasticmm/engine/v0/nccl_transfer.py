"""
NCCL-based KV Cache Transfer for V0 Engine
Implements GPU-to-GPU transfer using PyTorch's NCCL backend
"""

import asyncio
from typing import Dict, List, Optional, Tuple
import torch
import torch.distributed as dist


class NCCLTransferGroup:
    """
    NCCL communication group for KV cache transfer
    Manages a process group between source and destination workers
    """
    
    def __init__(
        self,
        rank: int,
        world_size: int,
        backend: str = "nccl",
        init_method: str = "tcp://localhost:29500",
    ):
        """
        Initialize NCCL transfer group
        
        Args:
            rank: Rank of this worker in the group
            world_size: Total number of workers in the group
            backend: Communication backend (default: nccl)
            init_method: Initialization method for process group
        """
        self.rank = rank
        self.world_size = world_size
        self.backend = backend
        self.init_method = init_method
        self.group = None
        self.initialized = False
        
    def initialize(self):
        """Initialize the NCCL process group"""
        if not self.initialized:
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=self.backend,
                    init_method=self.init_method,
                    rank=self.rank,
                    world_size=self.world_size,
                )
            self.initialized = True
            print(f"[NCCLTransferGroup] Initialized rank {self.rank}/{self.world_size}")
    
    def destroy(self):
        """Destroy the process group"""
        if self.initialized and dist.is_initialized():
            dist.destroy_process_group()
            self.initialized = False
    
    def send_tensor(self, tensor: torch.Tensor, dst_rank: int, tag: int = 0):
        """
        Send tensor to destination rank
        
        Args:
            tensor: Tensor to send (must be on GPU)
            dst_rank: Destination rank
            tag: Message tag for matching send/recv
        """
        assert self.initialized, "Group not initialized"
        assert tensor.is_cuda, "Tensor must be on GPU"
        
        # Use NCCL send
        dist.send(tensor, dst=dst_rank, tag=tag)
    
    def recv_tensor(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        src_rank: int,
        tag: int = 0,
        device: str = "cuda"
    ) -> torch.Tensor:
        """
        Receive tensor from source rank
        
        Args:
            shape: Shape of tensor to receive
            dtype: Data type of tensor
            src_rank: Source rank
            tag: Message tag for matching send/recv
            device: Device to receive on
            
        Returns:
            Received tensor
        """
        assert self.initialized, "Group not initialized"
        
        # Allocate tensor
        tensor = torch.empty(shape, dtype=dtype, device=device)
        
        # Use NCCL recv
        dist.recv(tensor, src=src_rank, tag=tag)
        
        return tensor
    
    def broadcast_tensor(
        self,
        tensor: Optional[torch.Tensor],
        src_rank: int,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional[torch.dtype] = None,
        device: str = "cuda"
    ) -> torch.Tensor:
        """
        Broadcast tensor from source to all ranks
        
        Args:
            tensor: Tensor to broadcast (only needed on src_rank)
            src_rank: Source rank
            shape: Shape for receiving ranks
            dtype: Data type for receiving ranks
            device: Device
            
        Returns:
            Broadcasted tensor
        """
        assert self.initialized, "Group not initialized"
        
        if self.rank == src_rank:
            assert tensor is not None, "Tensor required on source rank"
            dist.broadcast(tensor, src=src_rank)
            return tensor
        else:
            assert shape is not None and dtype is not None, "Shape and dtype required on recv ranks"
            recv_tensor = torch.empty(shape, dtype=dtype, device=device)
            dist.broadcast(recv_tensor, src=src_rank)
            return recv_tensor


class V0NCCLTransferManager:
    """
    Manager for NCCL-based KV cache transfers between stages
    """
    
    def __init__(self, base_port: int = 29500):
        """
        Initialize NCCL transfer manager
        
        Args:
            base_port: Base port for NCCL initialization
        """
        self.base_port = base_port
        self.transfer_groups: Dict[str, NCCLTransferGroup] = {}
        
        # Statistics
        self.total_transfers = 0
        self.total_bytes_transferred = 0
        
        print(f"[V0NCCLTransferManager] Initialized with base port {base_port}")
    
    def create_transfer_group(
        self,
        group_name: str,
        rank: int,
        world_size: int,
        port_offset: int = 0,
    ) -> NCCLTransferGroup:
        """
        Create a NCCL transfer group
        
        Args:
            group_name: Unique name for this group
            rank: Rank in this group
            world_size: Size of this group
            port_offset: Port offset from base port
            
        Returns:
            NCCLTransferGroup instance
        """
        port = self.base_port + port_offset
        init_method = f"tcp://localhost:{port}"
        
        group = NCCLTransferGroup(
            rank=rank,
            world_size=world_size,
            init_method=init_method,
        )
        
        self.transfer_groups[group_name] = group
        print(f"[V0NCCLTransferManager] Created group '{group_name}' with {world_size} ranks")
        
        return group
    
    def get_group(self, group_name: str) -> Optional[NCCLTransferGroup]:
        """Get transfer group by name"""
        return self.transfer_groups.get(group_name)
    
    async def transfer_kv_cache_blocks(
        self,
        group_name: str,
        src_worker_rank: int,
        dst_worker_rank: int,
        kv_cache: torch.Tensor,
        src_blocks: List[int],
        dst_blocks: List[int],
        is_sender: bool,
    ) -> Optional[torch.Tensor]:
        """
        Transfer KV cache blocks using NCCL
        
        Args:
            group_name: Transfer group name
            src_worker_rank: Source worker rank in group
            dst_worker_rank: Destination worker rank in group
            kv_cache: KV cache tensor (shape: [layers, 2, blocks, block_size, heads, head_dim])
            src_blocks: Source block indices
            dst_blocks: Destination block indices
            is_sender: True if this is the sending worker
            
        Returns:
            Received KV data (only for receiver)
        """
        group = self.get_group(group_name)
        if not group:
            raise ValueError(f"Group '{group_name}' not found")
        
        if not group.initialized:
            group.initialize()
        
        # Extract blocks to transfer
        if is_sender:
            # Sender: extract source blocks
            # kv_cache[:, :, src_blocks, :, :, :]
            num_layers = kv_cache.shape[0]
            num_kv = kv_cache.shape[1]  # 2 (K and V)
            block_size = kv_cache.shape[3]
            num_heads = kv_cache.shape[4]
            head_dim = kv_cache.shape[5]
            
            # Gather blocks
            kv_data = torch.stack([
                kv_cache[:, :, block_idx, :, :, :]
                for block_idx in src_blocks
            ], dim=2)  # [layers, 2, num_blocks, block_size, heads, head_dim]
            
            # Flatten for transfer
            kv_data_flat = kv_data.flatten()
            
            # Send to destination
            print(f"[Transfer] Sending {len(src_blocks)} blocks ({kv_data_flat.numel()} elements) "
                  f"from rank {src_worker_rank} to rank {dst_worker_rank}")
            
            group.send_tensor(kv_data_flat, dst_rank=dst_worker_rank, tag=0)
            
            self.total_transfers += 1
            self.total_bytes_transferred += kv_data_flat.numel() * kv_data_flat.element_size()
            
            return None
            
        else:
            # Receiver: receive and write to destination blocks
            num_layers = kv_cache.shape[0]
            num_kv = kv_cache.shape[1]
            block_size = kv_cache.shape[3]
            num_heads = kv_cache.shape[4]
            head_dim = kv_cache.shape[5]
            
            num_blocks = len(dst_blocks)
            total_elements = num_layers * num_kv * num_blocks * block_size * num_heads * head_dim
            
            # Receive flattened data
            print(f"[Transfer] Receiving {num_blocks} blocks ({total_elements} elements) "
                  f"at rank {dst_worker_rank} from rank {src_worker_rank}")
            
            kv_data_flat = group.recv_tensor(
                shape=(total_elements,),
                dtype=kv_cache.dtype,
                src_rank=src_worker_rank,
                tag=0,
            )
            
            # Reshape
            kv_data = kv_data_flat.view(
                num_layers, num_kv, num_blocks, block_size, num_heads, head_dim
            )
            
            # Write to destination blocks
            for i, block_idx in enumerate(dst_blocks):
                kv_cache[:, :, block_idx, :, :, :] = kv_data[:, :, i, :, :, :]
            
            print(f"[Transfer] Successfully wrote to {len(dst_blocks)} blocks")
            
            return kv_data
    
    async def transfer_vision_embeddings(
        self,
        group_name: str,
        src_worker_rank: int,
        dst_worker_rank: int,
        ve_cache: torch.Tensor,
        src_blocks: List[int],
        dst_blocks: List[int],
        is_sender: bool,
    ) -> Optional[torch.Tensor]:
        """
        Transfer vision embedding blocks using NCCL
        
        Args:
            group_name: Transfer group name
            src_worker_rank: Source worker rank
            dst_worker_rank: Destination worker rank
            ve_cache: Vision embedding cache (shape: [blocks, embed_tokens, embed_dim])
            src_blocks: Source block indices
            dst_blocks: Destination block indices
            is_sender: True if this is the sending worker
            
        Returns:
            Received vision embedding data (only for receiver)
        """
        group = self.get_group(group_name)
        if not group:
            raise ValueError(f"Group '{group_name}' not found")
        
        if not group.initialized:
            group.initialize()
        
        if is_sender:
            # Extract and send vision embeddings
            ve_data = torch.stack([
                ve_cache[block_idx]
                for block_idx in src_blocks
            ], dim=0)  # [num_blocks, embed_tokens, embed_dim]
            
            ve_data_flat = ve_data.flatten()
            
            print(f"[Transfer] Sending {len(src_blocks)} vision blocks "
                  f"from rank {src_worker_rank} to rank {dst_worker_rank}")
            
            group.send_tensor(ve_data_flat, dst_rank=dst_worker_rank, tag=1)
            
            return None
            
        else:
            # Receive vision embeddings
            embed_tokens = ve_cache.shape[1]
            embed_dim = ve_cache.shape[2]
            num_blocks = len(dst_blocks)
            total_elements = num_blocks * embed_tokens * embed_dim
            
            print(f"[Transfer] Receiving {num_blocks} vision blocks "
                  f"at rank {dst_worker_rank} from rank {src_worker_rank}")
            
            ve_data_flat = group.recv_tensor(
                shape=(total_elements,),
                dtype=ve_cache.dtype,
                src_rank=src_worker_rank,
                tag=1,
            )
            
            # Reshape and write
            ve_data = ve_data_flat.view(num_blocks, embed_tokens, embed_dim)
            
            for i, block_idx in enumerate(dst_blocks):
                ve_cache[block_idx] = ve_data[i]
            
            print(f"[Transfer] Successfully wrote {len(dst_blocks)} vision blocks")
            
            return ve_data
    
    def get_stats(self) -> Dict[str, any]:
        """Get transfer statistics"""
        return {
            "total_transfers": self.total_transfers,
            "total_bytes_transferred": self.total_bytes_transferred,
            "total_mb_transferred": self.total_bytes_transferred / (1024 ** 2),
            "num_groups": len(self.transfer_groups),
        }


# Export
__all__ = [
    'NCCLTransferGroup',
    'V0NCCLTransferManager',
]

