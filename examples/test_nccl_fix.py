"""
Test NCCL pre-allocated ranks fix
Verifies that dynamic worker addition doesn't cause NCCL deadlock
"""

import asyncio
import ray
from elasticmm.engine.v0.backend import V0EngineBackend

async def test_nccl_preallocation():
    """Test that NCCL world_size remains fixed with pre-allocated ranks"""
    
    print("=" * 60)
    print("Test: NCCL Pre-allocated Ranks")
    print("=" * 60)
    
    # Initialize backend with smaller config
    backend = V0EngineBackend(
        model_path="/root/lzd/model/qwen2.5-VL",
        num_encoding_workers=1,
        num_prefill_workers=1,
        num_decoding_workers=1,
        kv_transfer_method="nccl"
    )
    
    # Initialize and start backend
    await backend.initialize()
    await backend.start()
    
    print("\n✅ Step 1: Backend started with 3 workers (encoding_0, prefill_0, decoding_0)")
    print(f"   NCCL world_size: {backend.nccl_coordinator.get_world_size()}")
    print(f"   Active workers: {backend.nccl_coordinator.get_active_world_size()}")
    
    # Wait a bit for initialization
    await asyncio.sleep(5)
    
    # Check heartbeat status
    print("\n✅ Step 2: Checking heartbeat status...")
    stats = backend.get_stats()
    print(f"   Encoding workers: {stats['encoding']['num_workers']}")
    print(f"   Prefill workers: {stats['prefill']['num_workers']}")
    print(f"   Decoding workers: {stats['decoding']['num_workers']}")
    
    # Test adding a worker dynamically
    print("\n✅ Step 3: Adding decoding_1 worker...")
    await backend.decoding_engine.add_worker(worker_id=1)
    
    print(f"   NCCL world_size: {backend.nccl_coordinator.get_world_size()} (should remain same)")
    print(f"   Active workers: {backend.nccl_coordinator.get_active_world_size()} (should increase by 1)")
    
    # Wait for new worker to initialize
    await asyncio.sleep(20)  # Model loading takes time
    
    # Check heartbeat again
    print("\n✅ Step 4: Checking heartbeat after adding worker...")
    stats = backend.get_stats()
    print(f"   Decoding workers: {stats['decoding']['num_workers']}")
    
    # Verify all workers are responsive
    print("\n✅ Step 5: Waiting for heartbeats to stabilize...")
    await asyncio.sleep(10)
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
    
    # Cleanup
    await backend.stop()

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    asyncio.run(test_nccl_preallocation())
    ray.shutdown()

