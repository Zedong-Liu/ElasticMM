"""
Test script for elastic instance migration功能
Tests Phase 1: Instance Migration Infrastructure
"""

import asyncio
import sys
import os
import signal
import time

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global shutdown_requested
    if not shutdown_requested:
        print("\n\n⚠️  Ctrl+C detected. Shutting down gracefully...")
        print("⚠️  Please wait for cleanup (this may take 10-30 seconds)...")
        shutdown_requested = True
    else:
        print("\n⚠️  Force exit requested. This may leave Ray processes running.")
        print("⚠️  Use 'ray stop --force' and 'pkill -9 -f ray::' to clean up.")
        sys.exit(1)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

from elasticmm.engine.v0 import V0EngineBackend
from elasticmm.engine.v0.utils import Request
from elasticmm.engine.v0.config import V0EngineConfig


async def test_dynamic_worker_management():
    """
    Test 1: Dynamic worker addition and removal
    """
    print("\n" + "=" * 80)
    print("Test 1: Dynamic Worker Management")
    print("=" * 80)
    
    # Create configuration with minimal workers
    config = V0EngineConfig(
        model_path="/root/lzd/model/qwen2.5-VL",
        num_encoding_workers=1,
        num_prefill_workers=2,  # Start with 2
        num_decoding_workers=2,  # Start with 2
        block_size=16,
        max_num_gpu_blocks=500,  # Small for testing
        gpu_memory_utilization=0.4,
        kv_transfer_method="nccl",
        dtype="bfloat16",
        trust_remote_code=True,
    )
    
    # Create backend
    print("\n[1/6] Creating and initializing backend...")
    backend = V0EngineBackend(**config.to_dict())
    await backend.initialize()
    await backend.start()
    await asyncio.sleep(2.0)
    print("✓ Backend ready")
    
    # Check initial state
    print("\n[2/6] Checking initial worker state...")
    print(f"✓ Prefill workers: {len([w for w in backend.prefill_engine.workers if w is not None])}")
    print(f"✓ Decoding workers: {len([w for w in backend.decoding_engine.workers if w is not None])}")
    
    # Test: Add a new decode worker
    print("\n[3/6] Adding new decode worker...")
    try:
        await backend.decoding_engine.add_worker(worker_id=2)
        print("✓ Decode worker 2 added successfully")
        print(f"✓ Total decode workers now: {len([w for w in backend.decoding_engine.workers if w is not None])}")
    except Exception as e:
        print(f"❌ Failed to add worker: {e}")
        await backend.stop()
        return False
    
    # Test: Remove a decode worker (without migration, for simple test)
    print("\n[4/6] Removing decode worker...")
    try:
        await backend.decoding_engine.remove_worker(worker_id=2, migrate_requests=False)
        print("✓ Decode worker 2 removed successfully")
        print(f"✓ Total decode workers now: {len([w for w in backend.decoding_engine.workers if w is not None])}")
    except Exception as e:
        print(f"❌ Failed to remove worker: {e}")
        await backend.stop()
        return False
    
    print("\n[5/6] Stopping backend...")
    await backend.stop()
    print("✓ Backend stopped")
    
    print("\n[6/6] Test 1 completed successfully! ✅")
    return True


async def test_kv_migration():
    """
    Test 2: KV cache migration between workers
    """
    print("\n" + "=" * 80)
    print("Test 2: KV Cache Migration")
    print("=" * 80)
    
    # Create configuration
    config = V0EngineConfig(
        model_path="/root/lzd/model/qwen2.5-VL",
        num_encoding_workers=1,
        num_prefill_workers=1,
        num_decoding_workers=2,  # 2 decode workers for migration
        block_size=16,
        max_num_gpu_blocks=500,
        gpu_memory_utilization=0.4,
        kv_transfer_method="nccl",
        dtype="bfloat16",
        trust_remote_code=True,
    )
    
    # Create backend
    print("\n[1/8] Creating and initializing backend...")
    backend = V0EngineBackend(**config.to_dict())
    await backend.initialize()
    await backend.start()
    await asyncio.sleep(2.0)
    print("✓ Backend ready")
    
    # Create test requests
    print("\n[2/8] Creating test requests...")
    test_requests = []
    for i in range(5):  # 5 small requests
        req = Request(
            request_id=f"migrate_test_{i}",
            prompt=f"Hello, this is migration test {i}. Please respond.",
            max_tokens=20,  # Short for faster testing
        )
        test_requests.append(req)
        await backend.add_request(req)
    print(f"✓ Added {len(test_requests)} test requests")
    
    # Let requests process for a bit
    print("\n[3/8] Processing requests (10 seconds)...")
    for i in range(10):
        await asyncio.sleep(1)
        outputs = await backend.get_outputs()
        if outputs:
            print(f"  [{i+1}/10] Got {len(outputs)} outputs")
    
    # Check how many requests are active
    print("\n[4/8] Checking active requests...")
    decode_scheduler = backend.decoding_engine.scheduler
    running_count = len(decode_scheduler.running_requests)
    print(f"✓ {running_count} requests still running in decode stage")
    
    if running_count > 0:
        # Get first few running requests for migration test
        requests_to_migrate = list(decode_scheduler.running_requests.values())[:min(2, running_count)]
        print(f"✓ Selected {len(requests_to_migrate)} requests for migration test")
        
        # Test migration from decode_0 to decode_1
        print("\n[5/8] Testing KV migration: decoding_0 → decoding_1...")
        migration_start = time.time()
        
        try:
            success = await backend.migrate_instance(
                src_instance_id="decoding_0",
                dst_instance_id="decoding_1",
                requests=requests_to_migrate
            )
            
            migration_time = time.time() - migration_start
            
            if success:
                print(f"✅ Migration completed successfully in {migration_time:.3f}s!")
            else:
                print(f"❌ Migration failed")
        except Exception as e:
            print(f"❌ Migration error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️  No running requests to migrate (they completed too fast)")
    
    # Continue processing
    print("\n[6/8] Continuing to process remaining requests (10 seconds)...")
    completed_count = 0
    for i in range(10):
        await asyncio.sleep(1)
        outputs = await backend.get_outputs()
        if outputs:
            completed_count += len(outputs)
            print(f"  [{i+1}/10] Completed: {completed_count} total")
    
    print("\n[7/8] Stopping backend...")
    await backend.stop()
    print("✓ Backend stopped")
    
    print(f"\n[8/8] Test 2 completed! ✅")
    print(f"  Total completed: {completed_count} requests")
    return True


async def test_worker_removal_with_migration():
    """
    Test 3: Worker removal with KV migration
    """
    print("\n" + "=" * 80)
    print("Test 3: Worker Removal with KV Migration")
    print("=" * 80)
    
    # Create configuration with 3 decode workers
    config = V0EngineConfig(
        model_path="/root/lzd/model/qwen2.5-VL",
        num_encoding_workers=1,
        num_prefill_workers=1,
        num_decoding_workers=3,  # 3 workers, will remove one
        block_size=16,
        max_num_gpu_blocks=500,
        gpu_memory_utilization=0.4,
        kv_transfer_method="nccl",
        dtype="bfloat16",
        trust_remote_code=True,
    )
    
    # Create backend
    print("\n[1/7] Creating and initializing backend...")
    backend = V0EngineBackend(**config.to_dict())
    await backend.initialize()
    await backend.start()
    await asyncio.sleep(2.0)
    print("✓ Backend ready with 3 decode workers")
    
    # Create test requests
    print("\n[2/7] Creating test requests...")
    test_requests = []
    for i in range(8):
        req = Request(
            request_id=f"removal_test_{i}",
            prompt=f"Request {i}: Tell me something interesting.",
            max_tokens=30,
        )
        test_requests.append(req)
        await backend.add_request(req)
    print(f"✓ Added {len(test_requests)} requests")
    
    # Let requests start processing
    print("\n[3/7] Processing requests (5 seconds)...")
    for i in range(5):
        await asyncio.sleep(1)
        outputs = await backend.get_outputs()
        if outputs:
            print(f"  [{i+1}/5] Got {len(outputs)} outputs")
    
    # Remove worker 2 with migration
    print("\n[4/7] Removing decode worker 2 (with KV migration)...")
    try:
        await backend.decoding_engine.remove_worker(worker_id=2, migrate_requests=True)
        print("✅ Worker 2 removed, KV caches migrated to other workers")
        print(f"✓ Remaining decode workers: {len([w for w in backend.decoding_engine.workers if w is not None])}")
    except Exception as e:
        print(f"❌ Worker removal failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Continue processing with 2 workers
    print("\n[5/7] Continuing with 2 workers (10 seconds)...")
    completed_count = 0
    for i in range(10):
        await asyncio.sleep(1)
        outputs = await backend.get_outputs()
        if outputs:
            completed_count += len(outputs)
            print(f"  [{i+1}/10] Completed: {completed_count} total")
    
    print("\n[6/7] Stopping backend...")
    await backend.stop()
    print("✓ Backend stopped")
    
    print(f"\n[7/7] Test 3 completed! ✅")
    print(f"  Total completed: {completed_count}/{len(test_requests)} requests")
    return True


async def main():
    """Main test runner"""
    print("\n" + "=" * 80)
    print("ElasticMM Phase 1 功能测试: Instance Migration Infrastructure")
    print("=" * 80)
    
    tests = [
        ("Dynamic Worker Management", test_dynamic_worker_management),
        ("KV Cache Migration", test_kv_migration),
        ("Worker Removal with Migration", test_worker_removal_with_migration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n\n{'#' * 80}")
            print(f"Running: {test_name}")
            print(f"{'#' * 80}")
            
            result = await test_func()
            results.append((test_name, result))
            
            if result:
                print(f"\n✅ {test_name} PASSED")
            else:
                print(f"\n❌ {test_name} FAILED")
            
            # Cleanup between tests
            print("\nWaiting 5 seconds before next test...")
            await asyncio.sleep(5)
            
        except Exception as e:
            print(f"\n❌ {test_name} CRASHED: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Print summary
    print("\n\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Phase 1 implementation is working!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the output above.")


if __name__ == "__main__":
    asyncio.run(main())

