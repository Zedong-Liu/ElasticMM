"""
测试EMPScheduler的弹性调度功能
验证：
1. get_worker_allocation() API
2. 弹性调度循环
3. 历史数据收集
4. 资源重新平衡
"""

import asyncio
import time
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from elasticmm.engine.v0 import V0EngineBackend
from elasticmm.core.scheduler import EMPScheduler
from elasticmm.engine.v0.utils import Request
from elasticmm.core.balancer import ModalityType


async def test_basic_apis():
    """测试基础API"""
    print("\n" + "="*80)
    print("测试1: 基础API")
    print("="*80)
    
    # 创建backend (1E + 1P + 2D)
    backend = V0EngineBackend(
        model_path="/root/lzd/model/qwen2.5-VL",
        num_encoding_workers=1,
        num_prefill_workers=1,
        num_decoding_workers=2,
        block_size=16,
        max_num_gpu_blocks=3000,
        dtype="float16",
        gpu_memory_utilization=0.85,
        kv_transfer_method="nccl",
        limit_mm_per_prompt={"image": 1},
    )
    
    print("\n[Test] Initializing backend...")
    await backend.initialize()
    
    # 测试get_worker_allocation
    print("\n[Test] Testing get_worker_allocation()...")
    worker_alloc = backend.get_worker_allocation()
    print(f"✓ Worker allocation: {worker_alloc}")
    
    expected = {0: 'encoding', 1: 'prefill', 2: 'decoding', 3: 'decoding'}
    assert worker_alloc == expected, f"Expected {expected}, got {worker_alloc}"
    print("✓ Worker allocation API works correctly!")
    
    # 测试get_stats
    print("\n[Test] Testing get_stats()...")
    stats = backend.get_stats()
    print(f"✓ Stats: P({stats['prefill']['num_workers']}w) "
          f"D({stats['decoding']['num_workers']}w)")
    
    await backend.stop()
    print("\n✅ Test 1 passed!")
    return True


async def test_scheduler_integration():
    """测试调度器集成"""
    print("\n" + "="*80)
    print("测试2: EMPScheduler集成")
    print("="*80)
    
    # 创建backend
    backend = V0EngineBackend(
        model_path="/root/lzd/model/qwen2.5-VL",
        num_encoding_workers=1,
        num_prefill_workers=1,
        num_decoding_workers=2,
        block_size=16,
        max_num_gpu_blocks=3000,
        dtype="float16",
        gpu_memory_utilization=0.85,
        kv_transfer_method="nccl",
        limit_mm_per_prompt={"image": 1},
    )
    
    print("\n[Test] Initializing backend...")
    await backend.initialize()
    await backend.start()
    
    # 创建scheduler
    print("\n[Test] Creating EMPScheduler...")
    scheduler = EMPScheduler(backend=backend)
    
    # 测试继承
    print("\n[Test] Testing inheritance from Scheduler...")
    assert hasattr(scheduler, 'heartbeat'), "Should inherit heartbeat()"
    assert hasattr(scheduler, 'select_prefill'), "Should inherit select_prefill()"
    print("✓ EMPScheduler correctly inherits from Scheduler")
    
    # 测试新功能
    print("\n[Test] Testing new features...")
    assert hasattr(scheduler, 'start_elastic_scheduling'), "Should have elastic scheduling"
    assert hasattr(scheduler, '_elastic_scheduling_loop'), "Should have scheduling loop"
    print("✓ New elastic scheduling methods exist")
    
    # 测试allocator历史数据
    print("\n[Test] Testing allocator history...")
    from elasticmm.core.allocator import InferenceStage
    allocator = scheduler.stage_allocators[ModalityType.TEXT_ONLY]
    
    assert hasattr(allocator, 'workload_history'), "Should have workload_history"
    assert hasattr(allocator, 'record_step_stats'), "Should have record_step_stats"
    
    # 记录一些数据
    allocator.record_step_stats(InferenceStage.PREFILL, 10)
    allocator.record_step_stats(InferenceStage.PREFILL, 15)
    allocator.record_step_stats(InferenceStage.PREFILL, 12)
    
    assert len(allocator.workload_history[InferenceStage.PREFILL]) == 3
    print("✓ Workload history recording works")
    
    # 测试预测
    estimated = allocator._estimate_future_workload(InferenceStage.PREFILL, 20)
    print(f"✓ Future workload estimation: {estimated} (current: 20)")
    
    await backend.stop()
    print("\n✅ Test 2 passed!")
    return True


async def test_elastic_scheduling_short():
    """测试弹性调度（短时间运行）"""
    print("\n" + "="*80)
    print("测试3: 弹性调度循环（30秒测试）")
    print("="*80)
    
    # 创建backend
    backend = V0EngineBackend(
        model_path="/root/lzd/model/qwen2.5-VL",
        num_encoding_workers=1,
        num_prefill_workers=1,
        num_decoding_workers=2,
        block_size=16,
        max_num_gpu_blocks=3000,
        dtype="float16",
        gpu_memory_utilization=0.85,
        kv_transfer_method="nccl",
        limit_mm_per_prompt={"image": 1},
    )
    
    print("\n[Test] Initializing backend...")
    await backend.initialize()
    await backend.start()
    
    # 创建scheduler并启动弹性调度
    print("\n[Test] Starting elastic scheduling...")
    scheduler = EMPScheduler(backend=backend)
    scheduler.start_elastic_scheduling()
    
    print("✓ Elastic scheduling loop started")
    print("  Monitoring for 30 seconds...")
    
    # 提交一些请求模拟workload
    print("\n[Test] Submitting test requests...")
    from PIL import Image
    import numpy as np
    
    # 创建测试图像
    img_array = np.full((224, 224, 3), (128, 128, 128), dtype=np.uint8)
    test_image = Image.fromarray(img_array)
    
    # 提交10个请求
    for i in range(10):
        request = Request(
            request_id=f"test_req_{i}",
            prompt="Describe this image.",
            max_tokens=20,
            multi_modal_data={"image": [test_image]},
        )
        await backend.add_request(request)
        print(f"  Submitted request {i+1}/10")
        await asyncio.sleep(0.5)
    
    # 监控30秒
    start_time = time.time()
    check_count = 0
    
    while time.time() - start_time < 30:
        await asyncio.sleep(5)
        check_count += 1
        
        # 检查状态
        stats = backend.get_stats()
        worker_alloc = backend.get_worker_allocation()
        allocator = scheduler.stage_allocators[ModalityType.TEXT_ONLY]
        
        print(f"\n[Check {check_count}] Status:")
        print(f"  Worker allocation: {worker_alloc}")
        print(f"  Prefill: {stats['prefill']['num_workers']}w, "
              f"{stats['prefill']['num_waiting']}q")
        print(f"  Decoding: {stats['decoding']['num_workers']}w, "
              f"{stats['decoding']['num_waiting']}q")
        print(f"  Step counter: {allocator.step_counter}")
        
        # 检查历史数据
        from elasticmm.core.allocator import InferenceStage
        history_len = len(allocator.workload_history[InferenceStage.PREFILL])
        if history_len > 0:
            print(f"  History collected: {history_len} samples")
    
    print("\n[Test] Stopping elastic scheduling...")
    await scheduler.stop_elastic_scheduling()
    
    # 验证
    allocator = scheduler.stage_allocators[ModalityType.TEXT_ONLY]
    assert allocator.step_counter > 0, "Step counter should have incremented"
    print(f"✓ Step counter incremented to: {allocator.step_counter}")
    
    # 检查历史数据
    from elasticmm.core.allocator import InferenceStage
    history = allocator.workload_history[InferenceStage.PREFILL]
    if len(history) > 0:
        print(f"✓ Collected {len(history)} workload samples")
        stats = allocator.get_workload_stats(InferenceStage.PREFILL)
        print(f"  Workload stats: mean={stats['mean']:.1f}, "
              f"trend={stats['trend']:.2f}")
    
    await backend.stop()
    print("\n✅ Test 3 passed!")
    return True


async def test_worker_migration():
    """测试worker迁移功能"""
    print("\n" + "="*80)
    print("测试4: Worker迁移")
    print("="*80)
    
    # 创建backend (1E + 1P + 2D)
    backend = V0EngineBackend(
        model_path="/root/lzd/model/qwen2.5-VL",
        num_encoding_workers=1,
        num_prefill_workers=1,
        num_decoding_workers=2,
        block_size=16,
        max_num_gpu_blocks=3000,
        dtype="float16",
        gpu_memory_utilization=0.85,
        kv_transfer_method="nccl",
        limit_mm_per_prompt={"image": 1},
    )
    
    print("\n[Test] Initializing backend...")
    await backend.initialize()
    await backend.start()
    
    # 初始分配
    print("\n[Test] Initial allocation:")
    alloc_before = backend.get_worker_allocation()
    print(f"  {alloc_before}")
    stats_before = backend.get_stats()
    print(f"  Prefill: {stats_before['prefill']['num_workers']}w, "
          f"Decoding: {stats_before['decoding']['num_workers']}w")
    
    # 执行迁移: worker 2 (decoding -> prefill)
    print("\n[Test] Migrating worker 2: decoding -> prefill")
    success = await backend.switch_worker_role(
        worker_id=2,
        from_stage='decoding',
        to_stage='prefill',
        migrate_kv=True
    )
    
    if success:
        print("✓ Migration successful!")
        
        # 验证新分配
        await asyncio.sleep(2)  # 等待迁移完成
        alloc_after = backend.get_worker_allocation()
        print(f"\n[Test] New allocation:")
        print(f"  {alloc_after}")
        
        stats_after = backend.get_stats()
        print(f"  Prefill: {stats_after['prefill']['num_workers']}w, "
              f"Decoding: {stats_after['decoding']['num_workers']}w")
        
        # 验证变化
        assert alloc_after[2] == 'prefill', "Worker 2 should be prefill now"
        assert stats_after['prefill']['num_workers'] == 2, "Prefill should have 2 workers"
        assert stats_after['decoding']['num_workers'] == 1, "Decoding should have 1 worker"
        
        print("✓ Worker allocation updated correctly!")
    else:
        print("❌ Migration failed")
        return False
    
    await backend.stop()
    print("\n✅ Test 4 passed!")
    return True


async def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("ElasticMM Scheduler Integration Tests")
    print("="*80)
    
    import ray
    ray.init(ignore_reinit_error=True)
    
    try:
        # Test 1: 基础API
        result1 = await test_basic_apis()
        await asyncio.sleep(3)
        
        # Test 2: 调度器集成
        result2 = await test_scheduler_integration()
        await asyncio.sleep(3)
        
        # Test 3: 弹性调度循环（快速测试）
        result3 = await test_elastic_scheduling_short()
        await asyncio.sleep(3)
        
        # Test 4: Worker迁移
        result4 = await test_worker_migration()
        
        # 总结
        print("\n" + "="*80)
        print("测试总结")
        print("="*80)
        print(f"Test 1 (Basic APIs): {'✅ PASS' if result1 else '❌ FAIL'}")
        print(f"Test 2 (Scheduler Integration): {'✅ PASS' if result2 else '❌ FAIL'}")
        print(f"Test 3 (Elastic Scheduling): {'✅ PASS' if result3 else '❌ FAIL'}")
        print(f"Test 4 (Worker Migration): {'✅ PASS' if result4 else '❌ FAIL'}")
        
        if all([result1, result2, result3, result4]):
            print("\n🎉 所有测试通过！")
            return 0
        else:
            print("\n❌ 部分测试失败")
            return 1
    
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        ray.shutdown()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)


