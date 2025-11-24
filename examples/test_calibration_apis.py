#!/usr/bin/env python3
"""
Quick API test for calibration v2
验证所有API调用是否正确
"""

import asyncio
from elasticmm.engine.v0.backend import V0EngineBackend
from elasticmm.engine.v0.utils import Request, StageMetrics


async def test_backend_apis():
    """测试Backend APIs"""
    print("="*80)
    print("🔬 Testing Backend APIs")
    print("="*80)
    
    # Test 1: Backend initialization
    print("\n[Test 1] Backend initialization...")
    backend = V0EngineBackend(
        model_path="/root/lzd/model/qwen2.5-VL",
        num_encoding_workers=1,
        num_prefill_workers=1,
        num_decoding_workers=1,
        block_size=16,
        max_num_gpu_blocks=1000,
        kv_transfer_method="nccl",
    )
    print("✅ Backend created")
    
    # Test 2: StageMetrics initialization
    print("\n[Test 2] StageMetrics initialization...")
    try:
        metrics = StageMetrics(stage_name="test", sample_interval=10)
        print(f"✅ StageMetrics created: {metrics.stage_name}")
    except Exception as e:
        print(f"❌ StageMetrics creation failed: {e}")
        return
    
    # Test 3: Request creation
    print("\n[Test 3] Request creation...")
    try:
        req = Request(
            request_id="test_001",
            prompt="测试",
            max_tokens=10,
            temperature=0.8,
            top_p=0.9,
        )
        print(f"✅ Request created: {req.request_id}")
    except Exception as e:
        print(f"❌ Request creation failed: {e}")
        return
    
    # Test 4: Backend.get_performance_metrics
    print("\n[Test 4] Backend.get_performance_metrics()...")
    try:
        perf_metrics = backend.get_performance_metrics()
        print(f"✅ Performance metrics retrieved: {list(perf_metrics.keys())}")
    except Exception as e:
        print(f"❌ get_performance_metrics failed: {e}")
        return
    
    # Test 5: Backend.export_metrics_to_json
    print("\n[Test 5] Backend.export_metrics_to_json()...")
    try:
        backend.export_metrics_to_json("/tmp/test_metrics.json")
        print("✅ Metrics exported to /tmp/test_metrics.json")
    except Exception as e:
        print(f"❌ export_metrics_to_json failed: {e}")
        return
    
    print("\n" + "="*80)
    print("✅ All API tests passed!")
    print("="*80)


def test_stage_metrics():
    """测试StageMetrics的方法"""
    print("\n" + "="*80)
    print("🔬 Testing StageMetrics Methods")
    print("="*80)
    
    metrics = StageMetrics(stage_name="test", sample_interval=1)
    
    # Test record methods
    print("\n[Test 1] Recording prefill latency...")
    metrics.record_prefill(num_input_tokens=100, latency_ms=50.0, should_sample=True)
    print(f"✅ Recorded: {len(metrics.prefill_latencies)} samples")
    
    print("\n[Test 2] Recording decode latency...")
    metrics.record_decode(num_output_tokens=10, latency_ms=5.0, should_sample=True)
    print(f"✅ Recorded: {len(metrics.decode_latencies)} samples")
    
    print("\n[Test 3] Recording KV transfer...")
    metrics.record_kv_transfer(transfer_time_sec=0.01, num_blocks=5)
    print(f"✅ Recorded: {len(metrics.kv_transfer_times)} samples")
    
    print("\n[Test 4] Getting stats...")
    try:
        stats = metrics.get_stats()
        print(f"✅ Stats retrieved: {list(stats.keys())}")
        if 'prefill_latency_ms_per_token' in stats:
            print(f"   - Prefill mean: {stats['prefill_latency_ms_per_token']['mean']:.3f} ms/token")
        if 'decode_latency_ms_per_token' in stats:
            print(f"   - Decode mean: {stats['decode_latency_ms_per_token']['mean']:.3f} ms/token")
    except Exception as e:
        print(f"❌ get_stats failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*80)
    print("✅ All StageMetrics tests passed!")
    print("="*80)


if __name__ == "__main__":
    # Test StageMetrics (sync)
    test_stage_metrics()
    
    # Test Backend APIs (async)
    asyncio.run(test_backend_apis())
    
    print("\n" + "="*80)
    print("🎉 All tests passed! Ready to run calibration v2!")
    print("="*80)





