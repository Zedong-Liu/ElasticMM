#!/usr/bin/env python3
"""
Gain-Cost Model Parameter Calibration v2
按照用户建议的科学测试方法：
- Round 1: 基准测试 (1E + 0P + 2D, 泊松分布, 50 requests)
- Round 2: 抢占测试 (1E + 2P + 1D, Prefill抢占Decode)
- Round 3: 峰值压测 (1E + 2P + 1D, 最大token budget)
"""

import os
import asyncio
import numpy as np
import time
import json
import argparse
import ray
from typing import List, Dict, Any
from pathlib import Path

from elasticmm.engine.v0.backend import V0EngineBackend
from elasticmm.engine.v0.utils import Request, EngineStage
from elasticmm.core.balancer import ModalityType


class PoissonRequestGenerator:
    """泊松分布请求生成器"""
    
    def __init__(self, arrival_rate: float = 2.0, total_requests: int = 50):
        """
        Args:
            arrival_rate: 平均每秒到达的请求数 (lambda)
            total_requests: 总请求数
        """
        self.arrival_rate = arrival_rate
        self.total_requests = total_requests
    
    def generate_arrival_times(self) -> List[float]:
        """生成请求到达时间 (相对于开始时间的偏移，单位：秒)"""
        # 泊松过程：间隔时间服从指数分布
        intervals = np.random.exponential(1.0 / self.arrival_rate, self.total_requests)
        arrival_times = np.cumsum(intervals)
        return arrival_times.tolist()
    
    def create_test_requests(self, images: List[Any]) -> List[Request]:
        """创建测试请求（带图片的多模态请求）"""
        from PIL import Image
        
        prompts = [
            "<|user|>\n<|vision_start|><|image_pad|><|vision_end|>\n请描述这张图片。\n<|assistant|>\n",
            "<|user|>\n<|vision_start|><|image_pad|><|vision_end|>\n图片中有什么？\n<|assistant|>\n",
            "<|user|>\n<|vision_start|><|image_pad|><|vision_end|>\n这是什么场景？\n<|assistant|>\n",
            "<|user|>\n<|vision_start|><|image_pad|><|vision_end|>\n详细描述图片内容。\n<|assistant|>\n",
        ]
        
        requests = []
        for i in range(self.total_requests):
            img = images[i % len(images)]  # 循环使用图片
            prompt = prompts[i % len(prompts)]  # 循环使用prompts
            
            req = Request(
                request_id=f"poisson_req_{i:03d}",
                prompt=prompt,
                max_tokens=20,  # 固定输出长度
                temperature=0.8,
                top_p=0.9,
                multi_modal_data={"image": img},
            )
            requests.append(req)
        return requests


class GainCostCalibrator:
    """Gain-Cost模型参数校准器"""
    
    def __init__(self, model_path: str, output_dir: str = "./calibration_results"):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 校准结果
        self.results = {
            'round1_baseline': {},
            'round2_preemption': {},
            'round3_peak': {},
            'final_params': {},
        }
    
    async def round1_baseline(self):
        """
        Round 1: 基准测试
        配置: 1E + 1P + 2D
        负载: 50 requests, 泊松分布 (λ=2 req/s)
        测量: decode输出延迟, KV传输带宽
        """
        print("\n" + "="*80)
        print("🔬 Round 1: 基准测试 (Baseline)")
        print("="*80)
        print("配置: 1 Encoding + 1 Prefill + 2 Decoding")
        print("负载: 50 requests, 泊松分布 (λ=2 req/s)")
        print("目标: 测量prefill/decode延迟(ms/token), KV传输带宽")
        print("="*80)
        
        # 初始化backend
        backend = V0EngineBackend(
            model_path=self.model_path,
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
        
        print("\n[Round 1] Initializing backend...")
        await backend.initialize()
        
        print("[Round 1] Starting backend...")
        await backend.start()
        await asyncio.sleep(2)  # 等待稳定
        
        # 加载测试图片
        print("\n[Round 1] Loading test images...")
        from PIL import Image
        image_base_path = "/root/lzd/dataset/shareGPT-4o/images"
        
        images = []
        for i in range(10, 30):  # 加载20张图片
            image_path = f"{image_base_path}/{i}.jpg"
            try:
                img = Image.open(image_path)
                images.append(img)
            except Exception as e:
                print(f"  ⚠ Failed to load {image_path}, using dummy image")
                import numpy as np
                img_array = np.full((224, 224, 3), (128, 128, 128), dtype=np.uint8)
                img = Image.fromarray(img_array)
                images.append(img)
        
        print(f"[Round 1] Loaded {len(images)} images")
        
        # 生成请求
        print("\n[Round 1] Generating Poisson-distributed requests...")
        generator = PoissonRequestGenerator(arrival_rate=2.0, total_requests=50)
        requests = generator.create_test_requests(images)
        arrival_times = generator.generate_arrival_times()
        
        print(f"[Round 1] Generated {len(requests)} multimodal requests")
        print(f"[Round 1] Expected duration: ~{arrival_times[-1]:.1f}s")
        
        # 提交请求
        start_time = time.time()
        completed_requests = []
        
        async def submit_requests():
            for req, arrival_time in zip(requests, arrival_times):
                # 等待到达时间
                current_time = time.time() - start_time
                wait_time = arrival_time - current_time
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                
                await backend.add_request(req)
                print(f"[Round 1] Submitted {req.request_id} at t={time.time()-start_time:.2f}s")
        
        async def collect_outputs():
            while len(completed_requests) < len(requests):
                outputs = await backend.get_outputs()
                for output in outputs:
                    if output.finished:
                        completed_requests.append(output)
                        print(f"[Round 1] Completed {output.request_id} ({len(completed_requests)}/{len(requests)})")
                await asyncio.sleep(0.1)
        
        # 并发执行提交和收集
        await asyncio.gather(
            submit_requests(),
            collect_outputs(),
        )
        
        total_time = time.time() - start_time
        print(f"\n[Round 1] All requests completed in {total_time:.2f}s")
        
        # 获取性能指标
        print("\n[Round 1] Collecting performance metrics...")
        perf_metrics = backend.get_performance_metrics()
        
        # 计算结果
        results = {
            'config': '1E + 1P + 2D',
            'num_requests': len(requests),
            'total_time': total_time,
            'throughput': len(requests) / total_time,
            'metrics': perf_metrics,
        }
        
        # 提取关键指标
        if 'decoding' in perf_metrics and 'decode_latency_ms_per_token' in perf_metrics['decoding']:
            decode_latency = perf_metrics['decoding']['decode_latency_ms_per_token']
            results['decode_ms_per_token'] = decode_latency.get('mean', 0)
            print(f"\n📊 Decode延迟 (归一化): {decode_latency.get('mean', 0):.3f} ms/token")
            print(f"   - P50: {decode_latency.get('p50', 0):.3f} ms/token")
            print(f"   - P90: {decode_latency.get('p90', 0):.3f} ms/token")
            print(f"   - P99: {decode_latency.get('p99', 0):.3f} ms/token")
        
        if 'decoding' in perf_metrics and 'kv_transfer' in perf_metrics['decoding']:
            kv_transfer = perf_metrics['decoding']['kv_transfer']
            bandwidth = kv_transfer.get('bandwidth_blocks_per_sec', 0)
            results['kv_bandwidth_blocks_per_sec'] = bandwidth
            print(f"\n📡 KV传输带宽: {bandwidth:.1f} blocks/s")
            print(f"   - 平均传输时间: {kv_transfer.get('avg_time_sec', 0)*1000:.2f} ms")
            print(f"   - 平均传输块数: {kv_transfer.get('avg_blocks', 0):.1f} blocks")
        
        # 保存结果
        self.results['round1_baseline'] = results
        
        print("\n✅ Round 1 完成!")
        print("[Round 1] Backend保持运行，准备进行worker role切换...")
        
        # ✅ 保持backend运行，为Round 2的role切换做准备
        return results, backend  # 复用backend进行role切换测试
    
    async def round2_preemption(self, backend_from_round1=None):
        """
        Round 2: 抢占测试
        配置: 1E + 2P + 1D (通过worker role切换实现)
        负载: 60 requests, 泊松分布
        测量: 与Round 1对比，计算prefill扩容和decode缩容的影响
        """
        print("\n" + "="*80)
        print("🔬 Round 2: 抢占测试 (Preemption Simulation)")
        print("="*80)
        print("配置: 1 Encoding + 2 Prefill + 1 Decoding")
        print("负载: 60 requests, 泊松分布")
        print("目标: 测量扩容prefill/缩减decode对延迟的影响")
        print("(与Round 1的1E+1P+2D配置对比)")
        print("="*80)
        
        # ✅ 使用NCCL解耦架构进行真实的worker role切换
        if backend_from_round1 is None:
            raise RuntimeError("Round 2 requires backend from Round 1 for role switching test!")
        
        backend = backend_from_round1
        
        print("\n[Round 2] 🔄 执行Worker Role切换: 1E+1P+2D → 1E+2P+1D")
        print("  - Decoding Worker 1 (global_rank=3) → Prefill Worker 1")
        print("  这是NCCL解耦架构的关键测试！")
        
        # 执行role切换：将一个decode worker切换为prefill worker
        # Round 1配置: encoding(0), prefill(1), decoding(2,3)
        # Round 2目标: encoding(0), prefill(1,3), decoding(2)
        await backend.switch_worker_role(
            worker_id=1,  # decoding的第2个worker (global_rank=3)
            from_stage="decoding",
            to_stage="prefill",
            migrate_kv=True
        )
        
        print("[Round 2] ✅ Worker role切换完成!")
        await asyncio.sleep(2)
        
        # 加载测试图片
        print("\n[Round 2] Loading test images...")
        from PIL import Image
        image_base_path = "/root/lzd/dataset/shareGPT-4o/images"
        
        images = []
        for i in range(10, 30):
            image_path = f"{image_base_path}/{i}.jpg"
            try:
                img = Image.open(image_path)
                images.append(img)
            except:
                import numpy as np
                img_array = np.full((224, 224, 3), (128, 128, 128), dtype=np.uint8)
                img = Image.fromarray(img_array)
                images.append(img)
        
        # 生成请求
        print("\n[Round 2] Generating requests...")
        generator = PoissonRequestGenerator(arrival_rate=2.0, total_requests=60)
        requests = generator.create_test_requests(images)
        arrival_times = generator.generate_arrival_times()
        
        print(f"[Round 2] Generated {len(requests)} multimodal requests")
        
        # TODO: 实现抢占逻辑
        # 在系统运行到一定阶段时，手动触发Prefill抢占Decode worker
        
        start_time = time.time()
        completed_requests = []
        
        async def submit_requests():
            for i, (req, arrival_time) in enumerate(zip(requests, arrival_times)):
                current_time = time.time() - start_time
                wait_time = arrival_time - current_time
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                
                await backend.add_request(req)
                print(f"[Round 2] Submitted {req.request_id} at t={time.time()-start_time:.2f}s")
        
        async def collect_outputs():
            while len(completed_requests) < len(requests):
                outputs = await backend.get_outputs()
                for output in outputs:
                    if output.finished:
                        completed_requests.append(output)
                        print(f"[Round 2] Completed {output.request_id} ({len(completed_requests)}/{len(requests)})")
                await asyncio.sleep(0.1)
        
        await asyncio.gather(
            submit_requests(),
            collect_outputs(),
        )
        
        total_time = time.time() - start_time
        print(f"\n[Round 2] All requests completed in {total_time:.2f}s")
        print(f"[Round 2] Throughput: {len(requests)/total_time:.2f} req/s")
        
        # 获取最终性能指标
        perf_metrics_final = backend.get_performance_metrics()
        
        results = {
            'config': '1E + 2P + 1D (simulating prefill scale-up)',
            'num_requests': len(requests),
            'total_time': total_time,
            'throughput': len(requests) / total_time,
            'metrics': perf_metrics_final,
        }
        
        # 分析抢占对延迟的影响
        print("\n📊 Round 2 性能分析:")
        print("="*60)
        
        # Decode延迟分析
        if 'round1_baseline' in self.results and 'decode_ms_per_token' in self.results['round1_baseline']:
            baseline_decode = self.results['round1_baseline']['decode_ms_per_token']
            if 'decoding' in perf_metrics_final and 'decode_latency_ms_per_token' in perf_metrics_final['decoding']:
                current_decode = perf_metrics_final['decoding']['decode_latency_ms_per_token']['mean']
                # 抢占导致decode worker减少，延迟应该增加
                slowdown_ratio = current_decode / baseline_decode if baseline_decode > 0 else 1.0
                results['decode_slowdown'] = slowdown_ratio
                results['preemption_cost_decode'] = (slowdown_ratio - 1.0) * 100  # 百分比增加
                
                print(f"\n🔴 Decode延迟变化 (Preemption Cost):")
                print(f"   - Baseline (1E+1P+2D): {baseline_decode:.3f} ms/token")
                print(f"   - After preemption (1E+2P+1D): {current_decode:.3f} ms/token")
                print(f"   - Slowdown: {slowdown_ratio:.3f}x")
                print(f"   - Cost: +{(slowdown_ratio-1.0)*100:.1f}% 延迟增加")
        
        # Prefill延迟分析
        if 'prefill' in perf_metrics_final and 'prefill_latency_ms_per_token' in perf_metrics_final['prefill']:
            prefill_latency = perf_metrics_final['prefill']['prefill_latency_ms_per_token']['mean']
            results['prefill_ms_per_token'] = prefill_latency
            
            # 如果Round 1有prefill数据，对比
            if 'round1_baseline' in self.results and 'metrics' in self.results['round1_baseline']:
                r1_metrics = self.results['round1_baseline']['metrics']
                if 'prefill' in r1_metrics and 'prefill_latency_ms_per_token' in r1_metrics['prefill']:
                    baseline_prefill = r1_metrics['prefill']['prefill_latency_ms_per_token']['mean']
                    # 抢占增加了prefill worker，延迟应该减少
                    speedup_ratio = baseline_prefill / prefill_latency if prefill_latency > 0 else 1.0
                    results['prefill_speedup'] = speedup_ratio
                    results['preemption_gain_prefill'] = (speedup_ratio - 1.0) * 100  # 百分比减少
                    
                    print(f"\n🟢 Prefill延迟变化 (Preemption Gain):")
                    print(f"   - Baseline (1E+1P+2D): {baseline_prefill:.3f} ms/token")
                    print(f"   - After preemption (1E+2P+1D): {prefill_latency:.3f} ms/token")
                    print(f"   - Speedup: {speedup_ratio:.3f}x")
                    print(f"   - Gain: {(speedup_ratio-1.0)*100:.1f}% 延迟减少")
            else:
                print(f"\n🟢 Prefill延迟 (归一化): {prefill_latency:.3f} ms/token")
        
        print("="*60)
        
        self.results['round2_preemption'] = results
        
        print("\n✅ Round 2 完成!")
        print("💡 Round 2结束配置为 1E+2P+1D，与Round 3相同，复用backend...")
        
        # 不停止backend，直接返回给Round 3使用
        return results, backend  # 返回backend供Round 3复用
    
    async def round3_peak_stress(self, backend=None):
        """
        Round 3: 峰值压测
        配置: 1E + 2P + 1D (可以复用Round 2的backend)
        负载: 峰值负载，找到decode最大token budget
        测试: 抢占1个prefill worker，测量极限情况下的gain/cost
        
        Args:
            backend: 如果提供，则复用现有backend（来自Round 2）
        """
        print("\n" + "="*80)
        print("🔬 Round 3: 峰值压测 (Peak Stress Test)")
        print("="*80)
        print("配置: 1 Encoding + 2 Prefill + 1 Decoding")
        print("负载: 峰值负载，测试最大token budget")
        print("目标: 找到decode瓶颈，测试极限抢占场景")
        print("="*80)
        
        # 如果没有提供backend，则创建新的
        backend_created = False
        if backend is None:
            print("\n[Round 3] Creating new backend...")
            backend = V0EngineBackend(
                model_path=self.model_path,
                num_encoding_workers=1,
                num_prefill_workers=2,
                num_decoding_workers=1,
                block_size=16,
                max_num_gpu_blocks=3000,
                dtype="float16",
                gpu_memory_utilization=0.85,
                kv_transfer_method="nccl",
                limit_mm_per_prompt={"image": 1},
            )
            
            print("[Round 3] Initializing backend...")
            await backend.initialize()
            
            print("[Round 3] Starting backend...")
            await backend.start()
            await asyncio.sleep(2)
            backend_created = True
        else:
            print("\n[Round 3] ✅ Reusing backend from Round 2 (smart!)")
            # 等待一下，让系统稳定
            await asyncio.sleep(1)
        
        # 加载测试图片
        print("\n[Round 3] Loading test images...")
        from PIL import Image
        image_base_path = "/root/lzd/dataset/shareGPT-4o/images"
        
        images = []
        for i in range(10, 30):
            image_path = f"{image_base_path}/{i}.jpg"
            try:
                img = Image.open(image_path)
                images.append(img)
            except:
                import numpy as np
                img_array = np.full((224, 224, 3), (128, 128, 128), dtype=np.uint8)
                img = Image.fromarray(img_array)
                images.append(img)
        
        # 生成大量请求，模拟峰值负载
        print("\n[Round 3] Generating peak load requests...")
        num_requests = 100
        max_tokens_per_req = 50  # 更长的输出
        
        prompts = [
            "<|user|>\n<|vision_start|><|image_pad|><|vision_end|>\n请描述这张图片。\n<|assistant|>\n",
            "<|user|>\n<|vision_start|><|image_pad|><|vision_end|>\n图片中有什么？\n<|assistant|>\n",
            "<|user|>\n<|vision_start|><|image_pad|><|vision_end|>\n这是什么场景？\n<|assistant|>\n",
        ]
        
        requests = []
        for i in range(num_requests):
            img = images[i % len(images)]
            prompt = prompts[i % len(prompts)]
            
            req = Request(
                request_id=f"peak_req_{i:03d}",
                prompt=prompt,
                max_tokens=max_tokens_per_req,
                temperature=0.8,
                top_p=0.9,
                multi_modal_data={"image": img},
            )
            requests.append(req)
        
        print(f"[Round 3] Generated {len(requests)} multimodal requests (longer sequences)")
        
        # 快速提交所有请求 (模拟突发负载)
        start_time = time.time()
        print("\n[Round 3] Submitting all requests rapidly...")
        for i, req in enumerate(requests):
            await backend.add_request(req)
            if i % 10 == 0:
                print(f"[Round 3] Submitted {i+1}/{len(requests)} requests")
                await asyncio.sleep(0.1)  # 轻微延迟，避免过载
        
        print(f"[Round 3] All requests submitted at t={time.time()-start_time:.2f}s")
        
        # 监控系统状态和decode batch capacity
        completed_requests = []
        request_token_counts = {}  # 记录每个请求生成的token数
        max_decode_batch_tokens = 0  # 记录decode batch中的最大总token数
        
        while len(completed_requests) < len(requests):
            outputs = await backend.get_outputs()
            for output in outputs:
                # 累加每个请求的token数
                if output.request_id not in request_token_counts:
                    request_token_counts[output.request_id] = 0
                request_token_counts[output.request_id] += len(output.output_token_ids)
                
                if output.finished:
                    completed_requests.append(output)
            
            # 检查系统状态
            stats = backend.get_stats()
            decode_running = stats['decoding']['num_running']
            decode_waiting = stats['decoding']['num_waiting']
            
            # 计算当前decode batch中的总token数
            # 这是一个近似值：running请求数 * 平均每个请求的当前token数
            if decode_running > 0:
                # 获取当前正在decode的请求的平均token数
                running_request_tokens = []
                for req_id, token_count in request_token_counts.items():
                    # 检查这个请求是否还在running（未完成）
                    if req_id not in [out.request_id for out in completed_requests]:
                        running_request_tokens.append(token_count)
                
                if running_request_tokens:
                    current_batch_tokens = sum(running_request_tokens)
                    max_decode_batch_tokens = max(max_decode_batch_tokens, current_batch_tokens)
                    
                    if len(completed_requests) % 20 == 0 and len(completed_requests) > 0:
                        print(f"[Round 3] Current decode batch: {decode_running} reqs, "
                              f"~{current_batch_tokens} tokens total (max so far: {max_decode_batch_tokens})")
            
            
            await asyncio.sleep(0.1)
        
        total_time = time.time() - start_time
        print(f"\n[Round 3] All requests completed in {total_time:.2f}s")
        print(f"[Round 3] Throughput: {len(requests)/total_time:.2f} req/s")
        
        # 获取性能指标
        stats = backend.get_stats()
        
        # 计算总decode tokens：从request_token_counts统计
        total_tokens_decode = sum(request_token_counts.values())
        
        # 从decode metrics中获取更详细的信息
        decode_metrics = stats.get('decoding_metrics', {})
        
        results = {
            'config': '1E + 2P + 1D (peak load)',
            'num_requests': len(requests),
            'total_time': total_time,
            'throughput': len(requests) / total_time,
            'total_decode_tokens': total_tokens_decode,
            'max_decode_batch_tokens': max_decode_batch_tokens,  # 这才是你要的token budget！
            'decode_metrics': decode_metrics,
        }
        
        print(f"\n📊 Peak Load Results:")
        print(f"   - Total decode tokens generated: {total_tokens_decode}")
        print(f"   - Max decode batch capacity: {max_decode_batch_tokens} tokens (峰值并发)")
        print(f"   - Avg tokens per request: {total_tokens_decode/len(requests):.1f}")
        print(f"   - Throughput: {total_tokens_decode/total_time:.1f} tokens/s")
        
        self.results['round3_peak'] = results
        
        # 只在我们创建了新backend时才停止它
        if backend_created:
            await backend.stop()
        else:
            # 如果是复用的backend，现在停止它
            await backend.stop()
        
        print("\n✅ Round 3 完成!")
        return results
    
    def compute_final_parameters(self):
        """根据三轮测试结果，计算最终的Gain-Cost参数"""
        print("\n" + "="*80)
        print("📊 计算最终参数")
        print("="*80)
        
        params = {}
        
        # ===== 基础延迟 (从Round 1) =====
        print("\n1️⃣ 基础延迟 (Round 1 Baseline):")
        r1_metrics = self.results.get('round1_baseline', {}).get('metrics', {})
        
        # Encoding延迟
        if 'encoding' in r1_metrics and 'encoding_latency_ms_per_token' in r1_metrics['encoding']:
            enc_latency = r1_metrics['encoding']['encoding_latency_ms_per_token']['mean']
            params['encoding_latency_ms_per_token'] = enc_latency
            print(f"   ✓ Encoding: {enc_latency:.3f} ms/token")
        else:
            params['encoding_latency_ms_per_token'] = 10.0  # 默认值
            print(f"   ⚠️  Encoding: 使用默认值 10.0 ms/token")
        
        # Prefill延迟
        if 'prefill' in r1_metrics and 'prefill_latency_ms_per_token' in r1_metrics['prefill']:
            prefill_latency = r1_metrics['prefill']['prefill_latency_ms_per_token']['mean']
            params['prefill_latency_ms_per_token'] = prefill_latency
            print(f"   ✓ Prefill: {prefill_latency:.3f} ms/token")
        else:
            params['prefill_latency_ms_per_token'] = 5.0  # 默认值
            print(f"   ⚠️  Prefill: 使用默认值 5.0 ms/token")
        
        # Decode延迟
        if 'decoding' in r1_metrics and 'decode_latency_ms_per_token' in r1_metrics['decoding']:
            decode_latency = r1_metrics['decoding']['decode_latency_ms_per_token']['mean']
            params['decode_latency_ms_per_token'] = decode_latency
            print(f"   ✓ Decode: {decode_latency:.3f} ms/token")
        else:
            params['decode_latency_ms_per_token'] = 20.0  # 默认值
            print(f"   ⚠️  Decode: 使用默认值 20.0 ms/token")
        
        # ===== Migration Cost (从KV传输数据计算) =====
        print("\n2️⃣ Migration Cost (KV传输开销):")
        
        # 尝试从Round 2和Round 3中提取KV传输数据（这些轮次有迁移）
        kv_transfer_found = False
        for round_name in ['round2_preemption', 'round3_peak', 'round1_baseline']:
            round_data = self.results.get(round_name, {})
            round_metrics = round_data.get('metrics', {})
            
            if 'decoding' in round_metrics and 'kv_transfer' in round_metrics['decoding']:
                kv_transfer = round_metrics['decoding']['kv_transfer']
                avg_time = kv_transfer.get('avg_time_sec', 0)
                avg_blocks = kv_transfer.get('avg_blocks', 0)
                bandwidth = kv_transfer.get('bandwidth_blocks_per_sec', 0)
                
                # 使用平均传输时间作为迁移开销
                if avg_time > 0:
                    params['migration_cost'] = avg_time
                    params['kv_transfer_bandwidth'] = bandwidth
                    print(f"   ✓ Migration Cost: {avg_time:.4f}s (from {round_name})")
                    print(f"      平均传输 {avg_blocks:.1f} blocks")
                    print(f"      KV传输带宽: {bandwidth:.1f} blocks/s")
                    kv_transfer_found = True
                    break
        
        if not kv_transfer_found:
            params['migration_cost'] = 0.01  # 默认值
            params['kv_transfer_bandwidth'] = 0
            print(f"   ⚠️  Migration Cost: 使用默认值 0.01s (无KV传输数据)")
        
        # ===== Preemption Penalty (从Round 2计算) =====
        print("\n3️⃣ Preemption Penalty (抢占惩罚因子):")
        r2_results = self.results.get('round2_preemption', {})
        
        if 'preemption_cost_decode' in r2_results or 'decode_slowdown' in r2_results:
            # 从decode延迟增加计算惩罚
            slowdown = r2_results.get('decode_slowdown', 1.0)
            cost_pct = r2_results.get('preemption_cost_decode', 0)
            
            # 惩罚因子 = 延迟增加比例 (作为乘数因子，不是绝对时间)
            penalty_factor = slowdown - 1.0  # 例如1.2x slowdown => 0.2 penalty factor
            params['preemption_penalty'] = max(penalty_factor, 0.0)
            print(f"   ✓ Preemption Penalty: {penalty_factor:.3f} (slowdown={slowdown:.3f}x, cost=+{cost_pct:.1f}%)")
        else:
            params['preemption_penalty'] = 0.2  # 默认20%惩罚
            print(f"   ⚠️  Preemption Penalty: 使用默认值 0.2 (20% slowdown)")
        
        # ===== Scalability Coefficients =====
        print("\n4️⃣ Scalability Coefficients:")
        params['scalability_encode'] = 0.80
        params['scalability_prefill'] = 0.90
        params['scalability_decoding'] = 0.75
        print(f"   ✓ Encode: {params['scalability_encode']}")
        print(f"   ✓ Prefill: {params['scalability_prefill']}")
        print(f"   ✓ Decode: {params['scalability_decoding']}")
        
        # ===== Max Token Budget (从Round 3) =====
        print("\n5️⃣ Max Decode Token Budget:")
        if 'max_decode_batch_tokens' in self.results.get('round3_peak', {}):
            max_budget = self.results['round3_peak']['max_decode_batch_tokens']
            params['max_decode_token_budget'] = int(max_budget)
            print(f"   ✓ Max Decode Token Budget: {int(max_budget)} tokens (peak batch capacity)")
        else:
            params['max_decode_token_budget'] = 2000  # 默认值
            print(f"   ⚠️  Max Decode Token Budget: 使用默认值 2000 tokens")
        
        self.results['final_params'] = params
        
        print("\n" + "="*80)
        print("🎯 最终推荐参数:")
        print("="*80)
        print(f"  encoding_latency_ms_per_token: {params.get('encoding_latency_ms_per_token', 0):.3f}")
        print(f"  prefill_latency_ms_per_token: {params.get('prefill_latency_ms_per_token', 0):.3f}")
        print(f"  decode_latency_ms_per_token: {params.get('decode_latency_ms_per_token', 0):.3f}")
        print(f"  migration_cost: {params.get('migration_cost', 0):.4f}")
        print(f"  preemption_penalty: {params.get('preemption_penalty', 0):.3f}")
        print(f"  scalability_encode: {params.get('scalability_encode', 0)}")
        print(f"  scalability_prefill: {params.get('scalability_prefill', 0)}")
        print(f"  scalability_decoding: {params.get('scalability_decoding', 0)}")
        print(f"  max_decode_token_budget: {params.get('max_decode_token_budget', 0)}")
        
        return params
    
    def save_results(self):
        """保存所有结果到JSON文件"""
        # 保存为系统配置文件
        config_file = self.output_dir / "gain_cost_params.json"
        with open(config_file, 'w') as f:
            json.dump(self.results['final_params'], f, indent=2)
        print(f"\n💾 Gain-Cost parameters saved to {config_file}")
        print(f"    System will load this file on startup.")
        
        # 保存详细的校准数据（用于调试和分析）
        detail_file = self.output_dir / "calibration_details.json"
        with open(detail_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"💾 Detailed calibration data saved to {detail_file}")
        
        # 生成Python代码片段
        code_file = self.output_dir / "recommended_params.py"
        params = self.results['final_params']
        code = f'''# Gain-Cost Model Parameters (Calibrated on {time.strftime("%Y-%m-%d %H:%M:%S")})
# Hardware: {self.model_path}

# Base latencies (ms per token)
ENCODING_LATENCY_MS_PER_TOKEN = {params.get('encoding_latency_ms_per_token', 10.0):.3f}
PREFILL_LATENCY_MS_PER_TOKEN = {params.get('prefill_latency_ms_per_token', 5.0):.3f}
DECODE_LATENCY_MS_PER_TOKEN = {params.get('decode_latency_ms_per_token', 20.0):.3f}

# Migration cost (seconds per migration operation)
MIGRATION_COST = {params.get('migration_cost', 0.01):.4f}

# Preemption penalty (slowdown factor, e.g., 0.2 = 20% slowdown)
PREEMPTION_PENALTY = {params.get('preemption_penalty', 0.2):.3f}

# Scalability coefficients (parallel efficiency)
SCALABILITY_ENCODE = {params.get('scalability_encode', 0.80):.2f}
SCALABILITY_PREFILL = {params.get('scalability_prefill', 0.90):.2f}
SCALABILITY_DECODING = {params.get('scalability_decoding', 0.75):.2f}

# Max token budget (total tokens in decode batch at capacity)
MAX_DECODE_TOKEN_BUDGET = {params.get('max_decode_token_budget', 2000)}
'''
        with open(code_file, 'w') as f:
            f.write(code)
        print(f"💾 Python code saved to {code_file}")
    
    async def run_all(self):
        """运行所有三轮测试"""
        print("\n" + "="*80)
        print("🚀 Gain-Cost Model Calibration v2")
        print("="*80)
        print(f"Model: {self.model_path}")
        print(f"Output directory: {self.output_dir}")
        print("="*80)
        
        try:
            # Round 1: Baseline (返回backend供Round 2复用)
            round1_results, round1_backend = await self.round1_baseline()
            
            # Round 2: Preemption (复用Round 1的backend，返回backend供Round 3复用)
            round2_results, round2_backend = await self.round2_preemption(backend_from_round1=round1_backend)
            
            # Round 3: Peak stress (复用Round 2的backend)
            await self.round3_peak_stress(backend=round2_backend)
            
            # Compute final parameters
            self.compute_final_parameters()
            
            # Save results
            self.save_results()
            
            print("\n" + "="*80)
            print("✅ 校准完成！")
            print("="*80)
            
        except Exception as e:
            print(f"\n❌ Error during calibration: {e}")
            import traceback
            traceback.print_exc()


async def main():
    parser = argparse.ArgumentParser(description="Gain-Cost Model Calibration v2")
    parser.add_argument("--model_path", type=str, 
                        default="/root/lzd/model/qwen2.5-VL",
                        help="Path to model")
    parser.add_argument("--output_dir", type=str,
                        default="./calibration_results",
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    calibrator = GainCostCalibrator(
        model_path=args.model_path,
        output_dir=args.output_dir,
    )
    
    await calibrator.run_all()


if __name__ == "__main__":
    asyncio.run(main())

