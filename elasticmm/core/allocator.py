import asyncio
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
from collections import deque

from .balancer import ModalityType, WorkloadStats, InstanceInfo
from .gain_cost_config import get_gain_cost_config


class InferenceStage(Enum):
    """Inference stage enumeration"""
    ENCODE = "encode"
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass
class Request:
    """Request data structure"""
    request_id: str
    modality_type: ModalityType
    input_length: int
    estimated_output_length: int
    priority: int = 1  # 1=highest priority
    timestamp: float = 0.0
    images: Optional[List] = None  # Image data for multimodal requests
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
    
    def estimate_memory_usage(self) -> float:
        """Estimate memory usage (GB)"""
        # Simplified memory estimation based on input/output length
        base_memory = 1.0  # Base memory 1GB
        input_memory = self.input_length * 0.001  # ~1MB per token
        output_memory = self.estimated_output_length * 0.001
        
        return base_memory + input_memory + output_memory


@dataclass
class GainCostResult:
    """Gain-cost calculation result"""
    gain: float
    cost: float
    net_benefit: float
    is_beneficial: bool


@dataclass
class AllocationDecision:
    """Allocation decision result"""
    stage: InferenceStage
    instances: List[str]
    requests: List[Request]
    estimated_time: float
    gain_cost: Optional[GainCostResult] = None


class StageLevelResourceAllocator:
    """
    Stage-level resource allocator
    Implements Elastic Partition Scheduling strategy
    """
    
    def __init__(self):
        self.stage_instances = {
            InferenceStage.ENCODE: [],
            InferenceStage.PREFILL: [],
            InferenceStage.DECODE: []
        }
        self.request_queues = {
            InferenceStage.ENCODE: [],
            InferenceStage.PREFILL: [],
            InferenceStage.DECODE: []
        }
        self._lock = asyncio.Lock()
        
        # Load Gain-Cost model parameters
        self.gain_cost_config = get_gain_cost_config()
        
        # Configuration parameters
        self.memory_limit_gb = 32.0  # Total memory limit
        self.max_batch_size = 50  # Maximum batch size
        self.scaling_thresholds = {
            InferenceStage.ENCODE: 15,
            InferenceStage.PREFILL: 15,
            InferenceStage.DECODE: 30
        }
        self.min_instances_per_stage = 1
        
        # ✅ 未来收益计算参数
        self.future_rounds_horizon = 100  # 考虑未来100轮的累积收益
        self.discount_factor = 0.95  # 折扣因子（可选，越远的未来权重越低）
        
        # ✅ 对齐调度间隔：每N轮才重新评估资源分配
        self.step_counter = 0  # 总的decode迭代次数
        self.step_adjustment_interval = self.future_rounds_horizon  # 与未来收益轮数对齐
        self.last_step_adjustment = 0
        
        # ✅ 记录上一轮的worker分配（用于增量式调整）
        # 格式: {worker_id: stage}，例如 {0: 'encode', 1: 'prefill', 2: 'prefill', 3: 'decode'}
        self.previous_allocation = {}
        self.current_allocation = {}
        
        # ✅ 历史数据收集（用于预测未来workload）
        self.workload_history = {
            InferenceStage.ENCODE: deque(maxlen=200),
            InferenceStage.PREFILL: deque(maxlen=200),
            InferenceStage.DECODE: deque(maxlen=200),
        }
        self.latency_history = {
            InferenceStage.ENCODE: deque(maxlen=200),
            InferenceStage.PREFILL: deque(maxlen=200),
            InferenceStage.DECODE: deque(maxlen=200),
        }
    
    async def dispatch_requests(self, pending_requests: List[Request], 
                              stage: InferenceStage,
                              memory_limit: float = None) -> List[Request]:
        """
        请求分发
        根据内存和计算约束选择请求批次
        
        Args:
            pending_requests: 待处理的请求列表
            stage: 推理阶段
            memory_limit: 内存限制（GB）
            
        Returns:
            List[Request]: 选中的请求批次
        """
        if memory_limit is None:
            memory_limit = self.memory_limit_gb
        
        async with self._lock:
            selected_requests = []
            estimated_memory = 0.0
            
            # 按优先级排序请求
            sorted_requests = sorted(pending_requests, key=lambda r: (-r.priority, r.timestamp))
            
            for request in sorted_requests:
                request_memory = request.estimate_memory_usage()
                
                # 检查内存约束
                if estimated_memory + request_memory <= memory_limit:
                    # 检查计算约束（批次大小限制）
                    if len(selected_requests) < self.max_batch_size:
                        selected_requests.append(request)
                        estimated_memory += request_memory
                    else:
                        break  # 达到批次大小限制
                else:
                    break  # 内存不足，停止添加请求
            
            pass
            return selected_requests
    
    def calculate_gain_cost_model(self, stage: InferenceStage, 
                                current_instances: List[str],
                                candidate_instance: str,
                                requests: List[Request],
                                enable_future_benefit: bool = True) -> GainCostResult:
        """
        收益-成本模型计算（改进版：考虑未来多轮累积收益）
        对应论文公式(2)和(3)，并扩展到未来N轮
        
        Args:
            stage: 推理阶段
            current_instances: 当前实例列表
            candidate_instance: 候选实例
            requests: 请求列表
            enable_future_benefit: 是否启用未来收益计算（默认True）
            
        Returns:
            GainCostResult: 收益-成本计算结果
        """
        if not requests:
            return GainCostResult(gain=0.0, cost=0.0, net_benefit=0.0, is_beneficial=False)
        
        # 计算当前处理时间
        current_time = self._estimate_processing_time(
            stage, len(current_instances), len(requests)
        )
        
        # 计算添加实例后的处理时间
        new_time = self._estimate_processing_time(
            stage, len(current_instances) + 1, len(requests)
        )
        
        # 计算单轮收益和成本
        if stage == InferenceStage.PREFILL:
            # 预填充阶段抢占模型（论文公式2）
            gain_per_round = self._calculate_prefill_gain(requests, current_time, new_time)
            cost = self._calculate_prefill_cost(candidate_instance, requests)
        else:
            # 解码阶段扩容模型（论文公式3）
            gain_per_round = self._calculate_decode_gain(requests, current_time, new_time)
            cost = self._calculate_decode_cost(candidate_instance, requests)
        
        # ✅ 改进：考虑未来N轮的累积收益
        if enable_future_benefit and gain_per_round > 0:
            # 方案1: 简单累积（假设未来workload稳定）
            # cumulative_gain = gain_per_round * self.future_rounds_horizon
            
            # 方案2: 带折扣的累积（远期收益权重降低）
            cumulative_gain = 0.0
            for round_num in range(self.future_rounds_horizon):
                discount = self.discount_factor ** round_num
                cumulative_gain += gain_per_round * discount
            
            gain = cumulative_gain
        else:
            # 原始方法：只考虑单轮
            gain = gain_per_round
        
        net_benefit = gain - cost
        is_beneficial = net_benefit > 0
        
        return GainCostResult(
            gain=gain,
            cost=cost,
            net_benefit=net_benefit,
            is_beneficial=is_beneficial
        )
    
    def _estimate_processing_time(self, stage: InferenceStage, 
                                instances: int, requests: int) -> float:
        """估计处理时间（使用校准的scalability参数）"""
        if instances == 0:
            return float('inf')
        
        # ✅ 使用校准的scalability系数
        if stage == InferenceStage.DECODE:
            scalability = self.gain_cost_config.get('scalability_decoding', 0.75)
        elif stage == InferenceStage.ENCODE:
            scalability = self.gain_cost_config.get('scalability_encode', 0.80)
        else:  # PREFILL
            scalability = self.gain_cost_config.get('scalability_prefill', 0.90)
        
        # 基础处理时间（秒）
        # 使用Amdahl's Law: T(n) = T(1) / (s + (1-s)/n)
        # 简化版本：T(n) ≈ T(1) / (scalability * n)
        base_time = requests / (instances * scalability)
        
        # 添加一些非线性因素
        if instances > 4:
            # 实例过多时效率下降（通信开销增加）
            efficiency_factor = 0.9 ** (instances - 4)
            base_time *= (1 / efficiency_factor)
        
        return base_time
    
    def record_step_stats(self, stage: InferenceStage, num_requests: int, 
                         latency: float = 0.0):
        """
        记录每步的统计数据
        
        Args:
            stage: 推理阶段
            num_requests: 当前请求数（waiting + running）
            latency: 平均延迟（可选）
        """
        self.workload_history[stage].append(num_requests)
        if latency > 0:
            self.latency_history[stage].append(latency)
    
    def _estimate_future_workload(self, stage: InferenceStage, 
                                  current_queue_size: int) -> int:
        """
        估计未来workload（基于历史数据）
        
        Args:
            stage: 推理阶段
            current_queue_size: 当前队列大小
            
        Returns:
            估计的未来平均队列大小
        """
        history = self.workload_history[stage]
        
        # 如果有足够的历史数据，使用滑动窗口平均
        if len(history) >= 20:
            # 使用最近50步的数据
            recent_workload = list(history)[-50:]
            avg_workload = sum(recent_workload) / len(recent_workload)
            
            # 加权组合：70%历史平均 + 30%当前值
            # 这样可以平滑突发波动，同时快速响应趋势变化
            estimated = int(0.7 * avg_workload + 0.3 * current_queue_size)
            return estimated
        else:
            # 历史数据不足，使用当前值
            return current_queue_size
    
    def get_workload_stats(self, stage: InferenceStage) -> Dict[str, float]:
        """
        获取workload统计信息
        
        Returns:
            Dict包含: mean, std, min, max, recent_trend
        """
        history = self.workload_history[stage]
        
        if len(history) < 2:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'trend': 0}
        
        data = list(history)
        mean = sum(data) / len(data)
        
        # 标准差
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        std = variance ** 0.5
        
        # 趋势（最近10步 vs 之前10步）
        if len(data) >= 20:
            recent = sum(data[-10:]) / 10
            earlier = sum(data[-20:-10]) / 10
            trend = (recent - earlier) / max(earlier, 1)  # 百分比变化
        else:
            trend = 0
        
        return {
            'mean': mean,
            'std': std,
            'min': min(data),
            'max': max(data),
            'trend': trend
        }
    
    def calculate_incremental_gain_cost(self, 
                                       stage: InferenceStage,
                                       current_instances: List[str],
                                       candidate_instance: str,
                                       requests: List[Request],
                                       proposed_allocation: Dict[int, str]) -> GainCostResult:
        """
        增量式Gain-Cost计算（只计算变化部分的成本）
        
        Args:
            stage: 目标阶段
            current_instances: 当前该阶段的实例列表
            candidate_instance: 候选实例ID
            requests: 当前请求列表
            proposed_allocation: 提议的新分配方案 {worker_id: stage_name}
            
        Returns:
            GainCostResult: 考虑增量成本的收益-成本结果
        """
        # 计算标准的gain（基于未来收益）
        gain_cost_result = self.calculate_gain_cost_model(
            stage, current_instances, candidate_instance, requests,
            enable_future_benefit=True
        )
        
        # ✅ 增量成本计算：只计算需要迁移的worker
        incremental_cost = 0.0
        migration_cost_per_worker = self.gain_cost_config.get('migration_cost', 0.01)
        preemption_penalty = self.gain_cost_config.get('preemption_penalty', 0.2)
        
        # 找出需要改变角色的workers
        workers_to_migrate = []
        for worker_id, new_stage in proposed_allocation.items():
            old_stage = self.previous_allocation.get(worker_id)
            if old_stage and old_stage != new_stage:
                workers_to_migrate.append((worker_id, old_stage, new_stage))
        
        if workers_to_migrate:
            # 计算迁移成本
            num_migrations = len(workers_to_migrate)
            
            # 迁移成本 = 基础迁移开销 + 抢占惩罚（如果从正在工作的阶段抢占）
            for worker_id, old_stage, new_stage in workers_to_migrate:
                # 基础迁移成本（KV传输等）
                incremental_cost += migration_cost_per_worker
                
                # 如果从Prefill或Decode抢占，需要考虑中断正在运行请求的惩罚
                if old_stage in ['prefill', 'decoding']:
                    # 估计被中断的请求数（简化：假设每个worker平均处理batch_size个请求）
                    avg_requests_per_worker = len(requests) / max(len(current_instances), 1)
                    incremental_cost += preemption_penalty * avg_requests_per_worker
        
        # ✅ 更新cost为增量成本
        adjusted_gain_cost = GainCostResult(
            gain=gain_cost_result.gain,
            cost=incremental_cost if workers_to_migrate else gain_cost_result.cost,
            net_benefit=gain_cost_result.gain - incremental_cost if workers_to_migrate 
                       else gain_cost_result.net_benefit,
            is_beneficial=(gain_cost_result.gain - incremental_cost) > 0 if workers_to_migrate
                         else gain_cost_result.is_beneficial
        )
        
        return adjusted_gain_cost
    
    def update_allocation_state(self, new_allocation: Dict[int, str]):
        """
        更新worker分配状态（用于下一轮的diff计算）
        
        Args:
            new_allocation: 新的worker分配方案 {worker_id: stage_name}
        """
        self.previous_allocation = self.current_allocation.copy()
        self.current_allocation = new_allocation.copy()
    
    def should_trigger_reallocation(self) -> bool:
        """
        判断是否应该触发资源重新分配
        
        Returns:
            bool: 是否应该重新分配
        """
        # ✅ 对齐调度间隔：只在N轮后才重新评估
        steps_since_last = self.step_counter - self.last_step_adjustment
        
        if steps_since_last >= self.step_adjustment_interval:
            self.last_step_adjustment = self.step_counter
            return True
        
        return False
    
    def _calculate_prefill_gain(self, requests: List[Request], 
                              current_time: float, new_time: float) -> float:
        """计算预填充阶段的收益（论文公式2）"""
        if not requests or current_time <= new_time:
            return 0.0
        
        # 收益 = Σ(r∈Rp) [T(Rp, Ep) - T(Rp, Ep ∪ emax)] / r.input_len
        total_gain = 0.0
        for request in requests:
            time_saved = current_time - new_time
            normalized_gain = time_saved / max(request.input_length, 1)
            total_gain += normalized_gain
        
        return total_gain
    
    def _calculate_prefill_cost(self, candidate_instance: str, 
                              requests: List[Request]) -> float:
        #Σ(r∈Bd) [M(emax) + w · L(Bd, Ed - emax)] / r.output_len
        
        # 使用校准参数
        migration_cost = self.gain_cost_config['migration_cost']  # M(emax)
        preemption_penalty = self.gain_cost_config['preemption_penalty']  # w
        
        affected_requests = len(requests)
        preemption_cost = preemption_penalty * affected_requests
        
        total_output_length = sum(r.estimated_output_length for r in requests)
        if total_output_length == 0:
            return migration_cost + preemption_cost
        
        normalized_cost = (migration_cost + preemption_cost) / total_output_length
        
        return normalized_cost
    
    def _calculate_decode_gain(self, requests: List[Request], 
                             current_time: float, new_time: float) -> float:
        if not requests or current_time <= new_time:
            return 0.0
        
        # 收益 = Σ(r∈Bd) [AvgLatd - T(Bd, Ed ∪ emax)] / r.output_len
        total_gain = 0.0
        for request in requests:
            time_saved = current_time - new_time
            normalized_gain = time_saved / max(request.estimated_output_length, 1)
            total_gain += normalized_gain
        
        return total_gain
    
    def _calculate_decode_cost(self, candidate_instance: str, 
                             requests: List[Request]) -> float:
        # cost = Σ(r∈R'p) [M(emax) + w · L(R'p, E'p - emax)] / r.input_len
        
        migration_cost = 0.1
        
        preemption_penalty = 0.05
        
        affected_requests = len(requests)
        preemption_cost = preemption_penalty * affected_requests

        total_input_length = sum(r.input_length for r in requests)
        if total_input_length == 0:
            return migration_cost + preemption_cost
        
        normalized_cost = (migration_cost + preemption_cost) / total_input_length
        
        return normalized_cost
    
    async def elastic_instance_allocation(self, stage: InferenceStage, 
                                        requests: List[Request]) -> AllocationDecision:
        """
        
        Args:
            stage: Inference stage
            requests: Request batch
            
        Returns:
            AllocationDecision: Allocation decision
        """
        async with self._lock:
            current_instances = self.stage_instances[stage]
            
            if len(requests) <= len(current_instances):

                allocated_instances = current_instances[:len(requests)]
                estimated_time = self._estimate_processing_time(stage, len(allocated_instances), len(requests))
                
                return AllocationDecision(
                    stage=stage,
                    instances=allocated_instances,
                    requests=requests,
                    estimated_time=estimated_time
                )
            
            if stage != InferenceStage.DECODE:  # decode stage does not preempt other instances
                # try to preempt from decode stage
                decode_instances = self.stage_instances[InferenceStage.DECODE]
                if len(decode_instances) > self.min_instances_per_stage:
                    candidate_instance = decode_instances[0]  # 选择第一个解码实例
                    
                    gain_cost = self.calculate_gain_cost_model(
                        stage, current_instances, candidate_instance, requests
                    )
                    
                    if gain_cost.is_beneficial:
                        additional_instances = [candidate_instance]
                        total_instances = current_instances + additional_instances

                        needed_instances = min(len(requests), len(total_instances))
                        allocated_instances = total_instances[:needed_instances]
                        
                        estimated_time = self._estimate_processing_time(
                            stage, len(allocated_instances), len(requests)
                        )
                        
                        return AllocationDecision(
                            stage=stage,
                            instances=allocated_instances,
                            requests=requests,
                            estimated_time=estimated_time,
                            gain_cost=gain_cost
                        )
            
            estimated_time = self._estimate_processing_time(stage, len(current_instances), len(requests))
            
            return AllocationDecision(
                stage=stage,
                instances=current_instances,
                requests=requests,
                estimated_time=estimated_time
            )
    
    async def elastic_auto_scaling(self, stage: InferenceStage, 
                                 queue_length: int) -> str:

        current_instances = len(self.stage_instances[stage])
        threshold = self.scaling_thresholds.get(stage, 15)
        

        if self._should_scale_up(stage, queue_length, current_instances, threshold):
            return "scale_up"

        elif self._should_scale_down(stage, queue_length, current_instances, threshold):
            return "scale_down"
        
        else:
            return "no_action"
    
    def _should_scale_up(self, stage: InferenceStage, queue_len: int, 
                        instances: int, threshold: int) -> bool:
        """判断是否应该扩容"""
        if stage == InferenceStage.DECODE:
            # 解码阶段保守扩容
            return queue_len > threshold and instances < 3
        else:
            # 编码和预填充阶段积极扩容
            return queue_len > threshold and instances < 6
    
    def _should_scale_down(self, stage: InferenceStage, queue_len: int, 
                          instances: int, threshold: int) -> bool:
        """判断是否应该缩容"""
        return queue_len < threshold // 3 and instances > self.min_instances_per_stage
    
    def add_instance(self, instance_id: str, stage: InferenceStage):
        """添加实例到指定阶段"""
        if instance_id not in self.stage_instances[stage]:
            self.stage_instances[stage].append(instance_id)
            pass
    
    def remove_instance(self, instance_id: str):
        """从所有阶段中移除实例"""
        for stage in InferenceStage:
            if instance_id in self.stage_instances[stage]:
                self.stage_instances[stage].remove(instance_id)
                pass
                break
    
    def get_stage_status(self) -> Dict[InferenceStage, Dict]:
        """获取各阶段状态"""
        status = {}
        for stage in InferenceStage:
            instances = self.stage_instances[stage]
            queue_len = len(self.request_queues[stage])
            
            status[stage] = {
                "instance_count": len(instances),
                "queue_length": queue_len,
                "instances": instances
            }
        
        return status
    
    def increment_step_counter(self):
        """增加decode迭代次数计数器"""
        self.step_counter += 1
    
    def should_perform_step_adjustment(self) -> bool:
        """判断是否应该进行基于step的调整"""
        return (self.step_counter - self.last_step_adjustment) >= self.step_adjustment_interval
    
    async def step_based_adjustment(self) -> Dict[InferenceStage, str]:
        """
        基于step的模态组内调整
        """
        if not self.should_perform_step_adjustment():
            return {}
        
        pass
        
        adjustments = {}
        
        # 检查各阶段是否需要调整
        for stage in InferenceStage:
            queue_length = len(self.request_queues[stage])
            scaling_action = await self.elastic_auto_scaling(stage, queue_length)
            
            if scaling_action != "no_action":
                adjustments[stage] = scaling_action
        
        # 更新最后调整时间
        self.last_step_adjustment = self.step_counter
        
        pass
        
        return adjustments
    
    def get_step_status(self) -> Dict:
        """获取step相关状态"""
        return {
            "total_steps": self.step_counter,
            "steps_since_last_adjustment": self.step_counter - self.last_step_adjustment,
            "steps_until_next_adjustment": self.step_adjustment_interval - (self.step_counter - self.last_step_adjustment),
            "should_adjust": self.should_perform_step_adjustment(),
            "adjustment_interval": self.step_adjustment_interval
        }
