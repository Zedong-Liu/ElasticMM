import asyncio
import time
import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class ModalityType(Enum):
    """模态类型枚举"""
    TEXT_ONLY = "text_only"
    MULTIMODAL = "multimodal"


@dataclass
class WorkloadStats:
    """工作负载统计数据"""
    text_qps: float = 0.0
    multimodal_qps: float = 0.0
    text_peak_qps: float = 0.0
    multimodal_peak_qps: float = 0.0
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class InstanceInfo:
    """实例信息"""
    instance_id: str
    modality_group: ModalityType
    current_load: float = 0.0
    peak_capacity: float = 10.0  # 每个实例的峰值QPS容量
    avg_capacity: float = 5.0    # 每个实例的平均QPS容量
    last_heartbeat: float = 0.0
    is_active: bool = True
    
    def __post_init__(self):
        if self.last_heartbeat == 0.0:
            self.last_heartbeat = time.time()


@dataclass
class BurstTolerance:
    """突发容忍度信息"""
    modality_group: ModalityType
    burst_tolerance: float
    instance_count: int
    avg_qps: float
    peak_qps: float
    timestamp: float
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class ModalityAwareLoadBalancer:
    """
    模态感知负载均衡器
    """
    
    def __init__(self, history_window_size: int = 100):
        """
        初始化模态感知负载均衡器
        
        Args:
            history_window_size: 历史数据窗口大小
        """
        self.instance_groups = {
            ModalityType.TEXT_ONLY: [],
            ModalityType.MULTIMODAL: []
        }
        self.workload_history: List[WorkloadStats] = []
        self.burst_tolerance_history: List[BurstTolerance] = []
        self.history_window_size = history_window_size
        self._lock = asyncio.Lock()
        
        # 配置参数
        self.migration_threshold = 20  
        self.min_instances_per_group = 1  
        self.burst_detection_window = 60  
        
        # 周期性调整参数
        self.rebalance_interval = 600  
        self.prediction_window_size = 20  
        self.last_rebalance_time = 0.0
        
    def calculate_burst_tolerance(self, modality_group: ModalityType, 
                                workload: WorkloadStats) -> BurstTolerance:
        """
        计算突发容忍度
        bt(i) = N_peak_i / N_avg_i
        
        Args:
            modality_group: 模态组类型
            workload: 工作负载统计
            
        Returns:
            BurstTolerance: 突发容忍度信息
        """
        instances = self.instance_groups[modality_group]
        instance_count = len(instances)
        
        if instance_count == 0:
            return BurstTolerance(
                modality_group=modality_group,
                burst_tolerance=0.0,
                instance_count=0,
                avg_qps=0.0,
                peak_qps=0.0,
                timestamp=time.time()
            )
        
        # 获取对应模态的QPS数据
        if modality_group == ModalityType.TEXT_ONLY:
            avg_qps = workload.text_qps
            peak_qps = workload.text_peak_qps
        else:
            avg_qps = workload.multimodal_qps
            peak_qps = workload.multimodal_peak_qps
        
        avg_instance_capacity = 5.0
        peak_instance_capacity = 10.0
        
        if avg_qps == 0:
            burst_tolerance = float('inf')
        else:
            # 计算需要的实例数
            required_instances_avg = max(1, avg_qps / avg_instance_capacity)
            required_instances_peak = max(1, peak_qps / peak_instance_capacity)
            
            #bt(i) = N_peak_i / N_avg_i
            burst_tolerance = required_instances_peak / required_instances_avg
        
        return BurstTolerance(
            modality_group=modality_group,
            burst_tolerance=burst_tolerance,
            instance_count=instance_count,
            avg_qps=avg_qps,
            peak_qps=peak_qps,
            timestamp=time.time()
        )
    
    async def proactive_allocation(self, total_instances: int, workload: WorkloadStats) -> Dict[ModalityType, int]:
        """
        主动资源分配
        基于贪心算法最大化最小突发容忍度
        
        Args:
            total_instances: 总实例数
            workload: 当前工作负载统计
            
        Returns:
            Dict[ModalityType, int]: 每个模态组的实例分配
        """
        async with self._lock:
            # 计算当前各组的突发容忍度
            text_tolerance = self.calculate_burst_tolerance(ModalityType.TEXT_ONLY, workload)
            multimodal_tolerance = self.calculate_burst_tolerance(ModalityType.MULTIMODAL, workload)
            
            # 初始化分配结果
            allocation = {
                ModalityType.TEXT_ONLY: 0,
                ModalityType.MULTIMODAL: 0
            }
            
            # 确保每组至少有最小实例数
            remaining_instances = total_instances - 2 * self.min_instances_per_group
            if remaining_instances < 0:
                # 如果实例数不足，均匀分配
                allocation[ModalityType.TEXT_ONLY] = total_instances // 2
                allocation[ModalityType.MULTIMODAL] = total_instances - allocation[ModalityType.TEXT_ONLY]
                return allocation
            
            allocation[ModalityType.TEXT_ONLY] = self.min_instances_per_group
            allocation[ModalityType.MULTIMODAL] = self.min_instances_per_group
            
            # 贪心算法：迭代分配剩余实例给突发容忍度最低的组
            for _ in range(remaining_instances):
                # 重新计算突发容忍度（考虑当前分配）
                current_text_tolerance = self._calculate_tolerance_with_allocation(
                    ModalityType.TEXT_ONLY, workload, allocation[ModalityType.TEXT_ONLY]
                )
                current_multimodal_tolerance = self._calculate_tolerance_with_allocation(
                    ModalityType.MULTIMODAL, workload, allocation[ModalityType.MULTIMODAL]
                )
                
                # 选择突发容忍度最低的组
                if current_text_tolerance <= current_multimodal_tolerance:
                    allocation[ModalityType.TEXT_ONLY] += 1
                else:
                    allocation[ModalityType.MULTIMODAL] += 1
            
            return allocation
    
    def _calculate_tolerance_with_allocation(self, modality_group: ModalityType, 
                                           workload: WorkloadStats, 
                                           allocated_instances: int) -> float:
        """计算指定分配下的突发容忍度"""
        if allocated_instances == 0:
            return float('inf')
        
        # 获取对应模态的QPS数据
        if modality_group == ModalityType.TEXT_ONLY:
            avg_qps = workload.text_qps
            peak_qps = workload.text_peak_qps
        else:
            avg_qps = workload.multimodal_qps
            peak_qps = workload.multimodal_peak_qps
        
        if avg_qps == 0:
            return float('inf')
        
        # 计算需要的实例数
        avg_instance_capacity = 5.0
        peak_instance_capacity = 10.0
        
        required_instances_avg = max(1, avg_qps / avg_instance_capacity)
        required_instances_peak = max(1, peak_qps / peak_instance_capacity)
        
        # 考虑当前分配，计算突发容忍度
        return required_instances_peak / required_instances_avg
    
    async def reactive_scaling(self, queue_lengths: Dict[ModalityType, int], 
                             current_workload: WorkloadStats) -> Optional[Tuple[ModalityType, ModalityType, str]]:
        """
        响应式扩缩容决策引擎
        当检测到突发负载时，在模态组间迁移实例
        
        Args:
            queue_lengths: 各模态组的队列长度
            current_workload: 当前工作负载
            
        Returns:
            Optional[Tuple[ModalityType, ModalityType, str]]: (源组, 目标组, 实例ID) 或 None
        """
        async with self._lock:
            # 检测需要扩容的组
            overloaded_groups = []
            for modality_group, queue_len in queue_lengths.items():
                if queue_len > self.migration_threshold:
                    overloaded_groups.append((modality_group, queue_len))
            
            if not overloaded_groups:
                return None
            
            # 选择最需要扩容的组
            target_group, max_queue_len = max(overloaded_groups, key=lambda x: x[1])
            
            # 寻找可以迁移实例的源组
            source_group = self._find_best_source_group(target_group, current_workload)
            if source_group is None:
                return None
            
            # 选择最优的迁移候选实例
            instance_to_migrate = await self._select_migration_candidate(
                source_group, target_group, current_workload
            )
            
            if instance_to_migrate:
                return (source_group, target_group, instance_to_migrate)
            
            return None
    
    def _find_best_source_group(self, target_group: ModalityType, 
                              workload: WorkloadStats) -> Optional[ModalityType]:
        """寻找最佳的源组进行实例迁移"""
        source_candidates = []
        
        for modality_group in ModalityType:
            if modality_group == target_group:
                continue
            
            instances = self.instance_groups[modality_group]
            if len(instances) <= self.min_instances_per_group:
                continue  # 不能迁移，会低于最小实例数
            
            # 计算该组的负载压力
            if modality_group == ModalityType.TEXT_ONLY:
                current_qps = workload.text_qps
            else:
                current_qps = workload.multimodal_qps
            
            # 计算该组的资源利用率
            capacity = len(instances) * 5.0  # 每个实例5 QPS
            utilization = current_qps / capacity if capacity > 0 else 0
            
            source_candidates.append((modality_group, utilization))
        
        if not source_candidates:
            return None
        
        # 选择利用率最低的组作为源组
        return min(source_candidates, key=lambda x: x[1])[0]
    
    async def _select_migration_candidate(self, source_group: ModalityType, 
                                        target_group: ModalityType,
                                        workload: WorkloadStats) -> Optional[str]:
        """选择最优的迁移候选实例"""
        instances = self.instance_groups[source_group]
        if not instances:
            return None
        
        # 计算迁移成本，选择成本最低的实例
        best_instance = None
        min_cost = float('inf')
        
        for instance in instances:
            # 计算迁移该实例的成本
            cost = self._calculate_migration_cost(instance, source_group, target_group, workload)
            if cost < min_cost:
                min_cost = cost
                best_instance = instance
        
        return best_instance
    
    def _calculate_migration_cost(self, instance_id: str, source_group: ModalityType, 
                                target_group: ModalityType, workload: WorkloadStats) -> float:
        """计算实例迁移的成本"""
        # 基础迁移成本
        base_cost = 1.0
        
        # 对源组性能影响的成本
        source_instances = len(self.instance_groups[source_group])
        if source_instances <= 1:
            return float('inf')  # 不能迁移最后一个实例
        
        # 计算迁移后源组的负载压力
        if source_group == ModalityType.TEXT_ONLY:
            source_qps = workload.text_qps
        else:
            source_qps = workload.multimodal_qps
        
        remaining_capacity = (source_instances - 1) * 5.0
        source_utilization = source_qps / remaining_capacity if remaining_capacity > 0 else 0
        
        # 利用率越高，迁移成本越高
        utilization_cost = source_utilization * 2.0
        
        return base_cost + utilization_cost
    
    async def _migrate_instance(self, instance_id: str, source_group: ModalityType, 
                              target_group: ModalityType) -> bool:
        """执行实例迁移"""
        try:
            # 从源组移除实例
            if instance_id in self.instance_groups[source_group]:
                self.instance_groups[source_group].remove(instance_id)
            else:
                return False
            
            # 添加到目标组
            self.instance_groups[target_group].append(instance_id)
            
            pass
            return True
            
        except Exception as e:
            pass
            return False
    
    def add_workload_sample(self, workload: WorkloadStats):
        """添加工作负载样本到历史记录"""
        self.workload_history.append(workload)
        
        # 保持历史记录窗口大小
        if len(self.workload_history) > self.history_window_size:
            self.workload_history.pop(0)
    
    def get_average_workload(self) -> WorkloadStats:
        """获取平均工作负载"""
        if not self.workload_history:
            return WorkloadStats()
        
        total_text_qps = sum(w.text_qps for w in self.workload_history)
        total_multimodal_qps = sum(w.multimodal_qps for w in self.workload_history)
        total_text_peak = sum(w.text_peak_qps for w in self.workload_history)
        total_multimodal_peak = sum(w.multimodal_peak_qps for w in self.workload_history)
        
        count = len(self.workload_history)
        
        return WorkloadStats(
            text_qps=total_text_qps / count,
            multimodal_qps=total_multimodal_qps / count,
            text_peak_qps=total_text_peak / count,
            multimodal_peak_qps=total_multimodal_peak / count
        )
    
    def get_instance_distribution(self) -> Dict[ModalityType, int]:
        """获取当前实例分布"""
        return {
            ModalityType.TEXT_ONLY: len(self.instance_groups[ModalityType.TEXT_ONLY]),
            ModalityType.MULTIMODAL: len(self.instance_groups[ModalityType.MULTIMODAL])
        }
    
    def add_instance(self, instance_id: str, modality_group: ModalityType):
        """添加实例到指定模态组"""
        if instance_id not in self.instance_groups[modality_group]:
            self.instance_groups[modality_group].append(instance_id)
    
    def remove_instance(self, instance_id: str):
        """从所有组中移除实例"""
        for modality_group in ModalityType:
            if instance_id in self.instance_groups[modality_group]:
                self.instance_groups[modality_group].remove(instance_id)
                break
    
    def predict_workload(self) -> WorkloadStats:
        """
        基于窗口的简单工作负载预测
        使用移动平均和趋势分析
        """
        if len(self.workload_history) < 3:
            # 历史数据不足，返回最近的数据
            return self.workload_history[-1] if self.workload_history else WorkloadStats()
        
        # 获取最近的窗口数据
        recent_data = self.workload_history[-self.prediction_window_size:]
        
        # 计算移动平均
        avg_text_qps = sum(w.text_qps for w in recent_data) / len(recent_data)
        avg_multimodal_qps = sum(w.multimodal_qps for w in recent_data) / len(recent_data)
        avg_text_peak = sum(w.text_peak_qps for w in recent_data) / len(recent_data)
        avg_multimodal_peak = sum(w.multimodal_peak_qps for w in recent_data) / len(recent_data)
        
        # 简单的趋势分析（线性回归）
        if len(recent_data) >= 3:
            # 计算趋势
            text_trend = self._calculate_trend([w.text_qps for w in recent_data])
            multimodal_trend = self._calculate_trend([w.multimodal_qps for w in recent_data])
            
            # 应用趋势预测（预测下一个时间点的值）
            predicted_text_qps = avg_text_qps + text_trend
            predicted_multimodal_qps = avg_multimodal_qps + multimodal_trend
            predicted_text_peak = avg_text_peak + text_trend * 1.5  # 峰值趋势更明显
            predicted_multimodal_peak = avg_multimodal_peak + multimodal_trend * 1.5
        else:
            # 趋势数据不足，使用平均值
            predicted_text_qps = avg_text_qps
            predicted_multimodal_qps = avg_multimodal_qps
            predicted_text_peak = avg_text_peak
            predicted_multimodal_peak = avg_multimodal_peak
        
        # 确保预测值不为负数
        predicted_text_qps = max(0, predicted_text_qps)
        predicted_multimodal_qps = max(0, predicted_multimodal_qps)
        predicted_text_peak = max(predicted_text_qps, predicted_text_peak)
        predicted_multimodal_peak = max(predicted_multimodal_qps, predicted_multimodal_peak)
        
        return WorkloadStats(
            text_qps=predicted_text_qps,
            multimodal_qps=predicted_multimodal_qps,
            text_peak_qps=predicted_text_peak,
            multimodal_peak_qps=predicted_multimodal_peak
        )
    
    def _calculate_trend(self, values: List[float]) -> float:
        """计算线性趋势"""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))
        
        # 线性回归斜率
        if n * x2_sum - x_sum * x_sum == 0:
            return 0.0
        
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        return slope
    
    def should_perform_rebalance(self) -> bool:
        """判断是否应该进行重新平衡"""
        current_time = time.time()
        time_since_last = current_time - self.last_rebalance_time
        
        return time_since_last >= self.rebalance_interval
    
    async def periodic_rebalance(self) -> Optional[Dict[ModalityType, int]]:
        """
        周期性重新平衡
        每5分钟根据历史信息进行一次模态组调整
        """
        if not self.should_perform_rebalance():
            return None
        
        pass
        predicted_workload = self.predict_workload()
        
        # 获取当前实例分布
        current_distribution = self.get_instance_distribution()
        total_instances = sum(current_distribution.values())
        
        if total_instances == 0:
            return None
        
        # 基于预测工作负载计算新的分配
        new_allocation = await self.proactive_allocation(total_instances, predicted_workload)
        
        # 比较新旧分配
        if new_allocation != current_distribution:
            # 更新重新平衡时间
            self.last_rebalance_time = time.time()
            
            return new_allocation
        
        self.last_rebalance_time = time.time()
        return None
    
    def get_rebalance_status(self) -> Dict:
        """获取重新平衡状态"""
        current_time = time.time()
        time_since_last = current_time - self.last_rebalance_time
        time_until_next = max(0, self.rebalance_interval - time_since_last)
        
        return {
            "last_rebalance_time": self.last_rebalance_time,
            "time_since_last": time_since_last,
            "time_until_next": time_until_next,
            "should_rebalance": self.should_perform_rebalance(),
            "rebalance_interval": self.rebalance_interval
        }

    async def set_initial_allocation(self, allocation: Dict[ModalityType, List[int]]):
        """设置初始GPU分配"""
        async with self._lock:
            self.instance_groups = {
                ModalityType.TEXT_ONLY: [],
                ModalityType.MULTIMODAL: []
            }
            for modality, gpu_list in allocation.items():
                self.instance_groups[modality] = [f"gpu_{gpu_id}" for gpu_id in gpu_list]