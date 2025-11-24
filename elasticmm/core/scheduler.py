import threading
import time
import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .balancer import ModalityAwareLoadBalancer, ModalityType, WorkloadStats
from .allocator import StageLevelResourceAllocator, InferenceStage, Request
from ..engine.backend_interface import EngineBackend


@dataclass
class InstanceInfo:
    http_address: str
    zmq_address: str
    role: str  # "P" or "D"
    last_heartbeat_deadline: float
    load: float = 0.0


class Scheduler:
    def __init__(self, ping_seconds: int = 60) -> None:  
        self._ping_seconds = ping_seconds
        self._prefill: Dict[str, InstanceInfo] = {}
        self._decode: Dict[str, InstanceInfo] = {}
        self._lock = threading.Lock()
        self._rr_p = 0
        self._rr_d = 0

    def _gc(self) -> None:
        now = time.time()
        for pool in (self._prefill, self._decode):
            expired = [k for k, v in pool.items() if v.last_heartbeat_deadline < now]
            for k in expired:
                pool.pop(k, None)

    def heartbeat(self, http_address: str, zmq_address: str, role: str, instance_id: Optional[str] = None) -> None:
        """
        Register or update instance heartbeat
        
        Args:
            http_address: HTTP address of the instance
            zmq_address: ZMQ address of the instance
            role: Instance role ("P" for prefill, "D" for decode)
            instance_id: Optional instance ID. If provided (V0 backend), use it as key.
                        Otherwise (V1 backend), use http_address as key.
        """
        with self._lock:
            self._gc()
            info = InstanceInfo(
                http_address=http_address,
                zmq_address=zmq_address,
                role=role,
                last_heartbeat_deadline=time.time() + self._ping_seconds,
            )
            # Use instance_id as key if provided (V0 backend), otherwise use http_address (V1 backend)
            key = instance_id if instance_id is not None else http_address
            if role == "P":
                self._prefill[key] = info
            elif role == "D":
                self._decode[key] = info

    def remove(self, http_address: str) -> None:
        with self._lock:
            self._prefill.pop(http_address, None)
            self._decode.pop(http_address, None)

    def list_instances(self) -> Tuple[List[InstanceInfo], List[InstanceInfo]]:
        with self._lock:
            self._gc()
            return list(self._prefill.values()), list(self._decode.values())

    def select_prefill(self) -> Optional[InstanceInfo]:
        with self._lock:
            self._gc()
            items = list(self._prefill.values())
            if not items:
                return None
            self._rr_p = (self._rr_p + 1) % len(items)
            return items[self._rr_p]

    def select_decode(self) -> Optional[InstanceInfo]:
        with self._lock:
            self._gc()
            items = list(self._decode.values())
            if not items:
                return None
            self._rr_d = (self._rr_d + 1) % len(items)
            return items[self._rr_d]

    def select_prefills(self, k: int) -> List[InstanceInfo]:
        with self._lock:
            self._gc()
            items = list(self._prefill.values())
            if not items:
                return []
            # simple round-robin window
            start = self._rr_p
            out = []
            for i in range(min(k, len(items))):
                out.append(items[(start + i) % len(items)])
            self._rr_p = (start + 1) % len(items)
            return out

    def select_decodes(self, k: int) -> List[InstanceInfo]:
        with self._lock:
            self._gc()
            items = list(self._decode.values())
            if not items:
                return []
            start = self._rr_d
            out = []
            for i in range(min(k, len(items))):
                out.append(items[(start + i) % len(items)])
            self._rr_d = (start + 1) % len(items)
            return out

    def get_decode_with_lowest_utilization(self) -> Optional[InstanceInfo]:
        with self._lock:
            self._gc()
            items = list(self._decode.values())
            if not items:
                return None

            self._rr_d = (self._rr_d + 1) % len(items)
            return items[self._rr_d]
    
    def get_all_decode_nodes(self) -> List[InstanceInfo]:
        with self._lock:
            self._gc()
            return list(self._decode.values())


class EMPScheduler(Scheduler):
    """
    Elastic Multimodal Parallelism Scheduler
    Integrates heartbeat management with hierarchical EMP scheduling
    继承Scheduler以复用heartbeat管理
    """
    
    def __init__(self, backend: Optional[EngineBackend] = None, ping_seconds: int = 60):
        # ✅ 继承父类的heartbeat管理
        super().__init__(ping_seconds)
        
        # EMP调度组件
        self.modality_balancer = ModalityAwareLoadBalancer()
        
        # 阶段层级管理器
        self.stage_allocators = {
            ModalityType.TEXT_ONLY: StageLevelResourceAllocator(),
            ModalityType.MULTIMODAL: StageLevelResourceAllocator()
        }
        
        # 引擎后端，用于节点级KV迁移
        self.backend = backend
        
        # 全局系统状态
        self.system_status = {
            "total_instances": 0,
            "active_requests": 0,
            "last_rebalance_time": 0.0
        }
        
        # 配置参数
        self.rebalance_interval = 60 
        self.auto_scaling_enabled = True
        
        # ✅ 弹性调度循环控制
        self.elastic_scheduling_enabled = True
        self.running = False
        self._elastic_task = None
    
    # ✅ 所有heartbeat管理方法已从Scheduler继承，包括:
    # _gc(), heartbeat(), remove(), list_instances(), 
    # select_prefill(), select_decode(), select_prefills(), 
    # select_decodes(), get_decode_with_lowest_utilization(), get_all_decode_nodes()
    
    # EMP调度方法
    async def add_instance(self, instance_id: str, modality: ModalityType, 
                          stage: InferenceStage):
        """添加实例到系统"""
        with self._lock:  # 使用threading.Lock而不是asyncio.Lock
            self.modality_balancer.add_instance(instance_id, modality)
            
            self.stage_allocators[modality].add_instance(instance_id, stage)
            
            # Also add to _prefill/_decode dictionaries for backward compatibility
            # Create a dummy InstanceInfo for v0 backend instances
            info = InstanceInfo(
                http_address=f"http://127.0.0.1:10001",  # Dummy address
                zmq_address=f"tcp://127.0.0.1:10002",   # Dummy address
                role="P" if stage == InferenceStage.PREFILL else "D",
                last_heartbeat_deadline=time.time() + 3600  # 1 hour from now
            )
            
            if stage == InferenceStage.PREFILL:
                self._prefill[instance_id] = info
                print(f"ElasticMM LOG: Added {instance_id} to _prefill dict")
            elif stage == InferenceStage.DECODE:
                self._decode[instance_id] = info
                print(f"ElasticMM LOG: Added {instance_id} to _decode dict")
            
            print(f"ElasticMM LOG: _prefill size: {len(self._prefill)}, _decode size: {len(self._decode)}")
            
            self.system_status["total_instances"] += 1
            
            pass
    
    async def remove_instance(self, instance_id: str):
        """从系统移除实例"""
        async with self._lock:
            self.modality_balancer.remove_instance(instance_id)
            
            for allocator in self.stage_allocators.values():
                allocator.remove_instance(instance_id)
            
            if self.system_status["total_instances"] > 0:
                self.system_status["total_instances"] -= 1
            
            pass
    
    async def route_request(self, request: Request) -> ModalityType:
        """路由请求到适当的模态组"""
        return request.modality_type
    
    async def process_request_batch(self, requests: List[Request]) -> Dict:
        """
        处理请求批次
        实现完整的分层调度流程
        """
        async with self._lock:
            results = {}
            
            modality_groups = {}
            for request in requests:
                modality = request.modality_type
                if modality not in modality_groups:
                    modality_groups[modality] = []
                modality_groups[modality].append(request)
            
            for modality, modality_requests in modality_groups.items():
                allocator = self.stage_allocators[modality]
                modality_results = await self._process_modality_requests(
                    modality, modality_requests, allocator
                )
                results[modality] = modality_results
            
            return results
    
    async def _process_modality_requests(self, modality: ModalityType, 
                                       requests: List[Request],
                                       allocator: StageLevelResourceAllocator) -> Dict:
        """处理单个模态组的请求"""
        results = {}
        
        for stage in InferenceStage:
            stage_requests = [r for r in requests if self._should_process_in_stage(r, stage)]
            
            if stage_requests:
                selected_requests = await allocator.dispatch_requests(
                    stage_requests, stage
                )
                
                if selected_requests:
                    allocation_decision = await allocator.elastic_instance_allocation(
                        stage, selected_requests
                    )
                    
                    results[stage] = {
                        "requests": len(selected_requests),
                        "instances": allocation_decision.instances,
                        "estimated_time": allocation_decision.estimated_time,
                        "gain_cost": allocation_decision.gain_cost
                    }
        
        return results
    
    def _should_process_in_stage(self, request: Request, stage: InferenceStage) -> bool:
        """判断请求是否应该在指定阶段处理"""
        if stage == InferenceStage.ENCODE:
            return request.modality_type == ModalityType.MULTIMODAL and request.images
        elif stage == InferenceStage.PREFILL:
            return True  
        elif stage == InferenceStage.DECODE:
            return True  
        return False
    
    async def monitor_and_rebalance(self):
        """监控系统状态并触发重新平衡"""
        while True:
            try:
                await asyncio.sleep(self.rebalance_interval)
                
                if await self._should_rebalance():
                    await self._perform_rebalancing()
                
                if self.auto_scaling_enabled:
                    await self._perform_auto_scaling()
                
            except Exception as e:
                print(f"ElasticMM LOG: Error in monitoring and rebalancing: {e}")
    
    async def _should_rebalance(self) -> bool:
        """判断是否需要重新平衡"""
        current_time = time.time()
        time_since_last = current_time - self.system_status["last_rebalance_time"]
        
        
        if time_since_last >= self.rebalance_interval:
            return True
        
        
        for modality in ModalityType:
            allocator = self.stage_allocators[modality]
            stage_status = allocator.get_stage_status()
            
            for stage, status in stage_status.items():
                if status["queue_length"] > 50:  
                    return True
        
        return False
    
    async def _perform_rebalancing(self):
        """执行重新平衡"""
        pass
        
        
        workload = self.modality_balancer.get_average_workload()
        
        
        current_distribution = self.modality_balancer.get_instance_distribution()
        total_instances = sum(current_distribution.values())
        
        if total_instances == 0:
            return
        
        
        new_allocation = await self.modality_balancer.proactive_allocation(
            total_instances, workload
        )
        
        if new_allocation != current_distribution:
            pass
        
        self.system_status["last_rebalance_time"] = time.time()
    
    async def _perform_auto_scaling(self):
        """执行自动扩缩容"""
        for modality in ModalityType:
            allocator = self.stage_allocators[modality]
            stage_status = allocator.get_stage_status()
            
            for stage, status in stage_status.items():
                queue_length = status["queue_length"]
                scaling_action = await allocator.elastic_auto_scaling(stage, queue_length)
                
                if scaling_action != "no_action":
                    pass
    
    # ✅ 新增：弹性调度循环（与V0Backend集成）
    def start_elastic_scheduling(self):
        """启动弹性调度循环"""
        if not self.backend or not self.elastic_scheduling_enabled:
            print("[EMPScheduler] Elastic scheduling disabled or no backend")
            return
        
        self.running = True
        self._elastic_task = asyncio.create_task(self._elastic_scheduling_loop())
        print("[EMPScheduler] Elastic scheduling loop started")
    
    async def stop_elastic_scheduling(self):
        """停止弹性调度循环"""
        self.running = False
        if self._elastic_task:
            self._elastic_task.cancel()
            try:
                await self._elastic_task
            except asyncio.CancelledError:
                pass
    
    async def _elastic_scheduling_loop(self):
        """
        弹性调度主循环
        每N步（future_rounds_horizon）评估一次资源分配
        """
        while self.running:
            try:
                # 更新step counter
                for allocator in self.stage_allocators.values():
                    allocator.step_counter += 1
                    
                    # 记录当前workload（用于历史统计）
                    if hasattr(self.backend, 'get_stats'):
                        stats = self.backend.get_stats()
                        allocator.record_step_stats(
                            InferenceStage.PREFILL,
                            stats['prefill']['num_waiting'] + stats['prefill']['num_running']
                        )
                        allocator.record_step_stats(
                            InferenceStage.DECODE,
                            stats['decoding']['num_waiting'] + stats['decoding']['num_running']
                        )
                        
                        # ✅ 检查decode内存不足（基于token budget上限）
                        await self._check_decode_memory_pressure(stats, allocator)
                
                # 检查是否需要重新分配（每100步）
                for modality, allocator in self.stage_allocators.items():
                    if allocator.should_trigger_reallocation():
                        await self._rebalance_resources_v0(modality, allocator)
                
                await asyncio.sleep(1.0)  # 每秒检查一次
                
            except Exception as e:
                print(f"[EMPScheduler] Error in elastic scheduling loop: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(5.0)  # 错误后等待更长时间
    
    async def _rebalance_resources_v0(self, modality: ModalityType, 
                                     allocator: StageLevelResourceAllocator):
        """
        执行V0 Backend的资源重新平衡
        使用增量式Gain-Cost模型决策
        """
        if not self.backend or not hasattr(self.backend, 'get_worker_allocation'):
            return
        
        try:
            # 获取当前状态
            stats = self.backend.get_stats()
            worker_alloc = self.backend.get_worker_allocation()
            
            prefill_waiting = stats['prefill']['num_waiting']
            prefill_running = stats['prefill']['num_running']
            decode_waiting = stats['decoding']['num_waiting']
            decode_running = stats['decoding']['num_running']
            decode_workers = stats['decoding']['num_workers']
            prefill_workers = stats['prefill']['num_workers']
            
            print(f"[EMPScheduler] Rebalancing check: P({prefill_workers}w, {prefill_waiting}q) "
                  f"D({decode_workers}w, {decode_waiting}q)")
            
            # 策略1: Prefill拥堵 + Decode有余力 → 抢占Decode worker
            if prefill_waiting > 30 and decode_workers > 1 and decode_waiting < 10:
                # 选择一个decode worker迁移到prefill
                decode_worker_ids = [wid for wid, stage in worker_alloc.items() 
                                   if stage == 'decoding']
                if decode_worker_ids:
                    worker_to_migrate = decode_worker_ids[-1]  # 选择最后一个
                    
                    print(f"[EMPScheduler] 🔄 Migrating worker {worker_to_migrate}: "
                          f"decoding → prefill (prefill queue: {prefill_waiting})")
                    
                    success = await self.backend.switch_worker_role(
                        worker_id=worker_to_migrate,
                        from_stage='decoding',
                        to_stage='prefill',
                        migrate_kv=True
                    )
                    
                    if success:
                        # 更新allocator的分配记录
                        new_allocation = worker_alloc.copy()
                        new_allocation[worker_to_migrate] = 'prefill'
                        allocator.update_allocation_state(new_allocation)
                        print(f"[EMPScheduler] ✅ Migration successful")
                    else:
                        print(f"[EMPScheduler] ❌ Migration failed")
            
            # 策略2: Decode拥堵 + Prefill有余力 → 抢占Prefill worker  
            elif decode_waiting > 50 and prefill_workers > 1 and prefill_waiting < 5:
                prefill_worker_ids = [wid for wid, stage in worker_alloc.items()
                                    if stage == 'prefill']
                if prefill_worker_ids:
                    worker_to_migrate = prefill_worker_ids[-1]
                    
                    print(f"[EMPScheduler] 🔄 Migrating worker {worker_to_migrate}: "
                          f"prefill → decoding (decode queue: {decode_waiting})")
                    
                    success = await self.backend.switch_worker_role(
                        worker_id=worker_to_migrate,
                        from_stage='prefill',
                        to_stage='decoding',
                        migrate_kv=True
                    )
                    
                    if success:
                        new_allocation = worker_alloc.copy()
                        new_allocation[worker_to_migrate] = 'decoding'
                        allocator.update_allocation_state(new_allocation)
                        print(f"[EMPScheduler] ✅ Migration successful")
                    else:
                        print(f"[EMPScheduler] ❌ Migration failed")
        
        except Exception as e:
            print(f"[EMPScheduler] Error in _rebalance_resources_v0: {e}")
            import traceback
            traceback.print_exc()
    
    async def _check_decode_memory_pressure(self, stats: Dict, allocator: StageLevelResourceAllocator):
        """
        检查decode阶段的内存压力（基于token budget上限）
        如果超限，触发被动扩张
        
        Args:
            stats: Backend统计信息
            allocator: 资源分配器
        """
        try:
            # 获取decode阶段的当前token数量
            decode_running = stats['decoding']['num_running']
            decode_waiting = stats['decoding']['num_waiting']
            
            # 从backend获取当前decode batch的总token数
            if hasattr(self.backend, 'get_decode_token_count'):
                current_token_count = self.backend.get_decode_token_count()
            else:
                # 估算：假设每个请求平均100 tokens (简化)
                current_token_count = (decode_running + decode_waiting) * 100
            
            # 获取token budget上限
            max_token_budget = allocator.gain_cost_config['max_decode_token_budget']
            
            # 计算压力百分比
            pressure_ratio = current_token_count / max_token_budget if max_token_budget > 0 else 0
            
            # 如果超过90%阈值，触发被动扩张
            if pressure_ratio > 0.9:
                print(f"[EMPScheduler] ⚠️  Decode memory pressure detected! "
                      f"{current_token_count}/{max_token_budget} tokens ({pressure_ratio*100:.1f}%)")
                
                # 触发被动扩张
                await self._trigger_passive_scaling(stats, allocator, pressure_ratio)
            
        except Exception as e:
            print(f"[EMPScheduler] Error checking decode memory pressure: {e}")
    
    async def _trigger_passive_scaling(self, stats: Dict, 
                                      allocator: StageLevelResourceAllocator,
                                      pressure_ratio: float):
        """
        触发被动扩张：从prefill抢占一个worker到decode
        
        Args:
            stats: Backend统计信息
            allocator: 资源分配器
            pressure_ratio: 压力比例
        """
        if not self.backend or not hasattr(self.backend, 'get_worker_allocation'):
            return
        
        try:
            worker_alloc = self.backend.get_worker_allocation()
            prefill_workers = stats['prefill']['num_workers']
            decode_workers = stats['decoding']['num_workers']
            
            # 只有prefill还有多余worker时才扩张
            if prefill_workers <= 1:
                print(f"[EMPScheduler] ⚠️  Cannot scale decode: prefill only has {prefill_workers} worker(s)")
                return
            
            # 选择一个prefill worker迁移到decode
            prefill_worker_ids = [wid for wid, stage in worker_alloc.items() 
                                 if stage == 'prefill']
            
            if not prefill_worker_ids:
                print("[EMPScheduler] No prefill workers available for migration")
                return
            
            worker_to_migrate = prefill_worker_ids[-1]  # 选择最后一个
            
            print(f"[EMPScheduler] 🚨 PASSIVE SCALING: Migrating worker {worker_to_migrate}: "
                  f"prefill → decode (memory pressure: {pressure_ratio*100:.1f}%)")
            
            success = await self.backend.switch_worker_role(
                worker_id=worker_to_migrate,
                from_stage='prefill',
                to_stage='decoding',
                migrate_kv=True
            )
            
            if success:
                print(f"[EMPScheduler] ✅ Passive scaling completed: "
                      f"P({prefill_workers} → {prefill_workers-1}w) "
                      f"D({decode_workers} → {decode_workers+1}w)")
                
                # 更新allocator的分配记录
                allocator.previous_allocation = allocator.current_allocation.copy()
                allocator.current_allocation[worker_to_migrate] = 'decoding'
            else:
                print(f"[EMPScheduler] ❌ Passive scaling failed")
                
        except Exception as e:
            print(f"[EMPScheduler] Error in passive scaling: {e}")
            import traceback
            traceback.print_exc()
    
    def get_system_status(self) -> Dict:
        """获取系统状态"""
        status = self.system_status.copy()
        
        status["modality_groups"] = self.modality_balancer.get_instance_distribution()
        
        status["stage_status"] = {}
        for modality, allocator in self.stage_allocators.items():
            status["stage_status"][modality.value] = allocator.get_stage_status()
        
        return status
    
    async def update_workload_stats(self, workload: WorkloadStats):
        """更新工作负载统计"""
        self.modality_balancer.add_workload_sample(workload)
    
    def select_prefills(self, k: int) -> List[InstanceInfo]:
        """Select multiple prefill instances"""
        with self._lock:
            self._gc()
            items = list(self._prefill.values())
            if not items:
                return []
            # simple round-robin window
            start = self._rr_p
            out = []
            for i in range(min(k, len(items))):
                out.append(items[(start + i) % len(items)])
            self._rr_p = (start + 1) % len(items)
            return out
    
    def select_decodes(self, k: int) -> List[InstanceInfo]:
        """Select multiple decode instances"""
        with self._lock:
            self._gc()
            items = list(self._decode.values())
            if not items:
                return []
            # simple round-robin window
            start = self._rr_d
            out = []
            for i in range(min(k, len(items))):
                out.append(items[(start + i) % len(items)])
            self._rr_d = (start + 1) % len(items)
            return out

    async def migrate_node_in_scheduling(self, source_node: str, target_nodes: List[str], 
                                       reason: str = "elastic_scaling") -> bool:
        """
        在弹性调度过程中迁移节点
        
        Args:
            source_node: 要迁移的源节点
            target_nodes: 目标节点列表
            reason: 迁移原因 ("elastic_scaling", "load_balancing", "fault_recovery")
        
        Returns:
            bool: 迁移是否成功
        """
        if not self.backend:
            print("ElasticMM LOG: Scheduler not configured with backend, cannot execute node migration")
            return False
        
        print(f"ElasticMM LOG: Scheduler initiating node-level KV migration: {source_node} -> {target_nodes}")
        print(f"migration reason: {reason}")
        
        try:
            # Get source node info
            source_info = self.backend.get_instance_info(source_node)
            if not source_info:
                print(f"ElasticMM LOG: Source node {source_node} not found")
                return False
            
            # For v1 backend, get additional info from engine manager
            if hasattr(self.backend, 'get_engine_manager') and self.backend.get_engine_manager():
                engine_manager = self.backend.get_engine_manager()
                source_kv_info = engine_manager.get_node_kv_info(source_node)
                active_requests = engine_manager.get_node_active_requests(source_node)
                
                if not active_requests:
                    print(f"ElasticMM LOG: Source node {source_node} has no active requests, skipping KV migration")
                    return True
                
                print(f"ElasticMM LOG: Source node status: {len(active_requests)} active requests")
                print(f"ElasticMM LOG: KV cache usage: {source_kv_info.get('kv_cache_usage', 0):.2%}")
                
                # Select migration strategy
                migration_strategy = self._select_migration_strategy(
                    len(active_requests), len(target_nodes), reason
                )
                
                # Execute node-level KV migration
                migration_success = engine_manager.migrate_node_kv_cache(
                    source_node, target_nodes, migration_strategy
                )
            else:
                # For v0 backend, use backend's migrate_instance method
                # Create dummy requests for migration
                dummy_requests = []  # In real implementation, get actual requests
                migration_success = await self.backend.migrate_instance(
                    source_node, target_nodes[0], dummy_requests
                )
            
            if migration_success:
                print(f"ElasticMM LOG: Scheduler node migration successful: {source_node}")
                # Update scheduler state
                self._update_scheduler_state_after_migration(
                    source_node, target_nodes, reason
                )
            else:
                print(f"ElasticMM LOG: Scheduler node migration failed: {source_node}")
            
            return migration_success
            
        except Exception as e:
            print(f"ElasticMM LOG: Scheduler node migration exception: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _select_migration_strategy(self, request_count: int, target_count: int, 
                                 reason: str) -> str:
        """根据迁移场景选择最佳策略"""
        if reason == "load_balancing":
            # when load balancing, use round-robin strategy, average distribute load
            return "round_robin"
        elif reason == "fault_recovery":
            # when fault recovery, use single-target strategy, fast migration
            return "single_target"
        elif reason == "elastic_scaling":
            # when elastic scaling, select strategy based on request count
            if request_count <= target_count * 2:
                return "round_robin"
            else:
                return "load_balance"
        else:
            return "round_robin"
    
    def _update_scheduler_state_after_migration(self, source_node: str, 
                                              target_nodes: List[str], reason: str):
        """update scheduler state after migration"""
        try:
            # update system state record
            migration_record = {
                "timestamp": time.time(),
                "source_node": source_node,
                "target_nodes": target_nodes,
                "reason": reason,
                "status": "completed"
            }
            
            # record to system state
            if "migration_history" not in self.system_status:
                self.system_status["migration_history"] = []
            
            self.system_status["migration_history"].append(migration_record)
            
            # keep recent 50 migration records
            if len(self.system_status["migration_history"]) > 50:
                self.system_status["migration_history"] = \
                    self.system_status["migration_history"][-50:]
            
            print(f"📝 scheduler state updated: record node migration {source_node} -> {target_nodes}")
            
        except Exception as e:
            print(f"ElasticMM LOG: Failed to update scheduler status: {e}")
    
    def get_migration_history(self) -> List[Dict]:
        """get node migration history"""
        return self.system_status.get("migration_history", [])
    
    async def trigger_proactive_migration(self, load_threshold: float = 0.8) -> List[str]:
        """
        主动触发节点迁移（基于负载监控）
        
        Args:
            load_threshold: 负载阈值，超过此值触发迁移
        
        Returns:
            List[str]: 已迁移的节点列表
        """
        if not self.backend:
            return []
        
        migrated_nodes = []
        
        try:
            # For v1 backend, get detailed KV status
            if hasattr(self.backend, 'get_engine_manager') and self.backend.get_engine_manager():
                engine_manager = self.backend.get_engine_manager()
                all_nodes_status = engine_manager.get_all_nodes_kv_status()
                
                # 识别过载节点
                overloaded_nodes = []
                underloaded_nodes = []
                
                for node_name, status in all_nodes_status.items():
                    if 'error' in status:
                        continue
                    
                    kv_info = status.get('kv_info', {})
                    request_count = status.get('request_count', 0)
                    kv_usage = kv_info.get('kv_cache_usage', 0)
                    
                    if kv_usage > load_threshold and request_count > 0:
                        overloaded_nodes.append((node_name, kv_usage, request_count))
                    elif kv_usage < load_threshold * 0.5:
                        underloaded_nodes.append((node_name, kv_usage))
                
                # 执行负载重分布
                if overloaded_nodes and underloaded_nodes:
                    for node_name, usage, req_count in overloaded_nodes:
                        # 选择负载最低的节点作为目标
                        target_nodes = [node[0] for node in underloaded_nodes[:2]]
                        
                        print(f"ElasticMM LOG: Proactive migration of overloaded node: {node_name} (load: {usage:.2%}, requests: {req_count})")
                        
                        success = await self.migrate_node_in_scheduling(
                            node_name, target_nodes, "load_balancing"
                        )
                        
                        if success:
                            migrated_nodes.append(node_name)
            else:
                # For v0 backend, use simpler approach
                all_instances = self.backend.get_all_instances()
                print(f"ElasticMM LOG: V0 backend proactive migration not fully implemented, found {len(all_instances)} instances")
            
            return migrated_nodes
            
        except Exception as e:
            print(f"ElasticMM LOG: Proactive migration trigger failed: {e}")
            return []
    
    def _update_scheduler_state_after_migration(self, source_node: str, 
                                              target_nodes: List[str], reason: str):
        """update scheduler state after migration"""
        try:
            # update system state record
            migration_record = {
                "timestamp": time.time(),
                "source_node": source_node,
                "target_nodes": target_nodes,
                "reason": reason,
                "status": "completed"
            }
            
            # record to system state
            if "migration_history" not in self.system_status:
                self.system_status["migration_history"] = []
            
            self.system_status["migration_history"].append(migration_record)
            
            # keep recent 50 migration records
            if len(self.system_status["migration_history"]) > 50:
                self.system_status["migration_history"] = \
                    self.system_status["migration_history"][-50:]
            
            print(f"📝 scheduler state updated: record node migration {source_node} -> {target_nodes}")
            
        except Exception as e:
            print(f"ElasticMM LOG: Failed to update scheduler status: {e}")
    
    def get_migration_history(self) -> List[Dict]:
        """get node migration history"""
        return self.system_status.get("migration_history", [])
    
    async def trigger_proactive_migration(self, load_threshold: float = 0.8) -> List[str]:
        """
        主动触发节点迁移（基于负载监控）
        
        Args:
            load_threshold: 负载阈值，超过此值触发迁移
        
        Returns:
            List[str]: 已迁移的节点列表
        """
        if not self.backend:
            return []
        
        migrated_nodes = []
        
        try:
            # For v1 backend, get detailed KV status
            if hasattr(self.backend, 'get_engine_manager') and self.backend.get_engine_manager():
                engine_manager = self.backend.get_engine_manager()
                all_nodes_status = engine_manager.get_all_nodes_kv_status()
                
                # 识别过载节点
                overloaded_nodes = []
                underloaded_nodes = []
                
                for node_name, status in all_nodes_status.items():
                    if 'error' in status:
                        continue
                    
                    kv_info = status.get('kv_info', {})
                    request_count = status.get('request_count', 0)
                    kv_usage = kv_info.get('kv_cache_usage', 0)
                    
                    if kv_usage > load_threshold and request_count > 0:
                        overloaded_nodes.append((node_name, kv_usage, request_count))
                    elif kv_usage < load_threshold * 0.5:
                        underloaded_nodes.append((node_name, kv_usage))
                
                # 执行负载重分布
                if overloaded_nodes and underloaded_nodes:
                    for node_name, usage, req_count in overloaded_nodes:
                        # 选择负载最低的节点作为目标
                        target_nodes = [node[0] for node in underloaded_nodes[:2]]
                        
                        print(f"ElasticMM LOG: Proactive migration of overloaded node: {node_name} (load: {usage:.2%}, requests: {req_count})")
                        
                        success = await self.migrate_node_in_scheduling(
                            node_name, target_nodes, "load_balancing"
                        )
                        
                        if success:
                            migrated_nodes.append(node_name)
            else:
                # For v0 backend, use simpler approach
                all_instances = self.backend.get_all_instances()
                print(f"ElasticMM LOG: V0 backend proactive migration not fully implemented, found {len(all_instances)} instances")
            
            return migrated_nodes
            
        except Exception as e:
            print(f"ElasticMM LOG: Proactive migration trigger failed: {e}")
            return []