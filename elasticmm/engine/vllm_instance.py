import json
import os
import shlex
import socket
import subprocess
import threading
import time
import ray
import zmq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from elasticmm.engine.kv_migration import VLLMKVMigrationConnector


@dataclass
class VLLMEngineConfig:
    model_path: str
    http_host: str
    http_port: int
    kv_role: str  # "kv_producer" or "kv_consumer"
    kv_rank: int
    kv_parallel_size: int
    kv_port: int
    kv_buffer_size: str = "8e9"  # 恢复原始缓冲区大小
    proxy_ip: str = "0.0.0.0"
    proxy_port: int = 30002
    nccl_num_channels: int = 16  # 恢复原始NCCL通道数
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 32768
    seed: int = 1024
    dtype: str = "float16"
    tensor_parallel_size: int = 1
    enforce_eager: bool = True
    trust_remote_code: bool = True
    gpu_id: int = 0  # GPU ID for this engine


@dataclass
class KVSlotsInfo:
    """KV slots使用情况信息"""
    total_slots: int
    used_slots: int
    utilization_rate: float  # 0-1
    last_update: float
    
    def __post_init__(self):
        if self.last_update == 0.0:
            self.last_update = time.time()
    
    @property
    def unused_slots(self) -> int:
        return self.total_slots - self.used_slots
    
    @property
    def is_available(self) -> bool:
        return self.utilization_rate < 0.8  # 使用率低于80%认为可用


@ray.remote(num_gpus=1)
class VLLMEngineActor:
    """Ray actor for running vLLM engines on specific GPUs"""
    
    def __init__(self, config: VLLMEngineConfig, env: Optional[Dict[str, str]] = None):
        self.config = config
        self.env = dict(os.environ, **(env or {}))
        self.process: Optional[subprocess.Popen] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_heartbeat = False
        
        # 身份和状态管理
        self.current_role = config.kv_role  # 当前角色
        self.original_role = config.kv_role  # 原始角色
        self.is_active = True
        self.kv_slots_info = KVSlotsInfo(
            total_slots=1000,  # 默认1000个slots
            used_slots=0,
            utilization_rate=0.0,
            last_update=time.time()
        )
        
        # Don't start engine immediately, let the manager control it
        # self._start_engine()
        # self._start_heartbeat()
    
    def start(self):
        """Start the engine and heartbeat"""
        self._start_engine()
        self._start_heartbeat()
    
    def is_model_loaded(self) -> bool:
        """检查模型是否已加载完成"""
        if not self.is_running():
            return False
        
        # 检查进程是否还在运行
        if self.process and self.process.poll() is not None:
            return False
        
        # 这里可以添加更详细的模型加载检查
        # 例如检查特定的日志输出或HTTP端点响应
        return True
    
    def _build_cmd(self) -> List[str]:
        """Build vLLM command for this engine"""
        args = [
            "vllm", "serve", self.config.model_path,
            "--enforce-eager" if self.config.enforce_eager else "",
            "--host", self.config.http_host,
            "--port", str(self.config.http_port),
            "--tensor-parallel-size", str(self.config.tensor_parallel_size),
            "--seed", str(self.config.seed),
            "--dtype", self.config.dtype,
            "--max-model-len", str(self.config.max_model_len),
            "--max-num-batched-tokens", "10000",
            "--max-num-seqs", "256",
            "--trust-remote-code" if self.config.trust_remote_code else "",
            "--gpu-memory-utilization", str(self.config.gpu_memory_utilization),
            "--kv-transfer-config", json.dumps(self._build_kv_transfer_config()),
        ]
        # Remove empty strings
        return [a for a in args if a]

    def _build_kv_transfer_config(self) -> Dict:
        """Build KV transfer configuration"""
        return {
            "kv_connector": "P2pNcclConnector",
            "kv_role": self.config.kv_role,
            "kv_rank": self.config.kv_rank,
            "kv_parallel_size": self.config.kv_parallel_size,
            "kv_buffer_size": self.config.kv_buffer_size,
            "kv_port": str(self.config.kv_port),
            "kv_connector_extra_config": {
                "proxy_ip": self.config.proxy_ip,
                "proxy_port": str(self.config.proxy_port),
                "http_port": str(self.config.http_port),
                "send_type": "PUT_ASYNC",
                "nccl_num_channels": str(self.config.nccl_num_channels),
            }
        }
    
    def _start_engine(self):
        """Start the vLLM engine process"""
        cmd_list = self._build_cmd()
        env = self.env.copy()
        env["VLLM_USE_V1"] = env.get("VLLM_USE_V1", "1")
        env["CUDA_VISIBLE_DEVICES"] = str(self.config.gpu_id)
        
        # 优化NCCL配置
        env["NCCL_DEBUG"] = "WARN"  # 减少NCCL调试输出
        
        print(f"Starting vLLM engine on GPU {self.config.gpu_id}: {' '.join(shlex.quote(s) for s in cmd_list)}")
        
        self.process = subprocess.Popen(cmd_list, env=env)
        return self.process
    
    def is_running(self) -> bool:
        """Check if engine is running"""
        if self.process is None:
            return False
        return self.process.poll() is None
    
    def stop(self):
        """Stop the engine"""
        # Stop heartbeat first
        self._stop_heartbeat = True
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)
        
        # Stop the engine process
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            except Exception:
                pass
            self.process = None
    
    def _start_heartbeat(self):
        """Start heartbeat thread to register with scheduler"""
        def heartbeat_worker():
            import msgpack
            context = zmq.Context()
            zmq_socket = context.socket(zmq.REQ)  # Use REQ for simpler synchronous communication
            zmq_socket.setsockopt(zmq.LINGER, 0)  # Don't wait for messages on close
            zmq_socket.connect(f"tcp://{self.config.proxy_ip}:{self.config.proxy_port}")
            
            # Wait a bit for the scheduler to be ready
            print(f"🔗 {self.config.kv_role} engine connecting to scheduler at {self.config.proxy_ip}:{self.config.proxy_port}")
            time.sleep(2)  # 增加等待时间确保代理服务器完全启动
            
            # Determine the correct host address to use
            # If http_host is 0.0.0.0, we need to find the actual bindable address
            host_address = self.env.get('HOST')
            if not host_address:
                if self.config.http_host == "0.0.0.0":
                    # Try to get the actual IP address when binding to 0.0.0.0
                    try:
                        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        s.connect(("8.8.8.8", 80))
                        host_address = s.getsockname()[0]
                        s.close()
                    except Exception:
                        host_address = "127.0.0.1"
                else:
                    host_address = self.config.http_host
            
            while not self._stop_heartbeat:
                try:
                    # Send heartbeat to scheduler using msgpack format
                    heartbeat_data = {
                        "http_address": f"{host_address}:{self.config.http_port}",
                        "zmq_address": f"{host_address}:{self.config.kv_port}",
                        "type": "P" if self.config.kv_role == "kv_producer" else "D"  # Use 'type' not 'role'
                    }
                    
                    # Send as msgpack with REQ socket
                    zmq_socket.send(msgpack.packb(heartbeat_data))
                    
                    # Wait for response from REP socket
                    if zmq_socket.poll(5000):  # 5 second timeout
                        response = zmq_socket.recv()
                        if response == b"OK":
                            # Only print success message occasionally (减少输出频率)
                            if int(time.time()) % 120 == 0:  # 每2分钟打印一次
                                print(f"💓 {self.config.kv_role} engine on port {self.config.http_port} - OK")
                        else:
                            print(f"ElasticMM LOG: Unexpected heartbeat response: {response}")
                    else:
                        print(f"ElasticMM LOG: Heartbeat timeout for {self.config.kv_role} engine on port {self.config.http_port}")
                    
                    time.sleep(10)  # 适中的心跳频率：每10秒发送一次
                except Exception as e:
                    print(f"Heartbeat error for {self.config.kv_role} engine: {e}")
                    time.sleep(10)  # 错误时也等待10秒
            
            zmq_socket.close()
            context.term()
        
        self._heartbeat_thread = threading.Thread(target=heartbeat_worker, daemon=True)
        self._heartbeat_thread.start()
    
    def get_status(self) -> Dict:
        """Get engine status"""
        return {
            "gpu_id": self.config.gpu_id,
            "http_port": self.config.http_port,
            "kv_role": self.config.kv_role,
            "current_role": self.current_role,
            "original_role": self.original_role,
            "kv_rank": self.config.kv_rank,
            "is_running": self.is_running(),
            "is_active": self.is_active,
            "kv_slots_info": {
                "total_slots": self.kv_slots_info.total_slots,
                "used_slots": self.kv_slots_info.used_slots,
                "utilization_rate": self.kv_slots_info.utilization_rate,
                "unused_slots": self.kv_slots_info.unused_slots,
                "is_available": self.kv_slots_info.is_available
            },
            "pid": self.process.pid if self.process else None
        }
    
    def switch_role(self, new_role: str) -> bool:
        """切换实例角色（P <-> D），保留权重和KV缓存"""
        if new_role not in ["kv_producer", "kv_consumer"]:
            print(f"ElasticMM LOG: Invalid role: {new_role}")
            return False
        
        if new_role == self.current_role:
            print(f"ElasticMM LOG: Instance already has {new_role} role, no need to switch")
            return True
        
        print(f"ElasticMM LOG: Switching instance role: {self.current_role} -> {new_role}")
        
        # 更新配置和角色
        old_role = self.current_role
        self.current_role = new_role
        self.config.kv_role = new_role
        
        # 更新心跳中的角色信息
        # 注意：这里不需要重启引擎，只需要更新角色信息
        print(f"ElasticMM LOG: Instance role switch successful: {old_role} -> {new_role}")
        return True
    
    def update_kv_slots(self, used_slots: int) -> None:
        """更新KV slots使用情况"""
        self.kv_slots_info.used_slots = min(used_slots, self.kv_slots_info.total_slots)
        self.kv_slots_info.utilization_rate = self.kv_slots_info.used_slots / self.kv_slots_info.total_slots
        self.kv_slots_info.last_update = time.time()
    
    def get_kv_cache_stats(self) -> Dict:
        """获取KV缓存统计信息（模拟vLLM的KV缓存状态）"""
        # 这里可以集成真实的vLLM KV缓存API
        # 目前返回模拟数据
        return {
            "total_blocks": self.kv_slots_info.total_slots,
            "used_blocks": self.kv_slots_info.used_slots,
            "free_blocks": self.kv_slots_info.unused_slots,
            "utilization_rate": self.kv_slots_info.utilization_rate,
            "block_size": 16,  # 假设每个block包含16个tokens
            "total_tokens": self.kv_slots_info.total_slots * 16,
            "used_tokens": self.kv_slots_info.used_slots * 16,
            "last_update": self.kv_slots_info.last_update
        }
    
    def get_kv_slots_info(self) -> KVSlotsInfo:
        """获取KV slots信息"""
        return self.kv_slots_info
    
    def can_accept_kv_migration(self, required_slots: int) -> bool:
        """检查是否可以接受KV迁移"""
        return self.kv_slots_info.unused_slots >= required_slots and self.is_active
    
    def migrate_kv_to_instance(self, target_instance_ref, kv_data) -> bool:
        """将KV缓存迁移到目标实例"""
        # 检查目标实例是否可以接受KV迁移
        try:
            can_accept = ray.get(target_instance_ref.can_accept_kv_migration.remote(len(kv_data)))
            if not can_accept:
                print("ElasticMM LOG: Target instance cannot accept KV migration")
                return False
        except Exception as e:
            print(f"ElasticMM LOG: Failed to check target instance KV migration capability: {e}")
            return False
        
        print(f"ElasticMM LOG: Starting KV migration: {len(kv_data)} slots")
        
        # 这里实现实际的KV迁移逻辑
        # 在实际实现中，这里会调用NCCL进行GPU间数据传输
        # 模拟KV迁移过程
        time.sleep(0.1)  # 模拟迁移时间
        
        # 更新目标实例的slots使用情况
        try:
            target_slots_info = ray.get(target_instance_ref.get_kv_slots_info.remote())
            new_used_slots = target_slots_info.used_slots + len(kv_data)
            ray.get(target_instance_ref.update_kv_slots.remote(new_used_slots))
            print("ElasticMM LOG: KV migration completed")
            return True
        except Exception as e:
            print(f"ElasticMM LOG: Failed to update target instance KV slots: {e}")
            return False


class VLLMEngineManager:
    def __init__(self, env: Optional[Dict[str, str]] = None) -> None:
        self._env = dict(os.environ, **(env or {}))
        self._engines: Dict[str, ray.ObjectRef] = {}  # name -> actor_ref
        self._gpu_usage: Dict[int, bool] = {}  # gpu_id -> in_use
        # 使用补丁模式实现真实KV迁移
        self._kv_migration_connector = VLLMKVMigrationConnector(self, migration_mode="node_level")
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
    
    def _get_available_gpu(self) -> int:
        """Get next available GPU ID"""
        for gpu_id in range(8):  # Assuming 8 GPUs
            if not self._gpu_usage.get(gpu_id, False):
                return gpu_id
        raise RuntimeError("No available GPUs")
    
    def _mark_gpu_used(self, gpu_id: int):
        """Mark GPU as used"""
        self._gpu_usage[gpu_id] = True
    
    def _mark_gpu_free(self, gpu_id: int):
        """Mark GPU as free"""
        self._gpu_usage[gpu_id] = False

    def start_engine(self, cfg: VLLMEngineConfig, name: str) -> ray.ObjectRef:
        """Start a vLLM engine on an available GPU"""
        if name in self._engines:
            raise ValueError(f"Engine '{name}' already exists")
        
        # Get available GPU
        gpu_id = self._get_available_gpu()
        
        # Update config with GPU ID
        cfg.gpu_id = gpu_id
        
        # Create Ray actor
        actor_ref = VLLMEngineActor.remote(cfg, self._env)
        # Start the engine
        ray.get(actor_ref.start.remote())
        self._engines[name] = actor_ref
        self._mark_gpu_used(gpu_id)
        
        print(f"Started engine '{name}' on GPU {gpu_id}")
        return actor_ref
    
    def start_engines_parallel(self, configs: List[Tuple[VLLMEngineConfig, str]]) -> List[ray.ObjectRef]:
        """Start multiple vLLM engines in parallel"""
        print(f"ElasticMM LOG: Starting {len(configs)} engines in parallel...")
        
        # Create all actors first
        actors = []
        for cfg, name in configs:
            if name in self._engines:
                raise ValueError(f"Engine '{name}' already exists")
            
            gpu_id = self._get_available_gpu()
            cfg.gpu_id = gpu_id
            
            actor_ref = VLLMEngineActor.remote(cfg, self._env)
            actors.append((actor_ref, name, gpu_id))
            self._mark_gpu_used(gpu_id)
        
        # Start all engines in parallel
        start_tasks = []
        for actor_ref, name, gpu_id in actors:
            start_task = actor_ref.start.remote()
            start_tasks.append((start_task, actor_ref, name, gpu_id))
        
        # Wait for all engines to start
        results = []
        for start_task, actor_ref, name, gpu_id in start_tasks:
            ray.get(start_task)  # Wait for this engine to start
            self._engines[name] = actor_ref
            results.append(actor_ref)
            print(f"ElasticMM LOG: Started engine '{name}' on GPU {gpu_id}")
        
        print(f"Successfully started {len(results)} engines in parallel")
        return results
    
    def switch_engine_role(self, engine_name: str, new_role: str) -> bool:
        """切换引擎角色"""
        if engine_name not in self._engines:
            print(f"ElasticMM LOG: Engine '{engine_name}' does not exist")
            return False
        
        try:
            actor_ref = self._engines[engine_name]
            result = ray.get(actor_ref.switch_role.remote(new_role))
            print(f"ElasticMM LOG: Engine '{engine_name}' role switch successful: -> {new_role}")
            return result
        except Exception as e:
            print(f"ElasticMM LOG: Engine '{engine_name}' role switch failed: {e}")
            return False
    
    def get_engine_kv_slots(self, engine_name: str) -> Optional[Dict]:
        """获取引擎KV slots信息"""
        if engine_name not in self._engines:
            return None
        
        try:
            actor_ref = self._engines[engine_name]
            status = ray.get(actor_ref.get_status.remote())
            return status.get("kv_slots_info")
        except Exception as e:
            print(f"ElasticMM LOG: Failed to get engine '{engine_name}' KV slots info: {e}")
            return None
    
    def find_best_target_for_kv_migration(self, source_engine: str, required_slots: int) -> Optional[str]:
        """找到最适合的KV迁移目标实例"""
        if source_engine not in self._engines:
            return None
        
        best_target = None
        best_utilization = float('inf')
        
        for engine_name, actor_ref in self._engines.items():
            if engine_name == source_engine:
                continue
            
            try:
                status = ray.get(actor_ref.get_status.remote())
                kv_info = status.get("kv_slots_info", {})
                
                # 检查是否可以接受迁移
                if (kv_info.get("unused_slots", 0) >= required_slots and 
                    status.get("is_active", False)):
                    
                    utilization = kv_info.get("utilization_rate", 1.0)
                    if utilization < best_utilization:
                        best_utilization = utilization
                        best_target = engine_name
                        
            except Exception as e:
                print(f"ElasticMM LOG: Error checking engine '{engine_name}': {e}")
                continue
        
        return best_target
    
    def get_engine_status(self, engine_name: str) -> Optional[Dict]:
        """获取引擎状态"""
        if engine_name not in self._engines:
            return None
        
        try:
            actor_ref = self._engines[engine_name]
            return ray.get(actor_ref.get_status.remote())
        except Exception as e:
            print(f"ElasticMM LOG: Failed to get engine '{engine_name}' status: {e}")
            return None
    
    def migrate_kv_between_engines(self, source_engine: str, target_engine: str, kv_data) -> bool:
        """在两个引擎之间迁移KV缓存"""
        if source_engine not in self._engines or target_engine not in self._engines:
            print("ElasticMM LOG: Source or target engine does not exist")
            return False
        
        try:
            source_actor = self._engines[source_engine]
            target_actor = self._engines[target_engine]
            
            # 检查目标引擎是否可以接受迁移
            target_status = ray.get(target_actor.get_status.remote())
            kv_info = target_status.get("kv_slots_info", {})
            
            if kv_info.get("unused_slots", 0) < len(kv_data):
                print(f"ElasticMM LOG: Target engine '{target_engine}' has insufficient space")
                return False
            
            # 执行KV迁移
            result = ray.get(source_actor.migrate_kv_to_instance.remote(target_actor, kv_data))
            print(f"ElasticMM LOG: KV migration successful: {source_engine} -> {target_engine}")
            return result
            
        except Exception as e:
            print(f"ElasticMM LOG: KV migration failed: {e}")
            return False
    
    def migrate_request_kv_cache(self, request_id: str, source_decode_engine: str, target_decode_engine: str) -> bool:
        """迁移特定请求的KV缓存 - 使用真实的KV迁移实现"""
        print(f"ElasticMM LOG: Starting real KV migration: {request_id} ({source_decode_engine} -> {target_decode_engine})")
        
        try:
            # 检查引擎是否存在
            if source_decode_engine not in self._engines or target_decode_engine not in self._engines:
                print("ElasticMM LOG: Source or target engine does not exist")
                return False
            
            # 使用补丁方式实现真实KV迁移
            if not self._kv_migration_connector.setup_engine_migration(source_decode_engine, target_decode_engine):
                print("ElasticMM LOG: Failed to set up real KV migration connection")
                return False
            
            # 执行真实KV迁移
            success = self._kv_migration_connector.migrate_request_kv(
                request_id, source_decode_engine, target_decode_engine
            )
            
            if success:
                print(f"ElasticMM LOG: Real KV cache migration completed: {request_id}")
            else:
                print(f"ElasticMM LOG: Real KV cache migration failed: {request_id}")
            
            return success
            
        except Exception as e:
            print(f"ElasticMM LOG: Real KV cache migration failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_migration_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """获取迁移状态"""
        return self._kv_migration_connector.get_migration_status(request_id)
    
    def cleanup_migration(self, request_id: str) -> None:
        """清理迁移状态"""
        self._kv_migration_connector.cleanup_migration(request_id)
    
    # ============================================================================
    # 节点级别的KV迁移功能 - 新增核心功能
    # ============================================================================
    
    def migrate_node_kv_cache(self, source_node: str, target_nodes: List[str], 
                             migration_strategy: str = "round_robin") -> bool:
        """
        节点级别的KV缓存批量迁移
        
        当一个节点需要被撤销或切换身份时，将该节点上所有的KV cache迁移到其他节点
        
        Args:
            source_node: 源节点名称
            target_nodes: 目标节点列表
            migration_strategy: 迁移策略 ("round_robin", "load_balance", "single_target")
            
        Returns:
            bool: 迁移是否成功
        """
        print(f"ElasticMM LOG: Starting node-level KV migration: {source_node} -> {target_nodes}")
        print(f"ElasticMM LOG: Migration strategy: {migration_strategy}")
        
        try:
            # 检查源节点是否存在
            if source_node not in self._engines:
                print(f"ElasticMM LOG: Source node {source_node} does not exist")
                return False
            
            # 检查目标节点是否存在
            for target_node in target_nodes:
                if target_node not in self._engines:
                    print(f"ElasticMM LOG: Target node {target_node} does not exist")
                    return False
            
            # 执行节点级迁移
            success = self._kv_migration_connector.migrate_node_kv_cache(
                source_node, target_nodes, migration_strategy
            )
            
            if success:
                print(f"ElasticMM LOG: Node-level KV migration completed successfully: {source_node}")
            else:
                print(f"ElasticMM LOG: Node-level KV migration failed: {source_node}")
            
            return success
            
        except Exception as e:
            print(f"ElasticMM LOG: Node-level KV migration failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_node_kv_info(self, node_name: str) -> Dict[str, Any]:
        """获取节点的KV缓存信息"""
        try:
            return self._kv_migration_connector.get_node_kv_info(node_name)
        except Exception as e:
            print(f"ElasticMM LOG: Failed to get node KV info: {e}")
            return {}
    
    def get_node_active_requests(self, node_name: str) -> List[str]:
        """获取节点上的活跃请求"""
        try:
            return self._kv_migration_connector.get_node_active_requests(node_name)
        except Exception as e:
            print(f"ElasticMM LOG: Failed to get node active requests: {e}")
            return []
    
    def get_all_nodes_kv_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有节点的KV缓存状态"""
        status = {}
        for node_name in self._engines:
            try:
                node_kv_info = self.get_node_kv_info(node_name)
                active_requests = self.get_node_active_requests(node_name)
                
                status[node_name] = {
                    'kv_info': node_kv_info,
                    'active_requests': active_requests,
                    'request_count': len(active_requests)
                }
            except Exception as e:
                print(f"ElasticMM LOG: Failed to get node {node_name} status: {e}")
                status[node_name] = {'error': str(e)}
        
        return status
    
    def stop_engine(self, name: str) -> None:
        """Stop a specific engine"""
        if name not in self._engines:
            print(f"Engine '{name}' not found")
            return
        
        actor_ref = self._engines[name]
        
        # Get GPU ID before stopping
        try:
            status = ray.get(actor_ref.get_status.remote())
            gpu_id = status["gpu_id"]
        except Exception:
            gpu_id = None
        
        # Stop the engine
        ray.get(actor_ref.stop.remote())
        ray.kill(actor_ref)
        
        # Free the GPU
        if gpu_id is not None:
            self._mark_gpu_free(gpu_id)
        
        del self._engines[name]
        print(f"Stopped engine '{name}'")
    
    def stop_all(self) -> None:
        """Stop all engines"""
        for name in list(self._engines.keys()):
            self.stop_engine(name)
    
    def force_cleanup(self):
        """Force cleanup all engines and Ray resources"""
        print("Force cleaning up all engines...")
        
        # Stop all engines gracefully first
        for name in list(self._engines.keys()):
            try:
                print(f"Stopping engine: {name}")
                ray.get(self._engines[name].stop.remote())
                ray.kill(self._engines[name])
            except Exception as e:
                print(f"Error stopping engine {name}: {e}")
        
        # Clear the engines dictionary
        self._engines.clear()
        
        # Reset GPU usage
        self._gpu_usage = {i: False for i in range(self._num_gpus)}
        
        # Force garbage collection
        import gc
        gc.collect()
        
        print("Force cleanup completed")
    
    def get_engine_status(self, name: str) -> Optional[Dict]:
        """Get status of a specific engine"""
        if name not in self._engines:
            return None
        
        try:
            actor_ref = self._engines[name]
            return ray.get(actor_ref.get_status.remote())
        except Exception as e:
            print(f"Error getting status for engine '{name}': {e}")
            return None
    
    def list_engines(self) -> List[str]:
        """List all engine names"""
        return list(self._engines.keys())
    
    def get_gpu_usage(self) -> Dict[int, bool]:
        """Get GPU usage status"""
        return self._gpu_usage.copy()
    
    def wait_for_engines(self, timeout: int = 120) -> bool:
        """Wait for all engines to be ready with health checks"""
        import time
        import requests
        start_time = time.time()
        
        print("Waiting for engines to be ready...")
        
        while time.time() - start_time < timeout:
            all_ready = True
            for name in self._engines:
                status = self.get_engine_status(name)
                if not status or not status.get("is_running", False):
                    all_ready = False
                    continue
                
                # Additional health check: try to connect to the HTTP endpoint
                try:
                    http_port = status.get("http_port")
                    if http_port:
                        health_url = f"http://127.0.0.1:{http_port}/health"
                        response = requests.get(health_url, timeout=2)
                        if response.status_code != 200:
                            all_ready = False
                            print(f"Engine {name} HTTP health check failed")
                    else:
                        all_ready = False
                except Exception as e:
                    all_ready = False
                    print(f"Engine {name} not ready: {e}")
            
            if all_ready:
                print("ElasticMM LOG: All engines are ready and healthy!")
                return True
            
            # Only print waiting message every 10 seconds
            elapsed = int(time.time() - start_time)
            if elapsed % 10 == 0:
                print(f"⏳ Waiting for engines... ({elapsed}s elapsed)")
            time.sleep(3)
        
        print(f"Timeout waiting for engines to be ready after {timeout}s")
        return False
    
    def get_engine_addresses(self) -> Dict[str, str]:
        """Get HTTP addresses of all running engines"""
        addresses = {}
        for name in self._engines:
            status = self.get_engine_status(name)
            if status and status.get("is_running", False):
                addresses[name] = f"{self._env.get('HOST', '127.0.0.1')}:{status['http_port']}"
        return addresses
    
    def stop_engine_sync(self, name: str, wait_timeout: int = 10) -> bool:
        """Stop an engine and wait for complete cleanup"""
        if name not in self._engines:
            print(f"Engine {name} not found")
            return False
        
        print(f"ElasticMM LOG: Stopping engine {name}...")
        
        # Stop the engine
        success = self.stop_engine(name)
        if not success:
            return False
        
        # Wait for complete cleanup
        import time
        start_time = time.time()
        
        while time.time() - start_time < wait_timeout:
            # Check if the engine is still running
            status = self.get_engine_status(name)
            if not status or not status.get("is_running", False):
                print(f"ElasticMM LOG: Engine {name} stopped successfully")
                return True
            
            time.sleep(1)
        
        print(f"ElasticMM LOG: Engine {name} may not have stopped completely")
        return False
    
    def switch_configuration(self, old_engines: List[str], new_configs: List[tuple]) -> bool:
        """
        Switch configuration by stopping old engines and starting new ones
        
        Args:
            old_engines: List of engine names to stop
            new_configs: List of (name, config) tuples for new engines
            
        Returns:
            bool: True if all operations successful
        """
        print(f"ElasticMM LOG: Switching configuration: stopping {len(old_engines)} engines, starting {len(new_configs)} engines")
        
        # Phase 1: Stop old engines synchronously
        for name in old_engines:
            if not self.stop_engine_sync(name, wait_timeout=15):
                print(f"ElasticMM LOG: Failed to stop engine {name}")
                return False
        
        # Phase 2: Wait additional time for GPU memory cleanup
        print("⏳ Waiting for GPU memory cleanup...")
        time.sleep(3)
        
        # Phase 3: Start new engines
        for name, config in new_configs:
            if not self.start_engine(config, name):
                print(f"ElasticMM LOG: Failed to start engine {name}")
                return False
        
        # Phase 4: Wait for all new engines to be ready
        print("⏳ Waiting for new engines to be ready...")
        return self.wait_for_engines(timeout=200)