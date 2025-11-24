import aiohttp
import socket
import threading
import time
import uuid
from typing import Any, Dict, Optional

import msgpack
import zmq
from quart import Quart, make_response, request
from elasticmm.core.scheduler import Scheduler, EMPScheduler
import os
import asyncio


DEFAULT_PING_SECONDS = 15  # 适中的心跳超时时间


# Global references for cleanup
_zmq_context = None
_zmq_socket = None
_zmq_listener_thread = None

def cleanup_zmq_resources():
    """Clean up ZMQ resources"""
    global _zmq_context, _zmq_socket, _zmq_listener_thread
    try:
        if _zmq_socket is not None:
            _zmq_socket.close()
            _zmq_socket = None
        if _zmq_context is not None:
            _zmq_context.term()
            _zmq_context = None
        if _zmq_listener_thread is not None and _zmq_listener_thread.is_alive():
            # Thread will stop when socket is closed
            _zmq_listener_thread = None
    except Exception as e:
        print(f"ElasticMM LOG: Error cleaning up ZMQ resources: {e}")

def create_disagg_proxy_app(service_discovery_host: str = "0.0.0.0",
                            service_discovery_port: int = 30002,
                            api_host: str = "0.0.0.0",
                            api_port: int = 10001,
                            fanout_prefill: int = 1,
                            fanout_decode: int = 1,
                            scheduler: Optional[EMPScheduler] = None) -> Quart:
    app = Quart(__name__)
    
    # Use provided scheduler or create a new one
    if scheduler is None:
        scheduler = Scheduler()
    
    print(f"ElasticMM LOG: Using scheduler: {type(scheduler).__name__}")
    
    prefill_cv = threading.Condition()
    decode_cv = threading.Condition()

    def _remove_oldest_instances(instances: Dict[str, Any]) -> None:
        oldest_key = next(iter(instances), None)
        while oldest_key is not None:
            value = instances[oldest_key]
            if value[1] > time.time():
                break
            instances.pop(oldest_key, None)
            oldest_key = next(iter(instances), None)

    def _listen_for_register(poller, router_socket):
        while True:
            socks = dict(poller.poll())
            if router_socket in socks:
                try:
                    # For REQ-REP pattern, we get the message directly
                    message = router_socket.recv()
                    
                    # Parse msgpack data
                    try:
                        data = msgpack.loads(message)
                    except Exception as parse_error:
                        print(f"ElasticMM LOG: Heartbeat message parsing failed: {parse_error}")
                        try:
                            router_socket.send(b"ERROR")
                        except:
                            pass
                        continue
                    
                    role = data.get("type", "")
                    if role in ("P", "D"):
                        # Extract instance_id if present (V0 backend), otherwise None (V1 backend)
                        instance_id = data.get("instance_id", None)
                        
                        scheduler.heartbeat(
                            http_address=data["http_address"],
                            zmq_address=data["zmq_address"],
                            role=role,
                            instance_id=instance_id,  # Pass instance_id if present (V0) or None (V1)
                        )
                        # Only print registration message occasionally (reduce log spam)
                        # Print only every 12th heartbeat (about every 2 minutes with 10s interval)
                        if not hasattr(_listen_for_register, '_heartbeat_count'):
                            _listen_for_register._heartbeat_count = {}
                        # Use instance_id for counting if available (V0), otherwise use role (V1)
                        count_key = instance_id if instance_id else role
                        count = _listen_for_register._heartbeat_count.get(count_key, 0) + 1
                        _listen_for_register._heartbeat_count[count_key] = count
                        if count % 12 == 1:
                            if instance_id:
                                print(f"ElasticMM LOG: Registered {role} engine: {instance_id} ({data['http_address']}) (heartbeat #{count})")
                            else:
                                print(f"ElasticMM LOG: Registered {role} engine: {data['http_address']} (heartbeat #{count})")
                        # Send response back to client (REP socket)
                        router_socket.send(b"OK")
                    else:
                        print(f"ElasticMM LOG: Unknown role type: {role}")
                        try:
                            router_socket.send(b"ERROR")
                        except:
                            pass
                except Exception as e:
                    print(f"Error processing heartbeat: {e}")
                    # Send error response
                    try:
                        router_socket.send(b"ERROR")
                    except:
                        pass

    def start_service_discovery(hostname, port):
        global _zmq_context, _zmq_socket, _zmq_listener_thread
        
        if not hostname:
            hostname = socket.gethostname()
        if port == 0:
            raise ValueError("Port cannot be 0")
        
        # Clean up any existing resources first (in case of restart in same process)
        cleanup_zmq_resources()
        
        # Try multiple times with increasing delays
        max_retries = 3
        retry_delays = [1, 2, 3]  # seconds
        
        for attempt in range(max_retries):
            try:
                _zmq_context = zmq.Context()
                _zmq_socket = _zmq_context.socket(zmq.REP)  # Use REP for REQ-REP pattern
                
                # Set socket options for better port reuse
                _zmq_socket.setsockopt(zmq.LINGER, 0)  # Don't wait for messages on close
                
                # Try to bind
                _zmq_socket.bind(f"tcp://{hostname}:{port}")
                print(f"✓ ZMQ service discovery bound to tcp://{hostname}:{port}")
                break  # Success, exit retry loop
                
            except zmq.ZMQError as e:
                if e.errno == zmq.EADDRINUSE:
                    # Clean up before retry
                    cleanup_zmq_resources()
                    
                    if attempt < max_retries - 1:
                        delay = retry_delays[attempt]
                        print(f"⚠️  ZMQ port {port} is in use (attempt {attempt + 1}/{max_retries}). Waiting {delay}s...")
                        import time
                        time.sleep(delay)
                        continue
                    else:
                        # Last attempt failed
                        print(f"❌ Failed to bind ZMQ port {port} after {max_retries} attempts.")
                        print(f"   The port may be held by another process or in TIME_WAIT state.")
                        print(f"   Please wait a few seconds and try again, or kill the process using:")
                        print(f"   lsof -ti:{port} | xargs kill -9  (if lsof is available)")
                        cleanup_zmq_resources()
                        raise RuntimeError(f"ZMQ port {port} is in use and could not be freed after {max_retries} attempts")
                else:
                    # Other ZMQ error
                    print(f"❌ ZMQ bind error: {e}")
                    cleanup_zmq_resources()
                    raise
        
        poller = zmq.Poller()
        poller.register(_zmq_socket, zmq.POLLIN)
        _zmq_listener_thread = threading.Thread(
            target=_listen_for_register, args=[poller, _zmq_socket], daemon=True
        )
        _zmq_listener_thread.start()
        return _zmq_listener_thread

    AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=300, connect=30, sock_read=120)  # 增加超时时间

    async def forward_request(url, data, request_id):
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            headers = {"X-Request-Id": request_id}
            async with session.post(url=url, json=data, headers=headers) as resp:
                async for chunk in resp.content.iter_chunked(1024):
                    yield chunk

    def random_uuid() -> str:
        return str(uuid.uuid4().hex)

    @app.route("/v1/completions", methods=["POST"])
    @app.route("/v1/chat/completions", methods=["POST"])
    async def handle_request():
        original_request_data = await request.get_json()

        # Check if this is v0 backend (which doesn't use HTTP forwarding)
        is_v0_backend = False
        backend = None
        if isinstance(scheduler, EMPScheduler) and hasattr(scheduler, 'backend') and scheduler.backend:
            backend = scheduler.backend
            # Check if it's v0 backend by checking the class name
            if 'V0EngineBackend' in type(backend).__name__:
                is_v0_backend = True

        if is_v0_backend:
            # V0 backend: directly call backend.add_request()
            from elasticmm.engine.v0.utils import Request as V0Request
            
            # Convert HTTP request to V0Request
            messages = original_request_data.get("messages", [])
            prompt = ""
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if isinstance(content, str):
                    prompt += f"{role}: {content}\n"
                elif isinstance(content, list):
                    # Handle multimodal content
                    for item in content:
                        if item.get("type") == "text":
                            prompt += f"{role}: {item.get('text', '')}\n"
            
            v0_request = V0Request(
                request_id=random_uuid(),
                prompt=prompt.strip(),
                max_tokens=original_request_data.get("max_tokens", 100),
                temperature=original_request_data.get("temperature", 0.7),
            )
            
            # Handle multimodal data if present
            # ⚠️  IMPORTANT: Only collect the FIRST image (Qwen2.5-VL limitation: 1 image per request)
            first_image = None
            print(f"ElasticMM LOG: Processing request, messages count: {len(original_request_data.get('messages', []))}")
            if "messages" in original_request_data:
                for msg_idx, msg in enumerate(original_request_data["messages"]):
                    if first_image:
                        break  # Stop after finding first image
                    
                    content = msg.get("content", "")
                    print(f"ElasticMM LOG: Message {msg_idx} content type: {type(content)}")
                    if isinstance(content, list):
                        print(f"ElasticMM LOG: Message {msg_idx} has {len(content)} content items")
                        for item_idx, item in enumerate(content):
                            item_type = item.get("type", "")
                            print(f"ElasticMM LOG: Content item {item_idx}: type={item_type}")
                            if item_type == "image_url":
                                # Extract image data
                                image_url = item.get("image_url", {}).get("url", "")
                                print(f"ElasticMM LOG: Found image_url: {image_url[:100]}...")
                                if image_url.startswith("data:image"):
                                    # Base64 encoded image
                                    import base64
                                    try:
                                        header, encoded = image_url.split(",", 1)
                                        image_data = base64.b64decode(encoded)
                                        first_image = image_data
                                        print(f"ElasticMM LOG: Successfully decoded base64 image, size: {len(image_data)} bytes")
                                        break  # Stop after first image
                                    except Exception as e:
                                        print(f"ElasticMM LOG: Error decoding image: {e}")
                                        import traceback
                                        traceback.print_exc()
                                        continue
                                elif image_url.startswith("http://") or image_url.startswith("https://"):
                                    # TODO: Handle HTTP URLs (download image)
                                    print(f"ElasticMM LOG: HTTP image URLs not yet supported: {image_url}")
                                    continue
                                elif os.path.exists(image_url):
                                    # Handle file path
                                    try:
                                        with open(image_url, 'rb') as f:
                                            image_data = f.read()
                                        first_image = image_data
                                        print(f"ElasticMM LOG: Successfully read image file, size: {len(image_data)} bytes")
                                        break  # Stop after first image
                                    except Exception as e:
                                        print(f"ElasticMM LOG: Error reading image file {image_url}: {e}")
                                        continue
            
            # Set multi_modal_data if we found an image
            # ⚠️  IMPORTANT: Qwen2.5-VL only supports 1 image per request
            if first_image:
                v0_request.multi_modal_data = {"image": first_image}
                #print(f"ElasticMM LOG: ✅ Set multi_modal_data with 1 image for request {v0_request.request_id}")
            else:
                print(f"ElasticMM LOG: ⚠️  No images found for request {v0_request.request_id}, this is a text-only request")
            
            # Check if backend is initialized
            if not backend.encoding_engine:
                print(f"ElasticMM LOG: V0 backend not initialized, encoding_engine is None")
                return {"error": "Backend not initialized"}, 503
            
            # Get request_id from v0_request
            request_id = v0_request.request_id
            
            # Add request to backend
            print(f"ElasticMM LOG: Adding V0 request {request_id} to backend")
            print(f"ElasticMM LOG: Backend type: {type(backend).__name__}")
            print(f"ElasticMM LOG: Backend encoding_engine: {backend.encoding_engine}")
            try:
                await backend.add_request(v0_request)
                print(f"ElasticMM LOG: V0 request {request_id} added to backend successfully, waiting for outputs...")
            except Exception as e:
                print(f"ElasticMM LOG: ❌ Error adding request to backend: {e}")
                import traceback
                traceback.print_exc()
                return {"error": f"Failed to add request: {str(e)}"}, 500
            
            # Stream outputs from backend
            # Note: v0 backend outputs are StepOutput objects with token_ids
            accumulated_text = ""
            max_wait_time = 300  # 5 minutes timeout
            start_time = time.time()
            
            async def stream_outputs():
                nonlocal accumulated_text
                last_output_len = 0
                finished = False
                
                # Get or create event for this request
                output_event = None
                if hasattr(backend, '_request_output_events'):
                    output_event = backend._request_output_events.get(request_id)
                
                # Use shorter sleep intervals to check for shutdown more frequently
                while time.time() - start_time < max_wait_time and not finished:
                    # Get outputs for this specific request (non-blocking)
                    request_outputs = await backend.get_outputs(request_id=request_id)
                    
                    if request_outputs:
                        print(f"ElasticMM LOG: Got {len(request_outputs)} outputs for request {request_id}")
                        for output in request_outputs:
                            # Decode token_ids to text
                            if backend.tokenizer and output.output_token_ids:
                                # Decode only new tokens
                                current_len = len(output.output_token_ids)
                                if current_len > last_output_len:
                                    new_tokens = output.output_token_ids[last_output_len:]
                                    if new_tokens:
                                        new_text = backend.tokenizer.decode(new_tokens, skip_special_tokens=True)
                                        accumulated_text += new_text
                                        last_output_len = current_len
                                        
                                        # Escape JSON special characters
                                        new_text_escaped = new_text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
                                        
                                        # Stream the new text
                                        yield f'data: {{"choices": [{{"delta": {{"content": "{new_text_escaped}"}}}}]}}\n\n'
                            
                            if output.finished:
                                # Request completed
                                finish_reason = getattr(output, 'finish_reason', None) or "stop"
                                finish_reason_escaped = finish_reason.replace('\\', '\\\\').replace('"', '\\"')
                                yield f'data: {{"choices": [{{"delta": {{}}, "finish_reason": "{finish_reason_escaped}"}}]}}\n\n'
                                yield 'data: [DONE]\n\n'
                                finished = True
                                break
                    else:
                        # Wait for outputs using event if available, otherwise poll
                        # Use shorter intervals (0.1s) to allow signal handling
                        if output_event:
                            try:
                                # Wait up to 0.1 seconds for new output (shorter for better responsiveness)
                                await asyncio.wait_for(output_event.wait(), timeout=0.1)
                                output_event.clear()  # Reset event for next output
                            except asyncio.TimeoutError:
                                pass  # Continue polling
                        else:
                            # Fallback to polling with short sleep
                            await asyncio.sleep(0.1)
                    
                    # Check for KeyboardInterrupt periodically (allow signal handling)
                    # This gives the event loop a chance to process signals
                    await asyncio.sleep(0)  # Yield control to event loop
                
                if not finished:
                    # Timeout
                    error_msg = "Request timeout"
                    yield f'data: {{"error": "{error_msg}"}}\n\n'
            
            response = await make_response(stream_outputs())
            response.headers["Content-Type"] = "text/event-stream"
            return response
        else:
            # V1 backend: use HTTP forwarding (original logic)
            prefill_request = original_request_data.copy()
            prefill_request["max_tokens"] = 1
            if "max_completion_tokens" in prefill_request:
                prefill_request["max_completion_tokens"] = 1

            # pick fanout sets
            prefill_nodes = scheduler.select_prefills(max(1, fanout_prefill))
            decode_nodes = scheduler.select_decodes(max(1, fanout_decode))
            
            # Debug: print scheduler state
            if not prefill_nodes or not decode_nodes:
                print(f"ElasticMM LOG: Scheduler debug - prefill_nodes: {len(prefill_nodes) if prefill_nodes else 0}, decode_nodes: {len(decode_nodes) if decode_nodes else 0}")
                if hasattr(scheduler, '_prefill') and hasattr(scheduler, '_decode'):
                    print(f"ElasticMM LOG: Scheduler internal state - _prefill: {len(scheduler._prefill)}, _decode: {len(scheduler._decode)}")
            
            if not prefill_nodes or not decode_nodes:
                print(f"ElasticMM LOG: No available nodes: prefill={len(prefill_nodes) if prefill_nodes else 0}, decode={len(decode_nodes) if decode_nodes else 0}")
                return {"error": "No available prefill/decode nodes"}, 503

            # For xpyd we issue the same prefill to selected prefill nodes; then decode to one or many
            # Here we choose 1 decode target (first) to simplify response streaming.
            decode_target = decode_nodes[0]
            
            # Create request_id with the first prefill node's address for KV transfer
            # All prefill nodes will use the same request_id to ensure KV consistency
            prefill_target = prefill_nodes[0]
            request_id = (
                f"___prefill_addr_{prefill_target.zmq_address}___decode_addr_"
                f"{decode_target.zmq_address}_{random_uuid()}"
            )

            # multicast prefill - all prefill nodes use the same request_id
            for p in prefill_nodes:
                async for _ in forward_request(
                    f"http://{p.http_address}{request.path}", prefill_request, request_id
                ):
                    continue

            # decode return; if fanout > 1, drain others in background
            decode_request = original_request_data
            decoders = [
                forward_request(
                    f"http://{d.http_address}{request.path}", decode_request, request_id
                ) for d in decode_nodes
            ]
            # background drain for non-primary decoders
            for g in decoders[1:]:
                async def _drain(gen):
                    async for _ in gen:
                        continue
                asyncio.create_task(_drain(g))
            response = await make_response(decoders[0])
            response.timeout = None
            return response

    # test-only registration endpoint
    if os.environ.get("PIPELINE_TEST_REG", "0") == "1":
        @app.route("/_register", methods=["POST"])
        async def _register():
            data = await request.get_json()
            scheduler.heartbeat(
                http_address=data["http_address"],
                zmq_address=data["zmq_address"],
                role=data["type"],
            )
            return {"ok": True}

    # 健康检查端点 - 始终可用
    @app.route("/health", methods=["GET"])
    async def health_check():
        """代理健康检查端点"""
        return {"status": "ok", "service": "disagg_proxy"}

    # start background service discovery
    start_service_discovery(service_discovery_host, service_discovery_port)
    return app

# class InferencePipeline:
#     """
#     推理流水线控制器
#     协调多阶段推理流程
#     """
    
#     def __init__(self, cache_manager):
#         self.cache_manager = cache_manager
#         self.active_requests = {}
        
#     async def process_multimodal_request(self, request: Request, 
#                                        instances: Dict[InferenceStage, List[Any]]) -> str:
#         """
#         处理多模态请求的完整流水线
#         对应论文2.1节的MLLM推理流水线
#         """
#         try:
#             # 阶段1: 图像编码（支持缓存）
#             visual_tokens = await self._handle_image_encoding(request, instances)
            
#             # 阶段2: 预填充
#             hidden_states, kv_cache = await self._handle_prefill(
#                 request, visual_tokens, instances
#             )
            
#             # 阶段3: 解码
#             result = await self._handle_decode(
#                 request, hidden_states, kv_cache, instances
#             )
            
#             return result
            
#         except Exception as e:
#             print(f"处理多模态请求失败: {e}")
#             return f"错误: {e}"
    
#     async def process_text_request(self, request: Request,
#                                  instances: Dict[InferenceStage, List[Any]]) -> str:
#         """处理纯文本请求的流水线"""
#         try:
#             # 跳过编码阶段，直接预填充和解码
#             hidden_states, kv_cache = await self._handle_prefill(
#                 request, None, instances
#             )
            
#             result = await self._handle_decode(
#                 request, hidden_states, kv_cache, instances
#             )
            
#             return result
            
#         except Exception as e:
#             print(f"处理文本请求失败: {e}")
#             return f"错误: {e}"
    
#     async def _handle_image_encoding(self, request: Request, 
#                                    instances: Dict) -> Optional[Any]:
#         """处理图像编码阶段，支持缓存优化"""
#         if not request.images:
#             return None
            
#         # 检查统一多模态缓存
#         image_hash = self._generate_image_hash(request.images)
#         cached_tokens = self.cache_manager.get_visual_tokens(image_hash)
        
#         if cached_tokens:
#             print(f"缓存命中: {image_hash}")
#             return cached_tokens
        
#         # 缓存未命中，执行编码
#         encode_instances = instances.get(InferenceStage.ENCODE, [])
#         if not encode_instances:
#             raise ValueError("没有可用的编码实例")
            
#         instance = encode_instances[0]  # 简单选择策略
#         _, visual_tokens = await instance.encode_images_nonblocking.remote(request)
        
#         # 缓存编码结果
#         if visual_tokens:
#             self.cache_manager.cache_visual_tokens(image_hash, visual_tokens)
        
#         return visual_tokens
    
#     def _generate_image_hash(self, images: List[Any]) -> str:
#         """为图像生成哈希值用于缓存"""
#         import hashlib
#         content = str([str(img) for img in images])
#         return hashlib.md5(content.encode()).hexdigest()

    # test-only registration endpoint
    if os.environ.get("PIPELINE_TEST_REG", "0") == "1":
        @app.route("/_register", methods=["POST"])
        async def _register():
            data = await request.get_json()
            scheduler.heartbeat(
                http_address=data["http_address"],
                zmq_address=data["zmq_address"],
                role=data["type"],
            )
            return {"ok": True}

    # 健康检查端点 - 始终可用
    @app.route("/health", methods=["GET"])
    async def health_check():
        """代理健康检查端点"""
        return {"status": "ok", "service": "disagg_proxy"}

    # start background service discovery
    start_service_discovery(service_discovery_host, service_discovery_port)
    return app

# class InferencePipeline:
#     """
#     推理流水线控制器
#     协调多阶段推理流程
#     """
    
#     def __init__(self, cache_manager):
#         self.cache_manager = cache_manager
#         self.active_requests = {}
        
#     async def process_multimodal_request(self, request: Request, 
#                                        instances: Dict[InferenceStage, List[Any]]) -> str:
#         """
#         处理多模态请求的完整流水线
#         对应论文2.1节的MLLM推理流水线
#         """
#         try:
#             # 阶段1: 图像编码（支持缓存）
#             visual_tokens = await self._handle_image_encoding(request, instances)
            
#             # 阶段2: 预填充
#             hidden_states, kv_cache = await self._handle_prefill(
#                 request, visual_tokens, instances
#             )
            
#             # 阶段3: 解码
#             result = await self._handle_decode(
#                 request, hidden_states, kv_cache, instances
#             )
            
#             return result
            
#         except Exception as e:
#             print(f"处理多模态请求失败: {e}")
#             return f"错误: {e}"
    
#     async def process_text_request(self, request: Request,
#                                  instances: Dict[InferenceStage, List[Any]]) -> str:
#         """处理纯文本请求的流水线"""
#         try:
#             # 跳过编码阶段，直接预填充和解码
#             hidden_states, kv_cache = await self._handle_prefill(
#                 request, None, instances
#             )
            
#             result = await self._handle_decode(
#                 request, hidden_states, kv_cache, instances
#             )
            
#             return result
            
#         except Exception as e:
#             print(f"处理文本请求失败: {e}")
#             return f"错误: {e}"
    
#     async def _handle_image_encoding(self, request: Request, 
#                                    instances: Dict) -> Optional[Any]:
#         """处理图像编码阶段，支持缓存优化"""
#         if not request.images:
#             return None
            
#         # 检查统一多模态缓存
#         image_hash = self._generate_image_hash(request.images)
#         cached_tokens = self.cache_manager.get_visual_tokens(image_hash)
        
#         if cached_tokens:
#             print(f"缓存命中: {image_hash}")
#             return cached_tokens
        
#         # 缓存未命中，执行编码
#         encode_instances = instances.get(InferenceStage.ENCODE, [])
#         if not encode_instances:
#             raise ValueError("没有可用的编码实例")
            
#         instance = encode_instances[0]  # 简单选择策略
#         _, visual_tokens = await instance.encode_images_nonblocking.remote(request)
        
#         # 缓存编码结果
#         if visual_tokens:
#             self.cache_manager.cache_visual_tokens(image_hash, visual_tokens)
        
#         return visual_tokens
    
#     def _generate_image_hash(self, images: List[Any]) -> str:
#         """为图像生成哈希值用于缓存"""
#         import hashlib
#         content = str([str(img) for img in images])
#         return hashlib.md5(content.encode()).hexdigest()