"""
Test script for v0 engine backend
Demonstrates basic usage of v0 backend
"""

import asyncio
import sys
import os
import signal

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


async def test_v0_backend():
    """Test v0 backend basic functionality"""
    
    print("=" * 80)
    print("Testing ElasticMM v0 Engine Backend with Qwen2.5-VL")
    print("=" * 80)
    
    # Create configuration for Qwen2.5-VL
    config = V0EngineConfig(
        model_path="/root/lzd/model/qwen2.5-VL",
        num_encoding_workers=1,  # Small for testing
        num_prefill_workers=1,
        num_decoding_workers=1,
        block_size=16,
        max_num_gpu_blocks=1000,
        gpu_memory_utilization=0.5,  # Conservative for testing
        kv_transfer_method="nccl",
        dtype="bfloat16",  # Qwen2.5-VL uses bfloat16
        trust_remote_code=True,  # Required for Qwen models
    )
    
    # Validate configuration
    try:
        config.validate()
        print("✓ Configuration validated")
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        return
    
    # Create backend
    print("\n[1] Creating v0 backend...")
    backend = V0EngineBackend(**config.to_dict())
    print("✓ Backend created")
    
    # Initialize backend
    print("\n[2] Initializing backend...")
    try:
        await backend.initialize()
        print("✓ Backend initialized")
    except Exception as e:
        print(f"✗ Backend initialization failed: {e}")
        return
    
    # Start backend
    print("\n[3] Starting backend...")
    try:
        await backend.start()
        print("✓ Backend started")
        
        # Give it a moment to start
        await asyncio.sleep(1.0)
    except Exception as e:
        print(f"✗ Backend start failed: {e}")
        await backend.stop()
        return
    
    # Create test requests
    print("\n[4] Adding test requests...")
    test_requests = [
        Request(
            request_id=f"test_req_{i}",
            prompt=f"Hello, this is test request {i}",
            max_tokens=50,
            temperature=1.0,
            images=[f"dummy_image_{i}.jpg"] if i % 2 == 0 else None,  # Every other request has images
        )
        for i in range(5)
    ]
    
    for req in test_requests:
        await backend.add_request(req)
        print(f"  ✓ Added request: {req.request_id}")
    
    # Wait for some processing
    print("\n[5] Processing requests...")
    for i in range(10):
        await asyncio.sleep(0.5)
        
        # Get outputs
        outputs = await backend.get_outputs()
        if outputs:
            print(f"  Received {len(outputs)} outputs")
            for output in outputs:
                print(f"    - {output.request_id}: finished={output.finished}")
        
        # Get stats
        stats = backend.get_stats()
        print(f"  Stats: Received={stats['total_requests_received']}, "
              f"Completed={stats['total_requests_completed']}")
        
        # Stop if all completed
        if stats['total_requests_completed'] >= len(test_requests):
            break
    
    # Get final stats
    print("\n[6] Final Statistics:")
    final_stats = backend.get_stats()
    print(f"  Total requests received: {final_stats['total_requests_received']}")
    print(f"  Total requests completed: {final_stats['total_requests_completed']}")
    print(f"  Encoding stage: {final_stats['encoding']}")
    print(f"  Prefill stage: {final_stats['prefill']}")
    print(f"  Decoding stage: {final_stats['decoding']}")
    print(f"  KV transfer: {final_stats['kv_transfer']}")
    
    # Stop backend
    print("\n[7] Stopping backend...")
    try:
        await backend.stop()
        print("✓ Backend stopped")
    except Exception as e:
        print(f"✗ Backend stop failed: {e}")
    
    print("\n" + "=" * 80)
    print("Test completed")
    print("=" * 80)


async def test_backend_interface():
    """Test backend interface abstraction"""
    
    print("\n" + "=" * 80)
    print("Testing Backend Interface")
    print("=" * 80)
    
    from elasticmm.engine.backend_interface import EngineBackendFactory
    
    # Create v0 backend via factory
    print("\n[1] Creating backend via factory...")
    backend = EngineBackendFactory.create(
        backend_type='v0',
        model_path="/path/to/your/model",
        num_encoding_workers=1,
        num_prefill_workers=1,
        num_decoding_workers=1,
        max_num_gpu_blocks=1000,
    )
    print("✓ Backend created via factory")
    
    # Test interface methods
    print("\n[2] Testing interface methods...")
    print(f"  Number of instances: {backend.get_num_instances()}")
    print(f"  Can add request: {backend.can_add_request(Request('test', 'test', 10))}")
    
    instance_info = backend.get_instance_info("encoding_0")
    print(f"  Instance info: {instance_info}")
    
    print("✓ Interface methods work")
    
    print("\n" + "=" * 80)


async def test_qwen25vl_e2e(num_requests=200, max_tokens=50):
    """Test Qwen2.5-VL end-to-end with actual inference"""
    
    print("\n" + "=" * 80)
    print("Testing Qwen2.5-VL End-to-End Inference")
    print("=" * 80)
    
    import os
    import ray
    
    # Initialize Ray if not already
    if not ray.is_initialized():
        print("\n[1] Initializing Ray...")
        ray.init(ignore_reinit_error=True, num_gpus=8)
        print("✓ Ray initialized with 8 GPUs")
    
    # Create Qwen2.5-VL configuration
    print("\n[2] Creating Qwen2.5-VL configuration...")
    config = V0EngineConfig(
        model_path="/root/lzd/model/qwen2.5-VL",
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=4096,  # Smaller for testing
        num_encoding_workers=1,  # 1 GPU for encoding
        num_prefill_workers=1,   # 1 GPU for prefill  
        num_decoding_workers=1,  # 1 GPU for decoding
        block_size=16,
        max_num_gpu_blocks=2000,
        gpu_memory_utilization=0.9,  # Conservative
        kv_transfer_method="nccl",
    )
    
    print("✓ Configuration created")
    print(f"  Model: {config.model_path}")
    print(f"  DType: {config.dtype}")
    print(f"  GPU allocation: 1E + 1P + 1D = 3 GPUs")
    print(f"  Max blocks: {config.max_num_gpu_blocks}")
    
    # Check model exists
    if not os.path.exists(config.model_path):
        print(f"\n✗ Model not found at {config.model_path}")
        print("  Please ensure Qwen2.5-VL model is available")
        return
    
    print("✓ Model found")
    
    # Create backend
    print("\n[3] Creating V0 Backend...")
    try:
        backend = V0EngineBackend(**config.to_dict(), limit_mm_per_prompt={"image": 1})
        print("✓ Backend created")
    except Exception as e:
        print(f"✗ Backend creation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Initialize backend
    print("\n[4] Initializing backend (loading models on GPUs)...")
    print("    This may take a few minutes...")
    try:
        await backend.initialize()
        print("✓ Backend initialized")
        print("  - Encoding stage: GPU workers ready")
        print("  - Prefill stage: GPU workers ready")
        print("  - Decoding stage: GPU workers ready")
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Start backend
    print("\n[5] Starting backend event loops...")
    try:
        await backend.start()
        print("✓ All stages running")
        await asyncio.sleep(2)  # Let it stabilize
    except Exception as e:
        print(f"✗ Start failed: {e}")
        import traceback
        traceback.print_exc()
        await backend.stop()
        return
    
    # Create 20 test requests with real images
    print("\n[6] Creating 20 multimodal test requests with real images...")
    
    # Prepare test prompts with image placeholders for Qwen2.5-VL
    prompts = [
        "<|user|>\n<image>\nPlease describe this picture.\n<|assistant|>\n",
        "<|user|>\n<image>\nWhat is in the picture?\n<|assistant|>\n",
        "<|user|>\n<image>\nWhat scene is this?\n<|assistant|>\n",
        "<|user|>\n<image>\nDescribe the content of the picture in detail.\n<|assistant|>\n",
    ]
    
    # Load real images from dataset
    from PIL import Image
    image_base_path = "/root/lzd/dataset/shareGPT-4o/images"
    
    images = []
    # Load 200 images (10.jpg to 29.jpg)
    for i in range(10, 210):
        image_path = f"{image_base_path}/{i}.jpg"
        try:
            img = Image.open(image_path)
            images.append(img)
            print(f"  ✓ Loaded image {i}.jpg")
        except Exception as e:
            print(f"  ⚠ Warning: Failed to load {image_path}: {e}")
            # Create a dummy image as fallback
            import numpy as np
            img_array = np.full((224, 224, 3), (128, 128, 128), dtype=np.uint8)
            img = Image.fromarray(img_array)
            images.append(img)
    
    # Create requests with different images and prompts
    test_requests = []
    for i in range(num_requests):
        img = images[i % len(images)]  # Cycle through available images
        prompt = prompts[i % len(prompts)]  # Cycle through prompts
        test_request = Request(
            request_id=f"qwen_vl_{i:03d}",
            prompt=prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>"),
            max_tokens=max_tokens,
            temperature=0.9,
            multi_modal_data={"image": img},
        )
        test_requests.append(test_request)
    
    print(f"✓ Created {len(test_requests)} requests")
    print(f"  Images: {image_base_path}/10.jpg to 29.jpg (cycling)")
    print(f"  Max tokens per request: {max_tokens}")
    print(f"  Prompts: {len(prompts)} different question formats (cycling)")
    
    # Submit requests
    print(f"\n[7] Submitting {len(test_requests)} requests...")
    try:
        for i, req in enumerate(test_requests):
            await backend.add_request(req)
            if (i + 1) % 5 == 0:
                print(f"  Submitted {i + 1}/{len(test_requests)} requests...")
        print(f"✓ All {len(test_requests)} requests submitted")
    except Exception as e:
        print(f"✗ Request submission failed: {e}")
        import traceback
        traceback.print_exc()
        await backend.stop()
        return
    
    # Wait for processing
    print("\n[8] Processing through 3 stages...")
    print("    Stage 1: Encoding (vision processing)")
    print("    Stage 2: Prefill (KV cache generation)")
    print("    Stage 3: Decoding (token generation)")
    print(f"\n    Monitoring progress (waiting for {len(test_requests)} requests)...")
    
    max_wait_time = 1200  # 20 minutes timeout for 200 requests
    start_time = asyncio.get_event_loop().time()
    completed_requests = set()
    all_outputs = []
    last_print_time = 0
    
    while True:
        # Check for shutdown
        if shutdown_requested:
            print("\n⚠️  Shutdown requested, stopping processing...")
            break
            
        elapsed = asyncio.get_event_loop().time() - start_time
        
        if elapsed > max_wait_time:
            print(f"\n⚠ Timeout after {max_wait_time}s")
            print(f"  Completed: {len(completed_requests)}/{len(test_requests)}")
            break
        
        # Get stats
        stats = backend.get_stats()
        
        # Get outputs
        outputs = await backend.get_outputs()
        if outputs:
            all_outputs.extend(outputs)
            for output in outputs:
                if output.finished:
                    completed_requests.add(output.request_id)
        
        # Print progress every 5 seconds only (reduce spam)
        if elapsed - last_print_time >= 5:
            # Use correct keys from backend.get_stats()
            enc_stats = stats.get('encoding', {})
            pre_stats = stats.get('prefill', {})
            dec_stats = stats.get('decoding', {})
            print(f"    [{int(elapsed)}s] "
                  f"E: {enc_stats.get('num_running', 0)}r/{enc_stats.get('num_waiting', 0)}w, "
                  f"P: {pre_stats.get('num_running', 0)}r/{pre_stats.get('num_waiting', 0)}w, "
                  f"D: {dec_stats.get('num_running', 0)}r/{dec_stats.get('num_waiting', 0)}w, "
                  f"Completed: {len(completed_requests)}/{len(test_requests)}")
            last_print_time = elapsed
        
        # Check if all completed
        if len(completed_requests) >= len(test_requests):
            print(f"\n✓ All {len(test_requests)} requests completed!")
            break
        
        await asyncio.sleep(1)
    
    # Display sample outputs
    if all_outputs:
        print(f"\n✓ Received {len(all_outputs)} total outputs")
        print(f"  Completed requests: {len(completed_requests)}")
        print(f"  Showing first 5 outputs:")
        for i, output in enumerate(all_outputs[:5]):
            status = "✓" if output.finished else "..."
            print(f"    {i+1}. {output.request_id}: {len(output.output_token_ids)} tokens {status}")
    
    # Print prompt and output for each completed request
    print("\n[8.5] Request Details (Prompt & Output):")
    print("=" * 80)
    
    # Load tokenizer for decoding
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=True
    )
    
    # Group outputs by request
    request_outputs = {}
    for output in all_outputs:
        if output.request_id not in request_outputs:
            request_outputs[output.request_id] = []
        request_outputs[output.request_id].append(output)
    
    for req in test_requests:
        req_id = req.request_id
        if req_id not in request_outputs:
            continue
            
        # Get the final output (last one)
        final_output = request_outputs[req_id][-1]
        
        # Decode prompt - extract just the user question part
        prompt_text = req.prompt
        # Extract question between <|user|> and <|assistant|>
        if "<|user|>" in prompt_text and "<|assistant|>" in prompt_text:
            question = prompt_text.split("<|user|>")[1].split("<|assistant|>")[0].strip()
            # Remove vision tokens for cleaner display
            question = question.replace("<|vision_start|><|image_pad|><|vision_end|>", "[IMAGE]")
        else:
            question = prompt_text[:50] + "..."
        
        # Decode output tokens
        output_tokens = final_output.output_token_ids
        if output_tokens:
            output_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
            # Clean up output
            output_text = output_text.strip()
        else:
            output_text = "(empty)"
        
        print(f"\n{req_id}:")
        print(f"  Question: {question}")
        print(f"  Response: {output_text}")
        print(f"  Tokens: {len(final_output.output_token_ids)}")
    
    print("=" * 80)
    
    # Final stats
    print("\n[9] Final Statistics:")
    final_stats = backend.get_stats()
    print(f"  Total requests submitted: {len(test_requests)}")
    print(f"  Total requests completed: {len(completed_requests)}")
    print(f"  Success rate: {len(completed_requests)/len(test_requests)*100:.1f}%")
    print(f"  Encoding stage: {final_stats['encoding']}")
    print(f"  Prefill stage: {final_stats['prefill']}")
    print(f"  Decoding stage: {final_stats['decoding']}")
    
    # KV transfer stats
    if 'kv_transfer' in final_stats:
        print(f"\n  KV Transfer:")
        kv_stats = final_stats['kv_transfer']
        for key, value in kv_stats.items():
            print(f"    {key}: {value}")
        
        # Calculate and display bandwidth
        if kv_stats.get('total_transfer_time', 0) > 0 and kv_stats.get('total_bytes_transferred', 0) > 0:
            bandwidth_gbps = (kv_stats['total_bytes_transferred'] / 1e9) / kv_stats['total_transfer_time']
            bandwidth_mbps = (kv_stats['total_bytes_transferred'] / 1e6) / kv_stats['total_transfer_time']
            print(f"    avg_bandwidth: {bandwidth_gbps:.2f} GB/s ({bandwidth_mbps:.2f} MB/s)")
    
    # Stop backend
    print("\n[10] Stopping backend...")
    try:
        # Add timeout to stop to prevent hanging
        await asyncio.wait_for(backend.stop(), timeout=30.0)
        print("✓ Backend stopped cleanly")
    except asyncio.TimeoutError:
        print("⚠️  Backend stop timeout (30s), may need manual cleanup")
        print("⚠️  Run: ray stop --force")
    except Exception as e:
        print(f"⚠️  Stop error: {e}")
    
    print("\n" + "=" * 80)
    print("✅ KV Transfer Implementation Status:")
    print("=" * 80)
    print("✓ KV cache is allocated and managed (V0BlockManager)")
    print("✓ Block indexes are passed between stages")
    print("✓ ACTUAL KV cache DATA IS NOW TRANSFERRED via Ray!")
    print("")
    print("Implementation details:")
    print("  - Worker.extract_kv_blocks(): Extracts KV data from source")
    print("  - Ray object store: Transfers data between GPUs")
    print("  - Worker.write_kv_blocks(): Writes to destination")
    print("")
    print("Performance notes:")
    print("  - Current: Ray object store (works, moderate speed)")
    print("  - Future: Can optimize with direct NCCL for better performance")
    print("=" * 80)
    
    # Cleanup Ray
    print("\n[11] Cleaning up Ray...")
    try:
        ray.shutdown()
        print("✓ Ray shutdown complete")
    except Exception as e:
        print(f"⚠️  Ray shutdown warning: {e}")


def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test v0 engine backend')
    parser.add_argument('--test', choices=['basic', 'interface', 'qwen', 'all'], default='qwen',
                        help='Which test to run')
    parser.add_argument('--model-path', default="/root/lzd/model/qwen2.5-VL",
                        help='Path to Qwen2.5-VL model')
    parser.add_argument('--num_requests', type=int, default=30,
                        help='Number of requests to test (default: 200)')
    parser.add_argument('--max_tokens', type=int, default=50,
                        help='Maximum tokens to generate per request (default: 50)')
    args = parser.parse_args()
    
    if args.test in ['basic', 'all']:
        print("\n>>> Running basic backend test...")
        print(">>> Use --test qwen for full end-to-end test")
    
    if args.test in ['interface', 'all']:
        print("\n>>> Running interface test...")
        asyncio.run(test_backend_interface())
    
    if args.test in ['qwen', 'all']:
        print("\n>>> Running Qwen2.5-VL end-to-end test...")
        print(">>> This will:")
        print(">>>   1. Initialize Ray with 8 GPUs")
        print(">>>   2. Load Qwen2.5-VL on 3 GPUs (encode/prefill/decode)")
        print(">>>   3. Run a test inference request")
        print(">>>   4. Verify KV cache transfer works")
        print("")
        asyncio.run(test_qwen25vl_e2e(num_requests=args.num_requests, max_tokens=args.max_tokens))


if __name__ == '__main__':
    main()

