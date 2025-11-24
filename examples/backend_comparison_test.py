#!/usr/bin/env python3
"""
ElasticMM Backend Comparison Test

Simple test script to compare v0 vs v1 backends using existing components.
Reuses DynamicRequestGenerator and follows online_demo.py pattern.
"""

import asyncio
import time
import argparse
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from elasticmm.system import create_default_system, create_custom_system

# Import DynamicRequestGenerator with fallback
try:
    from tests.dynamic_request_generator import DynamicRequestGenerator
except ImportError:
    # Fallback import method
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "dynamic_request_generator", 
        os.path.join(project_root, "tests", "dynamic_request_generator.py")
    )
    dynamic_request_generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dynamic_request_generator)
    DynamicRequestGenerator = dynamic_request_generator.DynamicRequestGenerator


class BackendComparisonTest:
    """Simple backend comparison test"""
    
    def __init__(self, backend_type: str = "v0", test_duration: int = 300):
        self.backend_type = backend_type
        self.test_duration = test_duration
        self.proxy_port = 10001
        self.system = None
        self.generator = None
        
    async def create_system(self):
        """Create system with specified backend"""
        if self.backend_type == "v0":
            # V0 backend with 8 GPUs (2 text + 6 multimodal)
            self.system = create_custom_system(
                total_gpus=8,
                text_gpus=2,
                multimodal_gpus=6,
                model_path="/root/lzd/model/qwen2.5-VL",
                backend_type="v0",
                proxy_port=self.proxy_port
            )
        else:
            # V1 backend with default configuration
            self.system = create_default_system()
            # Override proxy port to avoid conflicts
            self.system.proxy_port = self.proxy_port
        
        print(f"[{self.backend_type.upper()}] System created")
    
    async def start_system(self):
        """Start the system"""
        print(f"[{self.backend_type.upper()}] Starting system...")
        await self.system.start()
        print(f"[{self.backend_type.upper()}] System started successfully")
        
        # Wait for system to stabilize
        await asyncio.sleep(10)
    
    async def run_test(self):
        """Run the test using DynamicRequestGenerator"""
        print(f"[{self.backend_type.upper()}] Starting test...")
        
        # Create request generator (reuse existing component)
        self.generator = DynamicRequestGenerator(
            proxy_url=f"http://127.0.0.1:{self.proxy_port}/v1/chat/completions",
            image_dir="/root/lzd/test_images",  # Adjust path as needed
            text_data_path="/root/lzd/test_data.jsonl",  # Adjust path as needed
            model_name="/root/lzd/model/qwen2.5-VL"
        )
        
        # Run generator for specified duration
        print(f"[{self.backend_type.upper()}] Running test for {self.test_duration} seconds...")
        await self.generator.run(
            duration_seconds=self.test_duration,
            text_base_rate=2.0,
            multimodal_base_rate=3.0,
            text_variance=0.5,
            multimodal_variance=1.5
        )
        
        # Get statistics
        stats = {
            "total_requests": self.generator.total_requests_sent,
            "successful_requests": self.generator.successful_requests,
            "failed_requests": self.generator.failed_requests,
            "success_rate": (self.generator.successful_requests / self.generator.total_requests_sent * 100) if self.generator.total_requests_sent else 0
        }
        
        self._print_results(stats)
        return stats
    
    def _print_results(self, stats):
        """Print test results"""
        print(f"\n📊 {self.backend_type.upper()} Backend Results:")
        print(f"{'─'*40}")
        print(f"Total Requests:     {stats.get('total_requests', 0)}")
        print(f"Successful:         {stats.get('successful_requests', 0)}")
        print(f"Failed:             {stats.get('failed_requests', 0)}")
        print(f"Avg Response Time:  {stats.get('avg_response_time', 0):.3f}s")
        print(f"Throughput:         {stats.get('throughput_rps', 0):.2f} req/s")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.generator:
            self.generator._is_running = False
        
        if self.system:
            await self.system.stop()
            print(f"[{self.backend_type.upper()}] System stopped")


async def test_single_backend(backend_type: str, duration: int = 300):
    """Test a single backend"""
    tester = BackendComparisonTest(backend_type, duration)
    
    try:
        await tester.create_system()
        await tester.start_system()
        stats = await tester.run_test()
        return stats
    finally:
        await tester.cleanup()


async def test_both_backends(duration: int = 300):
    """Test both backends and compare"""
    print("🔄 Testing V0 Backend...")
    v0_stats = await test_single_backend("v0", duration)
    
    print("\n⏳ Waiting 30 seconds...")
    await asyncio.sleep(30)
    
    print("🔄 Testing V1 Backend...")
    v1_stats = await test_single_backend("v1", duration)
    
    # Simple comparison
    print(f"\n📈 Comparison Results:")
    print(f"{'Metric':<20} {'V0':<15} {'V1':<15}")
    print(f"{'─'*50}")
    print(f"{'Throughput (req/s)':<20} {v0_stats.get('throughput_rps', 0):<15.2f} {v1_stats.get('throughput_rps', 0):<15.2f}")
    print(f"{'Avg Response (s)':<20} {v0_stats.get('avg_response_time', 0):<15.3f} {v1_stats.get('avg_response_time', 0):<15.3f}")
    print(f"{'Total Requests':<20} {v0_stats.get('total_requests', 0):<15} {v1_stats.get('total_requests', 0):<15}")


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="ElasticMM Backend Comparison Test")
    parser.add_argument(
        "--backend", 
        choices=["v0", "v1", "both"], 
        default="v0",
        help="Backend type to test (default: v0)"
    )
    parser.add_argument(
        "--duration", 
        type=int, 
        default=300,
        help="Test duration in seconds (default: 300)"
    )
    
    args = parser.parse_args()
    
    print(f"🎯 ElasticMM Backend Test")
    print(f"Backend: {args.backend}")
    print(f"Duration: {args.duration}s")
    
    try:
        if args.backend == "both":
            await test_both_backends(args.duration)
        else:
            await test_single_backend(args.backend, args.duration)
        
        print("\n✅ Test completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
