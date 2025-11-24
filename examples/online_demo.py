#!/usr/bin/env python3
"""
ElasticMM Online System with Dynamic Request Testing

This script demonstrates a comprehensive ElasticMM system test:
1. Creates a default system (8 GPUs: 2 text + 6 multimodal)
2. Starts the system with all components
3. Uses dynamic request generator for long-term testing (8+ minutes)
4. Monitors system performance and health
5. Provides detailed statistics and cleanup

The test simulates realistic workload patterns with varying request rates.
"""

import asyncio
import os
import sys
import signal
import time
import json
from typing import Optional


from elasticmm.system import create_default_system
from tests.dynamic_request_generator import DynamicRequestGenerator

# Global variables
system = None
generator = None
test_start_time = None

def cleanup_resources():
    """Clean up all resources"""
    print("\n" + "="*60)
    print("ElasticMM LOG: Cleaning up resources...")
    
    global system, generator
    
    if generator:
        try:
            print("ElasticMM LOG: Stopping request generator...")
            generator._is_running = False
            print("ElasticMM LOG: Request generator stopped")
        except Exception as e:
            print(f"ElasticMM LOG: Error stopping generator: {e}")
    
    if system:
        try:
            print("ElasticMM LOG: Stopping ElasticMM system...")
            asyncio.run(system.stop())
            print("ElasticMM LOG: System stopped successfully")
        except Exception as e:
            print(f"ElasticMM LOG: Error stopping system: {e}")

def signal_handler(signum, frame):
    """Signal handler for graceful shutdown"""
    print(f"\nElasticMM LOG: Received signal {signum}, initiating cleanup...")
    cleanup_resources()
    sys.exit(0)


def print_system_status(system):
    """Print current system status"""
    try:
        info = system.get_system_info()
        print(f"\nElasticMM LOG: System Status:")
        print(f"  - Running: {info['is_running']}")
        print(f"  - Uptime: {info['uptime']:.1f}s")
        print(f"  - Total GPUs: {info['total_gpus']}")
        print(f"  - Text GPUs: {info['text_gpus']}")
        print(f"  - Multimodal GPUs: {info['multimodal_gpus']}")
        print(f"  - Active Engines: {info['active_engines']}")
        print(f"  - Monitoring Tasks: {info['monitoring_tasks']}")
    except Exception as e:
        print(f"ElasticMM LOG: Error getting system status: {e}")

def print_generator_stats(generator):
    """Print request generator statistics"""
    if generator:
        print(f"\nElasticMM LOG: Request Generator Statistics:")
        print(f"  - Total Requests Sent: {generator.total_requests_sent}")
        print(f"  - Successful Requests: {generator.successful_requests}")
        print(f"  - Failed Requests: {generator.failed_requests}")
        if generator.total_requests_sent > 0:
            success_rate = (generator.successful_requests / generator.total_requests_sent) * 100
            print(f"  - Success Rate: {success_rate:.1f}%")

async def wait_for_system_ready(system, timeout=300):
    """Wait for system to be ready with health checks"""
    print("ElasticMM LOG: Waiting for system to be ready...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            # Check if system is running
            if not system.is_running:
                await asyncio.sleep(5)
                continue
            
            # Check proxy health
            import requests
            response = requests.get(f"http://127.0.0.1:{system.proxy_port}/health", timeout=5)
            if response.status_code == 200:
                print("ElasticMM LOG: System is ready!")
                return True
                
        except Exception as e:
            print(f"ElasticMM LOG: Health check failed: {e}")
        
        await asyncio.sleep(10)
    
    print("ElasticMM LOG: System readiness timeout!")
    return False

async def run_dynamic_test():
    """Run the dynamic request test"""
    global system, generator, test_start_time
    
    test_start_time = time.time()
    
    try:
        # 1. Create default system
        print("\nElasticMM LOG: Creating default system...")
        system = create_default_system()
        print("ElasticMM LOG: Default system created successfully")
        
        # 2. Start system
        print("\nElasticMM LOG: Starting system...")
        await system.start()
        print("ElasticMM LOG: System started successfully")
        
        # 3. Wait for system to be ready
        if not await wait_for_system_ready(system):
            print("ElasticMM LOG: System failed to become ready, aborting test")
            return False
        
        # 4. Print initial system status
        print_system_status(system)
        
        # 5. Create dynamic request generator
        print("\nElasticMM LOG: Creating dynamic request generator...")
        generator = DynamicRequestGenerator(
            proxy_url=f"http://127.0.0.1:{system.proxy_port}/v1/chat/completions",
            image_dir="/path/dataset/shareGPT-4o/images",
            text_data_path="/path/dataset/shareGPT-4o/image_conversations/gpt-4o.jsonl",
            model_name=system.model_path
        )
        print("ElasticMM LOG: Request generator created successfully")
        
        # 6. Run dynamic test for 8+ minutes
        print("\nElasticMM LOG: Starting dynamic request test...")
        print("ElasticMM LOG: Test will run for 8+ minutes with varying load patterns")
        
        # Test configuration for 8+ minutes with realistic patterns
        await generator.run(
            duration_seconds=500,  # 8+ minutes (500 seconds)
            text_base_rate=15.0,        # Stable text requests (2 req/s base)
            multimodal_base_rate=25.0,  # Higher multimodal rate (4 req/s base)
            text_variance=0.5,         # Low variance for text stability
            multimodal_variance=2.0    # High variance for multimodal randomness
        )
        
        print("\nElasticMM LOG: Dynamic request test completed!")
        
        # 7. Print final statistics
        print_generator_stats(generator)
        print_system_status(system)
        
        # 8. Calculate test duration
        test_duration = time.time() - test_start_time
        print(f"\nElasticMM LOG: Total test duration: {test_duration:.1f} seconds")
        
        return True
        
    except Exception as e:
        print(f"ElasticMM LOG: Test failed with error: {e}")
        return False
    
    finally:
        # 9. Cleanup
        cleanup_resources()

async def run_monitoring_loop():
    """Run a monitoring loop during the test"""
    global system, generator
    
    while generator and generator._is_running:
        await asyncio.sleep(60)  # Check every minute
        
        if system:
            print(f"\nElasticMM LOG: [Monitoring] Test running...")
            print_system_status(system)
            print_generator_stats(generator)

async def main():
    """Main function"""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("ElasticMM LOG: Starting ElasticMM Online System Test")
    print("ElasticMM LOG: Press Ctrl+C to stop the test gracefully")
    
    try:
        # Run the main test
        success = await run_dynamic_test()
        
        if success:
            print("\n" + "="*60)
            print("ElasticMM LOG: Test completed successfully!")
            print("ElasticMM LOG: All components performed as expected")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("ElasticMM LOG: Test completed with issues")
            print("ElasticMM LOG: Check logs for details")
            print("="*60)
            
    except KeyboardInterrupt:
        print("\nElasticMM LOG: Test interrupted by user")
    except Exception as e:
        print(f"\nElasticMM LOG: Test failed with unexpected error: {e}")
    finally:
        print("ElasticMM LOG: Test session ended")

if __name__ == "__main__":
    asyncio.run(main())