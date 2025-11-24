#!/usr/bin/env python3
"""
ElasticMM Demo with Dynamic Workload

Demonstrates the complete ElasticMM system with continuous request generation.
Runs for 10 minutes with automatic elastic scheduling.

Usage:
    python examples/demo_with_workload.py
"""

import asyncio
import sys
import signal
import os
from pathlib import Path
from datetime import datetime


sys.path.insert(0, str(Path(__file__).parent.parent))

from elasticmm.system import create_default_system
from tests.dynamic_request_generator import DynamicRequestGenerator


_generator_instance = None
_system_instance = None

def signal_handler(signum, frame):
    """Handle Ctrl+C - immediate force exit"""
    print("\n\n⚠️  Ctrl+C detected! Force exiting immediately...")
    print("⚠️  Attempting quick cleanup...")
    
    # Try to shutdown Ray quickly
    try:
        import ray
        if ray.is_initialized():
            print("   Shutting down Ray...")
            ray.shutdown()
    except Exception as e:
        print(f"   Ray shutdown warning: {e}")
    
    print("✓ Force exit now.")
    os._exit(0)  # Force immediate exit

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


async def main():
    global _generator_instance, _system_instance
    
    print("Starting ElasticMM system...")
    system = create_default_system()
    _system_instance = system  # Store for signal handler reference
    generator = None
    
    try:
        # Start system
        await system.start()
        print("System started successfully!\n")
        
        # For V0 backend, proxy runs in the main process and is ready immediately
        # Give it a brief moment to stabilize
        print("Waiting for proxy to stabilize...")
        await asyncio.sleep(2)
        print("✓ Proxy is ready")
        
        # Create request generator
        generator = DynamicRequestGenerator(
            proxy_url=f"http://127.0.0.1:{system.proxy_port}/v1/chat/completions",
            image_dir="/root/lzd/dataset/shareGPT-4o/images",
            text_data_path="/root/lzd/dataset/shareGPT-4o/image_conversations/gpt-4o.jsonl",
            model_name=system.model_path
        )
        _generator_instance = generator  # Store reference for signal handler
        
        # Run workload for 10 minutes
        print("\nRunning workload for 10 minutes...")
        print("Press Ctrl+C to stop gracefully.\n")
        
        # Run generator (Ctrl+C will force exit via signal handler)
        await generator.run(
            duration_seconds=600,
            text_base_rate=30.0,
            multimodal_base_rate=20.0,
            text_variance=0.3,
            multimodal_variance=1.5
        )
        
        # Print final statistics
        if generator:
            print(f"\n✓ Demo completed!")
            print(f"  Total requests: {generator.total_requests_sent}")
            print(f"  Success rate: {generator.successful_requests}/{generator.total_requests_sent}")
        
    finally:
        # Cleanup (only reached on normal exit, not on Ctrl+C)
        _generator_instance = None
        _system_instance = None
        
        print("\n🛑 Cleaning up (normal exit)...")
        try:
            await asyncio.wait_for(system.stop(), timeout=30.0)
            print("✓ System stopped gracefully")
        except Exception as e:
            print(f"⚠️  Cleanup error: {e}")
            print("⚠️  Run manually: ray stop --force")


if __name__ == "__main__":
    # Run main (Ctrl+C will be handled by signal_handler -> os._exit(0))
    asyncio.run(main())

