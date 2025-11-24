#!/usr/bin/env python3
"""
Simple ElasticMM Usage Example

This example shows the simplest way to use ElasticMM with different configurations.
Users only need to specify GPU allocation, everything else is handled automatically.
"""

import asyncio
import sys
import requests


from elasticmm.system import create_default_system, create_custom_system

def send_request(payload: dict, proxy_port: int = 10001, timeout: int = 60) -> dict:
    """
    Send HTTP request to ElasticMM proxy
    
    Args:
        payload: Request payload
        proxy_port: Proxy port number
        timeout: Request timeout in seconds
        
    Returns:
        Response JSON data
    """
    try:
        response = requests.post(
            f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Request failed: {e}")

async def simple_example():
    """Simple example using default configuration"""
    print("ElasticMM LOG: Simple usage example with default configuration")
    
    # Create system with default configuration (8 GPUs: 2 text + 6 multimodal)
    system = create_default_system()
    
    try:
        # Start system (automatically configures everything)
        await system.start()
        print("ElasticMM LOG: System started successfully")
        
        # Submit a text request
        try:
            text_result = await system.submit_request({
                "model": system.model_path,
                "messages": [{"role": "user", "content": "What is elastic computing?"}],
                "max_tokens": 50
            })
            print("ElasticMM LOG: Text request completed successfully")
        except Exception as e:
            print(f"ElasticMM LOG: Text request failed: {e}")
            return
        
        # Get system information
        info = system.get_system_info()
        print(f"ElasticMM LOG: System info: {info['total_gpus']} GPUs, uptime: {info['uptime']:.1f}s")
        
    finally:
        # Stop system
        await system.stop()
        print("ElasticMM LOG: System stopped")

async def main():
    """Main function"""
    print("ElasticMM LOG: Starting simple usage examples")
    
    # Run default configuration example
    await simple_example()
    
    print("\nElasticMM LOG: All examples completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
