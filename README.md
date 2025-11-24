# ElasticMM: Elastic and Efficient MLLM Serving System

<p align="center">
  <img src="assets/elasticmm-logo.svg" alt="ElasticMM Logo" width="420" />
</p>

## Latest News 🔥

- **[2025/11]** – Added **V1 backend** (beta) for next-generation scheduling and serving.
- **[2025/10]** – Open-sourced ElasticMM with full support for the **vLLM V0 backend**.
- **[2025/09]** – Our paper *ElasticMM* was accepted as an **Oral** presentation at **NeurIPS 2025**.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/elasticmm/elasticmm)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://elasticmm.readthedocs.io/)

ElasticMM is a high-performance framework for serving large multimodal language models with elastic resource allocation and intelligent load balancing. It implements the Elastic Multimodal Parallelism (EMP) framework to optimize resource utilization and system throughput for both text-only and multimodal inference workloads.

## 🚀 Key Features

- **Elastic Resource Allocation**: Dynamic GPU allocation between text and multimodal workloads
- **Hierarchical Scheduling**: Two-level scheduling architecture for optimal resource management
- **Modality-Aware Load Balancing**: Intelligent load balancing based on workload patterns
- **Real-time Auto-scaling**: Automatic scaling based on demand and performance metrics
- **Multi-GPU Support**: Efficient utilization of multiple GPU instances
- **OpenAI-Compatible API**: Easy integration with existing applications

## 🛠️ Installation

### Prerequisites

- **Python**: 3.8 or higher
- **CUDA**: 11.8 or higher (for GPU support)
- **NCCL**: For multi-GPU communication (usually included with PyTorch)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/CapitalLiu/ElasticMM.git
cd ElasticMM

# Install the package
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

## 🚀 Quick Start

### Step 1: System Calibration (Recommended)

Before running inference, we recommend calibrating the system parameters for your hardware configuration. This offline profiling step helps optimize performance for your specific GPU setup.

```bash
# Run calibration to profile your machine's parameters
python examples/calibrate_gain_cost.py
```

The calibration process will:
- Profile GPU memory and compute capabilities
- Measure KV cache transfer bandwidth
- Calculate optimal resource allocation parameters
- Generate configuration files for your hardware

**Note**: This step is optional but highly recommended for optimal performance.

### Step 2: Simple Usage Example

For a basic example demonstrating ElasticMM's core functionality:

```bash
python examples/simple_usage.py
```

This example shows:
- Basic system initialization
- Request submission and processing
- Output collection and handling

### Step 3: Online Service with Dynamic Workload

For a complete online service demonstration with dynamic request generation:

```bash
python examples/demo_with_workload.py
```

This demo includes:
- Full system deployment with proxy and scheduler
- Dynamic request generation (text-only and multimodal)
- Real-time load balancing and auto-scaling
- Performance monitoring and statistics

### System Requirements

⚠️ **Important**: ElasticMM requires a minimum of **8 GPUs** to run with the default configuration (2 for text-only workloads + 6 for multimodal workloads).

- **Minimum**: 8 GPUs
- **Recommended**: 8+ GPUs with high-bandwidth interconnects (NVLink/InfiniBand) for optimal performance
- **Memory**: Sufficient GPU memory for your target model (typically 20GB+ per GPU)

## 🏗️ Architecture

ElasticMM implements a hierarchical architecture with two main levels:

1. **Modality Level**: Allocates GPU instances between text-only and multimodal workloads
2. **Stage Level**: Manages encoding, prefill, and decoding stages within each modality group

The system automatically balances resources based on real-time demand and performance metrics, ensuring optimal utilization across different workload types.

## 📊 Performance

- **High Throughput**: Optimized for maximum requests per second
- **Low Latency**: Minimized time-to-first-token (TTFT)
- **Efficient Resource Usage**: Dynamic allocation prevents resource waste
- **Scalable**: Supports from single GPU to multi-node deployments


## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on top of [vLLM](https://github.com/vllm-project/vllm) for efficient LLM serving
- Inspired by research in elastic computing and multimodal systems
- Thanks to the open-source community for various dependencies

## 📚 Citation

If you find ElasticMM useful in your research or production deployments, please cite our NeurIPS 2025 paper:

```
@inproceedings{liu2025elasticmm,
  title     = {ElasticMM: Efficient Multimodal LLMs Serving with Elastic Multimodal Parallelism},
  author    = {Liu, Zedong and Cheng, Shenggan and Tan, Guangming and You, Yang and Tao, Dingwen},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  url       = {https://arxiv.org/abs/2507.10069}
}
```


**ElasticMM** - Making multimodal AI more efficient and accessible.
