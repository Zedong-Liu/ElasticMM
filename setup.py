#!/usr/bin/env python3
"""
ElasticMM - Elastic Multimodal Parallelism Framework
Setup script for package installation
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="elasticmm",
    version="1.0.0",
    author="ElasticMM Team",
    author_email="elasticmm@example.com",
    description="Elastic Multimodal Parallelism Framework for Large Language Model Services",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/elasticmm/elasticmm",
    project_urls={
        "Bug Reports": "https://github.com/elasticmm/elasticmm/issues",
        "Source": "https://github.com/elasticmm/elasticmm",
        "Documentation": "https://elasticmm.readthedocs.io/",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies (required versions)
        "ray>=2.8.0",
        "vllm>=0.10.0",
        "torch>=2.0.0",
        "transformers>=4.35.0",
        # HTTP and async frameworks
        "aiohttp",
        "httpx",
        "quart",
        "uvicorn",
        # Data processing
        "numpy",
        "msgpack",
        "Pillow",
        "pyyaml",
        # Communication
        "pyzmq",
        # Type hints
        "typing-extensions",
    ],
    include_package_data=True,
    package_data={
        "elasticmm": [
            "configs/*.yaml",
            "configs/*.json",
            "examples/*.py",
            "tests/*.py",
        ],
    },
    zip_safe=False,
    keywords=[
        "multimodal",
        "large-language-models",
        "elastic-computing",
        "distributed-systems",
        "parallelism",
        "vllm",
        "gpu-acceleration",
        "load-balancing",
        "auto-scaling",
    ],
)