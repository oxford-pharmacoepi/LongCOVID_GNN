#!/usr/bin/env python3
"""
Setup script for drug-disease prediction package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements from requirements.txt
def read_requirements():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Get version from src/__init__.py
def get_version():
    version_file = os.path.join("src", "__init__.py")
    if os.path.exists(version_file):
        with open(version_file, "r") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    return "1.0.0"

setup(
    name="drug-disease-prediction",
    version=get_version(),
    author="Your Name",
    author_email="your.email@example.com",
    description="Graph Neural Networks for Drug-Disease Prediction with Explainable AI",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/drug-disease-prediction",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/drug-disease-prediction/issues",
        "Source": "https://github.com/yourusername/drug-disease-prediction",
        "Documentation": "https://github.com/yourusername/drug-disease-prediction/wiki",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "jupyter>=1.0.0",
        ],
        "gpu": [
            "torch>=2.0.0+cu118",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "drug-disease-pipeline=run_pipeline:main",
            "create-graph=scripts.1_create_graph:main",
            "train-models=scripts.2_train_models:main",
            "test-evaluate=scripts.3_test_evaluate:main",
            "explain-predictions=scripts.4_explain_predictions:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml", "*.txt", "*.md"],
        "src": ["*.json"],
    },
    zip_safe=False,
    keywords=[
        "machine learning",
        "graph neural networks",
        "drug discovery",
        "biomedical informatics",
        "explainable ai",
        "pytorch",
        "drug repurposing",
        "knowledge graphs",
    ],
)
