# Transformers-Labs: Research Repository

## Abstract

Transformers-Labs is a comprehensive research-oriented repository for experimentation, benchmarking, and fine-tuning of transformer-based models. The project provides a unified platform for training, evaluation, quantization, benchmarking, and deployment of state-of-the-art models, with robust support for multimodal and distributed workflows. It is designed to facilitate reproducible research, scalable experimentation, and rapid prototyping for both academic and industrial use cases.

## Project Overview

This repository exists to advance research in transformer architectures, quantization techniques, and large-scale model deployment. Key goals include:
- Benchmarking transformer models across hardware and cloud platforms
- Fine-tuning and evaluating models for NLP and multimodal tasks
- Experimenting with quantization (GPTQ, 4/8-bit) and efficient inference
- Integrating with cloud infrastructure (Azure, AWS SageMaker)
- Supporting multimodal research (video, text)

Capabilities include:
- Model training and evaluation pipelines
- Inference benchmarking (latency, throughput, accuracy)
- Quantization and deployment workflows
- Infrastructure-as-code for reproducible cloud setups
- Multimodal experimentation (video-llava)

## Repository Structure

- `model-train/` – Jupyter notebooks and scripts for model training and fine-tuning
- `model-eval/` – Evaluation pipelines, metrics, and analysis notebooks
- `inference-benchmark/` – Scripts for benchmarking inference performance
- `optimum-benchmark/` – Advanced benchmarking using HuggingFace Optimum
- `sagemaker-benchmark/`, `sagemaker-labs/` – AWS SageMaker integration for distributed training and benchmarking
- `terraform/azure-workstation/` – Terraform scripts for provisioning Azure GPU workstations
- `video-llava/` – Multimodal (video+text) model experimentation
- `AutoGPTQ/` – GPTQ quantization, CUDA builds, and extension modules
- `mistral-common/` – Utilities and shared code for Mistral models
- `requirements.txt`, `pyproject.toml` – Python dependencies and environment configuration
- `benchmarks/`, `model-info/`, `model/` – Model artifacts, configs, and benchmark results

## Features

- Transformer fine-tuning (BERT, T5, LLaMA, Mistral, etc.)
- GPTQ quantization (4/8-bit) via AutoGPTQ
- HuggingFace TRL integration for RLHF and advanced training
- SageMaker benchmarking and distributed training
- Azure infrastructure provisioning with Terraform
- CUDA-enabled PyTorch builds for efficient GPU utilization
- Multimodal research (video-llava)
- Inference benchmarking and reporting
- Environment management with Conda and .env files
- Code formatting and linting with Ruff

## Environment Setup

### System Dependencies
- CUDA Toolkit (>=11.x recommended)
- NVIDIA drivers (latest)
- GCC (>=9.x)
- pkg-config, libmysqlclient-dev (for some quantization/builds)

### Conda Environment & Key Packages
```bash
conda env create -f environment.yml
conda activate transformers-labs
```
Key packages:
- torch
- transformers
- trl
- optimum
- auto-gptq
- langchain
- evaluate

### .env File Usage
- Store Hugging Face token and other secrets in `.env`
- Example:
  ```
  HF_TOKEN=your_huggingface_token
  AWS_ACCESS_KEY_ID=...
  AZURE_SUBSCRIPTION_ID=...
  ```

### Official Install Docs
- [PyTorch](https://pytorch.org/get-started/locally/)
- [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- [Transformers](https://huggingface.co/docs/transformers/installation)
- [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)
- [TRL](https://github.com/huggingface/trl)
- [Optimum](https://huggingface.co/docs/optimum/)

## Quickstart / Usage

### Train/Evaluate a Model
- See `model-train/` and `model-eval/` notebooks for training and evaluation workflows
- Example:
  ```python
  # Train
  python model-train/train-gpt2.ipynb
  # Evaluate
  python model-eval/eval.ipynb
  ```

### Run Inference Benchmarks
- Use scripts in `inference-benchmark/` and `optimum-benchmark/`
- Example:
  ```bash
  python inference-benchmark/benchmark.py
  ```

### SageMaker Integration
- See `sagemaker-benchmark/` and `sagemaker-labs/` for distributed training and benchmarking
- Example:
  ```bash
  python sagemaker-benchmark/run_benchmark.py
  ```

### Azure Workstation Provisioning
- Use Terraform scripts in `terraform/azure-workstation/`
- Example:
  ```bash
  cd terraform/azure-workstation
  terraform init
  terraform apply -auto-approve
  ```

### Multimodal Experimentation
- See `video-llava/` for video+text model workflows

## Benchmarks

- Run benchmarking pipelines in `inference-benchmark/`, `optimum-benchmark/`, and `benchmarks/`
- Results are stored in CSV/JSON format for reproducibility
- Example:
  ```bash
  python inference-benchmark/benchmark.py --model gpt2 --output results/gpt2_benchmark.csv
  ```
- Interpret results using provided analysis notebooks in `model-eval/`

## Development Notes

- Code formatting: Use Ruff (`ruff format ...`) for linting and formatting
- Jupyter/interactive workflow: Use `%load_ext autoreload` and `%autoreload 2` for live code reload
- Debugging: Common issues include CUDA setup, missing drivers, and environment variables
- Use `.env` for secrets and tokens

## Research Roadmap

- Extend to new transformer architectures (e.g., Mixtral, Phi-3)
- Larger scale distributed experiments (multi-node, multi-GPU)
- Advanced quantization and pruning strategies
- Multimodal fusion and cross-modal benchmarks
- Integration with additional cloud providers (GCP, OCI)
- Automated hyperparameter tuning and experiment tracking

## Contributing

- Fork the repository and submit pull requests
- Add new models, benchmarks, or infrastructure scripts
- Cite this work in academic publications
- See `CONTRIBUTING.md` for guidelines

## References

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [PyTorch](https://pytorch.org/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)
- [TRL](https://github.com/huggingface/trl)
- [Optimum](https://huggingface.co/docs/optimum/)
- [SageMaker](https://aws.amazon.com/sagemaker/)
- [Azure Machine Learning](https://azure.microsoft.com/en-us/products/machine-learning/)

## License

This repository is licensed under the MIT License. See `LICENSE` for details.
