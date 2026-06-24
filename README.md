# transformers-labs

> Hands-on labs for fine-tuning, quantizing, and rigorously evaluating transformer models on text-editing / grammar tasks — turning base LLMs into competitive grammar correctors on a single consumer GPU.

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.39-FFD21E)](https://huggingface.co/docs/transformers/)
[![PEFT / LoRA](https://img.shields.io/badge/PEFT-LoRA-blue)](https://huggingface.co/docs/peft/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A working notebook lab-book exploring the full lifecycle of adapting transformer language models to **grammatical error correction (GEC) and text editing**: instruction-style fine-tuning, parameter-efficient training (LoRA), low-bit quantization (BitsAndBytes 4/8-bit), a reproducible evaluation harness, and GPU/cloud benchmarking. Models are trained against the [`grammarly/coedit`](https://huggingface.co/datasets/grammarly/coedit) and [IteraTeR](https://huggingface.co/datasets/wanyu/IteraTeR_full_sent) datasets and published to the Hub.

---

## Why this repo

Production-grade text editing assistants are expensive to run. These labs ask a focused question: **how close can commodity base models (GPT-2, T5, BART, Phi-2, Gemma, Llama-2) get to a purpose-built editor like `grammarly/coedit-large` — when fine-tuned with LoRA and served in 4/8-bit on a single RTX 3080?** Every notebook is an experiment with the dataset, training recipe, quantization config, and measured outcome kept together so results are reproducible rather than anecdotal.

## What the labs cover

- **Fine-tuning across architectures** — seq2seq (T5, BART) and decoder-only (GPT-2, GPT-2-large, Phi-2, Gemma-2B, Falcon, Llama-2) on the CoEdIT text-editing tasks (`gec`, `fluency`, `coherence`, `clarity`, simplification, paraphrase, neutralize).
- **Parameter-efficient training** — LoRA adapters via 🤗 PEFT (~19 notebooks) instead of full fine-tunes.
- **Hardware-efficient methods** — 4-bit and 8-bit quantized training/inference with BitsAndBytes (`BitsAndBytesConfig`, ~27 notebooks) to fit larger models on limited VRAM.
- **Instruction / SFT training** — supervised fine-tuning with 🤗 TRL `SFTTrainer` for the Llama-2 and Falcon recipes.
- **A real evaluation harness** — batched generation with metrics computed via 🤗 `evaluate`: **ROUGE, SacreBLEU, SARI, and exact-match**, alongside live GPU/RAM utilization and throughput (samples/sec) logging.
- **Benchmarking & deployment** — inference benchmarks (latency/throughput) with [optimum-benchmark](https://github.com/huggingface/optimum-benchmark) (Hydra-driven), plus SageMaker and Terraform (Azure / Paperspace) provisioning for GPU workstations.

## Selected findings

Fine-tuning collapses the gap between a generic base model and a dedicated grammar editor. Measured on the GEC task (validation slice, RTX 3080, `max_length=350`):

| Model | Params | ROUGE-1 | SARI | Exact-match |
|---|--:|--:|--:|--:|
| `grammarly/coedit-large` *(reference editor)* | 783M | 0.942 | 87.3 | 0.555 |
| `google-t5/t5-large` *(base)* | 738M | 0.357 | 40.5 | 0.000 |
| **`iliazlobin/t5-large-coedit`** *(LoRA fine-tune)* | 738M | **0.889** | **67.2** | **0.101** |
| `facebook/bart-large` *(base)* | 406M | 0.760 | 55.0 | 0.000 |
| **`iliazlobin/bart-large-coedit`** | 406M | **0.886** | **64.7** | **0.075** |
| **`iliazlobin/gpt2-large-coedit`** | 774M | 0.882 | 67.0 | 0.097 |
| **`iliazlobin/gemma-2b-coedit`** | 2.5B | 0.903 | 74.5 | 0.219 |
| **`iliazlobin/phi-2-coedit`** | 2.8B | 0.903 | 75.0 | 0.225 |

Takeaways: LoRA fine-tuning lifts T5-large from **0.36 → 0.89 ROUGE-1** and **40 → 67 SARI** on GEC; quantized decoder-only models (Phi-2, Gemma-2B) close most of the remaining gap to the reference editor. Full per-task results for all models live in [`model-eval/results/`](model-eval/results/).

## Tech stack

`Python 3.12` · `PyTorch` · `Hugging Face Transformers 4.39` · `PEFT / LoRA` · `BitsAndBytes (4/8-bit)` · `TRL (SFTTrainer)` · `Datasets` · `Evaluate (ROUGE / SacreBLEU / SARI / EM)` · `optimum-benchmark` · `SageMaker` · `Terraform` · `Jupyter` · `Ruff`

## Repository layout

```
transformers-labs/
├── model-train/          # Fine-tuning notebooks: T5, BART, GPT-2, Phi-2, Gemma, Falcon, Llama-2 (LoRA / BnB / SFT)
├── model-info/           # Inference walkthroughs per model family (coedit, flan-t5, llama2, mistral, phi2)
├── model-eval/           # Evaluation harness (eval.py), metrics & analysis notebooks, results/ + samples/
├── utils/                # Dataset loaders (CoEdIT, IteraTeR), metric computation, GPU/RAM monitoring
├── inference-benchmark/  # optimum-benchmark configs + latency/throughput reports
├── sagemaker-labs/       # AWS SageMaker benchmarking notebook
├── terraform/            # GPU workstation provisioning (Azure, Paperspace)
├── transformers-tutorial/# Foundational fine-tuning tutorials
├── video-llava/          # Multimodal (video + text) Video-LLaVA experiment
├── transformers.md       # Working notes on the transformers ecosystem
├── requirements.txt / pyproject.toml
```

## How to run

```bash
git clone https://github.com/iliazlobin/transformers-labs.git
cd transformers-labs

# Environment (Python 3.12; a CUDA GPU is recommended for training)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install torch peft bitsandbytes trl datasets evaluate accelerate jupyter

# Secrets — provide a Hugging Face token to pull/push models & datasets
echo "HF_TOKEN=your_hf_token" > .env

# Explore: open the notebooks
jupyter lab
```

Then:
- **Fine-tune** — open a notebook in `model-train/` (e.g. `train-t5-coedit.ipynb`, `train-phi2-coedit.ipynb`).
- **Evaluate** — run `model-eval/eval.py` (or `eval.ipynb`) to generate and score against ROUGE / SacreBLEU / SARI / EM; results are written to `model-eval/results/`.
- **Benchmark inference** — see `inference-benchmark/` (optimum-benchmark, Hydra configs).

> Notebooks were developed on an RTX 3080 (and an A100 for larger runs). VRAM-heavy models rely on 4/8-bit quantization — adjust `batch_size` / quantization config to fit your GPU.

## Walkthrough videos

- [Fine-tuning transformers for grammar correction — part 1](https://www.youtube.com/watch?v=rY0f1GRK0h8)
- [Fine-tuning transformers for grammar correction — part 2](https://www.youtube.com/watch?v=k8XlLoGFIh0)

## Links

- **Portfolio:** [iliazlobin.com/portfolio](https://iliazlobin.com/portfolio)
- **Author:** Ilia Zlobin — Principal Software Engineer · [GitHub @iliazlobin](https://github.com/iliazlobin)
- **Fine-tuned models:** published under [`iliazlobin` on the Hugging Face Hub](https://huggingface.co/iliazlobin)

## License

Released under the [MIT License](LICENSE). © 2026 Ilia Zlobin.
