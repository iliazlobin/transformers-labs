from pprint import pprint

import psutil
import torch
from huggingface_hub import HfApi
from pynvml import *
from pynvml import nvmlInit
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer


def calculate_utilization(device=0):
    nvmlInit()
    total_memory = torch.cuda.get_device_properties(0).total_memory
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    memory_used = info.used
    cuda_allocated = torch.cuda.memory_allocated(device)
    cuda_reserved = torch.cuda.memory_reserved(device)
    ram_usage = psutil.virtual_memory().used
    return {
        "total_memory": total_memory,
        "memory_used": memory_used,
        "cuda_allocated": cuda_allocated,
        "cuda_reserved": cuda_reserved,
        "ram_usage": ram_usage,
    }


def format_utilization(utilization):
    total_memory = f"{utilization["total_memory"]/1024**3:15.2f}"
    memory_used = f"{utilization["memory_used"]/1024**3:15.2f}"
    cuda_allocated = f"{utilization["cuda_allocated"]/1024**3:15.2f}"
    cuda_reserved = f"{utilization["cuda_reserved"]/1024**3:15.2f}"
    ram_usage = f"{utilization["ram_usage"]/(1024**3):15.2f}"

    return {
        "total_memory": total_memory,
        "memory_used": memory_used,
        "cuda_allocated": cuda_allocated,
        "cuda_reserved": cuda_reserved,
        "ram_usage": ram_usage,
    }

def format_utilization_narrow(utilization):
    total_memory = f"{utilization["total_memory"]/1024**3:.2f}"
    memory_used = f"{utilization["memory_used"]/1024**3:.2f}"
    cuda_allocated = f"{utilization["cuda_allocated"]/1024**3:.2f}"
    cuda_reserved = f"{utilization["cuda_reserved"]/1024**3:.2f}"
    ram_usage = f"{utilization["ram_usage"]/(1024**3):.2f}"

    return {
        "total_memory": total_memory,
        "memory_used": memory_used,
        "cuda_allocated": cuda_allocated,
        "cuda_reserved": cuda_reserved,
        "ram_usage": ram_usage,
    }


def print_utilization_header():
    print(f"|    total_memory |     memory_used |  cuda_allocated |   cuda_reserved |       ram_usage |")


def print_utilization(utilization):
    utilization_str = format_utilization(utilization)
    print(
        f"| {utilization_str["total_memory"]} | {utilization_str["memory_used"]} | {utilization_str["cuda_allocated"]} | {utilization_str["cuda_reserved"]} | {utilization_str["ram_usage"]} |"
    )
