import os
import sys
import threading
import time
from pprint import pprint

import debugpy
import numpy as np
import pandas as pd
import torch
from datasets.utils.logging import disable_progress_bar
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.dataset import get_iterater_samples
from utils.metric import calculate_scores
from utils.monitoring import (
    calculate_utilization,
    format_utilization_narrow,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pprint(f"Device: {device}")
torch.cuda.empty_cache()


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = T5ForConditionalGeneration.from_pretrained(model_name, device_map=0)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device)
    print(
        f"Model loaded, allocated/reserved (Gb): {torch.cuda.memory_allocated(device)/1024**3:.2f}/{torch.cuda.memory_reserved(device)/1024**3:.2f}"
    )

    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total/trainable params: {total_params}/{total_trainable_params}")
    # total_memory_GB = total_params * 4 / (1024**3)
    # print(f"Estimated model memory: {total_memory_GB:.2f} GB")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    return model, tokenizer, total_params, total_trainable_params


# Utilization
utilization = calculate_utilization()
utilization_str = format_utilization_narrow(utilization)
print(
    f"total/used/cuda/res/ram(Gb): {utilization_str["total_memory"]}/{utilization_str["memory_used"]}/"
    f"{utilization_str["cuda_allocated"]}/{utilization_str["cuda_reserved"]}/{utilization_str["ram_usage"]}"
)

available_memory = utilization["total_memory"] - utilization["memory_used"]
recommended_fraction = available_memory / utilization["total_memory"]
print(f"Available memory: {available_memory:.2f} GB")
print(f"Recommended fraction: {recommended_fraction:.2f}")
# print(f"Set memory fraction: {recommended_fraction:.2f}")
# torch.cuda.set_per_process_memory_fraction(recommended_fraction, 0)
print(f"Set memory fraction: 0.95")
torch.cuda.set_per_process_memory_fraction(0.95, 0)


# Process
def process_batch(batch, idx, **kwargs):
    num_samples = len(batch["task"])
    start_time = time.time()

    model = kwargs.get("model")
    tokenizer = kwargs.get("tokenizer")
    total_samples = kwargs.get("total_samples")

    input_ids = tokenizer(batch["task"], padding=True, return_tensors="pt").input_ids.to(device)
    # input_ids = tokenizer(item["task"], return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=256)
    # print(f"outputs: {outputs}")
    processed = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    end_time = time.time()
    elapsed_time = end_time - start_time
    sps = num_samples / elapsed_time
    sps_str = f"{sps:.2f}"

    utilization = calculate_utilization()
    utilization_str = format_utilization_narrow(utilization)
    print(
        f"{idx[0]}-{idx[-1]}/{total_samples} | total/used/cuda/res/ram(Gb): {utilization_str["total_memory"]}/{utilization_str["memory_used"]}/"
        f"{utilization_str["cuda_allocated"]}/{utilization_str["cuda_reserved"]}/{utilization_str["ram_usage"]} | "
        f"batch/sps: {num_samples}/{sps_str}"
    )

    return {"processed": processed}


all_flat_frames = []
if os.path.exists("results/all-scores.csv"):
    all_flat_frames = pd.read_csv("results/all-scores.csv").to_dict("records")


def save_frame(model_name, model_alias, total_samples, processed_samples, processed_sps, total_params):
    scores = calculate_scores(processed_samples)
    # pprint(scores)

    score_paths = [
        "rouge.rouge1",
        # "rouge.rouge2",
        # "rouge.rougeL",
        # "rouge.rougeLsum",
        "sacreblue.score",
        "sari.sari",
        "em.exact_match",
    ]

    base_frame = {
        "model": model_name,
        "total_samples": total_samples,
        "total_params": total_params,
        "sps": processed_sps,
        "task": "fluency",
    }

    normalized_scores = {}
    for k, v in scores.items():
        for k2, v2 in v.items():
            if not isinstance(v2, list):
                # normalized_scores[f"score.{k}.{k2}"] = v2
                path = f"{k}.{k2}"
                if path in score_paths:
                    normalized_scores[f"score.{k}.{k2}"] = v2
    # pprint(normalized_scores)

    flat_frame = base_frame.copy()
    flat_frame.update(normalized_scores)
    # pprint(frame)

    flat_df = pd.DataFrame.from_records([flat_frame])
    # pprint(df)
    print(flat_df.head().to_markdown(index=False))
    flat_df.to_csv(f"results/{model_alias}.csv", index=False)

    all_flat_frames.append(flat_frame)
    all_flat_dfs = pd.DataFrame.from_records(all_flat_frames)
    all_flat_dfs.to_csv(f"results/all-scores.csv", index=False)

    full_frame = base_frame.copy()
    full_frame.update({"scores": scores})
    full_df = pd.DataFrame.from_records([full_frame])

    full_df.to_json(f"results/{model_alias}.json", orient="records")


disable_progress_bar()


def main():
    print(f"Start")

    utilization = calculate_utilization()
    utilization_str = format_utilization_narrow(utilization)
    print(
        f"total/used/cuda/res/ram (Gb): {utilization_str["total_memory"]}/{utilization_str["memory_used"]}/"
        f"{utilization_str["cuda_allocated"]}/{utilization_str["cuda_reserved"]}/{utilization_str["ram_usage"]}"
    )

    num_samples = 100
    batch_size = 20
    loaded_samples = get_iterater_samples(label="fluency", num_samples=num_samples)
    total_samples = len(loaded_samples)
    # pprint(loaded_samples)

    start_time = time.time()

    model_names = [
        "grammarly/coedit-large",
        "google/flan-t5-large",
    ]

    model_count = 0
    total_models = len(model_names)
    for model_name in model_names:
        model_count += 1
        model_alias = model_name.replace("/", "_")

        file_path = f"results/{model_alias}.json"
        if os.path.exists(file_path):
            print(f"Model has already been processed ({model_count}/{total_models}): {model_name}")
            continue

        print(f"Processing model ({model_count}/{total_models}): {model_name}")

        model, tokenizer, total_params, total_trainable_params = load_model(model_name)

        processed_samples = loaded_samples.map(
            process_batch,
            fn_kwargs={
                "model": model,
                "tokenizer": tokenizer,
                "total_samples": total_samples,
            },
            num_proc=1,
            batched=True,
            batch_size=batch_size,
            with_indices=True,
        )

        end_time = time.time()
        elapsed_time = end_time - start_time
        processed_sps = total_samples / elapsed_time

        # scores = calculate_scores(processed_samples)
        # pprint(scores)

        save_frame(
            model_name=model_name,
            model_alias=model_alias,
            total_samples=total_samples,
            processed_samples=processed_samples,
            processed_sps=processed_sps,
            total_params=total_params,
        )

    print(f"End")


def handle_exception(args):
    print(f"Exception occurred in thread {args.thread.ident}: {args.exc_type.__name__}: {args.exc_value}")


if __name__ == "__main__":
    # main_thread = threading.Thread(target=main)
    # main_thread.start()
    # main_thread.join()
    # multiprocessing.freeze_support()

    # threading.excepthook = handle_exception

    try:
        main()
    finally:
        debugpy.wait_for_client()
        print(f"Finalling threads")
        for t in threading.enumerate():
            print(f"Thread: ", t.getName)

        # print(f"Attempt to join threads to ensure all threads are finished")
        # for t in threading.enumerate():
        #     name = t.getName()
        #     print(f"About To Join : ", name)
        #     if name == "Thread-6":
        #         t.join()
