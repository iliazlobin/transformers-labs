import sys
import threading
import time
from pprint import pprint

import debugpy
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from utils.dataset import get_iterater_samples
from utils.metric import em_metric, rouge_metric, sacreblue_metric, sari_metric
from utils.monitoring import (
    calculate_utilization,
    format_utilization,
    format_utilization_narrow,
    print_utilization,
    print_utilization_header,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pprint(f"Device: {device}")
torch.cuda.empty_cache()

# Load model
tokenizer = AutoTokenizer.from_pretrained("grammarly/coedit-large")
# model = T5ForConditionalGeneration.from_pretrained("grammarly/coedit-large", device_map=0)
model = T5ForConditionalGeneration.from_pretrained("grammarly/coedit-large")
model = model.to(device)

print(f"Allocated: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved(device)/1024**3:.2f} GB")

# Model info
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params}")
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable Parameters: {total_trainable_params}")
# total_memory_GB = total_params * 4 / (1024**3)
# print(f"Estimated model memory: {total_memory_GB:.2f} GB")
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

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
def process_batch(batch):
    num_tokens = len(batch["task"])
    start_time = time.time()

    input_ids = tokenizer(batch["task"], padding=True, return_tensors="pt").input_ids.to(device)
    # input_ids = tokenizer(item["task"], return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=256)
    # print(f"outputs: {outputs}")
    processed = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    end_time = time.time()
    elapsed_time = end_time - start_time
    tps = num_tokens / elapsed_time
    tps_str = f"{tps:.2f}"

    utilization = calculate_utilization()
    utilization_str = format_utilization_narrow(utilization)
    print(
        f"total/used/cuda/res/ram(Gb): {utilization_str["total_memory"]}/{utilization_str["memory_used"]}/"
        f"{utilization_str["cuda_allocated"]}/{utilization_str["cuda_reserved"]}/{utilization_str["ram_usage"]}"
    )
    print(
        f"batch/tps: {num_tokens}/{tps_str}"
    )

    return {"processed": processed}


# Calculate scores
def calculate_scores(processed_samples):
    rouge_score = rouge_metric.compute(
        predictions=processed_samples["processed"], references=processed_samples["references"]
    )
    # pprint(rouge_score)

    sacreblue_score = sacreblue_metric.compute(
        predictions=processed_samples["processed"], references=processed_samples["references"]
    )
    # pprint(sacreblue_score)

    sari_score = sari_metric.compute(
        sources=processed_samples["source"],
        predictions=processed_samples["processed"],
        references=processed_samples["references"],
    )
    # pprint(sari_score)

    score = em_metric.compute(predictions=processed_samples["processed"], references=processed_samples["reference"])
    # pprint(score)

    return {
        "rouge": rouge_score,
        "sacreblue": sacreblue_score,
        "sari": sari_score,
        "em": score,
    }


def main():
    print(f"Start")

    utilization = calculate_utilization()
    utilization_str = format_utilization_narrow(utilization)
    print(
        f"total/used/cuda/res/ram(Gb): {utilization_str["total_memory"]}/{utilization_str["memory_used"]}/"
        f"{utilization_str["cuda_allocated"]}/{utilization_str["cuda_reserved"]}/{utilization_str["ram_usage"]}"
    )

    loaded_samples = get_iterater_samples(label="fluency", num_samples=100)
    pprint(loaded_samples)

    processed_samples = loaded_samples.map(process_batch, num_proc=1, batched=True, batch_size=20)
    pprint(processed_samples)
    pprint(processed_samples["processed"][:2])

    scores = calculate_scores(processed_samples)
    pprint(scores)

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
        debugpy.breakpoint()
        print(f"Finalling threads")
        for t in threading.enumerate():
            print(f"Thread: ", t.getName)

        # print(f"Attempt to join threads to ensure all threads are finished")
        # for t in threading.enumerate():
        #     name = t.getName()
        #     print(f"About To Join : ", name)
        #     if name == "Thread-6":
        #         t.join()
