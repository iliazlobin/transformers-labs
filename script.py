

from pprint import pprint

import evaluate
import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pprint(f"Device: {device}")
# torch.cuda.empty_cache()



# ### Lading coedit model

coedit_large_tokenizer = AutoTokenizer.from_pretrained("grammarly/coedit-large")
coedit_large_model = T5ForConditionalGeneration.from_pretrained("grammarly/coedit-large")
coedit_large_model=coedit_large_model.to(device)

print(f"Allocated: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved(device)/1024**3:.2f} GB")


prompt = "fix grammar: How is are you?"
input_ids = coedit_large_tokenizer(prompt, return_tensors="pt").input_ids.to(device)
outputs = coedit_large_model.generate(input_ids, max_new_tokens=200)
print(coedit_large_tokenizer.decode(outputs[0], skip_special_tokens=True))

# ### Loading flan-t5


flan_t5_large_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl", return_attention_mask=False)
flan_t5_large_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto")
# flan_t5_large_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")
# flan_t5_large_model = flan_t5_large_model.to(device)

print(f"Allocated: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved(device)/1024**3:.2f} GB")

total_params = sum(p.numel() for p in flan_t5_large_model.parameters())
print(f"Total Parameters: {total_params}")
total_trainable_params = sum(p.numel() for p in flan_t5_large_model.parameters() if p.requires_grad)
print(f"Trainable Parameters: {total_trainable_params}")
# total_memory_GB = total_params * 4 / (1024**3)
# print(f"Estimated model memory: {total_memory_GB:.2f} GB")
# for param_tensor in flan_t5_large_model.state_dict():
#     print(param_tensor, "\t", flan_t5_large_model.state_dict()[param_tensor].size())



prompt = "translate English to German: How old are you?"
input_ids = flan_t5_large_tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
outputs = flan_t5_large_model.generate(input_ids, max_new_tokens=200)
print(flan_t5_large_tokenizer.decode(outputs[0], skip_special_tokens=True))

# ### Datasets

# api = HfApi()
# coedit_info = api.dataset_info("grammarly/coedit")
# pprint(coedit_info)

grammarly_dataset = load_dataset("grammarly/coedit")
pprint(grammarly_dataset)

unique_categories = set(grammarly_dataset)
pprint(unique_categories)

unique_tasks = set(grammarly_dataset["train"]["task"])
pprint(unique_tasks)

def get_samples(dataset, category="validation", task="gec", num_samples=1, seed=42):
    return dataset[category].shuffle(seed=seed).filter(lambda item: item["task"] == task).select(range(num_samples))

def print_samples(samples) -> None:
    for item in samples:
        pfx, src = item["src"].split(": ", 1)
        print(f"[{item['task']}] {pfx}")
        print(f"src: {src}")
        print(f"tgt: {item['tgt']}")


print_samples(get_samples(grammarly_dataset, num_samples=2))

# input_ids = coedit_large_tokenizer(item["task"], return_tensors="pt").input_ids.to(device)
# outputs = coedit_large_model.generate(input_ids, max_length=256)
# corrected = coedit_large_tokenizer.decode(outputs[0], skip_special_tokens=True)
# return {"processed": corrected}


# ### Metrics

# #### Rouge metric

rouge_metric = evaluate.load("rouge")

samples = get_samples(grammarly_dataset, task="gec", num_samples=100)
pprint(samples)
print_samples([samples[0]])

score = rouge_metric.compute(
    predictions=samples['src'], references=samples['tgt']
)
pprint(score)

# #### _GLUE metric_

glue_metric = evaluate.load("glue", "stsb")

samples = get_samples(grammarly_dataset, task="gec", num_samples=2)
pprint(object=samples)
print_samples([samples[0]])

src_input_ids = coedit_large_tokenizer(samples["src"][0], return_tensors="pt", padding=True).input_ids
tgt_input_ids = coedit_large_tokenizer(samples["tgt"][0], return_tensors="pt", padding=True).input_ids
pprint(src_input_ids[0])
pprint(tgt_input_ids[0])

# score = glue_metric.compute(predictions=src_input_ids[0], references=tgt_input_ids[0])
# score = glue_metric.compute(predictions=samples["src"], references=samples["tgt"])
# pprint(score)

# #### SacreBLEU metric

sacreblue_metric = evaluate.load("sacrebleu")

samples = get_samples(grammarly_dataset, task="gec", num_samples=100)
pprint(samples)
print_samples([samples[0]])

score = sacreblue_metric.compute(predictions=samples["src"], references=samples["tgt"])
pprint(score)

# #### SARI metric

sari_metric = evaluate.load("sari")

samples = get_samples(grammarly_dataset, task="gec", num_samples=100)
pprint(samples)
print_samples([samples[0]])

new_samples = samples.map(lambda item: {"tgts": [item["tgt"]]})
new_samples["tgts"][:5]

# sources=["About 95 species are currently accepted.","About 95 species are currently accepted."]
# predictions=["About 95 you now get in.","About 95 you now get in."]
# references=[["About 95 species are currently known.","About 95 species are now accepted.","95 species are now accepted."],["About 95 species are currently known.","About 95 species are now accepted.","95 species are now accepted."]]

score = sari_metric.compute(
  sources=new_samples['src'],
  predictions=new_samples['src'],
  references=new_samples['tgts']
)
pprint(score)

# #### Exact match (EM) metric

em_metric = evaluate.load("exact_match")

samples = get_samples(grammarly_dataset, task="gec", num_samples=100)
pprint(samples)
print_samples([samples[0]])

score = em_metric.compute(
    predictions=samples['tgt'], references=samples['tgt']
)
pprint(score)

# ### Datasets

# #### IteraTeR
# * https://huggingface.co/datasets/wanyu/IteraTeR_v2
# * https://huggingface.co/datasets/wanyu/IteraTeR_full_sent

# iterater_dataset = load_dataset("wanyu/IteraTeR_v2") # human in the loop
iterater_dataset = load_dataset("wanyu/IteraTeR_full_sent")
pprint(iterater_dataset)
iterater_validation_dataset = load_dataset("wanyu/IteraTeR_full_sent", split="validation")
pprint(iterater_validation_dataset)
# pprint(iterater_validation_dataset['validation'][0])


verbolizers = {
    "gce": {
        "tokens": ["<fluency>"],
        "verbs": [
            "Fix grammar",
            "Fix grammar in this sentence",
            "Fix grammar in the sentence",
            "Fix grammar errors",
            "Fix grammatical errors",
            "Fix grammaticality",
            "Fix all grammatical errors",
            "Fix grammatical errors in this sentence",
            "Fix grammar errors in this sentence",
            "Fix grammatical mistakes in this sentence",
            "Fix grammaticality in this sentence",
            "Fix grammaticality of the sentence",
            "Fix disfluencies in the sentence",
            "Make the sentence grammatical",
            "Make the sentence fluent",
            "Fix errors in this text",
            "Update to remove grammar errors",
            "Remove all grammatical errors from this text",
            "Improve the grammar of this text",
            "Improve the grammaticality",
            "Improve the grammaticality of this text",
            "Improve the grammaticality of this sentence,",
            "Grammar improvements",
            "Remove grammar mistakes",
            "Remove grammatical mistakes",
            "Fix the grammar mistakes",
            "Fix grammatical mistakes",
        ],
    }
}

def substitute_verbolizer(text, verbolizer, count=[0]):
    verbs = verbolizers[verbolizer]["verbs"]

    verb = verbs[count[0]]
    tokens = verbolizers[verbolizer]["tokens"]
    replaced_text = text
    for t in tokens:
        replaced_text = text.replace(t, f"{verb}:")
        # pprint(f"> t: {t}, verb: {verb}, text: {text}, replaced_text: {replaced_text}")

    count[0] += 1
    if count[0] >= len(verbs):
        count[0] = 0

    return replaced_text


def get_iterater_samples(label, category="validation", num_samples=1, seed=42, confidence_threshold=0.9):
    samples = (
        iterater_dataset[category]
        .shuffle(seed=seed)
        .filter(lambda item: item["labels"] == label and float(item["confidence"]) >= confidence_threshold)
        .select(range(num_samples))
    )
    return samples.map(
        lambda item: {
            "task": substitute_verbolizer(item["before_sent_with_intent"], "gce"),
            "source": item["before_sent"],
            "reference": item["after_sent"],
            "references": [item["after_sent"]],
        },
        remove_columns=[
            "before_sent_with_intent",
            "before_sent",
            "after_sent",
            "labels",
            "confidence",
            "doc_id",
            "revision_depth",
        ],
    )


samples = get_iterater_samples(label="fluency", num_samples=5)
pprint(samples)
pprint(samples["task"][:2])

# ### GPU processing


samples = get_iterater_samples(label="fluency", num_samples=20)
pprint(samples)

process_samples = samples


def coedit_large_model_process(batch):
    input_ids = coedit_large_tokenizer(batch["task"], padding=True, return_tensors="pt").input_ids.to(device)
    # input_ids = coedit_large_tokenizer(item["task"], return_tensors="pt").input_ids
    outputs = coedit_large_model.generate(input_ids, max_length=256)
    # print(f"outputs: {outputs}")
    processed = coedit_large_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return {"processed": processed}


# process_samples = samples.map(coedit_large_model_process, num_proc=torch.cuda.device_count())
process_samples = samples.map(coedit_large_model_process, num_proc=1, batched=True, batch_size=10)
pprint(process_samples)
pprint(process_samples["processed"][:2])

# ### GPU processing


def flan_t5_large_model_process(item):
    input_ids = flan_t5_large_tokenizer(item["task"], return_tensors="pt").input_ids.to(device)
    outputs = flan_t5_large_model.generate(input_ids, max_length=256)
    processed = flan_t5_large_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"flan_t5_large_processed": processed}


process_samples = process_samples.map(flan_t5_large_model_process, num_proc=torch.cuda.device_count())
pprint(process_samples)


process_samples[:2]

rouge_score = rouge_metric.compute(
    predictions=process_samples['coedit_large_processed'], references=process_samples['references']
)
pprint(rouge_score)
# rouge_score = rouge_metric.compute(
#     predictions=process_samples['flan_t5_large_processed'], references=process_samples['references']
# )
# pprint(rouge_score)

sacreblue_score = sacreblue_metric.compute(predictions=process_samples['coedit_large_processed'], references=process_samples['references'])
pprint(sacreblue_score)
# sacreblue_score = sacreblue_metric.compute(predictions=process_samples['flan_t5_large_processed'], references=process_samples['references'])
# pprint(sacreblue_score)

sari_score = sari_metric.compute(
  sources=process_samples['source'],
  predictions=process_samples['coedit_large_processed'],
  references=process_samples['references']
)
pprint(sari_score)
# sari_score = sari_metric.compute(
#   sources=process_samples['source'],
#   predictions=process_samples['flan_t5_large_processed'],
#   references=process_samples['references']
# )
# pprint(sari_score)

score = em_metric.compute(
    predictions=process_samples['coedit_large_processed'], references=process_samples['reference']
)
pprint(score)
# score = em_metric.compute(
#     predictions=process_samples['flan_t5_large_processed'], references=process_samples['reference']
# )
# pprint(score)


