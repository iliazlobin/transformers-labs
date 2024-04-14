from pprint import pprint

from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer

grammarly_dataset = load_dataset("grammarly/coedit", num_proc=0)
# pprint(grammarly_dataset)

# unique_categories = set(grammarly_dataset)
# pprint(unique_categories)

# unique_tasks = set(grammarly_dataset["train"]["task"])
# pprint(unique_tasks)


def get_samples(dataset, category="validation", task="gec", num_samples=1, seed=42):
    return dataset[category].shuffle(seed=seed).filter(lambda item: item["task"] == task).select(range(num_samples))


def print_samples(samples) -> None:
    for item in samples:
        pfx, src = item["src"].split(": ", 1)
        print(f"[{item['task']}] {pfx}")
        print(f"src: {src}")
        print(f"tgt: {item['tgt']}")


# print_samples(get_samples(grammarly_dataset, num_samples=2))

# # input_ids = tokenizer(item["task"], return_tensors="pt").input_ids.to(device)
# # outputs = model.generate(input_ids, max_length=256)
# # corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
# # return {"processed": corrected}


# ## IteraTeR
# # * https://huggingface.co/datasets/wanyu/IteraTeR_v2
# # * https://huggingface.co/datasets/wanyu/IteraTeR_full_sent

# iterater_dataset = load_dataset("wanyu/IteraTeR_v2") # human in the loop
iterater_dataset = load_dataset("wanyu/IteraTeR_full_sent")
# pprint(iterater_dataset)
iterater_validation_dataset = load_dataset("wanyu/IteraTeR_full_sent", split="validation")
# pprint(iterater_validation_dataset)
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


def get_iterater_samples(label, category="validation", num_samples=0, seed=42, confidence_threshold=0.9):
    filtered_samples = (
        iterater_dataset[category]
        .shuffle(seed=seed)
        .filter(lambda item: item["labels"] == label and float(item["confidence"]) >= confidence_threshold)
    )
    max_samples = len(filtered_samples)
    selected = max_samples if num_samples == 0 else num_samples
    print(f"max_samples: {max_samples}, selected: {selected}, num_samples: {num_samples}")
    samples = filtered_samples.select(range(selected))

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
