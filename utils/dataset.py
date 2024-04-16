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
# iterater_validation_dataset = load_dataset("wanyu/IteraTeR_full_sent", split="validation")
# pprint(iterater_validation_dataset)
# pprint(iterater_validation_dataset['validation'][0])


verbolizers = {
    "fluency": {
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
    },
    "clarity": {
        "tokens": ["<clarity>"],
        "verbs": [
            "Clarify the sentence",
            "Clarify this sentence",
            "Clarify this text",
            "Write a clearer version for the sentence,",
            "Write a clarified version of the sentence",
            "Write a readable version of the sentence",
            "Write a better readable version of the sentence",
            "Rewrite the sentence more clearly",
            "Rewrite this sentence clearly",
            "Rewrite this sentence for clarity",
            "Rewrite this sentence for readability",
            "Improve this sentence for readability",
            "Make this sentence better readable",
            "Make this sentence more readable",
            "Make this sentence readable",
            "Make the sentence clear",
            "Make the sentence clearer",
            "Clarify",
            "Make the text more understandable",
            "Make this easier to read",
            "Clarification",
            "Change to clearer wording",
            "Clarify this paragraph",
            "Use clearer wording",
        ],
    },
    "coherence": {
        "tokens": ["<coherence>"],
        "verbs": [
            "Fix coherence",
            "Fix coherence in this sentence",
            "Fix coherence in the sentence",
            "Fix coherence in this text,",
            "Fix coherence in the text",
            "Fix coherence errors",
            "Fix sentence flow",
            "Fix sentence transition",
            "Fix coherence",
            "errors in this sentence",
            "Fix coherence mistakes in this sentence",
            "Fix coherence in this sentence",
            "Fix coherence of the sentence",
            "Fix lack of coherence in the sentence",
            "Make the text more coherent",
            "Make the text coherent",
            "Make the text more cohesive logically linked and consistent as a whole",
            "Make the text more cohesive",
            "Improve the cohesiveness of the text",
            "Make the text more logical",
            "Make the text more consistent",
            "Improve the consistency of the text",
            "Make the text clearer",
            "Improve the coherence of the text",
        ],
    },
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


def add_verbolizer(text, verbolizer, count=[0]):
    verbs = verbolizers[verbolizer]["verbs"]

    # print(f"count: {count[0]}, len(verbs): {len(verbs)}")
    verb = verbs[count[0]]
    replaced_text = f"{verb}: {text}"

    count[0] += 1
    if count[0] >= len(verbs):
        count[0] = 0

    return replaced_text


def get_iterater_samples_simplified(label, category="validation", num_samples=0, seed=42, confidence_threshold=0.9):
    filtered_samples = (
        iterater_dataset[category]
        .shuffle(seed=seed)
        .filter(lambda item: item["labels"] == label and float(item["confidence"]) >= confidence_threshold)
    )
    max_samples = len(filtered_samples)
    selected = max_samples if num_samples == 0 else num_samples
    print(f"max_samples: {max_samples}, num_samples: {num_samples}, selected: {selected}")
    samples = filtered_samples.select(range(selected))

    return samples.map(
        lambda item: {
            "task": add_verbolizer(item["before_sent"], label),
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


def get_iterater_samples_with_instruction(
    label,
    category="validation",
    num_samples=0,
    seed=42,
    confidence_threshold=0.9,
    pre_instruction="",
    post_instruction="",
):
    filtered_samples = (
        iterater_dataset[category]
        .shuffle(seed=seed)
        .filter(lambda item: item["labels"] == label and float(item["confidence"]) >= confidence_threshold)
    )
    max_samples = len(filtered_samples)
    selected = max_samples if num_samples == 0 else num_samples
    print(f"max_samples: {max_samples}, num_samples: {num_samples}, selected: {selected}")
    samples = filtered_samples.select(range(selected))

    if pre_instruction:
        pre_instruction = f"{pre_instruction} "
    if post_instruction:
        post_instruction = f" {post_instruction}"

    return samples.map(
        lambda item: {
            "task": f"{pre_instruction}{add_verbolizer(item['before_sent'], label)}{post_instruction}",
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
