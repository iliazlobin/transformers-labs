from pprint import pprint

import evaluate
from .dataset import get_samples, grammarly_dataset, print_samples
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer

# Rouge metric
rouge_metric = evaluate.load("rouge")

# GLUE metric
glue_metric = evaluate.load("glue", "stsb")

# SacreBLEU metric
sacreblue_metric = evaluate.load("sacrebleu")

# SARI metric
sari_metric = evaluate.load("sari")

# Exact match (EM) metric
em_metric = evaluate.load("exact_match")


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


# def main():
#     # Rouge metric
#     samples = get_samples(grammarly_dataset, task="gec", num_samples=100)
#     pprint(samples)
#     print_samples([samples[0]])

#     score = rouge_metric.compute(predictions=samples["src"], references=samples["tgt"])
#     pprint(score)

#     # GLUE metric
#     glue_metric = evaluate.load("glue", "stsb")

#     # src_input_ids = coedit_large_tokenizer(samples["src"][0], return_tensors="pt", padding=True).input_ids
#     # tgt_input_ids = coedit_large_tokenizer(samples["tgt"][0], return_tensors="pt", padding=True).input_ids
#     # pprint(src_input_ids[0])
#     # pprint(tgt_input_ids[0])

#     # score = glue_metric.compute(predictions=src_input_ids[0], references=tgt_input_ids[0])
#     # score = glue_metric.compute(predictions=samples["src"], references=samples["tgt"])
#     # pprint(score)

#     # SacreBLEU metric
#     samples = get_samples(grammarly_dataset, task="gec", num_samples=100)
#     pprint(samples)
#     print_samples([samples[0]])

#     score = sacreblue_metric.compute(predictions=samples["src"], references=samples["tgt"])
#     pprint(score)

#     # SARI metric
#     samples = get_samples(grammarly_dataset, task="gec", num_samples=100)
#     pprint(samples)
#     print_samples([samples[0]])

#     new_samples = samples.map(lambda item: {"tgts": [item["tgt"]]})
#     new_samples["tgts"][:5]

#     # sources=["About 95 species are currently accepted.","About 95 species are currently accepted."]
#     # predictions=["About 95 you now get in.","About 95 you now get in."]
#     # references=[["About 95 species are currently known.","About 95 species are now accepted.","95 species are now accepted."],["About 95 species are currently known.","About 95 species are now accepted.","95 species are now accepted."]]

#     score = sari_metric.compute(
#         sources=new_samples["src"], predictions=new_samples["src"], references=new_samples["tgts"]
#     )
#     pprint(score)

#     # Exact match (EM) metric
#     samples = get_samples(grammarly_dataset, task="gec", num_samples=100)
#     pprint(samples)
#     print_samples([samples[0]])

#     score = em_metric.compute(predictions=samples["tgt"], references=samples["tgt"])
#     pprint(score)


# if __name__ == "__main__":
#     main()
