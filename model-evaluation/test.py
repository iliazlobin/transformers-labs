import os
import sys
import threading
import time
from pprint import pprint

import debugpy
import numpy as np
import pandas as pd
import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pprint(f"Device: {device}")
torch.cuda.empty_cache()


def handle_exception(args):
    print(f"Exception occurred in thread {args.thread.ident}: {args.exc_type.__name__}: {args.exc_value}")


def main():
    model = LlamaForCausalLM.from_pretrained()
    tokenizer = LlamaTokenizer.from_pretrained()

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
    )


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
