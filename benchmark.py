import os
from datetime import datetime

import torch
from optimum_benchmark.backends.pytorch.config import PyTorchConfig
from optimum_benchmark.benchmarks.inference.config import InferenceConfig
from optimum_benchmark.experiment import ExperimentConfig, launch
from optimum_benchmark.launchers.inline.config import InlineConfig
from optimum_benchmark.launchers.process.config import ProcessConfig
from optimum_benchmark.launchers.torchrun.config import TorchrunConfig

if __name__ == "__main__":
    print("Start")

    print("PyTorch version:", torch.__version__)
    print("PyTorch install path:", os.path.dirname(torch.__file__))

    # launcher_config = TorchrunConfig(nproc_per_node=1)
    launcher_config = ProcessConfig()
    # launcher_config = InlineConfig()
    benchmark_config = InferenceConfig(latency=True, memory=True)
    # backend_config = PyTorchConfig(model="gpt2", device="cuda")
    # model_name = "gpt2"
    model_name = "google/flan-t5-xl"
    # backend_config = PyTorchConfig(model=model_name, device="cuda", device_ids="0", no_weights=True)
    backend_config = PyTorchConfig(model=model_name, device="cpu", no_weights=True)
    # backend_config = PyTorchConfig(model="grammarly/coedit-large", device="cuda", device_ids="0", no_weights=True)
    experiment_config = ExperimentConfig(
        experiment_name="api-launch",
        benchmark=benchmark_config,
        launcher=launcher_config,
        backend=backend_config,
    )

    benchmark_report = launch(experiment_config)
    print(benchmark_report)

    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    model_file_name = model_name.replace("/", "-")
    benchmark_report.to_json(f"benchmark/{model_file_name}_{now_str}_cpu_benchmark_report.json", flat=True)
    benchmark_report.to_csv(f"benchmark/{model_file_name}_{now_str}_cpu_benchmark_report.csv")

    # push artifacts to the hub
    # experiment_config.push_to_hub("IlyasMoutawwakil/benchmarks")
    # benchmark_report.push_to_hub("IlyasMoutawwakil/benchmarks")


print("End")
