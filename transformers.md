# conda environment pytorch
* https://pytorch.org/get-started/previous-versions/
```sh
conda deactivate
# conda env remove -n pytorch-3.10
# conda create -n pytorch-gptq python=3.12
conda create -n pytorch-3.10 python=3.10
# conda activate pytorch
conda activate pytorch-3.10
# conda activate pytorch-3.12
# conda activate pytorch-gptq
python --version
echo $CONDA_PREFIX
du -sh $CONDA_PREFIX

conda config --set channel_priority strict
conda clean -y --all
conda update -y --all

# all
# pip install sentencepiece pytz mpmath xxhash urllib3 tzdata typing-extensions tqdm sympy safetensors rouge regex python-dateutil pyarrow-hotfix psutil packaging nvidia-nvtx-cu12 nvidia-nvjitlink-cu12 nvidia-nccl-cu12 nvidia-curand-cu12 nvidia-cufft-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-cupti-cu12 nvidia-cublas-cu12 numpy networkx multidict MarkupSafe idna fsspec frozenlist filelock dill charset-normalizer certifi attrs async-timeout yarl triton requests pyarrow pandas nvidia-cusparse-cu12 nvidia-cudnn-cu12 multiprocess jinja2 gekko aiosignal nvidia-cusolver-cu12 huggingface-hub aiohttp torch tokenizers transformers datasets accelerate peft auto-gptq

# common packages
# pip install ipykernel
conda install -y ipykernel ipywidgets conda-forge
pip install python-dotenv pipreqs
pip install jupyter ipywidgets

# machine learning
pip install scikit-learn
pip install auto-gptq

# torch (https://pytorch.org/get-started/locally/)
nvcc --version
# conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

conda install -y pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia # wait long (3-5 min)
# conda install -y pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 -c pytorch
# conda install -y pytorch-cuda=12.1 -c pytorch -c nvidia

python -c "import os, torch; print(os.path.dirname(torch.__file__))"
python -c "import torch; print(torch.backends.cudnn.enabled)"
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.version.cuda)"
python -c "import torch; print(torch. __version__)"
python -m torch.utils.collect_env

# libs
conda install -c conda-forge scikit-learn

# bench
python benchmark.py

pip install -y -U "huggingface_hub[cli]"
HUGGING_FACE_TOKEN=$(cat .env | grep HUGGING_FACE_TOKEN) && HUGGING_FACE_TOKEN=${HUGGING_FACE_TOKEN#*=}
huggingface-cli login --token $HUGGING_FACE_TOKEN

# install pytorch (https://pytorch.org/get-started/locally/)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# check nvidia cuda
nvidia-smi
nvcc -V
python -c "import torch; print(torch.backends.cudnn.enabled)"
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch. __version__)"
python -m torch.utils.collect_env

# install transformers
pip install -y transformers
pip install -y transformers gradio datasets chardet cchardet librosa ipython sentencepiece plotly phonemizer
pip install -y sentencepiece
pip install -y accelerate

# debug
pip install -y torchsummary

# cache
find $CONDA_PREFIX

# pip install -y google
# pip install -y google-colab
# python -c "import google;"
# python -c "import google.colab;"

# pip install -y -q -U keras-nlp
# pip install -y -q -U keras>=3
# pip show keras
# pip show tensorflow

# pip install -y -U keras
# pip install -y keras==3.0.5

pip install -y --upgrade --quiet langchain langchain-openai
python -c "import langchain; print(langchain.__version__)"

```

## Benchmark
```sh
conda activate pytorch
echo $CONDA_PREFIX
python --version

# pip uninstall optimum-benchmark
# pip install -y optimum-benchmark@git+https://github.com/huggingface/optimum-benchmark.git@main # ERROR: doesn't install some dependencies
# pip show optimum-benchmark

git clone https://github.com/huggingface/optimum-benchmark.git
cd optimum-benchmark
pip install -y -e .

export BENCHMARK_INTERFACE=API
export LOG_LEVEL=DEBUG
python -benchmark.py

export HYDRA_FULL_ERROR=1
optimum-benchmark --config-dir benchmark/ --config-name pytorch_bert

```

## Ruff
```sh
/home/izlobin/.vscode-server/extensions/charliermarsh.ruff-2024.16.0-linux-x64/bundled/libs/bin/ruff --version

/home/izlobin/.vscode-server/extensions/charliermarsh.ruff-2024.16.0-linux-x64/bundled/libs/bin/ruff \
  format --stdin-filename /home/izlobin/ws/transformers-labs/model-evaluation/evaluate.py

ruff --version
ruff format /home/izlobin/ws/transformers-labs/model-evaluation/test.py
ruff format /home/izlobin/ws/transformers-labs/model-evaluation/evaluate.py

```

## ssh tunnel
```sh
ssh -L 16006:localhost:6006 paperspace@184.105.3.44

```
