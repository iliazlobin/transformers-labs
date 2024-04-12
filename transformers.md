# conda environment pytorch
* https://pytorch.org/get-started/previous-versions/
```sh
conda deactivate
conda create -n pytorch python=3.12
conda activate pytorch
python --version
echo $CONDA_PREFIX
du -sh $CONDA_PREFIX

conda config --set channel_priority strict
conda clean -y --all
conda update -y --all

# common packages
conda install -y ipykernel ipywidgets conda-forge
pip install python-dotenv pipreqs

# torch (https://pytorch.org/get-started/locally/)
nvcc --version
# conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# conda install -y pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -y pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 -c pytorch
conda install -y pytorch-cuda=12.1 -c pytorch -c nvidia

python -c "import os, torch; print(os.path.dirname(torch.__file__))"
python -c "import torch; print(torch.backends.cudnn.enabled)"
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch. __version__)"
python -m torch.utils.collect_env

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
