# conda environment pytorch
* https://pytorch.org/get-started/previous-versions/
```sh
conda deactivate
# conda env remove -n pytorch-benchmark
# conda create -n pytorch-benchmark python=3.10
# conda activate pytorch-benchmark
conda activate pytorch
python --version
echo $CONDA_PREFIX
du -sh $CONDA_PREFIX

# torch
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 -c pytorch
# conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia

python -c "import os, torch; print(os.path.dirname(torch.__file__))"
python -c "import torch; print(torch.backends.cudnn.enabled)"
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch. __version__)"
python -m torch.utils.collect_env

python benchmark.py

# common packages
conda install -y ipykernel ipywidgets
pip install python-dotenv pipreqs
conda install -c conda-forge ipywidgets

pip install -U "huggingface_hub[cli]"
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
pip install transformers
pip install transformers gradio datasets chardet cchardet librosa ipython sentencepiece plotly phonemizer
pip install sentencepiece
pip install accelerate

# debug
pip install torchsummary

# cache
find $CONDA_PREFIX

# pip install google
# pip install google-colab
# python -c "import google;"
# python -c "import google.colab;"

# pip install -q -U keras-nlp
# pip install -q -U keras>=3
# pip show keras
# pip show tensorflow

# pip install -U keras
# pip install keras==3.0.5

pip install --upgrade --quiet langchain langchain-openai
python -c "import langchain; print(langchain.__version__)"

```

## Benchmark
```sh
conda activate pytorch
echo $CONDA_PREFIX
python --version

# pip uninstall optimum-benchmark
# pip install optimum-benchmark@git+https://github.com/huggingface/optimum-benchmark.git@main # ERROR: doesn't install some dependencies
# pip show optimum-benchmark

git clone https://github.com/huggingface/optimum-benchmark.git
cd optimum-benchmark
pip install -e .

export BENCHMARK_INTERFACE=API
export LOG_LEVEL=DEBUG
python -benchmark.py

export HYDRA_FULL_ERROR=1
optimum-benchmark --config-dir benchmark/ --config-name pytorch_bert

```
