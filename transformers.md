# conda environment pytorch
```sh
conda deactivate
conda create -n pytorch python=3.12

# activate pytorch
conda activate pytorch
python --version
echo $CONDA_PREFIX
du -sh $CONDA_PREFIX

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
