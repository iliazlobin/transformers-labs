# env

- https://pytorch.org/get-started/previous-versions/

```sh

nvcc --version

# update system deps
sudo apt update
sudo apt upgrade -y

sudo apt install -y ubuntu-drivers-common


sudo apt install -y gcc
gcc -v

# install nvidia driver
# https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network
cd ~/dwl
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

sudo apt-get install -y cuda-toolkit-12-4
sudo apt-get install -y cuda-drivers

sudo ubuntu-drivers devices
sudo apt-get install -y cuda-drivers
sudo apt install nvidia-driver-535-server
sudo apt install nvidia-utils-535-server

lsmod | grep nvidia
sudo modprobe nvidia
nvidia-smi

```

# conda

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

# conda install
mkdir -p ~/dwl
cd ~/dwl
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda-installer.sh
chmod +x miniconda-installer.sh
./miniconda-installer.sh
conda init

# conda reset
# conda config --set channel_priority strict
# conda clean -y --all
# conda update -y --all

# all
# pip install sentencepiece pytz mpmath xxhash urllib3 tzdata typing-extensions tqdm sympy safetensors rouge regex python-dateutil pyarrow-hotfix psutil packaging nvidia-nvtx-cu12 nvidia-nvjitlink-cu12 nvidia-nccl-cu12 nvidia-curand-cu12 nvidia-cufft-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-cupti-cu12 nvidia-cublas-cu12 numpy networkx multidict MarkupSafe idna fsspec frozenlist filelock dill charset-normalizer certifi attrs async-timeout yarl triton requests pyarrow pandas nvidia-cusparse-cu12 nvidia-cudnn-cu12 multiprocess jinja2 gekko aiosignal nvidia-cusolver-cu12 huggingface-hub aiohttp torch tokenizers transformers datasets accelerate peft auto-gptq


# common packages
# pip install ipykernel
conda install -y ipykernel ipywidgets conda-forge
pip install python-dotenv pipreqs
pip install jupyter ipywidgets

# https://pytorch.org/get-started/locally/
# torch (https://pytorch.org/get-started/locally/)
# conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# conda install -y pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia # wait long (3-5 min)
# conda install -y pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 -c pytorch
# conda install -y pytorch-cuda=12.1 -c pytorch -c nvidia

python -c "import os, torch; print(os.path.dirname(torch.__file__))"
python -c "import torch; print(torch.backends.cudnn.enabled)"
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.version.cuda)"
python -c "import torch; print(torch. __version__)"
python -m torch.utils.collect_env

# ml libs
conda install -c conda-forge scikit-learn

pip install scikit-learn
pip install auto-gptq


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

## azure workstation

```sh
cd /home/izlobin/ws/transformers-labs/terraform/azure-workstation

az login
az account list --output table
az group list --output table

# infra
terraform init
terraform state list
# terraform untaint azurerm_network_interface.this
terraform taint azurerm_public_ip.this
terraform taint azurerm_linux_virtual_machine.workstation-nc24
terraform apply
terraform plan
terraform apply -auto-approve
terraform output

# update public-ip
terraform taint azurerm_public_ip.this
az network nic ip-config list -g machine-learning --nic-name workspace-nic
az network nic ip-config update --name internal --nic-name workspace-nic --resource-group machine-learning --remove PublicIPAddress

# provisioning
cd /home/izlobin/ws/transformers-labs/terraform/azure-workstation
PUBLIC_IP=$(terraform output public_ip_address | tr -d '"')
echo $PUBLIC_IP
echo $PUBLIC_IP | x
nmap -Pn -p 22,80,8080,6006 $PUBLIC_IP

ssh izlobin@$PUBLIC_IP
ssh izlobin@$PUBLIC_IP "whoami; pwd; ls -la"
ssh -L 16006:localhost:6006 izlobin@$PUBLIC_IP

ls ~/back.tar.gz
scp ~/back.tar.gz izlobin@$PUBLIC_IP:~/back.tar.gz
scp ~/.ssh/id_rsa izlobin@$PUBLIC_IP:~/.ssh/id_rsa
ssh izlobin@$PUBLIC_IP "tar xvzf ~/back.tar.gz"
ssh izlobin@$PUBLIC_IP "cat ~/.bashrc.iz >> ~/.bashrc"

```

## azure workstation disk
```sh
# disk init
ls -alF /dev/disk/azure/scsi1/
VOLUME=$(ls -alF /dev/disk/azure/scsi1/ | awk -F" " '$9 ~ /lun10/ {split($11,a,"/"); print a[length(a)]}')
VOLUME_DEV="/dev/$VOLUME"
echo $VOLUME_DEV
sudo mount -o rw $VOLUME_DEV /home/izlobin-new
ls -la /home/izlobin-new
sudo umount /home/izlobin-new

# UUID=$(sudo blkid -s UUID -o value /dev/disk/cloud/azure_resource-part1)
UUID=$(sudo blkid -s UUID -o value $VOLUME_DEV)
sudo cp /etc/fstab /etc/fstab.bak
sudo cp /etc/fstab.bak /etc/fstab
# cat /etc/fstab.bak
# sudo sed -iE "s|/dev/disk/cloud/azure_resource-part1.*\(auto.*\)|UUID=$UUID /home/izlobin \1|" /etc/fstab
sudo sed -iE "s|.*azure_resource-part1.*\(auto.*\)|UUID=$UUID /home/izlobin \1|" /etc/fstab
# sudo sed -iE "s|.*/home/izlobin.*\(auto.*\)|UUID=$UUID /home/izlobin \1|" /etc/fstab
cat /etc/fstab
sudo mount -a remount
ls -la /home/izlobin

# disk resize
df -h
lsblk
sudo fdisk -l
sudo resize2fs /dev/sda

sudo ls -alF /dev/disk/azure/scsi1/

echo 1 | sudo tee /sys/class/block/sdc/device/rescan
sudo fdisk -l /dev/sdc

echo 1 | sudo tee /sys/class/block/sda/device/rescan
sudo fdisk -l /dev/sda

df -Th

# swap
sudo fdisk -l /dev/sdc
# sudo swapon /dev/sdc


```


# azure spot prices
* https://learn.microsoft.com/en-us/rest/api/cost-management/retail-prices/azure-retail-prices
```sh
curl https://prices.azure.com/api/retail/prices?filter=serviceName eq 'Virtual Machines' > "azure-spot-prices/virtual-machines.json"
curl https://prices.azure.com/api/retail/prices?filter=serviceFamily eq 'Compute' > "azure-spot-prices/compute.json"

```
