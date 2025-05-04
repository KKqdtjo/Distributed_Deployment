# 部署环境配置指南

本文档详细介绍如何配置华为云ECS环境用于DeepSeek模型的分布式部署。

## 目录

- [基础环境配置](#基础环境配置)
- [CUDA和驱动安装](#CUDA和驱动安装)
- [Docker和NVIDIA-Docker安装](#Docker和NVIDIA-Docker安装)
- [Python环境配置](#Python环境配置)
- [依赖库安装](#依赖库安装)

## 基础环境配置

所有服务器节点都需要执行以下基础环境配置：

1. 更新系统并安装基础工具：

```bash
# 对于Ubuntu系统
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y build-essential wget curl git vim htop tmux nmon
```

2. 配置SSH免密登录（便于集群间通信）：

```bash
# 在头节点生成SSH密钥
ssh-keygen -t rsa -b 4096

# 将头节点的公钥复制到所有工作节点
ssh-copy-id username@worker-node-ip
```

3. 配置主机名解析（便于集群识别）：

```bash
# 在每个节点的/etc/hosts文件中添加所有节点信息
sudo vim /etc/hosts

# 添加类似以下内容
192.168.1.100 head-node
192.168.1.101 worker-node-1
192.168.1.102 worker-node-2
# ... 添加所有节点
```

4. 系统参数优化：

```bash
# 编辑系统限制配置
sudo vim /etc/security/limits.conf

# 添加以下内容
* soft nofile 1000000
* hard nofile 1000000
* soft nproc 65535
* hard nproc 65535

# 编辑sysctl配置
sudo vim /etc/sysctl.conf

# 添加以下内容
net.ipv4.tcp_fin_timeout = 30
net.ipv4.tcp_keepalive_time = 300
net.ipv4.tcp_max_syn_backlog = 8192
net.ipv4.tcp_max_tw_buckets = 5000
net.ipv4.tcp_tw_reuse = 1
net.ipv4.ip_local_port_range = 10000 65000
net.core.somaxconn = 65535
vm.overcommit_memory = 1

# 应用sysctl配置
sudo sysctl -p
```

## CUDA和驱动安装

在所有GPU节点上执行以下操作：

1. 安装NVIDIA驱动：

```bash
# 添加NVIDIA软件源
sudo add-apt-repository ppa:graphics-drivers/ppa -y
sudo apt-get update

# 安装最新驱动（此处以525为例，请根据实际需求选择驱动版本）
sudo apt-get install -y nvidia-driver-525
```

2. 重启系统并验证驱动安装：

```bash
sudo reboot
nvidia-smi
```

3. 安装CUDA（支持DeepSeek模型，此处以CUDA 11.8为例）：

```bash
# 下载CUDA安装包
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run

# 安装CUDA
sudo sh cuda_11.8.0_520.61.05_linux.run --toolkit --silent

# 配置环境变量
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## Docker和NVIDIA-Docker安装

在所有节点上执行以下操作：

1. 安装Docker：

```bash
# 设置Docker仓库
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

# 安装Docker
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# 将当前用户添加到docker组，无需sudo即可运行docker命令
sudo usermod -aG docker $USER
```

2. 在GPU节点上安装NVIDIA Docker：

```bash
# 设置NVIDIA Docker仓库
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# 安装NVIDIA Docker
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# 重启Docker服务
sudo systemctl restart docker
```

3. 验证NVIDIA Docker安装：

```bash
# 测试NVIDIA Docker是否正常工作
docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```

## Python环境配置

在所有节点上执行以下操作：

1. 安装Miniconda：

```bash
# 下载Miniconda安装脚本
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 安装Miniconda
bash Miniconda3-latest-Linux-x86_64.sh -b

# 初始化Miniconda
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

2. 创建Python环境：

```bash
# 创建Python 3.10环境（适用于DeepSeek和vLLM）
conda create -n deepseek python=3.10 -y
conda activate deepseek
```

## 依赖库安装

在所有节点上执行以下操作：

1. 安装基础Python依赖：

```bash
# 安装必要的Python库
pip install pydantic==1.10.8
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install accelerate==0.20.3
pip install transformers==4.33.2
pip install optimum==1.9.1
pip install sentencepiece==0.1.99
```

2. 安装Ray和vLLM：

```bash
# 安装Ray（用于分布式框架）
pip install ray==2.5.1 ray[default]==2.5.1 ray[serve]==2.5.1

# 安装vLLM（高性能推理引擎）
pip install vllm==0.2.0
```

3. 安装DeepSeek模型相关依赖：

```bash
# 克隆DeepSeek仓库
git clone https://github.com/deepseek-ai/DeepSeek-LLM.git
cd DeepSeek-LLM

# 安装依赖
pip install -e .
```

## 验证环境

执行以下命令验证环境配置是否正确：

```bash
# 验证PyTorch和CUDA
python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA是否可用:', torch.cuda.is_available()); print('可用GPU数量:', torch.cuda.device_count())"

# 验证Ray
python -c "import ray; ray.init(); print('Ray版本:', ray.__version__); ray.shutdown()"

# 验证vLLM
python -c "import vllm; print('vLLM版本:', vllm.__version__)"
```

至此，您已经完成了DeepSeek模型分布式部署的基本环境配置。接下来可以进行网络和存储配置，以及Ray集群的部署。
