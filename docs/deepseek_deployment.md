# DeepSeek模型部署指南

本文档详细介绍如何在Ray集群上部署DeepSeek模型，实现高效的分布式推理。

## 目录

- [DeepSeek模型介绍](#DeepSeek模型介绍)
- [vLLM框架介绍](#vLLM框架介绍)
- [模型权重下载与准备](#模型权重下载与准备)
- [单节点部署测试](#单节点部署测试)
- [分布式部署配置](#分布式部署配置)
- [推理服务配置](#推理服务配置)
- [性能调优](#性能调优)

## DeepSeek模型介绍

DeepSeek是一个强大的大语言模型，由DeepSeek AI开发，具有出色的理解能力和生成能力。在本部署中，我们使用DeepSeek-LLM系列模型，它具有以下特点：

- 支持中英双语
- 具有强大的代码理解和生成能力
- 提供多种规模（从7B到67B参数）的版本
- 支持多种任务如文本生成、问答、翻译等

## vLLM框架介绍

vLLM是一个高性能的LLM推理引擎，具有以下优势：

- PagedAttention技术，大幅降低GPU内存使用
- 高吞吐量和低延迟
- 支持多种优化和量化方法
- 与Ray无缝集成，实现分布式推理

## 模型权重下载与准备

1. 下载DeepSeek模型权重：

```bash
# 创建模型目录
mkdir -p /mnt/deepseek-shared/models

# 下载DeepSeek-LLM模型（以DeepSeek-LLM-7B-Chat为例）
cd /mnt/deepseek-shared/models
git lfs install
git clone https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat

# 对于更大的模型（如67B），可以使用以下命令
# git clone https://huggingface.co/deepseek-ai/deepseek-llm-67b-chat

# 验证模型文件
ls -la deepseek-llm-7b-chat
```

2. 准备模型配置：

```bash
# 创建模型配置目录
mkdir -p /mnt/deepseek-shared/configs

# 创建DeepSeek模型配置文件
cat > /mnt/deepseek-shared/configs/deepseek_config.json << EOF
{
    "model_name_or_path": "/mnt/deepseek-shared/models/deepseek-llm-7b-chat",
    "tensor_parallel_size": 2,
    "pipeline_parallel_size": 1,
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.9,
    "dtype": "bfloat16",
    "trust_remote_code": true,
    "disable_logs": false,
    "worker_use_ray": true,
    "max_concurrent_requests": 256
}
EOF
```

## 单节点部署测试

在进行分布式部署前，先在单个GPU节点上测试模型：

```bash
# 进入Python环境
conda activate deepseek

# 创建测试脚本
cat > test_deepseek.py << EOF
from vllm import LLM, SamplingParams

# 初始化DeepSeek模型
model_path = "/mnt/deepseek-shared/models/deepseek-llm-7b-chat"
llm = LLM(model=model_path, trust_remote_code=True, dtype="bfloat16")

# 定义采样参数
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512
)

# 测试输入
prompts = [
    "你好，请介绍一下自己。",
    "Write a Python function to calculate fibonacci numbers."
]

# 生成回答
outputs = llm.generate(prompts, sampling_params)

# 打印结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt}")
    print(f"Generated text: {generated_text}")
    print("-" * 50)
EOF

# 运行测试
python test_deepseek.py
```

## 分布式部署配置

在Ray集群上部署DeepSeek模型：

1. 创建部署脚本：

```bash
cat > deploy_deepseek.py << EOF
import ray
from ray import serve
from vllm.entrypoints.openai.api_server import app
from vllm.engine.arg_utils import AsyncEngineArgs
import os

# 连接到Ray集群
ray.init(address="auto", namespace="deepseek", _redis_password="StrongRedisPassword123!")

# 配置vLLM引擎参数
engine_args = AsyncEngineArgs(
    model="/mnt/deepseek-shared/models/deepseek-llm-7b-chat",
    tensor_parallel_size=2,  # 每个实例使用2个GPU进行张量并行
    max_model_len=4096,
    gpu_memory_utilization=0.9,
    dtype="bfloat16",
    trust_remote_code=True,
    worker_use_ray=True
)

# 部署配置
deployment_config = serve.deployment_config.deployment_config(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 8,  # 最大实例数
        "target_num_ongoing_requests_per_replica": 16
    },
    ray_actor_options={
        "num_gpus": 2,  # 每个副本使用2个GPU
        "num_cpus": 4,
        "memory": 32 * 1024 * 1024 * 1024,  # 32GB内存
    }
)

# 定义应用
@serve.deployment(name="deepseek-llm", **deployment_config)
class DeepSeekLLM:
    def __init__(self):
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.entrypoints.openai.api_server import OpenAIAPI
        
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.api = OpenAIAPI(self.engine)
        
    async def __call__(self, request):
        return await app(request)

# 部署应用
serve.run(DeepSeekLLM.bind(), route_prefix="/v1")

print("DeepSeek model deployed successfully!")
print("API accessible at: http://head-node-ip:8000/v1/")
EOF

# 在头节点上启动部署
python deploy_deepseek.py
```

2. 验证部署：

```bash
# 创建测试客户端脚本
cat > test_api.py << EOF
import requests
import json

# API端点
api_endpoint = "http://head-node-ip:8000/v1/chat/completions"

# 请求数据
payload = {
    "model": "deepseek-llm-7b-chat",
    "messages": [
        {"role": "user", "content": "你好，请简单介绍一下量子计算的基本原理。"}
    ],
    "temperature": 0.7,
    "max_tokens": 500
}

# 发送请求
response = requests.post(api_endpoint, json=payload)

# 打印结果
print(json.dumps(response.json(), indent=2, ensure_ascii=False))
EOF

# 运行测试
python test_api.py
```

## 推理服务配置

1. 创建Nginx配置为API提供反向代理：

```bash
# 安装Nginx
sudo apt-get install -y nginx

# 创建Nginx配置文件
sudo tee /etc/nginx/sites-available/deepseek-api.conf > /dev/null << EOF
server {
    listen 80;
    server_name api.deepseek.example.com;  # 替换为实际域名

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_read_timeout 300s;  # 增加超时时间，适应长输出
    }
}
EOF

# 启用配置
sudo ln -s /etc/nginx/sites-available/deepseek-api.conf /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

2. 创建启动脚本，确保服务自动启动：

```bash
cat > start_deepseek_service.sh << EOF
#!/bin/bash
set -e

# 激活Conda环境
source ~/miniconda3/bin/activate deepseek

# 启动DeepSeek服务
cd ~/deepseek-deployment
python deploy_deepseek.py
EOF

chmod +x start_deepseek_service.sh

# 创建systemd服务
sudo tee /etc/systemd/system/deepseek-service.service > /dev/null << EOF
[Unit]
Description=DeepSeek LLM API Service
After=network.target ray-head.service

[Service]
Type=simple
User=${USER}
WorkingDirectory=/home/${USER}/deepseek-deployment
ExecStart=/home/${USER}/start_deepseek_service.sh
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# 启用并启动服务
sudo systemctl daemon-reload
sudo systemctl enable deepseek-service
sudo systemctl start deepseek-service
```

## 性能调优

1. 模型量化（提高吞吐量）：

```bash
# 创建量化配置（适用于有限资源的环境）
cat > /mnt/deepseek-shared/configs/deepseek_quantized_config.json << EOF
{
    "model_name_or_path": "/mnt/deepseek-shared/models/deepseek-llm-7b-chat",
    "tensor_parallel_size": 2,
    "quantization": "awq",  # 可选: awq, squeezellm, gptq
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.9,
    "dtype": "half",
    "trust_remote_code": true,
    "worker_use_ray": true
}
EOF

# 修改部署脚本以使用量化模型
# 更新deploy_deepseek.py中的AsyncEngineArgs参数
```

2. 批处理优化：

```bash
# 更新部署脚本中的批处理参数
# 在deploy_deepseek.py的AsyncEngineArgs中添加以下参数
cat >> deploy_deepseek.py << EOF
# 添加批处理参数
engine_args = AsyncEngineArgs(
    # ... 现有参数 ...
    max_batch_size=32,  # 最大批处理大小
    max_batch_prefill_tokens=4096,  # 预填充阶段的最大批处理token数
    max_batch_total_tokens=32768,  # 总的最大批处理token数
)
EOF
```

3. 监控性能：

```bash
# 创建性能监控脚本
cat > monitor_performance.py << EOF
import time
import requests
import json
import threading
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 测试参数
api_endpoint = "http://head-node-ip:8000/v1/chat/completions"
num_requests = 100
concurrent_requests = 10
results = []

def send_request():
    payload = {
        "model": "deepseek-llm-7b-chat",
        "messages": [
            {"role": "user", "content": "Explain the theory of relativity in three sentences."}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    start_time = time.time()
    response = requests.post(api_endpoint, json=payload)
    end_time = time.time()
    
    latency = end_time - start_time
    success = response.status_code == 200
    
    results.append({
        "latency": latency,
        "success": success,
        "tokens": len(response.json()["choices"][0]["message"]["content"].split()) if success else 0
    })

# 创建并启动线程
threads = []
for i in range(num_requests):
    # 限制并发数
    while len(threads) >= concurrent_requests:
        for t in threads[:]:
            if not t.is_alive():
                threads.remove(t)
        time.sleep(0.1)
    
    t = threading.Thread(target=send_request)
    t.start()
    threads.append(t)

# 等待所有线程完成
for t in threads:
    t.join()

# 分析结果
latencies = [r["latency"] for r in results if r["success"]]
success_rate = sum(1 for r in results if r["success"]) / len(results)
tokens_per_second = sum(r["tokens"] for r in results if r["success"]) / sum(latencies)

print(f"测试结果摘要:")
print(f"成功率: {success_rate*100:.2f}%")
print(f"平均延迟: {np.mean(latencies):.2f}秒")
print(f"P50延迟: {np.percentile(latencies, 50):.2f}秒")
print(f"P95延迟: {np.percentile(latencies, 95):.2f}秒")
print(f"P99延迟: {np.percentile(latencies, 99):.2f}秒")
print(f"吞吐量: {tokens_per_second:.2f} tokens/秒")

# 生成延迟分布图
plt.figure(figsize=(10, 6))
plt.hist(latencies, bins=20)
plt.title(f'DeepSeek API延迟分布 ({datetime.now().strftime("%Y-%m-%d %H:%M")})')
plt.xlabel('延迟(秒)')
plt.ylabel('请求数')
plt.savefig('latency_distribution.png')
EOF

# 运行性能监控
python monitor_performance.py
```

至此，您已完成DeepSeek模型的分布式部署。该配置利用Ray的分布式能力和vLLM的高性能推理引擎，提供了一个高效、可扩展的大模型服务。

如需进一步优化模型性能或调整部署参数，请参考[性能优化](./inference_optimization.md)文档。
