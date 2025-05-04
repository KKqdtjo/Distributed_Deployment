# Ray集群部署指南

本文档详细介绍如何在华为云ECS上部署Ray集群，作为DeepSeek模型分布式部署的基础框架。

## 目录

- [Ray框架介绍](#Ray框架介绍)
- [集群架构设计](#集群架构设计)
- [头节点配置](#头节点配置)
- [工作节点配置](#工作节点配置)
- [集群启动与验证](#集群启动与验证)
- [集群监控与管理](#集群监控与管理)
- [故障恢复](#故障恢复)

## Ray框架介绍

Ray是一个用于构建分布式应用程序的开源框架，特别适合AI和机器学习工作负载。在DeepSeek模型部署中，我们使用Ray作为分布式计算的基础，结合vLLM进行高效的模型推理。

Ray的主要优势：

- 分布式任务调度和资源管理
- 内置的容错和扩展机制
- 高效的进程间通信
- 与Python生态系统紧密集成
- 专为AI应用优化的性能

## 集群架构设计

我们的Ray集群架构如下：

1. **Ray头节点(Head Node)**：负责集群管理、任务调度和资源分配
2. **Ray工作节点(Worker Nodes)**：提供计算资源，执行模型推理任务
3. **Redis服务器**：用于Ray的对象存储和任务队列

## 头节点配置

1. 登录到头节点服务器：

```bash
ssh username@head-node-ip
```

2. 确认已完成环境配置，包括Python、CUDA和必要的依赖库（详见[环境配置](./environment_setup.md)）。

3. 配置Ray头节点：

```bash
# 创建Ray配置目录
sudo mkdir -p /etc/ray
sudo chown -R $USER:$USER /etc/ray

# 创建头节点配置文件
cat > /etc/ray/head-node.yaml << EOF
cluster_name: deepseek-cluster

# 头节点配置
head_node:
  resources: {"CPU": 32, "memory": 64000000000}  # 32 CPU, 64GB内存

# 工作节点配置（仅供参考，工作节点会单独启动）
worker_nodes:
  worker_node_types:
    gpu_worker:
      min_workers: 0
      max_workers: 10
      resources: {"GPU": 4, "CPU": 64, "memory": 256000000000}  # 4 GPU, 64 CPU, 256GB内存

# Redis配置
redis:
  password: "StrongRedisPassword123!"
  port: 6379

# Ray对象存储配置
object_store:
  memory: 128000000000  # 128GB

# 系统配置
system_config:
  plasma_directory: "/mnt/deepseek-shared/ray-plasma"
  memory_monitor_refresh_ms: 500

# 网络配置
network:
  ssh_user: "ubuntu"
  ssh_private_key: "/home/ubuntu/.ssh/id_rsa"
EOF
```

4. 启动Ray头节点：

```bash
# 创建对象存储目录
sudo mkdir -p /mnt/deepseek-shared/ray-plasma
sudo chown -R $USER:$USER /mnt/deepseek-shared/ray-plasma

# 启动Ray头节点
ray start --head \
  --port=6379 \
  --redis-password="StrongRedisPassword123!" \
  --object-store-memory=128000000000 \
  --num-cpus=32 \
  --system-config='{"plasma_directory":"/mnt/deepseek-shared/ray-plasma","memory_monitor_refresh_ms":500}' \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265
```

5. 验证头节点启动：

```bash
# 检查Ray服务状态
ray status

# 检查Redis端口是否开放
netstat -tulpn | grep 6379
```

6. 设置Ray头节点开机自启动：

```bash
# 创建systemd服务文件
sudo tee /etc/systemd/system/ray-head.service > /dev/null << EOF
[Unit]
Description=Ray Head Node Service
After=network.target

[Service]
Type=simple
User=${USER}
ExecStart=/bin/bash -c 'ray start --head --port=6379 --redis-password="StrongRedisPassword123!" --object-store-memory=128000000000 --num-cpus=32 --system-config={"plasma_directory":"/mnt/deepseek-shared/ray-plasma","memory_monitor_refresh_ms":500} --dashboard-host=0.0.0.0 --dashboard-port=8265'
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# 启用并启动服务
sudo systemctl daemon-reload
sudo systemctl enable ray-head
sudo systemctl start ray-head
```

## 工作节点配置

对每个工作节点执行以下操作：

1. 登录到工作节点：

```bash
ssh username@worker-node-ip
```

2. 确认已完成环境配置，包括Python、CUDA和必要的依赖库（详见[环境配置](./environment_setup.md)）。

3. 配置Ray工作节点：

```bash
# 创建Ray配置目录
sudo mkdir -p /etc/ray
sudo chown -R $USER:$USER /etc/ray

# 创建工作节点配置文件
cat > /etc/ray/worker-node.yaml << EOF
# 工作节点配置
resources:
  GPU: 4      # 4 GPU
  CPU: 64     # 64 CPU
  memory: 256000000000  # 256GB内存

# 对象存储配置
object_store:
  memory: 128000000000  # 128GB

# 系统配置
system_config:
  plasma_directory: "/mnt/deepseek-shared/ray-plasma"
  memory_monitor_refresh_ms: 500
EOF
```

4. 启动Ray工作节点（连接到头节点）：

```bash
# 创建对象存储目录
sudo mkdir -p /mnt/deepseek-shared/ray-plasma
sudo chown -R $USER:$USER /mnt/deepseek-shared/ray-plasma

# 启动Ray工作节点
ray start --address=head-node-ip:6379 \
  --redis-password="StrongRedisPassword123!" \
  --object-store-memory=128000000000 \
  --num-cpus=64 \
  --num-gpus=4 \
  --resources='{"GPU": 4, "CPU": 64, "memory": 256000000000}' \
  --system-config='{"plasma_directory":"/mnt/deepseek-shared/ray-plasma","memory_monitor_refresh_ms":500}'
```

5. 验证工作节点连接：

```bash
# 检查Ray服务状态
ray status
```

6. 设置Ray工作节点开机自启动：

```bash
# 创建systemd服务文件
sudo tee /etc/systemd/system/ray-worker.service > /dev/null << EOF
[Unit]
Description=Ray Worker Node Service
After=network.target

[Service]
Type=simple
User=${USER}
ExecStart=/bin/bash -c 'ray start --address=head-node-ip:6379 --redis-password="StrongRedisPassword123!" --object-store-memory=128000000000 --num-cpus=64 --num-gpus=4 --resources={"GPU": 4, "CPU": 64, "memory": 256000000000} --system-config={"plasma_directory":"/mnt/deepseek-shared/ray-plasma","memory_monitor_refresh_ms":500}'
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# 启用并启动服务
sudo systemctl daemon-reload
sudo systemctl enable ray-worker
sudo systemctl start ray-worker
```

## 集群启动与验证

1. 确认所有节点已成功启动：

在头节点上执行：

```bash
# 查看集群状态
ray status
```

输出应该显示头节点和所有工作节点，以及每个节点的资源信息。

2. 验证集群性能：

```bash
# 创建简单的Ray测试脚本
cat > ray_test.py << EOF
import ray
import time
import numpy as np

ray.init(address="auto", _redis_password="StrongRedisPassword123!")

@ray.remote(num_cpus=1)
def f():
    time.sleep(1)
    return np.ones(1000000)

# 使用所有CPU创建并行任务
cpu_count = int(ray.cluster_resources()["CPU"])
results = ray.get([f.remote() for _ in range(cpu_count)])
print(f"Successfully ran {cpu_count} parallel tasks")

@ray.remote(num_gpus=1)
def g():
    time.sleep(1)
    return np.ones(1000000)

# 使用所有GPU创建并行任务
gpu_count = int(ray.cluster_resources().get("GPU", 0))
if gpu_count > 0:
    results = ray.get([g.remote() for _ in range(gpu_count)])
    print(f"Successfully ran {gpu_count} parallel GPU tasks")

ray.shutdown()
EOF

# 运行测试脚本
python ray_test.py
```

## 集群监控与管理

1. 访问Ray Dashboard：

通过浏览器访问Ray Dashboard：`http://head-node-ip:8265`

确保头节点安全组允许访问8265端口。

2. 设置预警监控：

```bash
# 安装监控工具
pip install prometheus-client

# 创建Ray监控脚本
cat > ray_monitor.py << EOF
import ray
import time
import socket
from prometheus_client import start_http_server, Gauge, Counter

# 指标定义
node_cpu_usage = Gauge('ray_node_cpu_usage', 'CPU usage of ray nodes', ['node_ip'])
node_memory_usage = Gauge('ray_node_memory_usage', 'Memory usage of ray nodes', ['node_ip'])
node_gpu_usage = Gauge('ray_node_gpu_usage', 'GPU usage of ray nodes', ['node_ip', 'gpu_id'])
cluster_tasks = Counter('ray_cluster_tasks', 'Number of tasks in the cluster')

def monitor_cluster():
    ray.init(address="auto", _redis_password="StrongRedisPassword123!")
    
    # 启动Prometheus导出器
    start_http_server(8000)
    
    while True:
        try:
            # 获取节点信息
            nodes = ray.nodes()
            
            for node in nodes:
                node_ip = node["NodeManagerAddress"]
                
                # 更新CPU指标
                node_cpu_usage.labels(node_ip=node_ip).set(node["Resources"]["CPU"] - node["AvailableResources"]["CPU"])
                
                # 更新内存指标
                node_memory_usage.labels(node_ip=node_ip).set(node["Resources"]["memory"] - node["AvailableResources"]["memory"])
                
                # 更新GPU指标（如果有）
                if "GPU" in node["Resources"]:
                    for gpu_id in range(int(node["Resources"]["GPU"])):
                        gpu_usage = 1.0 if gpu_id < int(node["Resources"]["GPU"] - node["AvailableResources"].get("GPU", 0)) else 0.0
                        node_gpu_usage.labels(node_ip=node_ip, gpu_id=str(gpu_id)).set(gpu_usage)
            
            # 更新任务计数
            tasks = ray.cluster_resources()
            cluster_tasks.inc(len(ray.tasks()))
            
        except Exception as e:
            print(f"Error in monitoring: {e}")
            
        time.sleep(5)  # 每5秒更新一次

if __name__ == "__main__":
    monitor_cluster()
EOF

# 将监控脚本设置为服务
sudo tee /etc/systemd/system/ray-monitor.service > /dev/null << EOF
[Unit]
Description=Ray Cluster Monitor
After=ray-head.service

[Service]
Type=simple
User=${USER}
ExecStart=/usr/bin/python /home/${USER}/ray_monitor.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# 启用并启动监控服务
sudo systemctl daemon-reload
sudo systemctl enable ray-monitor
sudo systemctl start ray-monitor
```

3. 集群扩缩容：

手动添加或移除工作节点：

```bash
# 添加工作节点（在新工作节点上运行）
ray start --address=head-node-ip:6379 --redis-password="StrongRedisPassword123!"

# 移除工作节点（在要移除的工作节点上运行）
ray stop
```

## 故障恢复

1. 头节点故障恢复：

```bash
# 检查头节点服务状态
sudo systemctl status ray-head

# 如果服务失败，尝试重启
sudo systemctl restart ray-head

# 查看日志以诊断问题
sudo journalctl -u ray-head
```

2. 工作节点故障恢复：

```bash
# 检查工作节点服务状态
sudo systemctl status ray-worker

# 如果服务失败，尝试重启
sudo systemctl restart ray-worker

# 查看日志以诊断问题
sudo journalctl -u ray-worker
```

3. 紧急情况下重置集群：

```bash
# 在所有节点上停止Ray服务
ray stop

# 在头节点上重新启动集群
ray start --head --port=6379 --redis-password="StrongRedisPassword123!"

# 在每个工作节点上重新连接到集群
ray start --address=head-node-ip:6379 --redis-password="StrongRedisPassword123!"
```

至此，您已成功配置Ray集群。接下来，可以在此基础上部署DeepSeek模型，详情请参考[DeepSeek模型部署](./deepseek_deployment.md)文档。
