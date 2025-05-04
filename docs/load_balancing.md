# 负载均衡优化指南

本文档详细介绍如何为DeepSeek模型分布式部署配置高效的负载均衡，以实现更好的服务可用性和性能。

## 目录

- [负载均衡架构](#负载均衡架构)
- [Nginx负载均衡配置](#Nginx负载均衡配置)
- [Ray Serve自动扩缩容](#Ray-Serve自动扩缩容)
- [请求路由策略](#请求路由策略)
- [会话保持机制](#会话保持机制)
- [健康检查配置](#健康检查配置)
- [故障转移机制](#故障转移机制)

## 负载均衡架构

我们采用多层负载均衡架构以确保系统的高可用性和可扩展性：

1. **外部负载均衡**：华为云ELB（弹性负载均衡）分发外部流量
2. **API网关层负载均衡**：Nginx反向代理分发请求到多个API服务实例
3. **推理服务负载均衡**：Ray Serve根据资源利用率分配推理任务

整体架构示意图：

```
                           外部流量
                              |
                              v
                        [华为云 ELB]
                         /        \
                        /          \
                       v            v
               [Nginx实例1]     [Nginx实例2]
                  /   \           /   \
                 /     \         /     \
                v       v       v       v
          [API服务1] [API服务2] [API服务3] [API服务4]
                \       /         \       /
                 \     /           \     /
                  v   v             v   v
              [Ray Serve头节点1]  [Ray Serve头节点2]
                /    |    \        /    |    \
               v     v     v      v     v     v
           [GPU1] [GPU2] [GPU3] [GPU4] [GPU5] [GPU6]
```

## Nginx负载均衡配置

### 基础负载均衡配置

```bash
# 安装Nginx
sudo apt-get install -y nginx

# 创建负载均衡配置
sudo tee /etc/nginx/sites-available/deepseek-lb.conf > /dev/null << EOF
upstream api_servers {
    # 定义多个API服务实例
    server 192.168.2.101:8080 weight=5 max_fails=3 fail_timeout=30s;
    server 192.168.2.102:8080 weight=5 max_fails=3 fail_timeout=30s;
    server 192.168.2.103:8080 weight=3 backup;  # 备用服务器，权重较低
    
    # 保持连接设置
    keepalive 32;
}

server {
    listen 80;
    listen [::]:80;
    server_name api.deepseek.example.com;
    
    # 重定向HTTP到HTTPS
    return 301 https://\$host\$request_uri;
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name api.deepseek.example.com;
    
    # SSL配置
    ssl_certificate /etc/ssl/certs/deepseek-api.crt;
    ssl_certificate_key /etc/ssl/private/deepseek-api.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers on;
    ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305';
    
    # 启用SSL会话缓存
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # 负载均衡配置
    location / {
        proxy_pass http://api_servers;
        proxy_http_version 1.1;
        proxy_set_header Connection "";  # 启用HTTP Keep-Alive
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # 超时设置
        proxy_connect_timeout 10s;
        proxy_send_timeout 120s;
        proxy_read_timeout 120s;
        
        # 缓冲设置
        proxy_buffers 16 16k;
        proxy_buffer_size 16k;
        
        # 错误处理
        proxy_next_upstream error timeout http_500 http_502 http_503 http_504;
        proxy_next_upstream_tries 3;
    }
    
    # 健康检查端点
    location /health {
        proxy_pass http://api_servers/health;
        proxy_next_upstream error timeout http_500 http_502 http_503 http_504;
        access_log off;
    }
    
    # 性能监控端点
    location /metrics {
        proxy_pass http://api_servers/metrics;
        # 仅允许内部访问
        allow 192.168.0.0/16;
        deny all;
    }
}
EOF

# 启用配置
sudo ln -s /etc/nginx/sites-available/deepseek-lb.conf /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 高级负载均衡配置

```bash
# 优化Nginx性能配置
sudo tee /etc/nginx/nginx.conf > /dev/null << EOF
user www-data;
worker_processes auto;  # 自动设置为CPU核心数
worker_rlimit_nofile 65535;  # 提高文件描述符限制
pid /run/nginx.pid;
include /etc/nginx/modules-enabled/*.conf;

events {
    worker_connections 10240;  # 增加每个worker连接数
    multi_accept on;  # 开启一次接受多个连接
    use epoll;  # 使用epoll事件模型
}

http {
    # 基础设置
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    server_tokens off;
    
    # 请求大小和缓冲区设置
    client_max_body_size 10m;
    client_body_buffer_size 128k;
    client_header_buffer_size 1k;
    large_client_header_buffers 4 4k;
    
    # 超时设置
    client_body_timeout 12;
    client_header_timeout 12;
    send_timeout 10;
    
    # 文件缓存设置
    open_file_cache max=1000 inactive=20s;
    open_file_cache_valid 30s;
    open_file_cache_min_uses 2;
    open_file_cache_errors on;
    
    # GZIP压缩
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_buffers 16 8k;
    gzip_http_version 1.1;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml;
    
    # 日志格式
    log_format main '\$remote_addr - \$remote_user [\$time_local] "\$request" '
                    '\$status \$body_bytes_sent "\$http_referer" '
                    '"\$http_user_agent" "\$http_x_forwarded_for" '
                    'rt=\$request_time uct=\$upstream_connect_time uht=\$upstream_header_time urt=\$upstream_response_time';
    
    access_log /var/log/nginx/access.log main;
    error_log /var/log/nginx/error.log warn;
    
    # 包含其他配置
    include /etc/nginx/conf.d/*.conf;
    include /etc/nginx/sites-enabled/*;
}
EOF

# 重新加载Nginx配置
sudo systemctl reload nginx
```

## Ray Serve自动扩缩容

### 基本扩缩容配置

```python
# 编辑Ray Serve部署配置，添加自动扩缩容逻辑
deployment_config = serve.deployment_config.deployment_config(
    autoscaling_config={
        "min_replicas": 1,  # 最小副本数
        "max_replicas": 8,  # 最大副本数
        "target_num_ongoing_requests_per_replica": 16,  # 每个副本的目标请求数
        "upscale_delay_s": 60,  # 扩容延迟（秒）
        "downscale_delay_s": 300,  # 缩容延迟（秒）
    },
    ray_actor_options={
        "num_gpus": 2,  # 每个副本使用的GPU数
        "num_cpus": 4,  # 每个副本使用的CPU数
        "memory": 32 * 1024 * 1024 * 1024,  # 32GB内存
    }
)
```

### 高级扩缩容策略

创建自定义扩缩容控制器：

```bash
# 创建自定义扩缩容控制器
cat > autoscale_controller.py << EOF
import ray
import time
import threading
import requests
import numpy as np
from ray import serve
from ray.serve.controller import BackendInfo, ReplicaState

class CustomAutoscaler(threading.Thread):
    def __init__(self, 
                 deployment_name="deepseek-llm", 
                 min_replicas=1, 
                 max_replicas=8, 
                 check_interval=30,
                 scale_up_threshold=0.7,  # CPU/GPU利用率阈值
                 scale_down_threshold=0.3,
                 scale_up_delay=60,
                 scale_down_delay=300):
        threading.Thread.__init__(self)
        self.daemon = True
        
        self.deployment_name = deployment_name
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.check_interval = check_interval
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.scale_up_delay = scale_up_delay
        self.scale_down_delay = scale_down_delay
        
        self.last_scale_up_time = 0
        self.last_scale_down_time = 0
        
        # 确保连接到Ray
        if not ray.is_initialized():
            ray.init(address="auto")
        
    def get_metrics(self):
        """从Prometheus获取CPU/GPU利用率指标"""
        try:
            # 假设Prometheus运行在监控服务器上
            response = requests.get("http://monitor-server:9090/api/v1/query", 
                                     params={"query": "ray_serve_deployment_replica_gpu_utilization"})
            if response.status_code == 200:
                results = response.json()["data"]["result"]
                for result in results:
                    if result["metric"]["deployment"] == self.deployment_name:
                        return float(result["value"][1])
            return None
        except Exception as e:
            print(f"获取指标错误: {e}")
            return None
    
    def get_current_replicas(self):
        """获取当前部署的副本数"""
        try:
            deployment_info = serve.get_deployment(self.deployment_name)
            return deployment_info.num_replicas
        except Exception as e:
            print(f"获取副本数错误: {e}")
            return None
    
    def scale_deployment(self, target_replicas):
        """扩展或缩减部署的副本数"""
        try:
            deployment = serve.get_deployment(self.deployment_name)
            deployment.options(num_replicas=target_replicas).deploy()
            print(f"部署 {self.deployment_name} 扩缩容至 {target_replicas} 个副本")
            return True
        except Exception as e:
            print(f"扩缩容错误: {e}")
            return False
    
    def run(self):
        """自动扩缩容主循环"""
        print(f"自动扩缩容控制器已启动，监控 {self.deployment_name}")
        while True:
            try:
                # 获取当前副本数和指标
                current_replicas = self.get_current_replicas()
                if current_replicas is None:
                    time.sleep(self.check_interval)
                    continue
                
                utilization = self.get_metrics()
                if utilization is None:
                    time.sleep(self.check_interval)
                    continue
                
                current_time = time.time()
                
                # 扩容逻辑
                if (utilization > self.scale_up_threshold and 
                    current_replicas < self.max_replicas and 
                    current_time - self.last_scale_up_time > self.scale_up_delay):
                    target_replicas = min(current_replicas + 1, self.max_replicas)
                    if self.scale_deployment(target_replicas):
                        self.last_scale_up_time = current_time
                
                # 缩容逻辑
                elif (utilization < self.scale_down_threshold and 
                      current_replicas > self.min_replicas and 
                      current_time - self.last_scale_down_time > self.scale_down_delay):
                    target_replicas = max(current_replicas - 1, self.min_replicas)
                    if self.scale_deployment(target_replicas):
                        self.last_scale_down_time = current_time
                
                print(f"当前状态: 副本数={current_replicas}, 利用率={utilization:.2f}")
            except Exception as e:
                print(f"自动扩缩容循环错误: {e}")
            
            time.sleep(self.check_interval)

# 启动自定义自动扩缩容控制器
if __name__ == "__main__":
    autoscaler = CustomAutoscaler()
    autoscaler.start()
    
    # 保持主线程运行
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("自动扩缩容控制器停止")
EOF

# 创建启动脚本
cat > start_autoscaler.sh << EOF
#!/bin/bash
source ~/miniconda3/bin/activate deepseek
python autoscale_controller.py
EOF

chmod +x start_autoscaler.sh

# 创建Systemd服务
sudo tee /etc/systemd/system/deepseek-autoscaler.service > /dev/null << EOF
[Unit]
Description=DeepSeek Autoscaler Service
After=network.target ray-head.service

[Service]
Type=simple
User=${USER}
WorkingDirectory=/home/${USER}/deepseek-deployment
ExecStart=/home/${USER}/deepseek-deployment/start_autoscaler.sh
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# 启用并启动服务
sudo systemctl daemon-reload
sudo systemctl enable deepseek-autoscaler
sudo systemctl start deepseek-autoscaler
```

## 请求路由策略

### 加权轮询策略

在Nginx负载均衡配置中使用加权轮询策略：

```bash
upstream api_servers {
    server 192.168.2.101:8080 weight=5;  # 权重高，获得更多请求
    server 192.168.2.102:8080 weight=3;
    server 192.168.2.103:8080 weight=2;
}
```

### 最少连接策略

在需要优化资源利用率的情况下使用最少连接策略：

```bash
upstream api_servers {
    least_conn;  # 使用最少连接策略
    server 192.168.2.101:8080;
    server 192.168.2.102:8080;
    server 192.168.2.103:8080;
}
```

### IP哈希策略

对于需要会话粘性的场景，使用IP哈希策略：

```bash
upstream api_servers {
    ip_hash;  # 使用IP哈希策略
    server 192.168.2.101:8080;
    server 192.168.2.102:8080;
    server 192.168.2.103:8080;
}
```

## 会话保持机制

### 基于Cookie的会话保持

启用基于Cookie的会话保持：

```bash
upstream api_servers {
    server 192.168.2.101:8080;
    server 192.168.2.102:8080;
    server 192.168.2.103:8080;
    
    # 启用基于Cookie的会话保持
    sticky cookie srv_id expires=1h domain=.deepseek.example.com path=/;
}
```

### 基于Ray会话的保持

对于需要在Ray级别保持会话的需求，使用自定义路由：

```python
# 创建基于会话ID的路由
@serve.deployment
class SessionRouter:
    def __init__(self):
        self.session_to_replica = {}
        
    async def route_request(self, session_id, request_data):
        # 根据会话ID路由到特定副本
        if session_id in self.session_to_replica:
            replica_id = self.session_to_replica[session_id]
            return await model_replicas[replica_id].handle_request.remote(request_data)
        else:
            # 分配新副本并记录
            # 这里简单使用轮询策略
            all_replicas = list(model_replicas.keys())
            replica_id = all_replicas[hash(session_id) % len(all_replicas)]
            self.session_to_replica[session_id] = replica_id
            return await model_replicas[replica_id].handle_request.remote(request_data)
```

## 健康检查配置

### Nginx健康检查配置

增强Nginx健康检查：

```bash
# 安装Nginx Plus或使用第三方模块
# 这里使用nginx_upstream_check_module
# 编译时需包含此模块

upstream api_servers {
    server 192.168.2.101:8080 max_fails=3 fail_timeout=30s;
    server 192.168.2.102:8080 max_fails=3 fail_timeout=30s;
    server 192.168.2.103:8080 max_fails=3 fail_timeout=30s;
    
    # 主动健康检查配置
    check interval=5000 rise=2 fall=3 timeout=1000 type=http;
    check_http_send "GET /health HTTP/1.0\r\n\r\n";
    check_http_expect_alive http_2xx http_3xx;
}
```

### Ray Serve健康检查

为Ray Serve部署添加健康检查端点：

```python
@serve.deployment
class HealthCheck:
    def __init__(self, target_deployments):
        self.target_deployments = target_deployments
        
    async def __call__(self, request):
        # 检查所有目标部署的健康状态
        results = {}
        for deployment_name in self.target_deployments:
            try:
                handle = serve.get_deployment(deployment_name).get_handle()
                response = await handle.check_health.remote()
                results[deployment_name] = {
                    "status": "healthy" if response else "unhealthy"
                }
            except Exception as e:
                results[deployment_name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                
        # 判断整体健康状态
        overall_healthy = all(r["status"] == "healthy" for r in results.values())
        return {"status": "healthy" if overall_healthy else "degraded", "components": results}

# 部署健康检查服务
serve.run(HealthCheck.bind(["deepseek-llm"]), route_prefix="/health")
```

## 故障转移机制

### Nginx故障转移配置

配置Nginx故障转移：

```bash
upstream api_servers {
    server 192.168.2.101:8080 max_fails=3 fail_timeout=30s;
    server 192.168.2.102:8080 max_fails=3 fail_timeout=30s backup;  # 备用服务器
    
    # 故障转移设置
    proxy_next_upstream error timeout http_500 http_502 http_503 http_504;
    proxy_next_upstream_timeout 10s;
    proxy_next_upstream_tries 3;
}
```

### Ray故障转移机制

配置Ray Serve的故障转移和恢复：

```python
# 编辑Ray Serve配置以增强容错性
serve.start(
    http_options={"host": "0.0.0.0", "port": 8000},
    _system_config={
        "max_pending_tasks_per_replica": 1000,
        "max_retries_on_actor_unavailable": 5,  # 当actor不可用时重试次数
        "health_check_period_s": 10,  # 健康检查间隔
        "health_check_failure_threshold": 3  # 健康检查失败阈值
    }
)
```

### 完整故障转移示例

创建包含故障转移逻辑的部署配置：

```bash
# 创建包含故障转移的部署配置
cat > deploy_with_failover.py << EOF
import ray
from ray import serve
import time
import threading
import os
import signal
import sys
from fastapi import FastAPI, Request
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.api_server import app as vllm_app

# 初始化Ray
ray.init(address="auto", namespace="deepseek", _redis_password="StrongRedisPassword123!")

# 启动Ray Serve
serve.start(
    http_options={"host": "0.0.0.0", "port": 8000},
    _system_config={
        "max_pending_tasks_per_replica": 1000,
        "max_retries_on_actor_unavailable": 5,
        "health_check_period_s": 10,
        "health_check_failure_threshold": 3
    }
)

# 配置vLLM引擎
engine_args = AsyncEngineArgs(
    model="/mnt/deepseek-shared/models/deepseek-llm-7b-chat",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.9,
    dtype="bfloat16",
    trust_remote_code=True,
    worker_use_ray=True
)

# 监控和恢复服务
class ServiceWatchdog(threading.Thread):
    def __init__(self, deployment_name, check_interval=30):
        threading.Thread.__init__(self)
        self.daemon = True
        self.deployment_name = deployment_name
        self.check_interval = check_interval
        self.running = True
        
    def run(self):
        while self.running:
            try:
                # 检查部署状态
                deployment = serve.get_deployment(self.deployment_name)
                status = deployment.status()
                
                # 检查每个副本状态
                replicas = status.replicas
                for replica_id, replica_info in replicas.items():
                    if replica_info.get("state") != "RUNNING":
                        print(f"检测到故障副本 {replica_id}，尝试重启...")
                        try:
                            # 尝试重启部署
                            deployment.restart()
                            print(f"部署 {self.deployment_name} 已重启")
                            break
                        except Exception as e:
                            print(f"重启失败: {e}")
            except Exception as e:
                print(f"监控检查错误: {e}")
                
            time.sleep(self.check_interval)
    
    def stop(self):
        self.running = False

# 部署配置
deployment_config = serve.deployment_config.deployment_config(
    autoscaling_config={
        "min_replicas": 2,  # 至少2个副本以确保高可用
        "max_replicas": 8,
        "target_num_ongoing_requests_per_replica": 16,
    },
    ray_actor_options={
        "num_gpus": 2,
        "num_cpus": 4,
        "memory": 32 * 1024 * 1024 * 1024,
        "resources": {"model_replica": 1},  # 自定义资源标签，用于分散部署
    },
    health_check_timeout_s=60,  # 健康检查超时
    health_check_period_s=30,    # 健康检查周期
)

# 主API应用
app = FastAPI()

# 转发到vLLM
@app.route("/{path:path}")
async def proxy_to_vllm(request: Request, path: str):
    return await vllm_app(request)

# 健康检查
@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "1.0"}

# DeepSeek LLM部署
@serve.deployment(name="deepseek-llm", route_prefix="/v1", **deployment_config)
class DeepSeekLLM:
    def __init__(self):
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    async def __call__(self, request: Request):
        return await app(request)
    
    async def check_health(self):
        # 简单检查模型是否加载
        return self.engine is not None

# 部署应用
deployment = DeepSeekLLM.deploy()

# 启动监控服务
watchdog = ServiceWatchdog("deepseek-llm")
watchdog.start()

# 优雅关闭处理
def handle_shutdown(signum, frame):
    print("接收到关闭信号，正在优雅关闭...")
    watchdog.stop()
    print("监控服务已停止")
    serve.shutdown()
    print("Ray Serve已关闭")
    sys.exit(0)

# 注册信号处理器
signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

# 保持脚本运行
try:
    while True:
        time.sleep(60)
except Exception as e:
    print(f"主循环错误: {e}")
    handle_shutdown(None, None)
EOF

# 运行带故障转移的部署
python deploy_with_failover.py
```

至此，您已完成DeepSeek模型的负载均衡配置。这些配置可以确保服务的高可用性和性能，并能够应对各种故障情况。
