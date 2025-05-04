# API服务配置指南

本文档详细介绍如何在DeepSeek模型分布式部署基础上配置公共API服务。

## 目录

- [API架构概述](#API架构概述)
- [FastAPI服务配置](#FastAPI服务配置)
- [API接口定义](#API接口定义)
- [认证与授权](#认证与授权)
- [限流与负载均衡](#限流与负载均衡)
- [日志与监控](#日志与监控)
- [API文档生成](#API文档生成)

## API架构概述

我们的API服务采用以下分层架构：

1. **前端负载均衡层**：Nginx负载均衡器，分发API请求
2. **API网关层**：处理认证、限流和请求路由
3. **应用服务层**：基于FastAPI的应用服务，处理业务逻辑
4. **推理服务层**：Ray Serve部署的DeepSeek模型
5. **监控层**：收集和展示API使用指标

整体架构图如下：

```
客户端 -> HTTPS负载均衡器 -> API网关 -> FastAPI服务 -> Ray Serve -> DeepSeek模型
                                          |
                                          v
                                      监控系统
```

## FastAPI服务配置

1. 安装必要的依赖：

```bash
# 安装FastAPI和相关依赖
pip install fastapi uvicorn pydantic python-jose[cryptography] passlib[bcrypt] prometheus-client
```

2. 创建FastAPI应用：

```bash
# 创建API服务目录
mkdir -p /home/${USER}/deepseek-api
cd /home/${USER}/deepseek-api

# 创建FastAPI应用主文件
cat > main.py << EOF
from fastapi import FastAPI, Depends, HTTPException, status, Request, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import httpx
import time
import json
import os
import uuid
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta

# 创建FastAPI应用
app = FastAPI(
    title="DeepSeek API",
    description="DeepSeek大语言模型API服务",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 身份验证配置
SECRET_KEY = os.getenv("API_SECRET_KEY", "YourSecretKeyHere")  # 生产环境中应使用环境变量
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# JWT相关模型
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# 用户相关模型
class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    api_key: Optional[str] = None

class UserInDB(User):
    hashed_password: str

# 聊天相关模型
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False

class ChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

# 密码哈希工具
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# 模拟用户数据库（生产环境应使用实际数据库）
fake_users_db = {
    "admin": {
        "username": "admin",
        "full_name": "Administrator",
        "email": "admin@example.com",
        "hashed_password": pwd_context.hash("admin123"),
        "disabled": False,
        "api_key": "sk-api-key-123456"
    }
}

# 身份验证函数
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_user(username: str):
    if username in fake_users_db:
        user_dict = fake_users_db[username]
        return UserInDB(**user_dict)
    return None

def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# 日志记录
def log_request(username, request_data, response_data, process_time):
    # 实际生产环境应使用结构化日志并存储到适当的系统
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "username": username,
        "request": request_data,
        "response_summary": {
            "id": response_data.get("id"),
            "usage": response_data.get("usage"),
        },
        "process_time_ms": process_time
    }
    print(json.dumps(log_entry))

# API端点
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def create_chat_completion(
    request: ChatRequest, 
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user)
):
    start_time = time.time()
    
    # 转发请求到DeepSeek推理服务
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://127.0.0.1:8000/v1/chat/completions",  # 内部推理服务地址
                json=request.dict(),
                timeout=120.0  # 长超时，适应大型模型推理
            )
            response_data = response.json()
            
            # 添加请求日志记录
            process_time = (time.time() - start_time) * 1000
            background_tasks.add_task(
                log_request, 
                current_user.username, 
                request.dict(), 
                response_data, 
                process_time
            )
            
            return response_data
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Error communicating with model service: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "1.0.0"}

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
EOF
```

3. 创建API服务启动脚本：

```bash
cat > start_api.sh << EOF
#!/bin/bash
set -e

# 激活conda环境
source ~/miniconda3/bin/activate deepseek

# 设置环境变量
export API_SECRET_KEY="$(openssl rand -hex 32)"

# 启动API服务
uvicorn main:app --host 0.0.0.0 --port 8080 --workers 4
EOF

chmod +x start_api.sh
```

4. 创建systemd服务：

```bash
sudo tee /etc/systemd/system/deepseek-api.service > /dev/null << EOF
[Unit]
Description=DeepSeek API Service
After=network.target

[Service]
Type=simple
User=${USER}
WorkingDirectory=/home/${USER}/deepseek-api
ExecStart=/home/${USER}/deepseek-api/start_api.sh
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# 启用并启动服务
sudo systemctl daemon-reload
sudo systemctl enable deepseek-api
sudo systemctl start deepseek-api
```

## API接口定义

我们的API服务提供以下主要接口：

1. **认证接口**：
   - `/token` - 获取访问令牌（POST请求）

2. **聊天接口**：
   - `/v1/chat/completions` - 聊天补全（POST请求）

3. **用户接口**：
   - `/users/me` - 获取当前用户信息（GET请求）

4. **健康检查接口**：
   - `/health` - 服务健康检查（GET请求）

### API使用示例

使用curl调用API：

```bash
# 获取认证令牌
TOKEN=$(curl -s -X POST "http://api.deepseek.example.com/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin123" | jq -r .access_token)

# 调用聊天补全API
curl -X POST "http://api.deepseek.example.com/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "model": "deepseek-llm-7b-chat",
    "messages": [
      {"role": "user", "content": "请介绍一下分布式系统的CAP理论"}
    ],
    "temperature": 0.7,
    "max_tokens": 500
  }'
```

## 认证与授权

我们使用JWT（JSON Web Token）进行API认证，支持以下认证方式：

1. **基于令牌的认证**：
   - 用户通过提供用户名和密码获取访问令牌
   - 访问令牌在请求头中使用`Authorization: Bearer <token>`传递

2. **基于API密钥的认证**（适用于生产环境）：

```bash
# 添加API密钥认证中间件
# 在main.py中添加以下代码

from fastapi import Security
from fastapi.security.api_key import APIKeyHeader, APIKey

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    for username, user in fake_users_db.items():
        if api_key_header == user["api_key"]:
            return get_user(username)
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid API key"
    )

# 修改chat_completion端点以支持API密钥认证
@app.post("/v1/chat/completions", response_model=ChatResponse)
async def create_chat_completion(
    request: ChatRequest, 
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    api_key_user: User = Depends(get_api_key)
):
    # 用户可以通过JWT令牌或API密钥进行认证
    user = current_user or api_key_user
    # ... 其余代码不变
```

## 限流与负载均衡

1. 在FastAPI应用中添加限流中间件：

```bash
# 创建限流配置文件
cat > rate_limiter.py << EOF
from fastapi import Request, HTTPException
import time
from collections import defaultdict

class RateLimiter:
    def __init__(
        self,
        times: int = 10,  # 默认每分钟最多10次请求
        seconds: int = 60
    ):
        self.times = times
        self.seconds = seconds
        self.requests = defaultdict(list)  # 用户ID -> 请求时间列表
        
    async def __call__(self, request: Request, user_id: str):
        current_time = time.time()
        
        # 清理过期请求记录
        self.requests[user_id] = [
            req_time for req_time in self.requests[user_id]
            if current_time - req_time < self.seconds
        ]
        
        # 检查是否超出限制
        if len(self.requests[user_id]) >= self.times:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {self.times} requests per {self.seconds} seconds"
            )
            
        # 记录本次请求
        self.requests[user_id].append(current_time)
        
        return True

# 创建限流实例
chat_rate_limiter = RateLimiter(times=20, seconds=60)  # 每分钟20次聊天请求
token_rate_limiter = RateLimiter(times=5, seconds=60)  # 每分钟5次认证请求
EOF

# 在main.py中应用限流器
cat >> main.py << EOF
# 导入限流器
from rate_limiter import chat_rate_limiter, token_rate_limiter

# 修改登录端点
@app.post("/token", response_model=Token)
async def login_for_access_token(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends()
):
    # 应用限流
    client_ip = request.client.host
    await token_rate_limiter(request, client_ip)
    
    # 现有的认证逻辑...

# 修改聊天端点
@app.post("/v1/chat/completions", response_model=ChatResponse)
async def create_chat_completion(
    request: Request,
    chat_request: ChatRequest, 
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user)
):
    # 应用限流
    await chat_rate_limiter(request, current_user.username)
    
    # 现有的聊天逻辑...
EOF
```

2. 配置Nginx负载均衡：

```bash
# 创建Nginx负载均衡配置
sudo tee /etc/nginx/sites-available/deepseek-api-lb.conf > /dev/null << EOF
upstream api_servers {
    server 127.0.0.1:8080;
    # 如果部署了多个API服务实例，可以添加更多服务器
    # server 192.168.1.101:8080;
    # server 192.168.1.102:8080;
}

server {
    listen 80;
    server_name api.deepseek.example.com;
    
    # HTTP重定向到HTTPS
    return 301 https://\$host\$request_uri;
}

server {
    listen 443 ssl;
    server_name api.deepseek.example.com;
    
    # SSL配置
    ssl_certificate /etc/ssl/certs/deepseek-api.crt;
    ssl_certificate_key /etc/ssl/private/deepseek-api.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers on;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305;
    
    # 配置API服务代理
    location / {
        proxy_pass http://api_servers;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # 超时设置
        proxy_connect_timeout 10s;
        proxy_send_timeout 120s;
        proxy_read_timeout 120s;
        
        # 缓冲设置
        proxy_buffering on;
        proxy_buffer_size 16k;
        proxy_buffers 8 16k;
        
        # 限制请求大小
        client_max_body_size 1m;
    }
    
    # 提供API文档
    location /docs {
        proxy_pass http://api_servers/docs;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
    
    # 健康检查
    location /health {
        proxy_pass http://api_servers/health;
        access_log off;
    }
}
EOF

# 启用配置
sudo ln -s /etc/nginx/sites-available/deepseek-api-lb.conf /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## 日志与监控

1. 配置结构化日志：

```bash
# 创建日志配置文件
cat > logger.py << EOF
import json
import logging
import time
from datetime import datetime
from pythonjsonlogger import jsonlogger

# 自定义日志格式化器
class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['level'] = record.levelname
        log_record['name'] = record.name
        
        # 添加请求信息（如果可用）
        if hasattr(record, 'request_id'):
            log_record['request_id'] = record.request_id
        if hasattr(record, 'user'):
            log_record['user'] = record.user
        if hasattr(record, 'latency_ms'):
            log_record['latency_ms'] = record.latency_ms

# 配置日志处理器
def setup_logger():
    logger = logging.getLogger("deepseek_api")
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    file_handler = logging.FileHandler("/var/log/deepseek/api.log")
    formatter = CustomJsonFormatter('%(timestamp)s %(level)s %(name)s %(message)s')
    file_handler.setFormatter(formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 创建API访问日志记录函数
def log_api_request(logger, user, request_data, response_data, process_time):
    logger.info(
        "API request processed",
        extra={
            'request_id': response_data.get('id'),
            'user': user,
            'request_type': 'chat_completion',
            'model': request_data.get('model'),
            'input_tokens': response_data.get('usage', {}).get('prompt_tokens', 0),
            'output_tokens': response_data.get('usage', {}).get('completion_tokens', 0),
            'latency_ms': process_time
        }
    )
EOF

# 在main.py中应用日志配置
cat >> main.py << EOF
# 导入日志配置
from logger import setup_logger, log_api_request

# 初始化日志记录器
api_logger = setup_logger()

# 修改聊天端点以使用结构化日志
@app.post("/v1/chat/completions", response_model=ChatResponse)
async def create_chat_completion(
    request: ChatRequest, 
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user)
):
    start_time = time.time()
    
    # ... 推理处理代码 ...
    
    # 使用结构化日志记录请求
    process_time = (time.time() - start_time) * 1000
    background_tasks.add_task(
        log_api_request, 
        api_logger,
        current_user.username, 
        request.dict(), 
        response_data, 
        process_time
    )
    
    return response_data
EOF
```

2. 设置Prometheus监控：

```bash
# 创建监控指标配置
cat > metrics.py << EOF
from prometheus_client import Counter, Histogram, Gauge, Summary

# 请求计数器
api_requests_total = Counter(
    'api_requests_total',
    'Total count of API requests',
    ['endpoint', 'status', 'user']
)

# 响应时间直方图
api_request_duration = Histogram(
    'api_request_duration_seconds',
    'API request duration in seconds',
    ['endpoint', 'status'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0)
)

# 并发请求数量
api_concurrent_requests = Gauge(
    'api_concurrent_requests',
    'Number of concurrent API requests',
    ['endpoint']
)

# 令牌使用量汇总
api_token_usage = Summary(
    'api_token_usage',
    'Token usage statistics',
    ['model', 'token_type']
)
EOF

# 在main.py中集成Prometheus指标
cat >> main.py << EOF
# 导入Prometheus指标
from metrics import (
    api_requests_total, 
    api_request_duration, 
    api_concurrent_requests,
    api_token_usage
)
from prometheus_client import make_asgi_app

# 创建Prometheus指标端点
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# 添加中间件记录指标
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    path = request.url.path
    
    # 增加并发请求计数
    api_concurrent_requests.labels(endpoint=path).inc()
    
    start_time = time.time()
    
    try:
        response = await call_next(request)
        status = response.status_code
        
        # 记录请求计数
        api_requests_total.labels(
            endpoint=path,
            status=status,
            user="unknown"  # 实际应从请求上下文中获取
        ).inc()
        
        # 记录请求时长
        api_request_duration.labels(
            endpoint=path,
            status=status
        ).observe(time.time() - start_time)
        
        return response
    except Exception as e:
        # 记录失败请求
        api_requests_total.labels(
            endpoint=path,
            status=500,
            user="unknown"
        ).inc()
        raise e
    finally:
        # 减少并发请求计数
        api_concurrent_requests.labels(endpoint=path).dec()
EOF
```

## API文档生成

FastAPI自动生成Swagger文档，可通过以下URL访问：

- Swagger UI: `https://api.deepseek.example.com/docs`
- ReDoc: `https://api.deepseek.example.com/redoc`

您可以通过以下方式自定义API文档：

```bash
# 在main.py中更新FastAPI应用定义
app = FastAPI(
    title="DeepSeek API",
    description="""
    # DeepSeek大语言模型API服务
    
    本API服务提供对DeepSeek大语言模型的访问。
    
    ## 功能特点
    
    * 支持聊天补全
    * 支持流式输出
    * 多种认证方式
    * 使用限制和计费
    
    ## 使用示例
    
    ```python
    import requests
    
    # 获取认证令牌
    response = requests.post(
        "https://api.deepseek.example.com/token",
        data={"username": "your_username", "password": "your_password"}
    )
    token = response.json()["access_token"]
    
    # 发送聊天请求
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(
        "https://api.deepseek.example.com/v1/chat/completions",
        headers=headers,
        json={
            "model": "deepseek-llm-7b-chat",
            "messages": [
                {"role": "user", "content": "请写一首关于人工智能的诗"}
            ]
        }
    )
    print(response.json())
    ```
    """,
    version="1.0.0",
    contact={
        "name": "DeepSeek API Support",
        "email": "api-support@example.com",
    },
)
```

至此，您已经完成了DeepSeek模型API服务的配置。该API服务提供了安全、高效、易于使用的接口，可以为不同客户端提供DeepSeek模型的推理能力。
