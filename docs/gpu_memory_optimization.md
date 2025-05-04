# GPU内存优化指南

本文档详细介绍如何优化DeepSeek模型分布式部署的GPU内存使用，实现更高效的资源利用。

## 目录

- [内存挑战分析](#内存挑战分析)
- [vLLM内存优化](#vLLM内存优化)
- [张量并行优化](#张量并行优化)
- [模型量化技术](#模型量化技术)
- [批处理与吞吐量优化](#批处理与吞吐量优化)
- [CUDA优化](#CUDA优化)
- [监控与调试](#监控与调试)

## 内存挑战分析

部署DeepSeek模型面临的主要内存挑战：

1. **模型权重占用**：
   - DeepSeek-LLM-7B：需要约14GB显存（FP16/BF16格式）
   - DeepSeek-LLM-67B：需要约134GB显存（FP16/BF16格式）

2. **注意力缓存**：对于长上下文生成，KV缓存会占用大量内存
   - 每生成1个token，需要存储所有前面token的KV值
   - 对于4096长度上下文，可能需要几GB额外内存

3. **批处理需求**：同时处理多个请求需要额外内存
   - 每增加一个并行请求，需要额外的KV缓存

## vLLM内存优化

vLLM通过PagedAttention技术实现高效内存管理：

1. **启用PagedAttention**：

```python
# 在vLLM配置中启用PagedAttention（默认启用）
engine_args = AsyncEngineArgs(
    model="/mnt/deepseek-shared/models/deepseek-llm-7b-chat",
    disable_paged_attention=False,  # 确保启用PagedAttention
    gpu_memory_utilization=0.9,  # 调整GPU内存利用率
    # 其他参数...
)
```

2. **调整块大小**：

```python
# 优化内存块大小
engine_args = AsyncEngineArgs(
    # 基础参数...
    block_size=16,  # KV缓存块大小（默认16）
)
```

3. **配置swap机制**（用于处理超长上下文）：

```python
# 启用CPU交换，处理特长序列
engine_args = AsyncEngineArgs(
    # 基础参数...
    swap_space=4,  # CPU交换空间，单位GB
    max_model_len=8192,  # 支持的最大序列长度
)
```

4. **内存利用率调整**：

```bash
# 测试不同内存利用率下的性能和稳定性
for util in 0.7 0.8 0.85 0.9 0.95; do
    echo "测试内存利用率: $util"
    python -c "
from vllm import LLM
model = LLM(
    model='/mnt/deepseek-shared/models/deepseek-llm-7b-chat',
    gpu_memory_utilization=$util,
    trust_remote_code=True
)
print('加载成功，显存利用率:', $util)
"
done
```

## 张量并行优化

对于大型模型（如67B版本），使用张量并行横跨多个GPU：

1. **基本张量并行配置**：

```python
# 4 GPU张量并行配置
engine_args = AsyncEngineArgs(
    model="/mnt/deepseek-shared/models/deepseek-llm-67b-chat",
    tensor_parallel_size=4,  # 使用4个GPU进行张量并行
    dtype="bfloat16",
)
```

2. **进阶参数调整**：

```python
# 微调张量并行设置
engine_args = AsyncEngineArgs(
    # 基础参数...
    max_parallel_loading_workers=2,  # 控制并行加载线程数
    enforce_eager=True,  # 使用PyTorch eager模式，提高部分情况下的性能
)
```

3. **混合并行策略**（用于超大模型）：

```python
# 张量并行和流水线并行混合使用
engine_args = AsyncEngineArgs(
    # 基础参数...
    tensor_parallel_size=4,  # 4-way 张量并行
    pipeline_parallel_size=2,  # 2-way 流水线并行
)
```

## 模型量化技术

量化技术可以显著减少模型内存占用：

1. **AWQ量化**（推荐方式，在精度和性能上较好）：

```python
# 使用AWQ量化DeepSeek-7B模型
engine_args = AsyncEngineArgs(
    model="/mnt/deepseek-shared/models/deepseek-llm-7b-chat",
    quantization="awq",
    dtype="half",  # AWQ推荐使用half精度
)
```

2. **GPTQ量化**（适用于部分设备）：

```python
# 对于GPTQ量化模型的加载
engine_args = AsyncEngineArgs(
    model="/mnt/deepseek-shared/models/deepseek-llm-7b-chat-gptq",
    quantization="gptq",
    dtype="half",
)
```

3. **量化模型转换脚本**：

```bash
# 创建AWQ量化转换脚本
cat > quantize_model.py << EOF
import os
import torch
from transformers import AutoTokenizer
from vllm.model_executor.quantization import quantize_awq

model_path = "/mnt/deepseek-shared/models/deepseek-llm-7b-chat"
output_path = "/mnt/deepseek-shared/models/deepseek-llm-7b-chat-awq"

# 加载原始模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 执行AWQ量化
quantize_awq(
    model_path=model_path,
    output_path=output_path,
    quant_config={
        "w_bit": 4,  # 4位量化
        "group_size": 128,  # 量化分组大小
        "zero_point": True,  # 是否使用零点
        "q_group_size": 128,  # 量化分组大小
    },
    tokenizer=tokenizer,
    trust_remote_code=True,
)

print(f"量化完成，模型保存在 {output_path}")
EOF

# 运行量化脚本
python quantize_model.py
```

4. **量化效果对比**：

| 量化方法 | 内存减少 | 推理速度 | 建议用例 |
|---------|---------|---------|---------|
| 无量化(BF16) | 0% | 100% | 高精度需求 |
| AWQ-4bit | ~75% | ~90% | 推荐平衡方案 |
| GPTQ-4bit | ~75% | ~80% | 广泛设备兼容 |
| INT8量化 | ~50% | ~95% | 简单快速部署 |

## 批处理与吞吐量优化

优化批处理设置提高GPU利用率：

1. **关键批处理参数**：

```python
# 优化批处理设置
engine_args = AsyncEngineArgs(
    # 基础参数...
    max_num_batched_tokens=4096,  # 每批次最大token数
    max_num_seqs=256,  # 最大序列数量
    max_paddings=256,  # 最大填充长度
)
```

2. **批处理大小调优脚本**：

```bash
# 创建批处理性能测试脚本
cat > batch_perf_test.py << EOF
import time
import torch
from vllm import LLM, SamplingParams
import numpy as np
import matplotlib.pyplot as plt

model_path = "/mnt/deepseek-shared/models/deepseek-llm-7b-chat"
batch_sizes = [1, 2, 4, 8, 16, 32, 64]
results = []

# 准备测试数据
sample_prompt = "请简洁回答：什么是人工智能？"
prompt_length = len(sample_prompt)

# 测试不同批处理大小
for batch_size in batch_sizes:
    print(f"测试批处理大小: {batch_size}")
    
    # 初始化模型
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
    )
    
    # 准备prompts
    prompts = [sample_prompt] * batch_size
    
    # 预热
    _ = llm.generate(prompts[:2], SamplingParams(max_tokens=10))
    
    # 正式测试
    torch.cuda.synchronize()
    start_time = time.time()
    
    outputs = llm.generate(prompts, SamplingParams(max_tokens=50))
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    # 计算吞吐量
    total_input_tokens = batch_size * prompt_length
    total_output_tokens = sum(len(output.outputs[0].text.split()) for output in outputs)
    throughput = total_output_tokens / (end_time - start_time)
    
    results.append({
        'batch_size': batch_size,
        'time': end_time - start_time,
        'throughput': throughput,
        'tokens_per_second': throughput
    })
    
    print(f"批处理大小 {batch_size}: {throughput:.2f} tokens/sec")
    
    # 清理
    del llm
    torch.cuda.empty_cache()
    time.sleep(2)

# 绘制结果
plt.figure(figsize=(10, 6))
plt.plot([r['batch_size'] for r in results], [r['tokens_per_second'] for r in results], 'o-')
plt.xlabel('批处理大小')
plt.ylabel('吞吐量 (tokens/sec)')
plt.title('批处理大小对吞吐量的影响')
plt.grid(True)
plt.savefig('batch_size_perf.png')
EOF

# 运行批处理性能测试
python batch_perf_test.py
```

3. **请求合并设置**：

```python
# 启用动态请求合并
engine_args = AsyncEngineArgs(
    # 基础参数...
    enable_chunked_prefill=True,  # 启用分块预填充
)
```

## CUDA优化

优化CUDA相关设置：

1. **CUDA内存分配优化**：

```bash
# 创建CUDA内存优化脚本
cat > cuda_optimize.py << EOF
import os
import torch

# 设置CUDA环境变量
os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'  # 限制CUDA连接数
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  # 限制分配块大小

# 显示CUDA信息
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA设备数量: {torch.cuda.device_count()}")
print(f"当前CUDA设备: {torch.cuda.current_device()}")
print(f"CUDA设备名称: {torch.cuda.get_device_name(0)}")

# 预分配显存（减少碎片）
torch.cuda.empty_cache()
dummy = torch.zeros(1024, 1024, 32, device='cuda')
del dummy
torch.cuda.empty_cache()

print("CUDA优化设置完成")
EOF

# 添加到启动脚本
echo "python cuda_optimize.py" >> start_deepseek_service.sh
```

2. **CUDA流优化**：

```python
# 在模型推理中使用CUDA流优化（添加到推理代码中）
with torch.cuda.stream(torch.cuda.Stream()):
    # 执行模型推理
    outputs = model(**inputs)
```

3. **设置GPU计算模式**：

```bash
# 在每个GPU节点上优化GPU计算模式
sudo nvidia-smi -ac 5001,1590  # 设置GPU的内存和图形时钟频率（根据GPU型号调整）

# 设置持久模式
sudo nvidia-smi -pm 1
```

## 监控与调试

1. **GPU内存监控脚本**：

```bash
# 创建GPU内存监控脚本
cat > monitor_gpu.py << EOF
import time
import datetime
import torch
import pandas as pd
import matplotlib.pyplot as plt
import threading
import nvidia_smi

def get_gpu_memory():
    nvidia_smi.nvmlInit()
    device_count = nvidia_smi.nvmlDeviceGetCount()
    results = []
    
    for i in range(device_count):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        
        results.append({
            'device': i,
            'total_memory': info.total / 1024**2,  # MB
            'used_memory': info.used / 1024**2,    # MB
            'free_memory': info.free / 1024**2,    # MB
            'utilization': info.used / info.total * 100  # %
        })
    
    nvidia_smi.nvmlShutdown()
    return results

def monitor_gpu(interval=1, duration=3600):
    """监控GPU内存使用情况"""
    records = []
    start_time = time.time()
    
    while time.time() - start_time < duration:
        timestamp = datetime.datetime.now()
        memory_info = get_gpu_memory()
        
        for device_info in memory_info:
            records.append({
                'timestamp': timestamp,
                'device': device_info['device'],
                'used_memory_mb': device_info['used_memory'],
                'utilization_percent': device_info['utilization']
            })
        
        time.sleep(interval)
    
    # 转换为DataFrame并保存
    df = pd.DataFrame(records)
    df.to_csv('gpu_memory_log.csv', index=False)
    
    # 绘制内存使用情况
    plt.figure(figsize=(12, 6))
    for device in df['device'].unique():
        device_data = df[df['device'] == device]
        plt.plot(device_data['timestamp'], device_data['used_memory_mb'], 
                 label=f'GPU {device}')
    
    plt.xlabel('时间')
    plt.ylabel('使用内存 (MB)')
    plt.title('GPU内存使用监控')
    plt.legend()
    plt.grid(True)
    plt.savefig('gpu_memory_usage.png')

# 后台启动监控
monitor_thread = threading.Thread(target=monitor_gpu, args=(5, 24*3600))
monitor_thread.daemon = True
monitor_thread.start()

print("GPU监控已启动，将在后台运行24小时")
EOF

# 安装依赖并运行监控
pip install nvidia-smi pandas matplotlib
python monitor_gpu.py &
```

2. **内存泄漏检测**：

```bash
# 创建内存泄漏检测脚本
cat > detect_leaks.py << EOF
import gc
import torch
from memory_profiler import profile

# 启用异常跟踪
torch._C._set_backcompat_keepdim_warning(True)
torch.autograd.set_detect_anomaly(True)

@profile
def test_inference():
    # 加载模型（使用小批量测试）
    from vllm import LLM, SamplingParams
    
    model = LLM(
        model="/mnt/deepseek-shared/models/deepseek-llm-7b-chat",
        trust_remote_code=True,
        dtype="bfloat16",
    )
    
    # 运行多次推理检测内存变化
    for i in range(10):
        prompts = ["你好，请介绍一下你自己。"] * 5
        outputs = model.generate(prompts, SamplingParams(max_tokens=100))
        
        # 强制垃圾回收
        del outputs
        gc.collect()
        torch.cuda.empty_cache()
        
        # 打印内存状态
        print(f"迭代 {i+1} 完成")
        print(f"分配的内存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"缓存的内存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        print("-" * 50)

if __name__ == "__main__":
    test_inference()
EOF

# 安装依赖并运行检测
pip install memory_profiler
python -m memory_profiler detect_leaks.py
```

3. **CUDA事件追踪**：

```python
# 创建CUDA事件追踪分析器
def profile_cuda_events(func):
    def wrapper(*args, **kwargs):
        # 创建CUDA事件
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        # 记录开始事件
        start.record()
        
        # 执行函数
        result = func(*args, **kwargs)
        
        # 记录结束事件
        end.record()
        
        # 等待事件完成
        torch.cuda.synchronize()
        
        # 计算时间
        time_ms = start.elapsed_time(end)
        print(f"{func.__name__} 执行时间: {time_ms:.2f} ms")
        
        return result
    return wrapper

# 使用方法
@profile_cuda_events
def run_inference(model, inputs):
    return model(**inputs)
```

至此，您已经掌握了优化DeepSeek模型GPU内存使用的关键技术。通过这些优化，可以显著提高模型推理效率，更好地利用有限的GPU资源。
