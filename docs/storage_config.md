# 存储配置指南

本文档详细介绍华为云ECS上DeepSeek模型分布式部署的存储配置方案。

## 目录

- [存储需求分析](#存储需求分析)
- [存储方案选择](#存储方案选择)
- [OBS对象存储配置](#OBS对象存储配置)
- [EVS云硬盘配置](#EVS云硬盘配置)
- [SFS文件存储配置](#SFS文件存储配置)
- [Redis缓存集群配置](#Redis缓存集群配置)
- [数据备份与恢复策略](#数据备份与恢复策略)

## 存储需求分析

DeepSeek模型分布式部署的存储需求主要包括：

1. **模型权重存储**：存储大型预训练模型权重文件，需要高吞吐量和共享访问能力
2. **模型缓存**：用于存储模型推理过程中的中间结果，需要低延迟和高IOPS
3. **应用数据**：存储应用配置、日志和监控数据等
4. **分布式状态管理**：存储Ray集群的状态信息和任务队列等

根据不同的存储需求，我们将使用华为云提供的多种存储服务进行配置。

## 存储方案选择

为满足DeepSeek模型不同的存储需求，我们采用混合存储方案：

| 存储需求 | 存储方案 | 优势 |
|---------|---------|-----|
| 模型权重 | 华为云OBS + SFS | 高吞吐、共享访问、持久化 |
| 模型缓存 | 本地SSD + Redis | 低延迟、高IOPS |
| 应用数据 | EVS云硬盘 | 易用、弹性扩展 |
| 分布式状态 | Redis集群 | 高可用、低延迟 |

## OBS对象存储配置

华为云对象存储服务(OBS)用于存储DeepSeek模型权重文件，支持所有节点共享访问。

1. 创建OBS桶：

```
华为云管理控制台 -> 存储 -> 对象存储服务 -> 创建桶
```

配置参数：

- 桶名称：`deepseek-model-weights`
- 区域：选择与ECS实例相同的区域（降低访问延迟）
- 存储类别：标准存储
- 桶策略：私有
- 版本控制：开启（便于模型版本管理）

2. 上传模型权重：

可通过华为云OBS Browser+客户端或API上传DeepSeek模型权重文件。

```bash
# 安装OBS Python SDK
pip install esdk-obs-python

# 使用Python脚本上传模型文件
python upload_model.py
```

3. 在各节点上配置OBS挂载工具：

```bash
# 安装obsfs工具
wget https://obs-community.obs.cn-north-1.myhuaweicloud.com/obsfs/current/obsfs_Ubuntu16.04_amd64.deb
sudo dpkg -i obsfs_Ubuntu16.04_amd64.deb
sudo apt-get -f install

# 创建挂载点
sudo mkdir -p /mnt/obs-deepseek-models

# 挂载OBS桶
sudo obsfs deepseek-model-weights /mnt/obs-deepseek-models -o url=https://obs.cn-north-4.myhuaweicloud.com -o passwd_file=~/.passwd-obsfs

# 设置开机自动挂载（添加到/etc/fstab）
echo "obsfs#deepseek-model-weights /mnt/obs-deepseek-models fuse _netdev,url=https://obs.cn-north-4.myhuaweicloud.com,passwd_file=/root/.passwd-obsfs 0 0" | sudo tee -a /etc/fstab
```

## EVS云硬盘配置

为每个节点配置适当的EVS云硬盘：

1. 创建并挂载系统盘：

```
华为云管理控制台 -> 存储 -> 云硬盘 -> 创建云硬盘
```

配置建议：

- 类型：通用型SSD
- 容量：100GB（系统盘）

2. 为GPU节点创建并挂载数据盘：

```
华为云管理控制台 -> 存储 -> 云硬盘 -> 创建云硬盘
```

配置建议：

- 类型：超高IO型SSD
- 容量：1TB（或根据需求调整）
- 多盘组RAID：否

3. 在操作系统中格式化并挂载数据盘：

```bash
# 查看可用磁盘
lsblk

# 创建分区
sudo fdisk /dev/vdb

# 格式化分区
sudo mkfs.ext4 /dev/vdb1

# 创建挂载点
sudo mkdir -p /data

# 挂载数据盘
sudo mount /dev/vdb1 /data

# 设置开机自动挂载
echo "/dev/vdb1 /data ext4 defaults 0 0" | sudo tee -a /etc/fstab
```

## SFS文件存储配置

使用华为云弹性文件服务(SFS)提供共享文件系统，用于存储需要多节点访问的数据：

1. 创建SFS文件系统：

```
华为云管理控制台 -> 存储 -> 弹性文件服务 -> 创建文件系统
```

配置参数：

- 文件系统名称：`deepseek-shared`
- 协议类型：NFS
- 类型：SFS Turbo（高性能）
- 容量：500GB（或根据需求调整）
- VPC和子网：选择compute-subnet

2. 在各节点上挂载SFS文件系统：

```bash
# 安装NFS客户端
sudo apt-get install -y nfs-common

# 创建挂载点
sudo mkdir -p /mnt/deepseek-shared

# 挂载SFS文件系统（替换为实际的挂载地址）
sudo mount -t nfs -o vers=3,timeo=600,noresvport,nolock sfs-turbo-ip:/share /mnt/deepseek-shared

# 设置开机自动挂载
echo "sfs-turbo-ip:/share /mnt/deepseek-shared nfs vers=3,timeo=600,noresvport,nolock 0 0" | sudo tee -a /etc/fstab
```

## Redis缓存集群配置

配置Redis集群用于分布式状态管理和缓存：

1. 在Redis节点上安装Redis：

```bash
# 安装Redis服务器
sudo apt-get install -y redis-server

# 配置Redis（启用集群模式）
sudo vim /etc/redis/redis.conf
```

主要配置项：

```
# 绑定所有接口（允许远程访问）
bind 0.0.0.0

# 设置密码
requirepass StrongPassword123!

# 启用集群模式
cluster-enabled yes
cluster-config-file nodes.conf
cluster-node-timeout 5000

# 禁用持久化（提高性能，注意数据可能丢失）
appendonly no
save ""

# 内存优化
maxmemory 100gb
maxmemory-policy allkeys-lru
```

2. 启动Redis服务：

```bash
# 重启Redis服务
sudo systemctl restart redis-server
```

3. 创建Redis集群：

```bash
# 安装Redis集群工具
sudo apt-get install -y redis-tools

# 创建Redis集群（替换为实际的节点IP）
redis-cli --cluster create \
  192.168.3.10:6379 \
  192.168.3.11:6379 \
  192.168.3.12:6379 \
  --cluster-replicas 0 \
  -a StrongPassword123!
```

## 数据备份与恢复策略

实施定期备份策略，保护重要数据：

1. OBS桶版本控制：

确保对象存储开启版本控制，定期为关键文件创建快照。

2. EVS云硬盘备份：

```
华为云管理控制台 -> 存储 -> 云硬盘备份 -> 创建备份策略
```

配置每周备份，保留最近4个备份。

3. 配置文件备份：

```bash
# 创建配置文件备份脚本
cat > backup_configs.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/mnt/deepseek-shared/backups/configs"
TIMESTAMP=$(date +%Y%m%d%H%M%S)
mkdir -p ${BACKUP_DIR}
tar -czf ${BACKUP_DIR}/configs_${TIMESTAMP}.tar.gz /etc/redis /etc/ray /etc/nginx /path/to/app/configs
EOF

# 添加执行权限
chmod +x backup_configs.sh

# 添加到crontab，每天执行一次
(crontab -l 2>/dev/null; echo "0 2 * * * /path/to/backup_configs.sh") | crontab -
```

## 性能优化

1. 系统IO调优：

```bash
# 安装工具
sudo apt-get install -y fio iotop

# 调整IO调度器为deadline（适合SSD）
echo 'deadline' | sudo tee /sys/block/vdb/queue/scheduler

# 增加读写缓冲区大小
sudo sysctl -w vm.dirty_ratio=80
sudo sysctl -w vm.dirty_background_ratio=5
```

2. 文件系统挂载优化：

```bash
# 重新挂载数据盘，使用优化选项
sudo mount -o remount,noatime,nodiratime,discard /data

# 更新/etc/fstab
sudo sed -i 's|/dev/vdb1 /data ext4 defaults|/dev/vdb1 /data ext4 noatime,nodiratime,discard|g' /etc/fstab
```

至此，您已完成DeepSeek模型分布式部署的存储配置。这些配置将确保模型权重、缓存数据和应用状态等能够高效存储和访问，同时提供数据保护和备份机制。
