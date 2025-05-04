# 监控系统搭建指南

本文档详细介绍如何为DeepSeek模型分布式部署配置完整的监控系统，实现性能跟踪和问题排查。

## 目录

- [监控架构设计](#监控架构设计)
- [Prometheus配置](#Prometheus配置)
- [Grafana仪表盘](#Grafana仪表盘)
- [AlertManager告警](#AlertManager告警)
- [Node-Exporter配置](#Node-Exporter配置)
- [GPU监控](#GPU监控)
- [API性能监控](#API性能监控)

## 监控架构设计

我们采用以下监控架构来监控DeepSeek模型的分布式部署：

```
                   +----------------+
                   |    Grafana     |
                   | (可视化仪表盘)  |
                   +--------+-------+
                            |
                            v
+------------+      +---------------+      +--------------+
| AlertManager|<---->|  Prometheus   |<---->| AlertManager |
| (告警处理)  |      | (指标收集存储) |      | (告警处理)   |
+------------+      +-------+-------+      +--------------+
                            |
            +---------------+----------------+
            |               |                |
            v               v                v
    +--------------+ +--------------+ +--------------+
    | Node Exporter | | GPU Exporter | | API Exporter |
    | (主机指标)    | | (GPU指标)    | | (API指标)    |
    +--------------+ +--------------+ +--------------+
            |               |                |
            v               v                v
    +--------------+ +--------------+ +--------------+
    |   服务器节点  | |   GPU节点    | |   API服务器  |
    |              | |              | |              |
    +--------------+ +--------------+ +--------------+
```

## Prometheus配置

### 安装 Prometheus

```bash
# 创建Prometheus用户
sudo useradd --no-create-home --shell /bin/false prometheus

# 创建所需目录
sudo mkdir -p /etc/prometheus /var/lib/prometheus

# 下载Prometheus
cd /tmp
wget https://github.com/prometheus/prometheus/releases/download/v2.40.7/prometheus-2.40.7.linux-amd64.tar.gz
tar xzf prometheus-2.40.7.linux-amd64.tar.gz

# 拷贝Prometheus文件
sudo cp prometheus-2.40.7.linux-amd64/prometheus /usr/local/bin/
sudo cp prometheus-2.40.7.linux-amd64/promtool /usr/local/bin/
sudo cp -r prometheus-2.40.7.linux-amd64/consoles /etc/prometheus
sudo cp -r prometheus-2.40.7.linux-amd64/console_libraries /etc/prometheus

# 设置权限
sudo chown -R prometheus:prometheus /etc/prometheus /var/lib/prometheus
sudo chown prometheus:prometheus /usr/local/bin/prometheus
sudo chown prometheus:prometheus /usr/local/bin/promtool
```

### 配置 Prometheus

```bash
# 创建Prometheus配置文件
sudo tee /etc/prometheus/prometheus.yml > /dev/null << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  scrape_timeout: 10s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - localhost:9093

rule_files:
  - "/etc/prometheus/rules/*.yml"

scrape_configs:
  # 监控Prometheus自身
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  # 监控主机系统指标
  - job_name: 'node_exporter'
    static_configs:
      - targets: ['head-node:9100', 'worker-node-1:9100', 'worker-node-2:9100', 'worker-node-3:9100', 'worker-node-4:9100']

  # 监控GPU指标
  - job_name: 'gpu_exporter'
    static_configs:
      - targets: ['worker-node-1:9835', 'worker-node-2:9835', 'worker-node-3:9835', 'worker-node-4:9835']

  # 监控Ray指标
  - job_name: 'ray_metrics'
    static_configs:
      - targets: ['head-node:8080']

  # 监控DeepSeek API
  - job_name: 'deepseek_api'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['frontend-1:8080', 'frontend-2:8080']

  # 监控Nginx
  - job_name: 'nginx'
    static_configs:
      - targets: ['frontend-1:9113', 'frontend-2:9113']
EOF

# 创建告警规则目录
sudo mkdir -p /etc/prometheus/rules
sudo chown prometheus:prometheus /etc/prometheus/rules

# 创建告警规则
sudo tee /etc/prometheus/rules/deepseek_alerts.yml > /dev/null << EOF
groups:
- name: deepseek_alerts
  rules:
  # GPU高温告警
  - alert: GPUHighTemperature
    expr: gpu_temperature_celsius > 80
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "GPU温度过高"
      description: "GPU {{ \$labels.gpu }} 温度达到 {{ \$value }}°C，超过安全阈值"

  # API高延迟告警
  - alert: APIHighLatency
    expr: api_request_duration_seconds{quantile="0.95"} > 2
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "API延迟过高"
      description: "API 95%请求延迟超过2秒，当前值: {{ \$value }}秒"

  # 节点内存使用率高告警
  - alert: NodeHighMemoryUsage
    expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100 > 90
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "节点内存使用率高"
      description: "节点 {{ \$labels.instance }} 内存使用率超过90%，当前值: {{ \$value | printf \"%.2f\" }}%"

  # GPU内存使用率高告警
  - alert: GPUHighMemoryUsage
    expr: gpu_memory_used_bytes / gpu_memory_total_bytes * 100 > 95
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "GPU内存使用率高"
      description: "GPU {{ \$labels.gpu }} 内存使用率超过95%，当前值: {{ \$value | printf \"%.2f\" }}%"

  # 节点负载高告警
  - alert: NodeHighLoad
    expr: node_load5 / count without(cpu, mode) (node_cpu_seconds_total{mode="idle"}) > 1.5
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "节点负载高"
      description: "节点 {{ \$labels.instance }} 5分钟平均负载超过CPU核心数的1.5倍，当前值: {{ \$value | printf \"%.2f\" }}"
EOF

# 创建Prometheus服务
sudo tee /etc/systemd/system/prometheus.service > /dev/null << EOF
[Unit]
Description=Prometheus Time Series Collection and Processing Server
Wants=network-online.target
After=network-online.target

[Service]
User=prometheus
Group=prometheus
Type=simple
ExecStart=/usr/local/bin/prometheus \
    --config.file /etc/prometheus/prometheus.yml \
    --storage.tsdb.path /var/lib/prometheus/ \
    --web.console.templates=/etc/prometheus/consoles \
    --web.console.libraries=/etc/prometheus/console_libraries \
    --web.listen-address=0.0.0.0:9090 \
    --web.enable-lifecycle

Restart=always

[Install]
WantedBy=multi-user.target
EOF

# 启动Prometheus服务
sudo systemctl daemon-reload
sudo systemctl enable prometheus
sudo systemctl start prometheus
```

## Grafana仪表盘

### 安装 Grafana

```bash
# 添加Grafana仓库
sudo apt-get install -y apt-transport-https software-properties-common
sudo add-apt-repository "deb https://packages.grafana.com/oss/deb stable main"
wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -
sudo apt-get update

# 安装Grafana
sudo apt-get install -y grafana

# 启动Grafana服务
sudo systemctl enable grafana-server
sudo systemctl start grafana-server
```

### 配置Grafana

1. 通过Web浏览器访问Grafana: `http://监控服务器IP:3000`
2. 使用默认凭据登录（admin/admin）并修改密码
3. 添加Prometheus数据源：

   - 名称: Prometheus
   - 类型: Prometheus
   - URL: <http://localhost:9090>
   - 访问方式: Server

4. 导入仪表盘：

```bash
# 创建Grafana仪表盘JSON文件
cat > deepseek_dashboard.json << EOF
{
  "annotations": {
    "list": [
      {
        "builtIn": 1,
        "datasource": "-- Grafana --",
        "enable": true,
        "hide": true,
        "iconColor": "rgba(0, 211, 255, 1)",
        "name": "Annotations & Alerts",
        "type": "dashboard"
      }
    ]
  },
  "editable": true,
  "gnetId": null,
  "graphTooltip": 0,
  "id": 1,
  "links": [],
  "panels": [
    {
      "aliasColors": {},
      "bars": false,
      "dashLength": 10,
      "dashes": false,
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {},
        "overrides": []
      },
      "fill": 1,
      "fillGradient": 0,
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 0,
        "y": 0
      },
      "hiddenSeries": false,
      "id": 2,
      "legend": {
        "avg": false,
        "current": false,
        "max": false,
        "min": false,
        "show": true,
        "total": false,
        "values": false
      },
      "lines": true,
      "linewidth": 1,
      "nullPointMode": "null",
      "options": {
        "alertThreshold": true
      },
      "percentage": false,
      "pluginVersion": "7.5.5",
      "pointradius": 2,
      "points": false,
      "renderer": "flot",
      "seriesOverrides": [],
      "spaceLength": 10,
      "stack": false,
      "steppedLine": false,
      "targets": [
        {
          "expr": "gpu_utilization",
          "interval": "",
          "legendFormat": "GPU {{gpu}}",
          "refId": "A"
        }
      ],
      "thresholds": [],
      "timeFrom": null,
      "timeRegions": [],
      "timeShift": null,
      "title": "GPU利用率",
      "tooltip": {
        "shared": true,
        "sort": 0,
        "value_type": "individual"
      },
      "type": "graph",
      "xaxis": {
        "buckets": null,
        "mode": "time",
        "name": null,
        "show": true,
        "values": []
      },
      "yaxes": [
        {
          "format": "percent",
          "label": null,
          "logBase": 1,
          "max": "100",
          "min": "0",
          "show": true
        },
        {
          "format": "short",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        }
      ],
      "yaxis": {
        "align": false,
        "alignLevel": null
      }
    },
    {
      "aliasColors": {},
      "bars": false,
      "dashLength": 10,
      "dashes": false,
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {},
        "overrides": []
      },
      "fill": 1,
      "fillGradient": 0,
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 12,
        "y": 0
      },
      "hiddenSeries": false,
      "id": 3,
      "legend": {
        "avg": false,
        "current": false,
        "max": false,
        "min": false,
        "show": true,
        "total": false,
        "values": false
      },
      "lines": true,
      "linewidth": 1,
      "nullPointMode": "null",
      "options": {
        "alertThreshold": true
      },
      "percentage": false,
      "pluginVersion": "7.5.5",
      "pointradius": 2,
      "points": false,
      "renderer": "flot",
      "seriesOverrides": [],
      "spaceLength": 10,
      "stack": false,
      "steppedLine": false,
      "targets": [
        {
          "expr": "gpu_memory_used_bytes / gpu_memory_total_bytes * 100",
          "interval": "",
          "legendFormat": "GPU {{gpu}}",
          "refId": "A"
        }
      ],
      "thresholds": [],
      "timeFrom": null,
      "timeRegions": [],
      "timeShift": null,
      "title": "GPU内存使用率",
      "tooltip": {
        "shared": true,
        "sort": 0,
        "value_type": "individual"
      },
      "type": "graph",
      "xaxis": {
        "buckets": null,
        "mode": "time",
        "name": null,
        "show": true,
        "values": []
      },
      "yaxes": [
        {
          "format": "percent",
          "label": null,
          "logBase": 1,
          "max": "100",
          "min": "0",
          "show": true
        },
        {
          "format": "short",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        }
      ],
      "yaxis": {
        "align": false,
        "alignLevel": null
      }
    },
    {
      "aliasColors": {},
      "bars": false,
      "dashLength": 10,
      "dashes": false,
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {},
        "overrides": []
      },
      "fill": 1,
      "fillGradient": 0,
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 0,
        "y": 8
      },
      "hiddenSeries": false,
      "id": 4,
      "legend": {
        "avg": false,
        "current": false,
        "max": false,
        "min": false,
        "show": true,
        "total": false,
        "values": false
      },
      "lines": true,
      "linewidth": 1,
      "nullPointMode": "null",
      "options": {
        "alertThreshold": true
      },
      "percentage": false,
      "pluginVersion": "7.5.5",
      "pointradius": 2,
      "points": false,
      "renderer": "flot",
      "seriesOverrides": [],
      "spaceLength": 10,
      "stack": false,
      "steppedLine": false,
      "targets": [
        {
          "expr": "api_requests_total",
          "interval": "",
          "legendFormat": "{{endpoint}}",
          "refId": "A"
        }
      ],
      "thresholds": [],
      "timeFrom": null,
      "timeRegions": [],
      "timeShift": null,
      "title": "API请求数量",
      "tooltip": {
        "shared": true,
        "sort": 0,
        "value_type": "individual"
      },
      "type": "graph",
      "xaxis": {
        "buckets": null,
        "mode": "time",
        "name": null,
        "show": true,
        "values": []
      },
      "yaxes": [
        {
          "format": "short",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": "0",
          "show": true
        },
        {
          "format": "short",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        }
      ],
      "yaxis": {
        "align": false,
        "alignLevel": null
      }
    },
    {
      "aliasColors": {},
      "bars": false,
      "dashLength": 10,
      "dashes": false,
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {},
        "overrides": []
      },
      "fill": 1,
      "fillGradient": 0,
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 12,
        "y": 8
      },
      "hiddenSeries": false,
      "id": 5,
      "legend": {
        "avg": false,
        "current": false,
        "max": false,
        "min": false,
        "show": true,
        "total": false,
        "values": false
      },
      "lines": true,
      "linewidth": 1,
      "nullPointMode": "null",
      "options": {
        "alertThreshold": true
      },
      "percentage": false,
      "pluginVersion": "7.5.5",
      "pointradius": 2,
      "points": false,
      "renderer": "flot",
      "seriesOverrides": [],
      "spaceLength": 10,
      "stack": false,
      "steppedLine": false,
      "targets": [
        {
          "expr": "api_request_duration_seconds{quantile=\"0.5\"}",
          "interval": "",
          "legendFormat": "P50 {{endpoint}}",
          "refId": "A"
        },
        {
          "expr": "api_request_duration_seconds{quantile=\"0.95\"}",
          "interval": "",
          "legendFormat": "P95 {{endpoint}}",
          "refId": "B"
        },
        {
          "expr": "api_request_duration_seconds{quantile=\"0.99\"}",
          "interval": "",
          "legendFormat": "P99 {{endpoint}}",
          "refId": "C"
        }
      ],
      "thresholds": [],
      "timeFrom": null,
      "timeRegions": [],
      "timeShift": null,
      "title": "API响应时间",
      "tooltip": {
        "shared": true,
        "sort": 0,
        "value_type": "individual"
      },
      "type": "graph",
      "xaxis": {
        "buckets": null,
        "mode": "time",
        "name": null,
        "show": true,
        "values": []
      },
      "yaxes": [
        {
          "format": "s",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": "0",
          "show": true
        },
        {
          "format": "short",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        }
      ],
      "yaxis": {
        "align": false,
        "alignLevel": null
      }
    }
  ],
  "refresh": "5s",
  "schemaVersion": 27,
  "style": "dark",
  "tags": [],
  "templating": {
    "list": []
  },
  "time": {
    "from": "now-1h",
    "to": "now"
  },
  "timepicker": {},
  "timezone": "",
  "title": "DeepSeek监控仪表盘",
  "uid": "deepseek",
  "version": 1
}
EOF
```

5. 在Grafana中导入上述JSON文件作为新仪表盘

## AlertManager告警

### 安装 AlertManager

```bash
# 创建AlertManager用户
sudo useradd --no-create-home --shell /bin/false alertmanager

# 创建所需目录
sudo mkdir -p /etc/alertmanager

# 下载AlertManager
cd /tmp
wget https://github.com/prometheus/alertmanager/releases/download/v0.24.0/alertmanager-0.24.0.linux-amd64.tar.gz
tar xzf alertmanager-0.24.0.linux-amd64.tar.gz

# 拷贝AlertManager文件
sudo cp alertmanager-0.24.0.linux-amd64/alertmanager /usr/local/bin/
sudo cp alertmanager-0.24.0.linux-amd64/amtool /usr/local/bin/

# 设置权限
sudo chown alertmanager:alertmanager /usr/local/bin/alertmanager
sudo chown alertmanager:alertmanager /usr/local/bin/amtool
```

### 配置 AlertManager

```bash
# 创建AlertManager配置文件
sudo tee /etc/alertmanager/alertmanager.yml > /dev/null << EOF
global:
  resolve_timeout: 5m
  smtp_smarthost: 'smtp.example.com:587'  # 替换为实际SMTP服务器
  smtp_from: 'alertmanager@example.com'   # 替换为实际发送地址
  smtp_auth_username: 'username'          # 替换为实际用户名
  smtp_auth_password: 'password'          # 替换为实际密码
  smtp_require_tls: true

# 告警路由
route:
  group_by: ['alertname', 'instance', 'severity']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 1h
  receiver: 'email-notifications'
  routes:
  - match:
      severity: critical
    receiver: 'pager-duty'
    continue: true
  - match:
      severity: warning
    receiver: 'email-notifications'

# 告警接收器
receivers:
- name: 'email-notifications'
  email_configs:
  - to: 'alerts@example.com'  # 替换为实际接收邮箱
    send_resolved: true

- name: 'pager-duty'
  webhook_configs:
  - url: 'https://example.pagerduty.com/webhook'  # 替换为实际PagerDuty webhook URL
    send_resolved: true

# 告警模板
templates:
- '/etc/alertmanager/template/*.tmpl'

# 禁止重复提示
inhibit_rules:
- source_match:
    severity: 'critical'
  target_match:
    severity: 'warning'
  equal: ['alertname', 'instance']
EOF

# 创建AlertManager服务
sudo tee /etc/systemd/system/alertmanager.service > /dev/null << EOF
[Unit]
Description=Alertmanager for Prometheus
Wants=network-online.target
After=network-online.target

[Service]
User=alertmanager
Group=alertmanager
Type=simple
ExecStart=/usr/local/bin/alertmanager \
  --config.file=/etc/alertmanager/alertmanager.yml \
  --storage.path=/var/lib/alertmanager \
  --web.external-url=http://localhost:9093

Restart=always

[Install]
WantedBy=multi-user.target
EOF

# 启动AlertManager服务
sudo systemctl daemon-reload
sudo systemctl enable alertmanager
sudo systemctl start alertmanager
```

## Node-Exporter配置

在所有服务器上安装Node Exporter以收集主机指标：

```bash
# 创建Node Exporter用户
sudo useradd --no-create-home --shell /bin/false node_exporter

# 下载Node Exporter
cd /tmp
wget https://github.com/prometheus/node_exporter/releases/download/v1.3.1/node_exporter-1.3.1.linux-amd64.tar.gz
tar xzf node_exporter-1.3.1.linux-amd64.tar.gz

# 拷贝Node Exporter
sudo cp node_exporter-1.3.1.linux-amd64/node_exporter /usr/local/bin/
sudo chown node_exporter:node_exporter /usr/local/bin/node_exporter

# 创建Node Exporter服务
sudo tee /etc/systemd/system/node_exporter.service > /dev/null << EOF
[Unit]
Description=Node Exporter
Wants=network-online.target
After=network-online.target

[Service]
User=node_exporter
Group=node_exporter
Type=simple
ExecStart=/usr/local/bin/node_exporter \
  --collector.filesystem.ignored-mount-points='^/(sys|proc|dev|host|etc)($$|/)' \
  --collector.textfile.directory=/var/lib/node_exporter/textfile_collector

Restart=always

[Install]
WantedBy=multi-user.target
EOF

# 启动Node Exporter服务
sudo systemctl daemon-reload
sudo systemctl enable node_exporter
sudo systemctl start node_exporter
```

## GPU监控

在所有GPU节点上安装NVIDIA Data Center GPU Manager (DCGM) Exporter来监控GPU指标：

```bash
# 安装NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2

# 拉取DCGM Exporter容器
docker pull nvidia/dcgm-exporter:2.3.1-2.6.1-ubuntu20.04

# 运行DCGM Exporter
docker run -d --restart=always \
  --name=dcgm-exporter \
  --gpus all \
  -p 9835:9835 \
  nvidia/dcgm-exporter:2.3.1-2.6.1-ubuntu20.04
```

## API性能监控

为DeepSeek API服务添加Prometheus指标导出：

```bash
# 在API服务代码中添加以下代码
# 已在之前创建的API服务配置中集成
```

至此，您已完成DeepSeek模型分布式部署的监控系统配置。该监控系统能够全面跟踪GPU使用、API性能和系统资源，并在出现问题时及时发出告警。
