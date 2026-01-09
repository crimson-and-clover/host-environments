# 主机开发环境

基于 NVIDIA CUDA 的 Docker 开发环境，支持多版本 CUDA，包含常用的开发工具和依赖。

## 功能特性

- 支持多个 CUDA 版本（11.8, 12.4, 12.8）
- 预装常用开发工具（SSH、Git、CMake、Python 环境等）
- 自动配置用户权限和 SSH 访问
- 支持 GPU 资源管理和 CUDA 环境配置
- 支持 SSD/HDD 数据卷挂载

## 系统要求

- Docker 和 Docker Compose
- NVIDIA Docker Runtime（nvidia-docker2）
- NVIDIA GPU 驱动

## 快速开始

### 构建镜像

使用 Makefile 构建不同 CUDA 版本的镜像：

```bash
# 构建 CUDA 11.8 镜像
make cuda/11.8

# 构建 CUDA 12.4 镜像
make cuda/12.4

# 构建 CUDA 12.8 镜像
make cuda/12.8

# 构建所有版本
make all
```

### 部署容器

1. 编辑 `deploy.sh` 文件，修改以下配置：
   - `CONTAINER_NAME`: 容器名称
   - `HOSTNAME`: 容器主机名
   - `IMAGE_NAME`: 使用的镜像名称
   - `SSD_ROOT`: SSD 挂载路径数组
   - `HDD_ROOT`: HDD 挂载路径数组
   - `SSH_PORT`: SSH 端口
   - `USER_ID` 和 `GROUP_ID`: 用户和组 ID

2. 运行部署脚本：

```bash
./deploy.sh
```

或者通过环境变量覆盖配置：

```bash
CONTAINER_NAME=my-container \
HOSTNAME=my-hostname \
IMAGE_NAME=huyu/cuda:11.8-ubuntu22.04 \
SSH_PORT=22001 \
USER_ID=1000 \
GROUP_ID=1000 \
./deploy.sh
```

## 配置说明

### 环境变量

部署脚本支持以下环境变量：

- `USER_NAME`: 容器内用户名（默认: developer）
- `CONTAINER_NAME`: 容器名称
- `HOSTNAME`: 容器主机名
- `IMAGE_NAME`: Docker 镜像名称
- `SSH_PORT`: SSH 端口（默认: 22000）
- `USER_ID`: 用户 ID
- `GROUP_ID`: 组 ID（默认与 USER_ID 相同）
- `ROOT_PASSWORD`: root 用户密码（默认: 000000，**建议修改**）
- `USER_PASSWORD`: 普通用户密码（默认: 000000，**建议修改**）

### 数据卷挂载

- 第一个 HDD 的 `home` 目录会挂载到用户的 home 目录
- 所有 SSD 的 `data` 目录会挂载到 `~/ssd1`, `~/ssd2`, ...
- 所有 HDD 的 `data` 目录会挂载到 `~/hdd1`, `~/hdd2`, ...

## 安全注意事项

⚠️ **重要**: 默认密码为 `000000`，首次登录后请立即修改密码！

可以通过环境变量设置密码：

```bash
ROOT_PASSWORD=your_secure_password \
USER_PASSWORD=your_secure_password \
./deploy.sh
```

## 项目结构

```
.
├── README.md              # 项目文档
├── Makefile              # 构建配置
├── deploy.sh             # 部署脚本
├── .dockerignore         # Docker 忽略文件
└── src/                  # 源代码目录
    ├── Dockerfile        # Docker 镜像定义
    ├── build.sh          # 构建脚本
    ├── init.sh           # 初始化脚本
    ├── init_user.sh      # 用户初始化脚本
    ├── create_cuda_env.sh # CUDA 环境配置脚本
    ├── authorized_keys   # SSH 公钥
    ├── pause.c           # 容器暂停程序
    └── cuda_occupier.cu  # CUDA 资源占用程序
```

## 已安装的软件包

- **开发工具**: openssh-server, git, git-lfs, cmake, ninja-build, vim, curl, wget
- **编译依赖**: build-essential, 各种 C/C++ 库
- **Python 环境**: pyenv 相关依赖
- **GPU 工具**: CUDA 工具链
- **其他工具**: ffmpeg, rclone, proxychains, sshfs 等

## 故障排除

### 容器启动失败

检查 Docker 日志：
```bash
docker logs <container_name>
```

### SSH 连接失败

1. 检查 SSH 端口是否正确映射
2. 确认容器内 SSH 服务已启动：`docker exec <container> service ssh status`
3. 检查防火墙设置

### 权限问题

确保挂载的目录有正确的权限，或者使用与主机相同的 UID/GID。

## 许可证

本项目中的 `pause.c` 文件使用 Apache License 2.0 许可证（来自 Kubernetes 项目）。

## 贡献

欢迎提交 Issue 和 Pull Request！
