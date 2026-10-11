"""部署辅助工具与诊断脚本包 (deploy.tools).

包含部署与感知系统调试相关的实用工具：
- `etam_track_live.py`:
  基于 ZED Mini 与 RealtimeEfficientTAM 的实时球体跟踪与掩码深度图可视化调试工具。
- `patch_efficienttam_py38.py`:
  自动修补 RealtimeEfficientTAM 子模块源码使其完全兼容 Python 3.8 的补丁工具。
- `fetch_jetson_etam_wheels.sh`:
  针对 Jetson aarch64 边缘计算平台自动下载与镜像 PyTorch/Torchvision 离线安装轮子包的脚本。
"""
