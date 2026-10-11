"""部署通用公共模块 (deploy.common).

包含实机与仿真部署中跨进程复用的核心基础组件与协议定义，设计为独立于 mjlab/Warp 的纯 NumPy 实现：
1. `command.py`:
   - 遥控手柄摇杆映射与名义速度指令 (vx, vy, yaw_rate) 生成，包含死区处理与增益缩放。
2. `g1_deploy_constants.py`:
   - 固化的 G1 机器人 29 自由度常量（关节顺序、默认起立姿态、PD 控制刚度与阻尼、动作缩放比例、深度图参数），
     脱离 mjlab 直接在 Jetson 运行。
3. `udp_sync.py`:
   - 三进程间低延迟 UDP 数据包封包与解包协议定义，包括 State (状态)、Depth (深度)、Action (动作) 三种固定结构。
4. `zed_depth_source.py`:
   - 基于多线程后台采集的 ZED Mini 深度数据源驱动，提供高性能百分位数软最小值池化 (Percentile Pooling)
     及目标球体掩码远端填充工具。
"""
