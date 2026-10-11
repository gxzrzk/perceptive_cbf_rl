"""Unitree G1 机器人实机与仿真部署模块 (deploy).

本模块包含训练好的深度躲避策略 (Single-Camera Depth Dodge) 在真实 Unitree G1 人形机器人
（配备 ZED Mini 头部相机与 Jetson Orin 算力平台）以及 MuJoCo 仿真环境下的完整部署栈。

系统架构概览：
-------------
部署栈采用无 ROS 的轻量级三进程本地 UDP 通信架构，实现故障隔离（Fault Isolation）：
1. 相机节点 (`camera_node` / `camera_node_etam`)：
   - 独立进程运行，避免 ZED SDK / pyzed 崩溃或 USB 阻塞拖垮电机控制。
   - 采集 ZED Mini 原始图像，通过百分位数池化 (Percentile Pooling) 或 EfficientTAM 实时分割目标球体生成专用掩码深度图。
   - 以指定控制频率（默认 50 Hz）通过 UDP (:9872) 将下采样后的深度图发送给策略节点。
2. 策略推理节点 (`dodge_policy`)：
   - 基于 ONNXRuntime（CPU 推理），解耦深度学习框架。
   - 接收硬件状态 (:9871) 与深度数据 (:9872)，严格按训练时的观测排布拼接本体感受历史与堆叠深度帧。
   - 推理计算 29 个关节位置目标，支持躲避 (Dodge) 与行走 (Walk) 双策略平滑插值切换，输出目标关节位置 (:9870)。
3. 硬件桥接节点 (`hardware_node`)：
   - 核心保障进程，绝不允许崩溃。负责持有 Unitree SDK2 接口，直接与 G1 底层硬件通信。
   - 以 50 Hz 发送 LowCmd、接收 LowState 及无线遥控手柄输入，维护安全状态机（阻尼保护、起立插值、一键急停）。

仿真验证 (Sim2sim)：
-------------------
提供 `sim_node`，可在 x86 开发机上利用 MuJoCo 仿真环境完全模拟硬件与相机行为，
通过完全一致的 UDP 协议连接 `dodge_policy`，用于零硬件风险验证观测拼接、动作反归一化及策略表现。
"""
