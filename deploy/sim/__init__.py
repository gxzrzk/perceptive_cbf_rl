"""仿真部署与 Sim2sim 验证模块 (deploy.sim).

包含在开发机 (Dev Box) 上脱离实体机器人硬件进行全栈部署验证的核心组件：
- `sim_node.py`:
  Sim2sim 仿真驱动节点。在开发机上构建真实的 MuJoCo 动力学与渲染仿真环境，
  同时扮演 hardware_node 与 camera_node 的角色，通过相同的 UDP 数据包协议
  连接并驱动后台运行的实机策略节点 (dodge_policy)。
  用于在无硬件风险的情况下端到端验证观测拼接、深度堆叠、动作反归一化及避障响应。
"""
