"""部署端测试套件 (Deployment Test Suite)。

本测试包覆盖真机与部署环境相关功能的离线测试与数据流校验，包括：
1. 部署常数漂移防护 (Deploy Constants Drift Guard)，对比真机常量与仿真来源的一致性；
2. 深度图有效掩码过滤与远距离填充 (Mask Depth to Far)；
3. 基于物体视线高度与距离的尺寸过滤 (Mask Too Tall)，防止大尺寸背景或人员误判为球体；
4. 本体感知与深度图环形缓冲区的观测向量装配契约 (Obs Assembly)；
5. UDP 状态包、控制动作包以及多相机深度图包的序列化/反序列化及时间戳对齐 (UDP Sync)。
"""
