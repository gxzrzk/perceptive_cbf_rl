教室行走 + 静态障碍物 + 飞球避障：近一年对比论文
检索与代码核对日期：2026-10-08

一、先看结论

已收集 10 篇外部候选论文，以及 1 篇本仓库上游 PAC-MAN，PDF 位于 papers/。
优先阅读：01 DODGER、02 LP-NavOA、04 ConstrainedMimic、03 CoFiT、05 Egocentric Dodgeball。
若先考虑复现条件，优先 DODGER、LP-NavOA、ConstrainedMimic；它们已有可访问的官方代码。
这些优先级是结合当前代码作出的判断，不是论文作者的结论，也不是已完成的性能比较。

本次核验的论文中，没有发现同时覆盖“G1 在桌椅教室中持续沿路线行走、任意方向飞球、逐身体部位避碰”且可直接运行于本仓库的现成方法。
因此应分别设置导航、飞球躲避、全身安全控制的对比，再统一迁移到同一个教室环境评测。
多数收录项目前是 arXiv 预印本；05 明确为 ICRA 2026 RoboTac Workshop 接收，不能写成 ICRA 主会论文。07 原文仅称投稿 Humanoids 2026，不能写成已接收。

时间口径：严格按首次公开提交日落在 2025-10-08 至 2026-10-08 内筛选，采用截止日之前的指定版本。旧论文在最近一年更新版本，不自动视为近一年新论文。
所有正式条目都核验了 arXiv 原始页面；主要候选进一步检查了正文和官方项目/代码。未开展完整复现实验。

文件索引：
  papers/             11 篇完整 PDF，文件名前缀与下文编号一致。
  comparison.csv      可用 Excel 打开的对比表，含日期、适配建议、代码和来源链接。
  references.bib      按所下载版本的作者元数据生成的 BibTeX。
  manifest.json       来源、版本、字节数、SHA-256 和 PDF 校验信息。
  sources/            arXiv 页面、可获得的 HTML 正文与官方项目/代码页面快照。

二、当前代码真正对应的研究问题

1. 平台与任务：Unitree G1，PPO + AMP，mjlab / MuJoCo Warp。
   入口为 train_classroom.sh；环境为 Unitree-G1-AMP-Dodge-MimicKit-Classroom-Flat。
   主要配置：src/tasks/amp_loco/config/g1/dodge_env_cfgs.py:1258。
   上游来源：仓库 README.md 明确对应 PAC-MAN，而教室是当前仓库的扩展。

2. 感知：当前教室是 state oracle，不是已经完成的视觉教室导航。
   球相对位置/速度来自仿真；家具与墙的几何也来自仿真。
   最近 8 个障碍物，每个 9 维：相对中心 xyz、半尺寸 xyz、相对朝向 sin/cos、水平边界距离，共 72 维。
   src/tasks/amp_loco/mdp/observations.py:862 的 dodge_obstacle_geometry_b 按边界距离选择障碍物。
   最后一维是根位置到障碍物水平投影的距离，不能当作整个机器人净空。

3. 静态安全：15 套桌椅 + 4 面墙，共 34 个静态碰撞体，家具用保守包围盒。
   wall_link_cbf_reward 在 src/tasks/amp_loco/mdp/rewards.py:690，按 link 到盒子表面的距离和接近速度塑形。
   球的逐 link 安全奖励见同文件 dodge_link_cbf_reward；Joint-CBF 及是否真正修改执行动作另有开关。
   CBF 奖励本身不等于运行时硬约束，也不能仅凭名称宣称部署策略具有形式化安全保证。

4. 任务结构：沿预生成的教室通道路线行走，有前视目标和转弯逻辑；这不是完全不依赖路线提示的自主目标导航。
   最高前进指令 0.8 m/s，回合 25 s；投球距离 3–5 m，方位 360°，混合高低弹道。
   不应把投球距离、机器人速度或其他任务配置中的球速混为当前教室的实际投球速度。

5. 评测中必须说明的物理设置：球只和机器人碰撞，穿透桌椅、墙和地面，无反弹。
   这是当前代码的基准设置，不能直接称为有遮挡和家具挡球的真实物理教室。
   所有对比方法要使用相同规则；若研究真实教室，再单列启用球—场景碰撞的实验。

6. 奖励：教室使用 0.75 × 加权任务奖励 + 0.25 × AMP 风格分数，任务奖励不乘 dt。
   对照方法需记录该缩放、AMP 数据及训练预算；不能把继承的奖励缩放差异误归因于新方法。

代码依据还包括 CLASSROOM.md、mdp/classroom.py、mdp/walk_path.py 和 mdp/terminations.py。

三、候选论文与如何比较

01 | DODGER: Safety-Guided Reinforcement Learning for Robot Navigation Among Dynamic Obstacles
首次提交 2026-09-30；下载 v2，2026-10-05 修订；arXiv 预印本。
原文：https://arxiv.org/abs/2609.38873v2
项目：https://psh0823.github.io/dodger-homepage/
代码：https://github.com/PSH0823/dodger

原文事实：G1 高层动态避障导航；高层策略 10 Hz，冻结的低层速度跟踪策略 50 Hz。DPCBF 安全参考引导训练，但 rollout 执行策略原动作，部署不使用在线安全过滤器。正文还包含约束相关终止机制和正负价值分支。
适合作为最优先的外部动态避障基线：比较训练期安全引导及目标前进与避碰的权衡。
适配建议：将它的导航层接到统一 G1 行走控制器，使用同一教室路线目标；需要保留或明确替换原方法的训练机制。
边界：原任务是地面移动障碍/行人；二维导航安全模型和躯干移动不能自动覆盖飞球撞头、手臂或膝盖。论文方法移植和“DODGER-inspired 全身扩展”必须分别命名。
代码状态：官方仓库可访问，有 training/ 和 deployment/；未验证可在本项目直接运行。

02 | LP-NavOA: Integrated Local Navigation and Obstacle Avoidance for Humanoid Robots under Limited Perception
首次提交 2026-06-22；v1；arXiv 预印本。
原文：https://arxiv.org/abs/2606.23249v1
项目：https://shenqiqishi.github.io/LP-NavOA/
代码：https://github.com/shenqiqishi/LP-NavOA-code

原文事实：先训练以局部测距为条件的 PPO 行走策略，再冻结它，蒸馏循环局部规划器以修正 heading；报告 G1 部署检查与静态墙体绕行、绕后恢复目标。正文明确把移动障碍列为后续扩展。
适合作为教室静态绕障和避障后恢复路线的基线。
适配建议：先运行无球教室，使用相同局部目标和行走速度限制；再增加飞球作为扩展测试。需建立局部 ray/range 观测，或将所有方法统一到状态输入并标记为改编版本。
边界：原文的有限感知、无持续 waypoint 输入与本仓库的全状态及通道路线提示不同；也不能把 G1 可执行性检查写成大规模实机统计。
代码状态：已核验项目页和官方代码，代码是 mjlab 的研究分支，与本仓库框架接近；具体版本和运行兼容性尚未测试。

03 | Filter-Aware Fine-Tuning for Safe Humanoid Whole-Body Tracking（CoFiT）
首次提交 2026-10-01；v1；arXiv 预印本。
原文：https://arxiv.org/abs/2610.02341v1

原文事实：给预训练全身 tracker 增加约束上下文、历史过滤修正及相关奖励，通过微调缓解策略和运行时过滤器之间的不匹配；使用 TWIST2/SONIC，包含 G1 实机实验。
适合比较 Joint-CBF 的“直接加 filter”和“策略与 filter 联合适应”。
适配建议：以相同名义策略、同一过滤器比较不微调、普通微调、filter-aware 微调；记录约束违反时长、过滤幅度与任务进度。
边界：实机篮球实验是对静止篮球附近的 reaching 约束，不是躲飞来的篮球。迁移到飞球需显式考虑时变障碍和相对速度；AMP 行走策略也不是原文 tracker。
代码状态：本次核验未确认作者公开的独立 CoFiT 代码仓库；ConstrainedMimic 代码不能自动视为 CoFiT 已开源。

04 | Constrained Whole-Body Tracking for Humanoid Robots（ConstrainedMimic）
首次提交 2026-05-29；v1；arXiv 预印本。编号 2606 不代表首次提交日一定在六月。
原文：https://arxiv.org/abs/2606.00374v1
代码：https://github.com/StanfordASL/constrainedmimic

原文事实：利用接触约束下的全身运动学/动力学与 CBF，在参考动作及控制输出层施加约束；G1 仿真实验包含身体/环境避碰等约束。
适合作为本仓库训练期 CBF 奖励之外的运行时安全过滤基线。
适配建议：在同一名义行走/躲球策略上比较无 filter、本仓库 Joint-CBF、ConstrainedMimic 风格全身 filter；统一球与盒子的几何、安全余量和感知输入。
边界：依赖接触模式、动力学/状态信息；把它迁到 mjlab 和现有动作接口需要工程工作，不能等同于原论文复现。原文验证以仿真为主。
代码状态：官方仓库存在，README 标记仍在建设中；代码栈与本仓库不同，尚未运行验证。

05 | Egocentric Tactile and Proximity Sensors as Observation Priors for Humanoid Collision Avoidance
首次提交 2026-04-28；v1；ICRA 2026 RoboTac Workshop 接收。
原文：https://arxiv.org/abs/2604.25554v1

原文事实：H1-2 仿真躲球，PPO 输出关节目标，研究身体分布式接近/触觉传感器。球初速 4–8 m/s、发射距离 4–6 m；主要比较不同感知表达，奖励没有显式物体净空塑形。
适合作为飞球子任务的感知与奖励基线，任务相关性高。
适配建议：在 G1 上生成同样的局部 proximity 特征；比较状态输入、接近距离输入，并分别开关 CBF 塑形。固定 PPO/AMP 和网络预算，区分传感器收益与奖励收益。
边界：H1-2 与 G1 身体/动作维度不同；原文未验证边走边绕桌椅。理想感知、纯仿真结果不能当成实机传感器证据。
代码状态：本次核验未确认作者官方实现。

06 | Safety-Critical Whole-Body Control for Humanoid Robots via Input-to-State Safe Control Barrier Functions（SafeWBC）
首次提交 2026-05-25；v1；arXiv 预印本。
原文：https://arxiv.org/abs/2605.25546v1
项目：https://kwlee365.github.io/SafeWBC-Website/
代码：https://github.com/dyroshumanoid/safeWBC/tree/nightly

原文事实：KinWBC + ISSf-CBF filter + DynWBC，考虑有界扰动下的安全约束和接触可行性，有仿真与实机全身控制实验。
适合模型驱动鲁棒安全控制对比，尤其是加执行误差/状态噪声后的净空和稳定性。
适配建议：先比较静态家具旁的移动和全身避碰，再讨论时变球约束；需统一动力学与控制频率。
边界：不能把原文的扰动有界条件和控制架构所支持的保证，直接套到当前 PPO 策略上；不是现成教室躲球管线。
代码状态：官方项目链接 safeWBC 的 nightly 分支，README 对应 TOCABI / ROS1 / C++ 控制栈，迁到 G1 成本较高；未运行验证。

07 | Collision-Aware Humanoid Whole-Body Control under Imperfect Tracking Targets（RECAL）
首次提交 2026-09-14；下载 v2，2026-09-30 修订；原文称投稿 Humanoids 2026。
原文：https://arxiv.org/abs/2609.16405v2

原文事实：用机器人/物体点与环境点云的 cross-attention 修正盲 WBC 指令；覆盖行走、搬物和操作，含 Digit V3 实机演示。
适合比较“逐 link 几何安全塑形”与“学习机器人—环境几何关系”，尤其是桌边手臂/躯干碰撞。
适配建议：由相同仿真几何生成点云，与当前 top-8 × 9 维盒子表示比较；固定动作/奖励时可做编码器对照。
边界：机器人平台不同，原控制任务及点云输入不同；无已核实的飞球实验。仅替换编码器时应写 RECAL-inspired，而不是完整 RECAL 复现。
代码状态：本次核验未确认作者官方实现。

08 | RAVEN: Reinforcement-Adaptive Visibility-Graph Planning for Robust Humanoid Navigation with Collision-Free MPC
首次提交 2026-07-17；v1；arXiv 预印本。
原文：https://arxiv.org/abs/2607.15701v1

原文事实：RL 调整可视图的障碍膨胀等几何参数，MPC 执行约束跟踪；正文实验使用静态圆形障碍与控制延迟/观测噪声，实机定位使用 MoCap。
适合传统规划 + RL/MPC 类静态导航对比。
适配建议：桌椅保守投影到平面，统一目标与机器人外形膨胀；评估绕行代价、窄通道通行、延迟鲁棒性。
边界：正文把移动障碍列为未来工作，不能因摘要提及 dynamic environments 就归为已验证动态避障；平面模型也不体现低头/侧身躲球。
代码状态：本次核验未确认作者官方实现。

09 | Whole-Body Planning for Humanoids Navigating Confined Spaces via Self-Collision Avoidance References
首次提交 2026-08-10；v1；arXiv 预印本。
原文：https://arxiv.org/abs/2608.10220v1
项目：https://carlosiglezb.github.io/confined-space-wbp-humanoid/

原文事实：以可达刚体体积为基础的三阶段全身规划，生成轨迹优化参考，再用残差 RL 跟踪；G1 狭窄空间、多接触任务，主要为仿真验证。
适合空间受限时全身几何建模的相关工作，以及狭窄通道附加实验。
适配建议：比较家具间转身、通过空间的身体净空与动作可行性。
边界：长时域规划参考与突发飞球反应任务不同；当前教室包围盒不允许钻桌底，不应拿原文复杂攀爬/多接触任务直接比较。
代码状态：已核验作者项目页，本次未确认完整方法代码；引用的通用 G1 底层仓库不是该方法开源证明。

10 | FutureRay: Control-Aligned Future Range for Agile Quadruped Navigation
首次提交 2026-09-26；v1；arXiv 预印本。
原文：https://arxiv.org/abs/2609.32158v1

原文事实：由深度测距历史预测未来各方向净空和相遇风险，局部规划考虑外形与反应/制动限制，低层行走策略固定；评测包含静态和动态场景。
适合研究是否需要动态预测的补充基线。
适配建议：可在统一 G1 速度控制器上比较当前距离、恒速/Kalman 预测、学习预测三种高层方法；须标记为跨平台移植。
边界：原平台为四足，预测测距导航不等同于人形全身飞球避让；不建议作为唯一或首要对比。
代码状态：本次核验未确认作者官方实现。

00 | PAC-MAN: Perception-Aware CBF-RL for Whole-Body Safety in Humanoid Dodgeball
首次提交 2026-07-30；下载 v2，2026-09-29 修订；arXiv 预印本。
原文：https://arxiv.org/abs/2607.28623v2
项目：https://lzyang2000.github.io/perceptive_cbf_rl/
代码：https://github.com/lzyang2000/perceptive_cbf_rl

这是本仓库上游方法，必须引用，建议作为继承关系/内部基线单列，不计入 10 篇独立外部候选。
与当前工作的关系是全身躲球、CBF-RL、AMP 与感知训练基础；新增贡献需要由教室移动与静态障碍实验说明。
建议保留同一教室输入/路线/训练预算，仅逐项移除新增静态安全和联合训练机制进行消融；不要直接加载无法接收 72 维家具观测的上游模型，称其为公平同条件对比。

四、建议的最小实验组合（以下均为针对本项目的建议，尚未实施）

A. 内部消融：相同 PPO/AMP、同一观测、同一训练预算；分别比较无球/静态 CBF、仅球 CBF、仅静态 CBF、联合 CBF。无 CBF 的组要关闭所有相关奖励和实际动作投影，不能只关闭一个环境变量。
B. 上游关系：PAC-MAN 风格躲球 + 相同路线跟随，去掉新增静态安全项，再与完整教室方法比较。准确称为上游方法的教室适配。
C. 外部动态导航：DODGER 移植；先报告地面动态障碍，再单独报告飞球扩展，不能混合两种任务的论文原始结果。
D. 外部静态导航：LP-NavOA 移植；报告纯静态教室与联合威胁教室两个子集。
E. 运行时安全：ConstrainedMimic 风格 filter；若资源允许，加 CoFiT 思路的 filter-aware 微调。
F. 感知附加实验：05 的 proximity 输入，与 state oracle 及未来视觉版本比较。

公平性约束：
  - 固定 G1 模型、接触几何、安全距离、球半径与实际弹道分布、出生状态、路线、控制周期、试验时长。
  - 状态输入与感知输入分开成表；有限视场/噪声/延迟按同一条件控制。
  - 导航层比较时冻结同一个低层控制器；全身策略比较时统一动作空间。跨架构结果可同时报告，但需明确差异。
  - 所有方法得到同级别的路线/目标提示，避免一个获得完整通道路线、另一个只有遮挡后的局部目标。
  - 相同训练环境步数与多随机种子；可另外报告原作者权重 zero-shot 结果，但与重新训练结果分开。
  - 固定评测随机种子、保留成对场景；低头、侧移、转向、靠近桌边投球分别分层统计。

建议指标：
  1. 联合成功率：预先规定最低路线进度，并要求回合内无任意 link 球碰撞、无家具/墙碰撞、无跌倒。
  2. 每次投球躲避率、每回合任意球接触率；报告投球次数与分母，避免短回合或提前失败造成虚高。
  3. 静态碰撞率和跌倒率，分别统计；最低真实几何净空与 CBF 代理净空分开。
  4. 沿路线进度/速度、安全时停滞比例、躲避后恢复路线时间，防止原地不走获得高安全分。
  5. 求解/推理延迟、filter 干预率和修正幅度、约束违反时间、训练样本效率。
  6. 新布局、投球方位/速度/高度、观测噪声/延迟泛化；安全率与任务进度一起报告。

五、时间窗口外的重要近邻（未混入 PDF 主列表）

SHIELD: Safety on Humanoids via CBFs In Expectation on Learned Dynamics
https://arxiv.org/abs/2505.11494
首次公开在 2025 年 5 月，虽然之后有更新，按本次“首次公开近一年”口径排除；如扩展到近两年，它是值得补充的 G1 安全导航基线。

End-to-End Humanoid Robot Safe and Comfortable Locomotion Policy
https://arxiv.org/abs/2508.07611
首次公开在 2025 年 8 月；与 CBF/CMDP 行走相关，但同样在本次窗口外。

六、检索与验证记录

检索方向包含 humanoid / quadruped + dynamic obstacle avoidance、dodgeball、control barrier、whole-body collision avoidance、limited-perception navigation，以及 2025 年第四季度至 2026 年论文。
搜索结果只用于发现线索；条目事实以 arXiv 正文、作者项目页、作者官方仓库为准。
没有用转载站的“发表时间”代替首次提交日，也没有把“投稿”或“计划开源”写成“正式接收”或“已发布代码”。
未确认官方代码表示本次查找未找到可核实入口，不表示不存在代码。代码可访问也不代表完整、可直接运行或已复现。
本目录完成的是文献筛选、原文归档与对比设计，没有修改训练代码或启动训练。
