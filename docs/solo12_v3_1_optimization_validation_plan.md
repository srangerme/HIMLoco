# Solo12 v3.1 HIMLoco 步态与连续地形优化验证计划

## 结论与边界

本计划只定义优化方向、短测和准入门槛。当前未启动正式训练。

候选起点仍冻结为
`Sep15_17-51-13_formal_r6_height032_latency40_seed1/model_2900.pt`。
`Sep16_00-39-49_validation_only_optimization_probe_5it/model_2905.pt` 只是 64 环境、
5 iteration 的训练链 smoke 产物，不得进入模型候选或部署流程。

本计划保持 RS02 48 V 转速—扭矩包络、44 rad/s 电机输出转速边界和 calf 2:1
额外传动不变。过载持续时间只做训练后验收，不进入 reward、termination 或课程。

## 已完成的可行性验证

- 连续随机高度场已能覆盖整个 8 m 地形块，不再使用带 3 m 中央平整平台的短视频
  作为不平整路面证据。0–4 级对应约 ±1–4 cm 起伏。
- 新增的 terrain-relative swing clearance、饱和 swing time、stance slip、hip posture、
  diagonal action symmetry、带容差的 0.32 m 高度奖励均可在 GPU 环境中计算。
- 64 环境、5 iteration 从 model_2900 恢复到 model_2905 成功；共采集 32,000
  transitions，HIM/PPO 更新、保存和配置快照均正常，无 NaN 或发散。
- smoke 暴露并修正了两个仅在非零权重时出现的接口问题：`feet_stumble` 与
  `_reward_stumble` 命名不一致，以及实际足高接口为 `_get_feet_heights()`。
- smoke checkpoint 的同 seed A/B 未出现立即退化：平地/连续起伏的短测环境通过率
  从 87.5%/75% 提高到 100%/100%，最长停滞由 6.62/8.82 s 降为 0，平地平均绝对
  pitch 从 7.49°降到 5.81°。但 FR/RL 摆腿仍只有约 2.6/2.4 cm，逐足站宽仍未通过，
  因而该结果只支持继续做 100 iteration 诊断 A/B，不构成模型晋级证据。
- 新验收 trace 已包含逐足离地高度、接触、足速、足位、关节位置、动作、机身姿态、
  世界位移和 terrain level；因此可以识别“走到地形边缘后停住”，不能再只凭画面判断。

当前 model_2900 在 8 环境、15 s 平地测试中，平均绝对 pitch 为 7.49°，P95 为
11.03°；FL/FR/RL/RR 摆腿峰值中位数分别为 4.60/2.50/2.37/5.04 cm；逐足
站立横向半宽分别为 16.00/6.34/7.90/6.09 cm。指标与视频中的抬头、前腿跛行和
后腿内收一致。

在全程连续起伏面上，8 环境、15 s 中只有 75% 环境达到平均前进速度不低于命令的
60%，最长连续停滞 8.82 s。16 环境、30 s 独立测试中，无随机化只有 5/16 环境同时
满足不摔倒和平均速度 ≥0.3 m/s；当前模型不具备可靠连续不平整地面能力。

## 正式训练前的冻结 A/B 验证

正式训练前先从同一 model_2900、同一 seed 和同一地形样本做三个 100 iteration
诊断分支。三个分支都只是参数选择试验，不能直接晋级：

1. Control：只更换连续地形和验收记录，不增加新 reward。
2. Clearance：增加 terrain-relative swing clearance、饱和 swing time 和 stance slip。
3. Gait：在 Clearance 基础上增加轻量 hip posture、diagonal symmetry，并将高度奖励
   改成 0.32 m ±1 cm 容差，同时增强 pitch/roll 稳定。

只接受同时改善逐足抬腿高度、姿态、站宽和连续地形前进率的 Pareto 分支。若只是
速度提高但 pitch、包络触碰或步态不对称恶化，则拒绝该分支。100 iteration 只用于
判断方向；没有形成两个相邻 checkpoint 的一致趋势时，不进入正式阶段。

## 获得授权后的正式训练课程

### G1：步态修复

- 从 model_2900 恢复，先用 1024 环境训练 300–500 iteration。
- 使用平地到 ±1.7 cm 连续微起伏，不使用 3 m 平台，也不加入楼梯或离散高障碍。
- 速度范围以低中速为主，收窄横移和 yaw，先形成稳定的前进对角步态。
- 随机化采用 smoke 已验证的窄范围：payload `[-0.3,0.5] kg`、link mass
  `[0.9,1.1]`、friction `[0.8,1.5]`、电机/PD `[0.95,1.05]`、push ≤0.3 m/s、
  外力 ±10 N、延迟和恢复系数固定为 0。
- 每 50 iteration 做冻结评估；diagonal symmetry 只用于这一阶段，后续逐渐衰减，
  避免限制崎岖地形上的独立落足。

G1 晋级要求：平地 32 环境、3 个 seed、每个 60 s 均无摔倒；环境通过率 ≥95%；
平均速度不低于命令的 80%；最长停滞 ≤0.5 s；机身平均绝对 pitch ≤4°、P95 ≤8°；
四足摆腿峰值中位数均 ≥4 cm，最大足间差 ≤2 cm；每只脚站立横向半宽均 ≥8 cm，
足间最大差 ≤4 cm；平均高度保持在 0.32 m ±1 cm。

### G2：连续不平整地面

- 由 G1 最佳 checkpoint 恢复，逐级扩展到 ±1/2/3/4 cm 连续随机高度场。
- 建议 2048–4096 环境、400–600 iteration；只有当前级别通过 held-out 测试才提高
  起伏，不按固定轮数强制晋级。
- swing clearance 目标由 6 cm 逐步增至 8 cm；diagonal symmetry 降低或关闭，保留
  stance slip、逐足站宽和局部地形相对离地高度。
- episode 后段提高到 30–40 s，并把连续停滞超过 2 s 视为课程失败，不能只依赖
  base contact termination。

G2 晋级要求：连续 ±1–4 cm 地形 32 环境 × 60 s，nominal 环境通过率 ≥95%，窄随机化
环境通过率 ≥90%；无连续停滞超过 2 s；速度比 ≥60%；不得靠降低 0.32 m 高度或重新
形成不对称拖步换取通过率。

### G3：动力学与延迟鲁棒性

- 按 payload/COM/link mass → friction → motor/PD → joint friction/damping → restitution
  → observation latency → push/disturbance 的顺序单变量扩展。
- 每次只扩大一个维度，并同时保留 nominal 回归；任何 nominal 步态回退都回到上一
  checkpoint。
- 最终才恢复既定边界：payload `[-0.635,1.906] kg`、COM ±0.02 m、link mass
  `[0.7,1.3]`、friction `[0.7,3.0]`、电机/PD `[0.9,1.1]`、joint friction
  `[0,0.02]`、joint damping `[0,0.01]`、restitution `[0,0.2]`、latency
  `[0,40] ms`、push ±1 m/s 和外力 ±30 N。

### 楼梯能力的独立决策

当前 Actor 只接收 6 帧本体观测历史，地形高度扫描只提供给 Critic。连续低幅起伏可用
盲走鲁棒性处理，但可靠楼梯和离散落脚通常需要 Actor 获得高度图或深度感知。

因此楼梯不与 G1/G2 混为一个门槛。先做 3 cm 台阶的盲走探针；若无法达到 80% 通过率，
则新增 Actor 地形观测并建立独立策略。改变 Actor 输入维度后不能直接把 model_2900
作为完整网络恢复，只能迁移兼容层权重或重新训练。

## 最终验收

最终候选必须同时通过以下独立测试：

- 平地步态：3 个 seed、32 环境 × 60 s，并人工复核近景慢放视频。
- 连续起伏：±1–4 cm，nominal 和完整随机化分别 64 环境 × 120 s；地形从出生点开始
  连续起伏，视频必须展示完整路线和实时速度，禁止中央平台替代。
- 楼梯：只在楼梯方案被明确纳入模型能力后验收，逐级报告台阶高度，不能用走到台阶
  前方后停止的视频作为通过证据。
- 扰动矩阵：0/10/20/30/40 ms latency 与 restitution 0/0.2 交叉端点，同时保留
  nominal 回归。
- 执行器：48 V 转速—扭矩包络零违规；calf 按 2:1 换算到电机输出坐标；按手册
  25℃、24 rpm 的过载持续时间做训练后验收；硬转速限制提高 10% 的 A/B 不得造成
  速度跟踪或稳定性显著变化。

任一项失败都报告为未通过，不再用综合平均掩盖某只脚、某个地形等级或某个随机化
端点的失败。

## 100-iteration 诊断结果

优化配置的单分支 100-iteration 验证已完成。连续地形前进率、停滞和 pitch 显著改善，
执行器约束保持通过，但 FR/RL 抬腿和 RL 站宽仍失败。当前瞬时 diagonal action
symmetry 会允许两组对角腿形成不同幅度，必须改成逐足 worst-case clearance、对角组间
摆动统计平衡和 body-frame 逐足站宽下限。详细结果见
`solo12_v3_1_optimization_100it_validation_report.md`。

V2 又完成了一次同规模短验证：逐足最差 clearance 与 body-frame 站宽约束在不牺牲
连续起伏通过率的情况下，使 FR/RL 抬脚和 RL 站宽继续改善。因此修复路径已完成
“可行性验证”；短验证 checkpoint 不需要、也不应被要求通过完整发布验收。

正式训练采用以下修正版：摆腿高度按每只脚一次完整 swing 的峰值在 touchdown 时结算，
目标 5–6 cm，避免逐帧奖励诱导悬脚；站距按机身坐标系 9–12 cm 容差区间约束，既纠正
RL 内收，也抑制 FL 过度外展。保留 0.32 m 高度容差、orientation、slip、action-rate、
joint-power 与 RS02 48 V 转速—扭矩包络。瞬时 action symmetry 仅作弱正则，不作为
步态正确性的代理指标。
