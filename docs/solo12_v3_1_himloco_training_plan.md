# Solo12 v3.1 HIMLoco 训练计划

> 本文是训练前的历史计划。最终模型实际采用的有效训练链、A/B/C 阶段、
> checkpoint 继承关系和验收结果见
> [`solo12_v3_1_himloco_training_process.md`](solo12_v3_1_himloco_training_process.md)。

## 1. 计划依据

训练主流程沿用原始 Go1 HIMLoco：同一个 rough-locomotion task 从零开始，4096
个并行环境，每轮每环境采集 100 个策略步，再进行 HIM estimator 与 PPO 联合
更新；地形课程和速度指令课程从训练开始自动运行，不采用 jumping 的
Upward→Forward 两任务迁移。

机器人、控制和动力学边界参考仓库内已经验收的 Solo12 jumping 配置。延迟与
恢复系数仍按 jumping 的经验逐级引入，因为最终训练记录表明一次性开放这些
维度会产生策略捷径。

## 2. 固定训练设置

- 仿真步长：5 ms（200 Hz）。
- control decimation：4，策略周期 20 ms（50 Hz）。
- 站立高度：0.30 m。
- 默认关节角：hip `0`、thigh `0.7220`、calf `-1.4441`。
- PD：全部关节 `Kp=16, Kd=0.5`，与已验收 Solo jumping 控制一致。
- action scale：0.25；hip reduction：0.5。
- PhysX TGS position iterations：8。
- Actor：6×45 维本体观测历史；Critic 使用 HIMLoco 特权观测。
- 每轮：`4096 × 100 = 409,600` transitions。
- PPO：5 epochs、4 mini-batches、初始学习率 `1e-3`、adaptive KL `0.01`。
- 每 20 轮保存 checkpoint。

## 3. 随机化边界

| 随机项 | 训练最终边界 | 来源/说明 |
|---|---:|---|
| Base 附加质量 | `[-0.635, 1.906] kg` | Solo jump 公共范围 |
| COM XYZ | 每轴 `[-0.02, 0.02] m` | 最终有效 Upward/Forward 范围 |
| Link 质量倍率 | `[0.7, 1.3]` | Solo jump 公共范围 |
| 地面摩擦 | `[0.7, 3.0]` | 采用接触丰富的 Solo Forward 有效范围 |
| 电机强度倍率 | `[0.9, 1.1]` | Solo jump 公共范围 |
| Kp 倍率 | `[0.9, 1.1]` | Solo jump 公共范围 |
| Kd 倍率 | `[0.9, 1.1]` | Solo jump 公共范围 |
| 关节摩擦 | `[0, 0.02]` | Solo jump 最终准出边界 |
| 关节阻尼 | `[0, 0.01]` | Solo Forward 范围 |
| 恢复系数 | `[0, 0.2]` | 按阶段从 0 扩展 |
| 观测延迟 | `[0, 40] ms` | 按 10/20/30/40 ms 逐级扩展 |
| 初始关节角倍率 | `[0.8, 1.2]` | Solo locomotion 启动保护范围 |
| 外力 XYZ | 每轴 `[-30, 30] N` | 保留 Go1 HIMLoco 扰动方式 |
| Push 后 XY 速度 | 每轴 `[-1, 1] m/s` | 保留 Go1 HIMLoco 范围，周期 16 s |

观测噪声保持 HIMLoco 默认开启。40 ms 作为准出边界工况，低延迟仍是主工作域。
极低摩擦 `[0.01,0.7)` 不纳入首轮 locomotion 准出范围；如有明确硬件需求，
应在基线通过后单独做低摩擦课程，而不是混入主训练。

## 4. 正式训练前的实现门槛

当前 HIMLoco 与 jumping 的随机化实现并不等价，以下项目必须先补齐并测试：

1. 实现 0–40 ms 的观测历史延迟；现有 `delay` 是每个控制周期内的动作延迟，
   只有 0/5/10/15 ms，不能替代观测延迟。
2. 实现关节摩擦 `[0,0.02]` 和附加关节阻尼 `[0,0.01]`。
3. 确认 motor-strength factor 实际作用于输出力矩；原 HIMLoco 代码虽然采样该
   factor，但当前力矩公式没有使用它。
4. 将电机、Kp、Kd 随机化改为每个关节独立采样，与 jumping 的 12 维采样一致；
   原 HIMLoco 是每台机器人共享一个倍率。
5. 为 observation latency 和 restitution 提供可复现的阶段配置，禁止训练中
   临时手改同一配置文件。
6. 每个阶段保存实际解析后的配置、随机化端点和父 checkpoint。

在这些门槛完成前，只允许进行 R0 基线和接口验证，不能声称已经覆盖 jump 的
完整随机化。

## 5. Phase 0：非训练验证

1. Adapter smoke 必须识别 `solo12_v3_1` 并找到挂载 URDF。
2. 单环境初始化必须得到 12 个驱动关节和 4 个独立 foot body。
3. 核对 Isaac Gym 原生顺序 `FR, FL, RR, RL` 以及每腿
   `hip, thigh, calf`，为部署端保留显式 permutation。
4. 零动作站姿应无穿地、异常自碰撞、饱和或立即终止。
5. 逐关节施加小幅正负动作，验证方向、限位、力矩和 hip 下标。
6. 对每个随机化参数做固定最小值/标称值/最大值三点测试。

## 6. Phase 1：短测与学习 sanity check

先运行 64 环境、5 轮，仅检查 rollout、HIM 和 PPO 数据链：

```bash
python3 -m trains.cli train himloco-ada \
  --task solo12_v3_1 --num_envs 64 --max_iterations 5 \
  --seed 1 --run_name r0_integration_seed1
```

通过后运行 1024 环境、200 轮：

```bash
python3 -m trains.cli train himloco-ada \
  --task solo12_v3_1 --num_envs 1024 --max_iterations 200 \
  --seed 1 --run_name r0_sanity_seed1
```

R0 使用表中质量、COM、link mass、摩擦、电机、PD、初态、外力和 push 范围；
观测延迟固定 0，恢复系数固定 0。若大多数 episode 早停、HIM loss 发散、动作
持续高频振荡或 terrain level 塌缩，不进入正式训练。

## 7. Phase 2：Go1 式主训练与 jump 式鲁棒性课程

所有阶段使用同一个 `solo12_v3_1` 网络结构，从前一阶段选定 checkpoint 恢复。
`--max_iterations` 在恢复时表示追加轮数。每阶段是否晋级由 held-out 评估决定，
不能只看训练 reward。

| 阶段 | 追加轮数建议 | 观测延迟 | 恢复系数 | 目的 |
|---|---:|---:|---:|---|
| R0 | 600 | 固定 0 ms | 固定 0 | 从零形成 rough locomotion |
| R1 | 300 | `[0,10] ms` | 固定 0 | 首级延迟适应 |
| R2 | 300 | `[0,20] ms` | 固定 0 | 覆盖一个策略周期 |
| R3 | 400 | `[0,30] ms` | 固定 0 | 扩展历史错位 |
| R4 | 300 | `[0,30] ms` | `[0,0.1]` | 引入轻微回弹 |
| R5 | 400 | `[0,30] ms` | `[0,0.2]` | 达到 jump 恢复系数边界 |
| R6 | 500 | `[0,40] ms` | `[0,0.2]` | 40 ms 边界适应与低延迟恢复 |

总建议上限约 2800 轮，即约 11.47 亿 transitions。某阶段提前平台化或退化时，
应回到该阶段最佳 checkpoint，而不是机械跑满。

Go1 的自动课程在所有阶段持续工作：地形按 episode 表现升降；X 速度初始范围
逐步扩展到 `[-2,2] m/s`。Solo 首轮仍采用较温和的地形起点和初始速度范围，
但课程目标不变。

## 8. 每阶段晋级指标

每 20 轮检查：

- episode length、base-contact termination、terrain level 分布；
- X/Y 线速度和 yaw 跟踪误差；
- PPO KL、entropy、value loss、action std；
- HIM velocity MSE、swap loss、prototype 使用分布；
- 关节位置/速度/力矩端点、饱和比例、action rate；
- body/hip/thigh/calf 非足端接触；
- 按随机化参数分桶后的最差分位数，而不只是总体均值。

建议初始晋级门槛：无 NaN、无越限；nominal 平地完成率 ≥95%；当前课程覆盖的
rough terrain 完成率 ≥80%；平面速度 RMSE ≤0.25 m/s；无持续力矩饱和和高频
振荡。上述数值是项目初始门槛，需要结合硬件能力校准，并非论文宣称值。

R6 必须单列 `0/10/20/30/40 ms × restitution 0/0.2` 端点矩阵。40 ms 可以作为
允许退化的边界，但 0–20 ms 的主工作域不得因 R6 明显回退。

## 9. 复现与部署

1. 选定 seed 1 的训练链后，用 seed 2、3 完整复现。
2. 按 held-out 多场景聚合结果选 checkpoint，不按最高训练 reward 选择。
3. Actor 与 HIM estimator 一起导出，冻结归一化、45 维观测布局、6 帧历史顺序、
   command scale 和 action scale。
4. 显式实现策略顺序、URDF 顺序和实机电机顺序的双向映射。
5. 用记录观测做训练端与部署端逐帧回放，要求 estimator 和 actor 输出一致。
6. 真机从吊架/安全绳、低速度命令、保守力矩限制和独立急停开始。

已有 Solo jumping checkpoint 不作为 HIMLoco 初始化权重；两者观察布局、网络、
reward 和算法不同。复用的是机器人控制参数、动力学边界和随机化课程经验。
