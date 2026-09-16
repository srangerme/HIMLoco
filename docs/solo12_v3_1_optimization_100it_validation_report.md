# Solo12 v3.1 优化配置 100-iteration 短训验证报告

## 结论

100-iteration 短训已完成，训练链和连续地形优化方向可行，但当前 reward 组合未通过
完整步态验收。诊断 checkpoint 不晋级，正式训练未启动。

正式候选仍冻结为
`Sep15_17-51-13_formal_r6_height032_latency40_seed1/model_2900.pt`。

本次诊断 run 为
`Sep16_00-57-42_validation_only_optimization_probe_100it_seed1`。它从 model_2900
恢复，使用 1024 环境、100 iteration、每环境每轮 100 steps，共采集 10,240,000
transitions，训练耗时 293.77 s。保存了 2920、2940、2960、2980 和 3000 检查点；
训练过程无 NaN，HIM/PPO 更新与 checkpoint 元数据正常。

## 有效改善

同 seed、同连续 ±1–4 cm 起伏地形短测中，model_2900 的环境通过率为 75%，最长
连续停滞 8.82 s，平均绝对 pitch 6.71°。model_3000 的环境通过率为 100%，最长
停滞为 0，平均绝对 pitch 1.47°，P95 pitch 3.29°，8/8 环境无摔倒。

更长的 16 环境 × 30 s 测试中：

- 平地环境通过率 100%，无停滞、无摔倒，平均绝对 pitch 1.36°，P95 2.61°。
- 完整训练随机化下的连续起伏环境通过率 100%，最长停滞 0.12 s，无摔倒，平均
  绝对 pitch 1.44°，P95 3.33°。
- RS02 48 V 转速—扭矩包络零违规，手册过载时间通过。
- 将硬转速上限提高 10% 后，tracking reward 绝对变化 0.00277，平均前进速度变化
  0.0104 m/s，硬限速依赖 A/B 通过。

因此，连续地形、带容差的 0.32 m 高度和更强姿态约束的总体方向有效；模型没有通过
停止或依赖硬转速裁剪来获取表面成绩。

## 未通过项

model_3000 的平地 FL/FR/RL/RR 摆腿峰值中位数为 6.3/2.8/2.6/6.2 cm；连续起伏
完整随机化下为 6.6/3.8/3.5/6.5 cm。FR 和 RL 仍低于 4 cm 门槛，足间差明显超过
2 cm。

平地逐足站立横向半宽为 13.5/8.8/5.8/10.9 cm；连续起伏完整随机化下为
13.6/9.0/6.0/10.9 cm。RL 仍明显内收，四足站宽差超过 4 cm。

所以 locomotion checker 的 swing clearance、per-foot symmetry 和 stance width
仍失败。model_3000 不是合格候选。

## 原因定位和计划修正

当前 `diagonal_action_symmetry` 只约束 FR–RL 和 FL–RR 各对内部同步，没有约束两组
对角腿之间的幅度、摆动高度和 duty factor 一致。短训因此收敛到“一组高抬、另一组
低拖”的局部最优。统一的 hip posture 二次惩罚也无法直接保证每只脚达到最小站宽。

下一次诊断不能简单延长当前 run，应回到 model_2900 做新的单变量 A/B：

1. 将抬腿 reward 从四足求和改为逐足门槛与 worst-foot 聚合，使 FR/RL 的不足不能被
   FL/RR 的高抬抵消。
2. 增加两组对角腿之间的 swing peak、air time 和 contact duty 平衡，而不只比较瞬时
   action。
3. 用 body-frame 逐足站宽下限替代或补充统一 hip posture；只惩罚低于目标站宽，保留
   崎岖地形上的外展自由度。
4. 下一轮仍只做 100 iteration 诊断，并以逐足指标决定是否继续；当前 model_3000
   不作为其 parent。

## 状态

100-iteration 短训验证完成；结果为“方向部分有效、完整验收失败”。没有启动正式训练。

## V2 可行性补充验证

第二个 100-iteration 诊断 run
`Sep16_01-18-07_validation_only_gait_v2_100it_seed1` 仍从正式候选
`model_2900.pt` 恢复，只用于验证逐足 reward 的方向。它把 clearance 改成最差足聚合，
并加入 body-frame 逐足站宽下限。训练链稳定，无 NaN，未启动正式训练。

连续 ±1–4 cm 起伏地面上，checkpoint 3000 的 8/8 环境均通过，零摔倒、零连续停滞，
平均绝对 pitch 为 2.52°。FL/FR/RL/RR 摆腿峰值由 V1 的约
6.4/3.6/3.3/6.5 cm 变为 6.0/3.9/3.5/5.8 cm；逐足站立横向半宽由约
13.6/8.8/6.0/10.8 cm 变为 12.7/8.8/7.8/9.9 cm。FR/RL 抬脚与 RL 站宽均沿目标
方向改善，证明逐足 clearance 和 body-frame stance shaping 可行。

本轮目的为验证修复方案可行，而不是让 100-iteration 诊断模型完成全部验收，因此不再
继续 V3 超参数搜索。正式方案应保留 V2 的逐足约束，但把很弱的 swing-height EMA
均衡项替换为“每次摆动结束时按该足峰值结算”的 clearance 缺口；站宽从单侧下限改成
带容差的目标区间，避免一只脚过宽而另一只脚仍内收。正式训练期间再按冻结 checkpoint
调权，不把本次 model_3000 当作发布候选。
