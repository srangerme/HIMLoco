# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .base.legged_robot import LeggedRobot
from .a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from legged_gym.envs.go1.go1_config import Go1RoughCfg, Go1RoughCfgPPO
from legged_gym.envs.aliengo.aliengo_config import AlienGoRoughCfg, AlienGoRoughCfgPPO
from legged_gym.envs.solo12.solo12_v3_1_config import (
    Solo12V31RoughCfg,
    Solo12V31RoughCfgPPO,
    Solo12V31R1Cfg,
    Solo12V31R2Cfg,
    Solo12V31R3Cfg,
    Solo12V31R4Cfg,
    Solo12V31R5Cfg,
    Solo12V31R6Cfg,
    Solo12V31OptimizationProbeCfg,
    Solo12V31OptimizationProbeV2Cfg,
    Solo12V31FormalGaitRepairCfg,
    Solo12V31FormalGaitRefineCfg,
    Solo12V31FormalGaitBalanceCfg,
    Solo12V31FormalRobustCfg,
    Solo12V31FormalPostureFixCfg,
    Solo12V31FormalPostureFinishCfg,
    Solo12V31FormalStairFeasibilityCfg,
    Solo12V31FormalFullSceneCurriculumCfg,
    Solo12V31FormalFullSceneFinalCfg,
    Solo12V31ObstacleCurriculumAConfig,
    Solo12V31ObstacleCurriculumBConfig,
    Solo12V31ObstacleCurriculumCConfig,
    Solo12V31ObstacleMixedStabilityConfig,
    Solo12V31ObstacleMixedStabilityHighStairConfig,
    Solo12V31ObstacleMixedStabilityReplayConfig,
    Solo12V31ArmatureAdaptConfig,
    Solo12V31ArmatureReplayConfig,
    Solo12V31Payload25Config,
    Solo12V31Payload50Config,
    Solo12V31Payload75Config,
    Solo12V31Payload100Config,
    Solo12V31Payload100TorqueRecoveryConfig,
    Solo12V31Payload100Envelope125Config,
    Solo12V31Payload100Envelope150Config,
    Solo12V31Payload100Envelope200Config,
    Solo12V31RS06CalfAdaptConfig,
    Solo12V31RS06CalfWarmupConfig,
    Solo12V31RS06CalfIntermediateConfig,
    Solo12V31RS06Payload25Config,
    Solo12V31RS06Payload50Config,
    Solo12V31RS06Payload75Config,
    Solo12V31RS06Payload100Config,
    Solo12V31RS06Payload100TorqueRecoveryConfig,
    Solo12V31RS06Payload100TorqueRecoveryStrongConfig,
    Solo12V31RS06Thigh2Payload100AdaptConfig,
    Solo12V31RS06Thigh2Payload100HardReplayConfig,
    Solo12V31RS06Thigh2Payload100SlopeFocusConfig,
    Solo12V31RS06Thigh2Payload100SlopeHighConfig,
    Solo12V31RS06Thigh2Payload100FinalReplayConfig,
    Solo12V31RS06Thigh2Payload100UnevenTorqueRecoveryConfig,
    Solo12V31RS03ThighPayload100AdaptConfig,
    Solo12V31RS06Thigh2StraightObstacleAConfig,
    Solo12V31RS06Thigh2StraightObstacleBConfig,
    Solo12V31RS06Thigh2StraightObstacleFinalConfig,
    Solo12V31RS06Thigh2StraightSlope45Config,
    Solo12V31RS06Thigh2StraightSlope45CapabilityConfig,
    Solo12V31RS06Thigh3StraightObstacleAdaptConfig,
    Solo12V31RS06Thigh3StraightObstacleReplayConfig,
    Solo12V31RS06Thigh3Speed15AdaptConfig,
    Solo12V31RS06Thigh3Speed20AdaptConfig,
    Solo12V31RS06Thigh3StraightSlopeAdaptConfig,
    Solo12V31RS06Payload100HardReplayConfig,
    Solo12V31ArmatureDeployNominalConfig,
    Solo12V31ObstacleMixedStabilityFlatReplayConfig,
    Solo12V31ObstacleHardReplayConfig,
    Solo12V31ObstacleFocusedReplayConfig,
    Solo12V31ObstacleTargetedReplayConfig,
    Solo12V31TargetedFinetuneCfgPPO,
    Solo12V31ObstacleBalancedReplayConfig,
    Solo12V31ObstacleFinalDynamicsConfig,
    Solo12V31ObstacleFinalLatency10Config,
    Solo12V31ObstacleFinalLatency15Config,
    Solo12V31ObstacleFinalLatency20Config,
    Solo12V31ObstacleFinalLatency25Config,
    Solo12V31ObstacleFinalLatency30Config,
    Solo12V31ObstacleFinalLatency35Config,
    Solo12V31ObstacleFinalLatencyConfig,
    Solo12V31ObstacleStairPolishConfig,
    Solo12V31ObstacleFinalExternalWarmupConfig,
    Solo12V31ObstacleFinalActionDelayConfig,
    Solo12V31ObstacleFinalConfig,
)

import os

from legged_gym.utils.task_registry import task_registry

task_registry.register( "a1", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO() )
task_registry.register( "go1", LeggedRobot, Go1RoughCfg(), Go1RoughCfgPPO() )
task_registry.register( "aliengo", LeggedRobot, AlienGoRoughCfg(), AlienGoRoughCfgPPO() )
task_registry.register( "solo12_v3_1", LeggedRobot, Solo12V31RoughCfg(), Solo12V31RoughCfgPPO() )
task_registry.register( "solo12_v3_1_r0", LeggedRobot, Solo12V31RoughCfg(), Solo12V31RoughCfgPPO() )
task_registry.register( "solo12_v3_1_r1", LeggedRobot, Solo12V31R1Cfg(), Solo12V31RoughCfgPPO() )
task_registry.register( "solo12_v3_1_r2", LeggedRobot, Solo12V31R2Cfg(), Solo12V31RoughCfgPPO() )
task_registry.register( "solo12_v3_1_r3", LeggedRobot, Solo12V31R3Cfg(), Solo12V31RoughCfgPPO() )
task_registry.register( "solo12_v3_1_r4", LeggedRobot, Solo12V31R4Cfg(), Solo12V31RoughCfgPPO() )
task_registry.register( "solo12_v3_1_r5", LeggedRobot, Solo12V31R5Cfg(), Solo12V31RoughCfgPPO() )
task_registry.register( "solo12_v3_1_r6", LeggedRobot, Solo12V31R6Cfg(), Solo12V31RoughCfgPPO() )
task_registry.register( "solo12_v3_1_optimization_probe", LeggedRobot, Solo12V31OptimizationProbeCfg(), Solo12V31RoughCfgPPO() )
task_registry.register( "solo12_v3_1_optimization_probe_v2", LeggedRobot, Solo12V31OptimizationProbeV2Cfg(), Solo12V31RoughCfgPPO() )
task_registry.register( "solo12_v3_1_formal_gait_repair", LeggedRobot, Solo12V31FormalGaitRepairCfg(), Solo12V31RoughCfgPPO() )
task_registry.register( "solo12_v3_1_formal_gait_refine", LeggedRobot, Solo12V31FormalGaitRefineCfg(), Solo12V31RoughCfgPPO() )
task_registry.register( "solo12_v3_1_formal_gait_balance", LeggedRobot, Solo12V31FormalGaitBalanceCfg(), Solo12V31RoughCfgPPO() )
task_registry.register( "solo12_v3_1_formal_robust", LeggedRobot, Solo12V31FormalRobustCfg(), Solo12V31RoughCfgPPO() )
task_registry.register( "solo12_v3_1_formal_posture_fix", LeggedRobot, Solo12V31FormalPostureFixCfg(), Solo12V31RoughCfgPPO() )
task_registry.register( "solo12_v3_1_formal_posture_finish", LeggedRobot, Solo12V31FormalPostureFinishCfg(), Solo12V31RoughCfgPPO() )
task_registry.register( "solo12_v3_1_formal_stair_feasibility", LeggedRobot, Solo12V31FormalStairFeasibilityCfg(), Solo12V31RoughCfgPPO() )
task_registry.register( "solo12_v3_1_formal_fullscene_curriculum", LeggedRobot, Solo12V31FormalFullSceneCurriculumCfg(), Solo12V31RoughCfgPPO() )
task_registry.register( "solo12_v3_1_formal_fullscene_final", LeggedRobot, Solo12V31FormalFullSceneFinalCfg(), Solo12V31RoughCfgPPO() )
task_registry.register( "solo12_v3_1_obstacle_a", LeggedRobot, Solo12V31ObstacleCurriculumAConfig(), Solo12V31RoughCfgPPO() )
task_registry.register( "solo12_v3_1_obstacle_b", LeggedRobot, Solo12V31ObstacleCurriculumBConfig(), Solo12V31RoughCfgPPO() )
task_registry.register( "solo12_v3_1_obstacle_c", LeggedRobot, Solo12V31ObstacleCurriculumCConfig(), Solo12V31RoughCfgPPO() )
task_registry.register( "solo12_v3_1_obstacle_mixed_stability", LeggedRobot, Solo12V31ObstacleMixedStabilityConfig(), Solo12V31RoughCfgPPO() )
task_registry.register( "solo12_v3_1_obstacle_mixed_stability_high_stair", LeggedRobot, Solo12V31ObstacleMixedStabilityHighStairConfig(), Solo12V31RoughCfgPPO() )
task_registry.register( "solo12_v3_1_obstacle_mixed_stability_replay", LeggedRobot, Solo12V31ObstacleMixedStabilityReplayConfig(), Solo12V31RoughCfgPPO() )
task_registry.register( "solo12_v3_1_armature_adapt", LeggedRobot, Solo12V31ArmatureAdaptConfig(), Solo12V31TargetedFinetuneCfgPPO() )
task_registry.register( "solo12_v3_1_armature_replay", LeggedRobot, Solo12V31ArmatureReplayConfig(), Solo12V31TargetedFinetuneCfgPPO() )
task_registry.register( "solo12_v3_1_payload_2_5", LeggedRobot, Solo12V31Payload25Config(), Solo12V31TargetedFinetuneCfgPPO() )
task_registry.register( "solo12_v3_1_payload_5", LeggedRobot, Solo12V31Payload50Config(), Solo12V31TargetedFinetuneCfgPPO() )
task_registry.register( "solo12_v3_1_payload_7_5", LeggedRobot, Solo12V31Payload75Config(), Solo12V31TargetedFinetuneCfgPPO() )
task_registry.register( "solo12_v3_1_payload_10", LeggedRobot, Solo12V31Payload100Config(), Solo12V31TargetedFinetuneCfgPPO() )
task_registry.register( "solo12_v3_1_payload_10_torque_recovery", LeggedRobot, Solo12V31Payload100TorqueRecoveryConfig(), Solo12V31TargetedFinetuneCfgPPO() )
task_registry.register( "solo12_v3_1_payload_10_envelope125", LeggedRobot, Solo12V31Payload100Envelope125Config(), Solo12V31TargetedFinetuneCfgPPO() )
task_registry.register( "solo12_v3_1_payload_10_envelope150", LeggedRobot, Solo12V31Payload100Envelope150Config(), Solo12V31TargetedFinetuneCfgPPO() )
task_registry.register( "solo12_v3_1_payload_10_envelope200", LeggedRobot, Solo12V31Payload100Envelope200Config(), Solo12V31TargetedFinetuneCfgPPO() )
task_registry.register( "solo12_v3_1_rs06_adapt", LeggedRobot, Solo12V31RS06CalfAdaptConfig(), Solo12V31TargetedFinetuneCfgPPO() )
task_registry.register( "solo12_v3_1_rs06_warmup", LeggedRobot, Solo12V31RS06CalfWarmupConfig(), Solo12V31TargetedFinetuneCfgPPO() )
task_registry.register( "solo12_v3_1_rs06_intermediate", LeggedRobot, Solo12V31RS06CalfIntermediateConfig(), Solo12V31TargetedFinetuneCfgPPO() )
task_registry.register( "solo12_v3_1_rs06_payload_2_5", LeggedRobot, Solo12V31RS06Payload25Config(), Solo12V31TargetedFinetuneCfgPPO() )
task_registry.register( "solo12_v3_1_rs06_payload_5", LeggedRobot, Solo12V31RS06Payload50Config(), Solo12V31TargetedFinetuneCfgPPO() )
task_registry.register( "solo12_v3_1_rs06_payload_7_5", LeggedRobot, Solo12V31RS06Payload75Config(), Solo12V31TargetedFinetuneCfgPPO() )
task_registry.register( "solo12_v3_1_rs06_payload_10", LeggedRobot, Solo12V31RS06Payload100Config(), Solo12V31TargetedFinetuneCfgPPO() )
task_registry.register( "solo12_v3_1_rs06_payload_10_torque_recovery", LeggedRobot, Solo12V31RS06Payload100TorqueRecoveryConfig(), Solo12V31TargetedFinetuneCfgPPO() )
task_registry.register( "solo12_v3_1_rs06_payload_10_torque_recovery_strong", LeggedRobot, Solo12V31RS06Payload100TorqueRecoveryStrongConfig(), Solo12V31TargetedFinetuneCfgPPO() )
task_registry.register( "solo12_v3_1_rs06_thigh2_payload_10_adapt", LeggedRobot, Solo12V31RS06Thigh2Payload100AdaptConfig(), Solo12V31TargetedFinetuneCfgPPO() )
task_registry.register( "solo12_v3_1_rs06_thigh2_payload_10_hard_replay", LeggedRobot, Solo12V31RS06Thigh2Payload100HardReplayConfig(), Solo12V31TargetedFinetuneCfgPPO() )
task_registry.register( "solo12_v3_1_rs06_thigh2_payload_10_slope_focus", LeggedRobot, Solo12V31RS06Thigh2Payload100SlopeFocusConfig(), Solo12V31TargetedFinetuneCfgPPO() )
task_registry.register( "solo12_v3_1_rs06_thigh2_payload_10_slope_high", LeggedRobot, Solo12V31RS06Thigh2Payload100SlopeHighConfig(), Solo12V31TargetedFinetuneCfgPPO() )
task_registry.register( "solo12_v3_1_rs06_thigh2_payload_10_final_replay", LeggedRobot, Solo12V31RS06Thigh2Payload100FinalReplayConfig(), Solo12V31TargetedFinetuneCfgPPO() )
task_registry.register( "solo12_v3_1_rs06_thigh2_payload_10_uneven_torque_recovery", LeggedRobot, Solo12V31RS06Thigh2Payload100UnevenTorqueRecoveryConfig(), Solo12V31TargetedFinetuneCfgPPO() )
task_registry.register( "solo12_v3_1_rs03_thigh_payload_10_adapt", LeggedRobot, Solo12V31RS03ThighPayload100AdaptConfig(), Solo12V31TargetedFinetuneCfgPPO() )
task_registry.register( "solo12_v3_1_rs06_thigh2_straight_obstacle_a", LeggedRobot, Solo12V31RS06Thigh2StraightObstacleAConfig(), Solo12V31TargetedFinetuneCfgPPO() )
task_registry.register( "solo12_v3_1_rs06_thigh2_straight_obstacle_b", LeggedRobot, Solo12V31RS06Thigh2StraightObstacleBConfig(), Solo12V31TargetedFinetuneCfgPPO() )
task_registry.register( "solo12_v3_1_rs06_thigh2_straight_obstacle_final", LeggedRobot, Solo12V31RS06Thigh2StraightObstacleFinalConfig(), Solo12V31TargetedFinetuneCfgPPO() )
task_registry.register( "solo12_v3_1_rs06_thigh2_straight_slope45", LeggedRobot, Solo12V31RS06Thigh2StraightSlope45Config(), Solo12V31TargetedFinetuneCfgPPO() )
task_registry.register( "solo12_v3_1_rs06_thigh2_straight_slope45_capability", LeggedRobot, Solo12V31RS06Thigh2StraightSlope45CapabilityConfig(), Solo12V31TargetedFinetuneCfgPPO() )
task_registry.register( "solo12_v3_1_rs06_thigh3_straight_obstacle_adapt", LeggedRobot, Solo12V31RS06Thigh3StraightObstacleAdaptConfig(), Solo12V31TargetedFinetuneCfgPPO() )
task_registry.register( "solo12_v3_1_rs06_thigh3_straight_obstacle_replay", LeggedRobot, Solo12V31RS06Thigh3StraightObstacleReplayConfig(), Solo12V31TargetedFinetuneCfgPPO() )
task_registry.register( "solo12_v3_1_rs06_thigh3_speed15_adapt", LeggedRobot, Solo12V31RS06Thigh3Speed15AdaptConfig(), Solo12V31TargetedFinetuneCfgPPO() )
task_registry.register( "solo12_v3_1_rs06_thigh3_speed20_adapt", LeggedRobot, Solo12V31RS06Thigh3Speed20AdaptConfig(), Solo12V31TargetedFinetuneCfgPPO() )
task_registry.register( "solo12_v3_1_rs06_thigh3_straight_slope_adapt", LeggedRobot, Solo12V31RS06Thigh3StraightSlopeAdaptConfig(), Solo12V31TargetedFinetuneCfgPPO() )
task_registry.register( "solo12_v3_1_rs06_payload_10_hard_replay", LeggedRobot, Solo12V31RS06Payload100HardReplayConfig(), Solo12V31TargetedFinetuneCfgPPO() )
task_registry.register( "solo12_v3_1_armature_deploy_nominal", LeggedRobot, Solo12V31ArmatureDeployNominalConfig(), Solo12V31TargetedFinetuneCfgPPO() )
task_registry.register( "solo12_v3_1_obstacle_mixed_stability_flat_replay", LeggedRobot, Solo12V31ObstacleMixedStabilityFlatReplayConfig(), Solo12V31RoughCfgPPO() )
task_registry.register( "solo12_v3_1_obstacle_hard_replay", LeggedRobot, Solo12V31ObstacleHardReplayConfig(), Solo12V31RoughCfgPPO() )
task_registry.register( "solo12_v3_1_obstacle_focused_replay", LeggedRobot, Solo12V31ObstacleFocusedReplayConfig(), Solo12V31RoughCfgPPO() )
task_registry.register( "solo12_v3_1_obstacle_targeted_replay", LeggedRobot, Solo12V31ObstacleTargetedReplayConfig(), Solo12V31TargetedFinetuneCfgPPO() )
task_registry.register( "solo12_v3_1_obstacle_balanced_replay", LeggedRobot, Solo12V31ObstacleBalancedReplayConfig(), Solo12V31TargetedFinetuneCfgPPO() )
task_registry.register( "solo12_v3_1_obstacle_final_dynamics", LeggedRobot, Solo12V31ObstacleFinalDynamicsConfig(), Solo12V31RoughCfgPPO() )
task_registry.register( "solo12_v3_1_obstacle_final_latency10", LeggedRobot, Solo12V31ObstacleFinalLatency10Config(), Solo12V31RoughCfgPPO() )
task_registry.register( "solo12_v3_1_obstacle_final_latency15", LeggedRobot, Solo12V31ObstacleFinalLatency15Config(), Solo12V31RoughCfgPPO() )
task_registry.register( "solo12_v3_1_obstacle_final_latency20", LeggedRobot, Solo12V31ObstacleFinalLatency20Config(), Solo12V31RoughCfgPPO() )
task_registry.register( "solo12_v3_1_obstacle_final_latency25", LeggedRobot, Solo12V31ObstacleFinalLatency25Config(), Solo12V31RoughCfgPPO() )
task_registry.register( "solo12_v3_1_obstacle_final_latency30", LeggedRobot, Solo12V31ObstacleFinalLatency30Config(), Solo12V31RoughCfgPPO() )
task_registry.register( "solo12_v3_1_obstacle_final_latency35", LeggedRobot, Solo12V31ObstacleFinalLatency35Config(), Solo12V31RoughCfgPPO() )
task_registry.register( "solo12_v3_1_obstacle_final_latency", LeggedRobot, Solo12V31ObstacleFinalLatencyConfig(), Solo12V31RoughCfgPPO() )
task_registry.register( "solo12_v3_1_obstacle_stair_polish", LeggedRobot, Solo12V31ObstacleStairPolishConfig(), Solo12V31RoughCfgPPO() )
task_registry.register( "solo12_v3_1_obstacle_final_external_warmup", LeggedRobot, Solo12V31ObstacleFinalExternalWarmupConfig(), Solo12V31RoughCfgPPO() )
task_registry.register( "solo12_v3_1_obstacle_final_action_delay", LeggedRobot, Solo12V31ObstacleFinalActionDelayConfig(), Solo12V31RoughCfgPPO() )
task_registry.register( "solo12_v3_1_obstacle_final", LeggedRobot, Solo12V31ObstacleFinalConfig(), Solo12V31RoughCfgPPO() )
