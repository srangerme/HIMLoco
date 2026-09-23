"""HIMLoco rough-terrain locomotion configuration for Solo12 v3.1."""

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class Solo12V31RoughCfg(LeggedRobotCfg):
    class terrain(LeggedRobotCfg.terrain):
        # Solo's 0.4 m legs benefit from a gentler starting curriculum than the
        # Unitree defaults. The curriculum can still advance through all rows.
        max_init_terrain_level = 2

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.32]
        default_joint_angles = {
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.6000,
            "FL_calf_joint": -1.2000,
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": 0.6000,
            "FR_calf_joint": -1.2000,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.6000,
            "RL_calf_joint": -1.2000,
            "RR_hip_joint": 0.0,
            "RR_thigh_joint": 0.6000,
            "RR_calf_joint": -1.2000,
        }

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {"joint": 16.0}
        damping = {"joint": 0.5}
        action_scale = 0.25
        decimation = 4
        hip_reduction = 0.5
        use_torque_speed_envelope = True
        # Approximate samples read from the RS02 48 V T-N curve. The terminal
        # point uses the documented 44 rad/s actuator speed limit and zero
        # available drive torque so the URDF limit remains a safety boundary.
        torque_speed_curve_rpm = [
            0.0, 60.0, 120.0, 180.0, 210.0, 225.0, 240.0, 255.0,
            270.0, 285.0, 300.0, 315.0, 330.0, 345.0, 360.0, 375.0,
            390.0, 405.0, 420.0,
        ]
        torque_speed_curve_nm = [
            17.0, 16.8, 16.7, 16.9, 16.8, 16.6, 16.0, 15.4,
            14.7, 13.4, 12.0, 10.8, 9.5, 8.2, 6.1, 4.7,
            3.4, 1.9, 0.0,
        ]
        joint_gear_ratios = {"hip": 1.0, "thigh": 1.0, "calf": 2.0}
        # No external-transmission efficiency data is available in the manual;
        # use the explicitly ideal value for the first model.
        transmission_efficiencies = {"joint": 1.0}

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 2.0
        resampling_time = 10.0
        heading_command = True

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.5, 0.5]
            lin_vel_y = [-0.5, 0.5]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-3.14, 3.14]

    class asset(LeggedRobotCfg.asset):
        file = "/workspace/models/solo12_v3_1/urdf/solo12_v3_1.urdf"
        name = "solo12_v3_1"
        foot_name = "foot"
        penalize_contacts_on = ["hip", "thigh", "calf", "Body"]
        terminate_after_contacts_on = ["Body"]
        privileged_contacts_on = ["Body", "hip", "thigh", "calf"]
        collapse_fixed_joints = True
        self_collisions = 0
        flip_visual_attachments = False
        armature = 0.0

    class domain_rand(LeggedRobotCfg.domain_rand):
        # R0 baseline: retain Go1's all-on training structure while using the
        # robot-specific ranges validated by the Solo jumping project.
        payload_mass_range = [-0.635, 1.906]
        com_displacement_range = [-0.02, 0.02]
        randomize_link_mass = True
        link_mass_range = [0.7, 1.3]
        friction_range = [0.7, 1.25]
        # Restitution is introduced by checkpoint curriculum after R0.
        randomize_restitution = False
        restitution_range = [0.0, 0.2]
        motor_strength_range = [0.9, 1.1]
        kp_range = [0.9, 1.1]
        kd_range = [0.9, 1.1]
        per_joint_randomization = True
        randomize_joint_friction = True
        joint_friction_range = [0.0, 0.02]
        randomize_joint_damping = True
        joint_damping_range = [0.0, 0.01]
        initial_joint_pos_range = [0.8, 1.2]
        disturbance_range = [-30.0, 30.0]
        max_push_vel_xy = 1.0
        # The legacy flag implements action delay, not the validated jumping
        # observation latency. Keep it off until the history-latency curriculum
        # described in the training plan is implemented.
        delay = False
        observation_latency = False
        observation_latency_range = [0.0, 0.0]
        observation_latency_min_probability = 0.0
        observation_latency_max_probability = 0.0

    class rewards(LeggedRobotCfg.rewards):
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.2
            dof_acc = -2.5e-7
            joint_power = -2e-5
            base_height = -500.0
            foot_clearance = -0.01
            action_rate = -0.01
            smoothness = -0.01
            feet_air_time = 0.0
            collision = -0.1
            feet_stumble = -0.0
            stand_still = -0.0
            torques = -0.0
            dof_vel = -0.0
            dof_pos_limits = -0.0
            dof_vel_limits = -0.0
            torque_limits = -0.0

        only_positive_rewards = False
        tracking_sigma = 0.25
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.95
        soft_torque_limit = 0.95
        base_height_target = 0.32
        max_contact_force = 80.0
        clearance_height_target = -0.20

    class sim(LeggedRobotCfg.sim):
        class physx(LeggedRobotCfg.sim.physx):
            # The Solo model's small spherical feet need the additional contact
            # iterations already validated by the jumping task.
            num_position_iterations = 8

    class viewer(LeggedRobotCfg.viewer):
        pos = [0.9, -0.75, 0.55]
        lookat = [0.0, 0.0, 0.22]


class Solo12V31RoughCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "rough_solo12_v3_1"


class Solo12V31R1Cfg(Solo12V31RoughCfg):
    class domain_rand(Solo12V31RoughCfg.domain_rand):
        observation_latency = True
        observation_latency_range = [0.0, 10.0]
        observation_latency_min_probability = 0.1
        observation_latency_max_probability = 0.1


class Solo12V31R2Cfg(Solo12V31RoughCfg):
    class domain_rand(Solo12V31RoughCfg.domain_rand):
        observation_latency = True
        observation_latency_range = [0.0, 20.0]
        observation_latency_min_probability = 0.1
        observation_latency_max_probability = 0.1


class Solo12V31R3Cfg(Solo12V31RoughCfg):
    class domain_rand(Solo12V31RoughCfg.domain_rand):
        observation_latency = True
        observation_latency_range = [0.0, 30.0]
        observation_latency_min_probability = 0.1
        observation_latency_max_probability = 0.1


class Solo12V31R4Cfg(Solo12V31R3Cfg):
    class domain_rand(Solo12V31R3Cfg.domain_rand):
        randomize_restitution = True
        restitution_range = [0.0, 0.1]


class Solo12V31R5Cfg(Solo12V31R3Cfg):
    class domain_rand(Solo12V31R3Cfg.domain_rand):
        randomize_restitution = True
        restitution_range = [0.0, 0.2]


class Solo12V31R6Cfg(Solo12V31R5Cfg):
    class domain_rand(Solo12V31R5Cfg.domain_rand):
        observation_latency_range = [0.0, 40.0]


class Solo12V31OptimizationProbeCfg(Solo12V31RoughCfg):
    """Short-run probe only; not a formal-training or release task."""

    class terrain(Solo12V31RoughCfg.terrain):
        continuous_uneven = True
        terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0]
        max_init_terrain_level = 1

    class domain_rand(Solo12V31RoughCfg.domain_rand):
        payload_mass_range = [-0.3, 0.5]
        link_mass_range = [0.9, 1.1]
        friction_range = [0.8, 1.5]
        motor_strength_range = [0.95, 1.05]
        kp_range = [0.95, 1.05]
        kd_range = [0.95, 1.05]
        initial_joint_pos_range = [0.9, 1.1]
        disturbance_range = [-10.0, 10.0]
        max_push_vel_xy = 0.3
        observation_latency = False
        observation_latency_range = [0.0, 0.0]

    class rewards(Solo12V31RoughCfg.rewards):
        class scales(Solo12V31RoughCfg.rewards.scales):
            termination = -2.0
            tracking_lin_vel = 1.5
            ang_vel_xy = -0.1
            orientation = -0.5
            base_height = -250.0
            foot_clearance = 0.0
            swing_clearance = -10.0
            swing_time = 0.3
            feet_slip = -0.2
            hip_posture = -0.05
            diagonal_action_symmetry = -0.01
            collision = -0.2
            stumble = -0.5
            feet_contact_forces = -0.002

        base_height_target = 0.32
        base_height_tolerance = 0.01
        swing_clearance_target = 0.06
        swing_time_target = 0.25


class Solo12V31OptimizationProbeV2Cfg(Solo12V31OptimizationProbeCfg):
    """Second validation probe with per-foot worst-case gait shaping."""

    class rewards(Solo12V31OptimizationProbeCfg.rewards):
        class scales(Solo12V31OptimizationProbeCfg.rewards.scales):
            swing_clearance = 0.0
            swing_clearance_worst = -40.0
            swing_height_balance = -20.0
            swing_time = 0.6
            hip_posture = 0.0
            diagonal_action_symmetry = -0.005
            stance_width = -20.0

        swing_clearance_target = 0.06
        swing_height_ema_alpha = 0.05
        stance_half_width_target = 0.10


class Solo12V31FormalGaitRepairCfg(Solo12V31OptimizationProbeV2Cfg):
    """Formal forward-gait repair before restoring the full randomization set."""

    class commands(Solo12V31OptimizationProbeV2Cfg.commands):
        class ranges(Solo12V31OptimizationProbeV2Cfg.commands.ranges):
            lin_vel_x = [0.0, 0.6]
            lin_vel_y = [-0.15, 0.15]
            ang_vel_yaw = [-0.3, 0.3]
            heading = [-0.5, 0.5]

    class rewards(Solo12V31OptimizationProbeV2Cfg.rewards):
        class scales(Solo12V31OptimizationProbeV2Cfg.rewards.scales):
            swing_clearance_worst = 0.0
            swing_height_balance = 0.0
            swing_peak_clearance = -20.0
            stance_width = 0.0
            stance_width_band = -2.0
            diagonal_action_symmetry = -0.002

        swing_peak_clearance_target = 0.055
        stance_half_width_min = 0.09
        stance_half_width_max = 0.12


class Solo12V31FormalGaitRefineCfg(Solo12V31FormalGaitRepairCfg):
    """Refine repaired gait with a bounded, per-foot swing-peak target."""

    class rewards(Solo12V31FormalGaitRepairCfg.rewards):
        class scales(Solo12V31FormalGaitRepairCfg.rewards.scales):
            swing_peak_clearance = 0.0
            swing_peak_height_band = -20.0

        swing_peak_height_min = 0.055
        swing_peak_height_max = 0.075


class Solo12V31FormalGaitBalanceCfg(Solo12V31FormalGaitRefineCfg):
    """Final per-foot balance pass with a narrow swing-peak band."""

    class rewards(Solo12V31FormalGaitRefineCfg.rewards):
        class scales(Solo12V31FormalGaitRefineCfg.rewards.scales):
            swing_peak_height_band = -60.0

        swing_peak_height_min = 0.06
        swing_peak_height_max = 0.07


class Solo12V31FormalRobustCfg(Solo12V31FormalGaitRefineCfg):
    """Formal gait repair with the complete R6 dynamics/latency envelope."""

    class commands(Solo12V31RoughCfg.commands):
        pass

    class domain_rand(Solo12V31R6Cfg.domain_rand):
        pass


class Solo12V31FormalPostureFixCfg(Solo12V31FormalRobustCfg):
    """Restore the commanded 0.32 m posture without constraining natural footfall."""

    class rewards(Solo12V31FormalRobustCfg.rewards):
        class scales(Solo12V31FormalRobustCfg.rewards.scales):
            base_height = -1000.0
            orientation = -5.0
            ang_vel_xy = -0.2

        base_height_target = 0.32
        base_height_tolerance = 0.005


class Solo12V31FormalPostureFinishCfg(Solo12V31FormalPostureFixCfg):
    """Finish convergence to the 0.32 m target with a low-side linear term."""

    class rewards(Solo12V31FormalPostureFixCfg.rewards):
        class scales(Solo12V31FormalPostureFixCfg.rewards.scales):
            base_height_low = -30.0


class Solo12V31FormalStairFeasibilityCfg(Solo12V31FormalPostureFinishCfg):
    """Warm-start the complete original terrain mix at low difficulty."""

    class terrain(Solo12V31FormalPostureFinishCfg.terrain):
        continuous_uneven = False
        up_down_stairs = False
        spawn_offset_x = None
        geometry_scale = 0.94
        terrain_length = 7.52
        terrain_width = 7.52
        terrain_proportions = [0.1, 0.2, 0.3, 0.3, 0.1]
        max_init_terrain_level = 1

    class commands(Solo12V31FormalPostureFinishCfg.commands):
        curriculum = False
        heading_command = False

        class ranges(Solo12V31FormalPostureFinishCfg.commands.ranges):
            lin_vel_x = [0.2, 0.6]
            lin_vel_y = [-0.15, 0.15]
            ang_vel_yaw = [-0.3, 0.3]
            heading = [-0.5, 0.5]

    class domain_rand(Solo12V31FormalPostureFinishCfg.domain_rand):
        payload_mass_range = [-0.3, 0.5]
        link_mass_range = [0.9, 1.1]
        friction_range = [0.8, 1.5]
        motor_strength_range = [0.95, 1.05]
        kp_range = [0.95, 1.05]
        kd_range = [0.95, 1.05]
        initial_joint_pos_range = [0.9, 1.1]
        push_robots = False
        disturbance = False
        observation_latency = False
        observation_latency_range = [0.0, 0.0]

    class rewards(Solo12V31FormalPostureFinishCfg.rewards):
        class scales(Solo12V31FormalPostureFinishCfg.rewards.scales):
            tracking_lin_vel = 2.5
            commanded_stall = -3.0


class Solo12V31FormalFullSceneCurriculumCfg(Solo12V31FormalStairFeasibilityCfg):
    """Expand commands and dynamics while retaining the complete terrain mix."""

    class terrain(Solo12V31FormalStairFeasibilityCfg.terrain):
        max_init_terrain_level = 4

    class commands(Solo12V31FormalStairFeasibilityCfg.commands):
        curriculum = False
        heading_command = True

        class ranges(Solo12V31FormalStairFeasibilityCfg.commands.ranges):
            lin_vel_x = [-0.7, 0.7]
            lin_vel_y = [-0.4, 0.4]
            ang_vel_yaw = [-1.5, 1.5]
            heading = [-3.14, 3.14]

    class domain_rand(Solo12V31FormalStairFeasibilityCfg.domain_rand):
        payload_mass_range = [-0.5, 1.0]
        link_mass_range = [0.85, 1.15]
        friction_range = [0.7, 2.0]
        motor_strength_range = [0.93, 1.07]
        kp_range = [0.93, 1.07]
        kd_range = [0.93, 1.07]
        initial_joint_pos_range = [0.85, 1.15]


class Solo12V31FormalFullSceneFinalCfg(Solo12V31FormalFullSceneCurriculumCfg):
    """Final unscaled command range with the complete R6 randomization set."""

    class terrain(Solo12V31FormalFullSceneCurriculumCfg.terrain):
        max_init_terrain_level = 9

    class commands(Solo12V31FormalFullSceneCurriculumCfg.commands):
        curriculum = False
        heading_command = True

        class ranges(Solo12V31FormalFullSceneCurriculumCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-3.14, 3.14]
            heading = [-3.14, 3.14]

    class domain_rand(Solo12V31FormalPostureFinishCfg.domain_rand):
        pass


class Solo12V31ObstacleCurriculumAConfig(Solo12V31FormalStairFeasibilityCfg):
    """Relearn obstacle clearance on low terrain levels without an upper swing cap."""

    class terrain(Solo12V31FormalStairFeasibilityCfg.terrain):
        max_init_terrain_level = 2

    class commands(Solo12V31FormalStairFeasibilityCfg.commands):
        curriculum = False
        heading_command = False

        class ranges(Solo12V31FormalStairFeasibilityCfg.commands.ranges):
            lin_vel_x = [0.2, 0.5]
            lin_vel_y = [-0.15, 0.15]
            ang_vel_yaw = [-0.3, 0.3]
            heading = [-0.5, 0.5]

    class rewards(Solo12V31FormalStairFeasibilityCfg.rewards):
        class scales(Solo12V31FormalStairFeasibilityCfg.rewards.scales):
            # The previous 5.5--7.5 cm band punished the clearance needed by
            # medium and high steps.  Use a minimum-only touchdown signal.
            swing_peak_height_band = 0.0
            swing_peak_clearance = -30.0
            tracking_lin_vel = 2.0
            commanded_stall = -4.0
            base_height = -250.0
            base_height_low = -10.0
            orientation = -0.5
            ang_vel_xy = -0.1
            collision = -0.3
            stumble = -0.8

        swing_peak_clearance_target = 0.07


class Solo12V31ObstacleCurriculumBConfig(Solo12V31ObstacleCurriculumAConfig):
    """Extend clearance learning through medium terrain levels."""

    class terrain(Solo12V31ObstacleCurriculumAConfig.terrain):
        max_init_terrain_level = 5

    class commands(Solo12V31ObstacleCurriculumAConfig.commands):
        class ranges(Solo12V31ObstacleCurriculumAConfig.commands.ranges):
            lin_vel_x = [0.15, 0.7]
            lin_vel_y = [-0.25, 0.25]
            ang_vel_yaw = [-0.6, 0.6]
            heading = [-1.0, 1.0]

    class rewards(Solo12V31ObstacleCurriculumAConfig.rewards):
        swing_peak_clearance_target = 0.09


class Solo12V31ObstacleCurriculumCConfig(Solo12V31ObstacleCurriculumBConfig):
    """Expose the policy to the complete terrain-height range."""

    class terrain(Solo12V31ObstacleCurriculumBConfig.terrain):
        # Starting every resumed rollout across levels 0--9 caused the
        # curriculum mean to collapse before the policy could adapt.  Levels
        # 8--9 remain reachable through the normal terrain curriculum; only
        # the reset-time distribution is softened here.
        max_init_terrain_level = 7
        curriculum_use_command_progress = True

    class commands(Solo12V31ObstacleCurriculumBConfig.commands):
        class ranges(Solo12V31ObstacleCurriculumBConfig.commands.ranges):
            lin_vel_x = [0.1, 0.7]
            lin_vel_y = [-0.4, 0.4]
            ang_vel_yaw = [-1.5, 1.5]
            heading = [-3.14, 3.14]

    class rewards(Solo12V31ObstacleCurriculumBConfig.rewards):
        class scales(Solo12V31ObstacleCurriculumBConfig.rewards.scales):
            base_height = -500.0
            base_height_low = -15.0
            orientation = -1.0

        swing_peak_clearance_target = 0.11


class Solo12V31ObstacleMixedStabilityConfig(Solo12V31ObstacleCurriculumCConfig):
    """Mixed replay: retain rough terrain while sampling 11.5--15 cm stairs."""

    class terrain(Solo12V31ObstacleCurriculumCConfig.terrain):
        mixed_stair_probability = 0.25
        mixed_stair_height_range = [0.115, 0.15]


class Solo12V31ObstacleMixedStabilityHighStairConfig(Solo12V31ObstacleMixedStabilityConfig):
    """Mixed replay with a higher stair share for the 15 cm target."""

    class terrain(Solo12V31ObstacleMixedStabilityConfig.terrain):
        mixed_stair_probability = 0.40


class Solo12V31ObstacleMixedStabilityReplayConfig(Solo12V31ObstacleMixedStabilityConfig):
    """Replay mix with a small pure-plane share to prevent flat-gait forgetting."""

    class terrain(Solo12V31ObstacleMixedStabilityConfig.terrain):
        mixed_plane_probability = 0.10


class Solo12V31ArmatureAdaptConfig(Solo12V31ObstacleMixedStabilityReplayConfig):
    """Adapt 8040 to deployment armature without changing legacy damping/friction ranges."""

    class domain_rand(Solo12V31ObstacleMixedStabilityReplayConfig.domain_rand):
        randomize_armature = True
        armature_ranges = {
            "hip": [0.0, 0.00504],
            "thigh": [0.0, 0.00504],
            "calf": [0.0, 0.02016],
        }


class Solo12V31ArmatureReplayConfig(Solo12V31ArmatureAdaptConfig):
    """Retain the legacy zero-armature endpoint during armature adaptation."""

    class domain_rand(Solo12V31ArmatureAdaptConfig.domain_rand):
        armature_zero_probability = 0.25


class Solo12V31ArmatureDeployNominalConfig(Solo12V31ArmatureAdaptConfig):
    """Validation endpoint for deployment armature with legacy damping/friction ranges."""

    class domain_rand(Solo12V31ArmatureAdaptConfig.domain_rand):
        armature_ranges = {
            "hip": [0.0042, 0.0042],
            "thigh": [0.0042, 0.0042],
            "calf": [0.0168, 0.0168],
        }


class Solo12V31ObstacleMixedStabilityFlatReplayConfig(Solo12V31ObstacleMixedStabilityReplayConfig):
    """Replay mix with extra plane samples for command robustness."""

    class terrain(Solo12V31ObstacleMixedStabilityReplayConfig.terrain):
        mixed_plane_probability = 0.20


class Solo12V31ObstacleHardReplayConfig(Solo12V31ObstacleMixedStabilityReplayConfig):
    """Hard-case replay used to improve 8040's high uneven/stair performance.

    This changes terrain sampling only: the policy architecture, observations,
    rewards, and dynamics randomization remain inherited from the validated
    mixed-replay chain.
    """

    class terrain(Solo12V31ObstacleMixedStabilityReplayConfig.terrain):
        max_init_terrain_level = 9
        mixed_plane_probability = 0.15
        mixed_stair_probability = 0.35
        mixed_stair_height_range = [0.115, 0.149]


class Solo12V31ObstacleFocusedReplayConfig(Solo12V31ObstacleHardReplayConfig):
    """Stable hard replay with the full requested command-speed envelope."""

    class terrain(Solo12V31ObstacleHardReplayConfig.terrain):
        # Keep the validated level-7 reset distribution; do not jump directly
        # to level 9, which caused the prior fine-tune to lose stability.
        max_init_terrain_level = 7

    class commands(Solo12V31ObstacleHardReplayConfig.commands):
        curriculum = False
        heading_command = False

        class ranges(Solo12V31ObstacleHardReplayConfig.commands.ranges):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-0.4, 0.4]
            ang_vel_yaw = [-1.5, 1.5]


class Solo12V31ObstacleTargetedReplayConfig(Solo12V31ObstacleFocusedReplayConfig):
    """Replay the exact acceptance terrain families without changing policy I/O."""

    class terrain(Solo12V31ObstacleFocusedReplayConfig.terrain):
        max_init_terrain_level = 9
        continuous_uneven = True
        mixed_plane_probability = 0.15
        mixed_stair_probability = 0.35
        mixed_stair_height_range = [0.115, 0.149]


class Solo12V31TargetedFinetuneCfgPPO(Solo12V31RoughCfgPPO):
    """Conservative PPO settings for retaining 8040 while adapting terrain."""

    class algorithm(Solo12V31RoughCfgPPO.algorithm):
        learning_rate = 0.0002
        entropy_coef = 0.003
        desired_kl = 0.005


class Solo12V31ObstacleBalancedReplayConfig(Solo12V31ObstacleFocusedReplayConfig):
    """Incremental replay: add continuous uneven terrain without forgetting rough terrain."""

    class terrain(Solo12V31ObstacleFocusedReplayConfig.terrain):
        max_init_terrain_level = 7
        continuous_uneven = False
        mixed_plane_probability = 0.10
        mixed_stair_probability = 0.25
        mixed_continuous_uneven_probability = 0.35
        mixed_stair_height_range = [0.115, 0.149]


class Solo12V31ObstacleFinalConfig(Solo12V31ObstacleCurriculumCConfig):
    """Restore the full command and R6 randomization envelopes."""

    class commands(Solo12V31ObstacleCurriculumCConfig.commands):
        heading_command = True

        class ranges(Solo12V31ObstacleCurriculumCConfig.commands.ranges):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-3.14, 3.14]
            heading = [-3.14, 3.14]

    class domain_rand(Solo12V31R6Cfg.domain_rand):
        # Observation latency is intentionally out of scope for this policy.
        observation_latency = False
        observation_latency_range = [0.0, 0.0]
        observation_latency_min_probability = 0.0
        observation_latency_max_probability = 0.0

    class rewards(Solo12V31ObstacleCurriculumCConfig.rewards):
        class scales(Solo12V31ObstacleCurriculumCConfig.rewards.scales):
            base_height = -800.0
            base_height_low = -20.0
            orientation = -2.5
            ang_vel_xy = -0.2


class Solo12V31ObstacleFinalDynamicsConfig(Solo12V31ObstacleFinalConfig):
    """Adapt to full commands and R6 dynamics before temporal/external disturbances."""

    class commands(Solo12V31ObstacleFinalConfig.commands):
        forward_only_fraction = 0.3
        forward_only_x_range = [0.2, 1.0]

    class domain_rand(Solo12V31ObstacleFinalConfig.domain_rand):
        observation_latency = False
        push_robots = False
        disturbance = False


class Solo12V31ObstacleFinalLatency10Config(Solo12V31ObstacleFinalDynamicsConfig):
    """Introduce observation latency through a 10 ms envelope."""

    class domain_rand(Solo12V31ObstacleFinalDynamicsConfig.domain_rand):
        observation_latency = True
        observation_latency_range = [0.0, 10.0]


class Solo12V31ObstacleFinalLatency15Config(Solo12V31ObstacleFinalLatency10Config):
    class domain_rand(Solo12V31ObstacleFinalLatency10Config.domain_rand):
        observation_latency_range = [0.0, 15.0]


class Solo12V31ObstacleFinalLatency20Config(Solo12V31ObstacleFinalLatency15Config):
    class domain_rand(Solo12V31ObstacleFinalLatency15Config.domain_rand):
        observation_latency_range = [0.0, 20.0]


class Solo12V31ObstacleFinalLatency25Config(Solo12V31ObstacleFinalLatency20Config):
    class domain_rand(Solo12V31ObstacleFinalLatency20Config.domain_rand):
        observation_latency_range = [0.0, 25.0]


class Solo12V31ObstacleFinalLatency30Config(Solo12V31ObstacleFinalLatency25Config):
    class domain_rand(Solo12V31ObstacleFinalLatency25Config.domain_rand):
        observation_latency_range = [0.0, 30.0]


class Solo12V31ObstacleFinalLatency35Config(Solo12V31ObstacleFinalLatency30Config):
    class domain_rand(Solo12V31ObstacleFinalLatency30Config.domain_rand):
        observation_latency_range = [0.0, 35.0]


class Solo12V31ObstacleFinalLatencyConfig(Solo12V31ObstacleFinalLatency35Config):
    """Complete R6 observation-latency envelope without external impulses."""

    class domain_rand(Solo12V31ObstacleFinalLatency35Config.domain_rand):
        observation_latency_range = [0.0, 40.0]


class Solo12V31ObstacleStairPolishConfig(Solo12V31ObstacleFinalDynamicsConfig):
    """Short specialization pass on repeated up/down stairs across full heights."""

    class terrain(Solo12V31ObstacleFinalDynamicsConfig.terrain):
        curriculum = False
        up_down_stairs = True
        up_down_stair_height_range = [0.081, 0.199]
        spawn_offset_x = 1.5
        terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0]

    class commands(Solo12V31ObstacleFinalDynamicsConfig.commands):
        heading_command = False
        forward_only_fraction = 0.0

        class ranges(Solo12V31ObstacleFinalDynamicsConfig.commands.ranges):
            lin_vel_x = [0.2, 0.7]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]


class Solo12V31ObstacleFinalExternalWarmupConfig(Solo12V31ObstacleFinalDynamicsConfig):
    """Introduce external disturbances gradually before the complete R6 envelope."""

    class domain_rand(Solo12V31ObstacleFinalDynamicsConfig.domain_rand):
        push_robots = True
        max_push_vel_xy = 0.3
        disturbance = True
        disturbance_range = [-10.0, 10.0]


class Solo12V31ObstacleFinalActionDelayConfig(Solo12V31ObstacleFinalConfig):
    """Final full-randomization task with Go1-style 0--15 ms action delay."""

    class domain_rand(Solo12V31ObstacleFinalConfig.domain_rand):
        delay = True
