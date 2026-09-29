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
        rated_motor_torque_nm = 6.0

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


class Solo12V31Payload25Config(Solo12V31ArmatureReplayConfig):
    """First payload curriculum stage: exact 0/2.5 kg replay plus interior samples."""

    class domain_rand(Solo12V31ArmatureReplayConfig.domain_rand):
        randomize_payload_mass = True
        payload_mass_range = [0.0, 2.5]
        payload_min_endpoint_probability = 0.15
        payload_max_endpoint_probability = 0.15

    class rewards(Solo12V31ArmatureReplayConfig.rewards):
        class scales(Solo12V31ArmatureReplayConfig.rewards.scales):
            # This is a preference, not a 6 Nm clamp. The physical 48 V
            # torque-speed envelope remains the only actuator torque clamp.
            rated_torque_excess = -0.002


class Solo12V31Payload50Config(Solo12V31Payload25Config):
    class domain_rand(Solo12V31Payload25Config.domain_rand):
        payload_mass_range = [0.0, 5.0]


class Solo12V31Payload75Config(Solo12V31Payload50Config):
    class domain_rand(Solo12V31Payload50Config.domain_rand):
        payload_mass_range = [0.0, 7.5]


class Solo12V31Payload100Config(Solo12V31Payload75Config):
    class domain_rand(Solo12V31Payload75Config.domain_rand):
        payload_mass_range = [0.0, 10.0]


class Solo12V31Payload100TorqueRecoveryConfig(Solo12V31Payload100Config):
    """Final real-envelope recovery with a stronger rated-torque preference."""

    class rewards(Solo12V31Payload100Config.rewards):
        class scales(Solo12V31Payload100Config.rewards.scales):
            rated_torque_excess = -0.02


class Solo12V31Payload100Envelope125Config(Solo12V31Payload100Config):
    """Diagnostic only: scale both envelope and URDF effort limits by 1.25."""

    class control(Solo12V31Payload100Config.control):
        torque_speed_envelope_scale = 1.25
        rated_motor_torque_nm = 7.5

    class asset(Solo12V31Payload100Config.asset):
        dof_torque_limit_scale = 1.25


class Solo12V31Payload100Envelope150Config(Solo12V31Payload100Config):
    """Diagnostic only: scale both envelope and URDF effort limits by 1.5."""

    class control(Solo12V31Payload100Config.control):
        torque_speed_envelope_scale = 1.5
        rated_motor_torque_nm = 9.0

    class asset(Solo12V31Payload100Config.asset):
        dof_torque_limit_scale = 1.5


class Solo12V31Payload100Envelope200Config(Solo12V31Payload100Config):
    """Diagnostic only: scale both envelope and URDF effort limits by 2.0."""

    class control(Solo12V31Payload100Config.control):
        torque_speed_envelope_scale = 2.0
        rated_motor_torque_nm = 12.0

    class asset(Solo12V31Payload100Config.asset):
        dof_torque_limit_scale = 2.0


class Solo12V31RS06CalfAdaptConfig(Solo12V31ArmatureReplayConfig):
    """Adapt checkpoint 8520 to RS06 calf actuators and a 3:1 external stage."""

    class control(Solo12V31ArmatureReplayConfig.control):
        # RS02 stays on hip/thigh. RS06 is used only by calf joints. The RS06
        # points are digitized from the official 2026-07-13 48 V T-N plot; the
        # documented 480 rpm no-load point closes the curve at zero torque.
        torque_speed_curves = {
            "hip": {
                "rpm": [
                    0.0, 60.0, 120.0, 180.0, 210.0, 225.0, 240.0, 255.0,
                    270.0, 285.0, 300.0, 315.0, 330.0, 345.0, 360.0, 375.0,
                    390.0, 405.0, 420.0,
                ],
                "nm": [
                    17.0, 16.8, 16.7, 16.9, 16.8, 16.6, 16.0, 15.4,
                    14.7, 13.4, 12.0, 10.8, 9.5, 8.2, 6.1, 4.7,
                    3.4, 1.9, 0.0,
                ],
            },
            "thigh": {
                "rpm": [
                    0.0, 60.0, 120.0, 180.0, 210.0, 225.0, 240.0, 255.0,
                    270.0, 285.0, 300.0, 315.0, 330.0, 345.0, 360.0, 375.0,
                    390.0, 405.0, 420.0,
                ],
                "nm": [
                    17.0, 16.8, 16.7, 16.9, 16.8, 16.6, 16.0, 15.4,
                    14.7, 13.4, 12.0, 10.8, 9.5, 8.2, 6.1, 4.7,
                    3.4, 1.9, 0.0,
                ],
            },
            "calf": {
                "rpm": [
                    0.0, 100.0, 250.0, 280.0, 300.0, 320.0, 340.0,
                    360.0, 380.0, 400.0, 410.0, 420.0, 425.0, 430.0, 480.0,
                ],
                "nm": [
                    36.0, 36.0, 36.0, 36.0, 33.0, 30.0, 27.0,
                    24.5, 21.5, 18.0, 16.0, 13.5, 11.5, 5.0, 0.0,
                ],
            },
        }
        joint_gear_ratios = {"hip": 1.0, "thigh": 1.0, "calf": 3.0}
        rated_motor_torque_nm = 6.0
        rated_motor_torques_nm = {"hip": 6.0, "thigh": 6.0, "calf": 11.0}

    class asset(Solo12V31ArmatureReplayConfig.asset):
        dof_velocity_limits = {"calf": 16.755160819145562}
        dof_torque_limits = {"calf": 108.0}
        # The RS06 housing is mounted at the knee on the thigh-side link.
        body_mass_additions = {"thigh": 0.241}

    class domain_rand(Solo12V31ArmatureReplayConfig.domain_rand):
        randomize_payload_mass = False
        payload_mass_range = [0.0, 0.0]
        # RS06 output inertia is not published. This range scales the known
        # RS02 estimate by actuator mass/radius and the new external ratio.
        armature_ranges = {
            "hip": [0.0, 0.00504],
            "thigh": [0.0, 0.00504],
            "calf": [0.045, 0.095],
        }
        armature_zero_probability = 0.0

    class rewards(Solo12V31ArmatureReplayConfig.rewards):
        class scales(Solo12V31ArmatureReplayConfig.rewards.scales):
            rated_torque_excess = -0.002


class Solo12V31RS06CalfWarmupConfig(Solo12V31RS06CalfAdaptConfig):
    """Conservative first handoff: 3:1 and mass are final, capability is staged."""

    class control(Solo12V31RS06CalfAdaptConfig.control):
        torque_speed_envelope_scales = {"calf": 0.35}

    class domain_rand(Solo12V31RS06CalfAdaptConfig.domain_rand):
        armature_ranges = {
            "hip": [0.0, 0.00504],
            "thigh": [0.0, 0.00504],
            "calf": [0.0168, 0.035],
        }


class Solo12V31RS06CalfIntermediateConfig(Solo12V31RS06CalfAdaptConfig):
    """Intermediate handoff before exposing the complete RS06 capability."""

    class control(Solo12V31RS06CalfAdaptConfig.control):
        torque_speed_envelope_scales = {"calf": 0.65}

    class domain_rand(Solo12V31RS06CalfAdaptConfig.domain_rand):
        armature_ranges = {
            "hip": [0.0, 0.00504],
            "thigh": [0.0, 0.00504],
            "calf": [0.03, 0.065],
        }


class Solo12V31RS06Payload25Config(Solo12V31RS06CalfAdaptConfig):
    class domain_rand(Solo12V31RS06CalfAdaptConfig.domain_rand):
        randomize_payload_mass = True
        payload_mass_range = [0.0, 2.5]
        payload_min_endpoint_probability = 0.15
        payload_max_endpoint_probability = 0.15


class Solo12V31RS06Payload50Config(Solo12V31RS06Payload25Config):
    class domain_rand(Solo12V31RS06Payload25Config.domain_rand):
        payload_mass_range = [0.0, 5.0]


class Solo12V31RS06Payload75Config(Solo12V31RS06Payload50Config):
    class domain_rand(Solo12V31RS06Payload50Config.domain_rand):
        payload_mass_range = [0.0, 7.5]


class Solo12V31RS06Payload100Config(Solo12V31RS06Payload75Config):
    class domain_rand(Solo12V31RS06Payload75Config.domain_rand):
        payload_mass_range = [0.0, 10.0]
        payload_min_endpoint_probability = 0.15
        payload_max_endpoint_probability = 0.25


class Solo12V31RS06Payload100TorqueRecoveryConfig(Solo12V31RS06Payload100Config):
    class rewards(Solo12V31RS06Payload100Config.rewards):
        class scales(Solo12V31RS06Payload100Config.rewards.scales):
            rated_torque_excess = -0.02


class Solo12V31RS06Payload100TorqueRecoveryStrongConfig(
    Solo12V31RS06Payload100TorqueRecoveryConfig
):
    """Stronger recovery stage used only if the 95% rated-torque gate fails."""

    class rewards(Solo12V31RS06Payload100TorqueRecoveryConfig.rewards):
        class scales(Solo12V31RS06Payload100TorqueRecoveryConfig.rewards.scales):
            rated_torque_excess = -0.08


class Solo12V31RS06Thigh2Payload100AdaptConfig(
    Solo12V31RS06Payload100TorqueRecoveryStrongConfig
):
    """Adapt checkpoint 10320 to RS02 hip 1:1, RS02 thigh 2:1, RS06 calf 3:1."""

    class control(Solo12V31RS06Payload100TorqueRecoveryStrongConfig.control):
        joint_gear_ratios = {"hip": 1.0, "thigh": 2.0, "calf": 3.0}

    class asset(Solo12V31RS06Payload100TorqueRecoveryStrongConfig.asset):
        # RS02/RS06 documented motor-speed endpoints divided by each external ratio.
        dof_velocity_limits = {
            "thigh": 21.991148575128552,
            "calf": 16.755160819145562,
        }
        # Joint-side peak torque from the unscaled physical T-N envelopes.
        dof_torque_limits = {"thigh": 34.0, "calf": 108.0}

    class domain_rand(Solo12V31RS06Payload100TorqueRecoveryStrongConfig.domain_rand):
        # RS02 reflected inertia scales with ratio squared: 0.00504 * 2^2.
        armature_ranges = {
            "hip": [0.0, 0.00504],
            "thigh": [0.0, 0.02016],
            "calf": [0.045, 0.095],
        }


class Solo12V31RS06Thigh2Payload100HardReplayConfig(
    Solo12V31RS06Thigh2Payload100AdaptConfig
):
    """Terrain replay after the thigh 2:1 drivetrain handoff is stable."""

    class terrain(Solo12V31RS06Thigh2Payload100AdaptConfig.terrain):
        max_init_terrain_level = 9
        mixed_plane_probability = 0.10
        mixed_stair_probability = 0.20
        mixed_slope_probability = 0.35
        mixed_slope_angle_range_deg = [35.0, 45.0]
        mixed_continuous_uneven_probability = 0.45
        mixed_stair_height_range = [0.115, 0.15]
        terrain_length = 12.0
        slope_ramp_start_distance = 7.1
        slope_treshold = 1.05

    class commands(Solo12V31RS06Thigh2Payload100AdaptConfig.commands):
        class ranges(Solo12V31RS06Thigh2Payload100AdaptConfig.commands.ranges):
            lin_vel_x = [0.1, 1.0]


class Solo12V31RS06Thigh2Payload100SlopeFocusConfig(
    Solo12V31RS06Thigh2Payload100HardReplayConfig
):
    """Use a 20--45 degree curriculum around the 0.7 m/s acceptance point."""

    class terrain(Solo12V31RS06Thigh2Payload100HardReplayConfig.terrain):
        # Start near 20--28 degrees, then promote successful environments one
        # row at a time until curriculum level 9 reaches exactly 45 degrees.
        curriculum = True
        max_init_terrain_level = 3
        mixed_plane_probability = 0.0
        mixed_stair_probability = 0.0
        mixed_slope_probability = 1.0
        mixed_slope_angle_range_deg = [20.0, 45.0]
        mixed_continuous_uneven_probability = 0.0
        terrain_length = 20.0
        slope_ramp_start_distance = 3.5
        spawn_offset_x = 2.0

    class commands(Solo12V31RS06Thigh2Payload100HardReplayConfig.commands):
        heading_command = False
        forward_only_fraction = 0.0

        class ranges(Solo12V31RS06Thigh2Payload100HardReplayConfig.commands.ranges):
            lin_vel_x = [0.55, 0.80]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]

    class domain_rand(Solo12V31RS06Thigh2Payload100HardReplayConfig.domain_rand):
        # Bias the specialization pass toward the exact worst-case payload.
        payload_min_endpoint_probability = 0.10
        payload_max_endpoint_probability = 0.50


class Solo12V31RS06Thigh2Payload100SlopeHighConfig(
    Solo12V31RS06Thigh2Payload100SlopeFocusConfig
):
    """High-angle polish after the broad 20--45 degree curriculum succeeds."""

    class terrain(Solo12V31RS06Thigh2Payload100SlopeFocusConfig.terrain):
        max_init_terrain_level = 5
        mixed_slope_angle_range_deg = [35.0, 45.0]

    class commands(Solo12V31RS06Thigh2Payload100SlopeFocusConfig.commands):
        class ranges(Solo12V31RS06Thigh2Payload100SlopeFocusConfig.commands.ranges):
            lin_vel_x = [0.65, 0.75]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]


class Solo12V31RS06Thigh2Payload100FinalReplayConfig(
    Solo12V31RS06Thigh2Payload100SlopeHighConfig
):
    """Recover stairs/uneven terrain while retaining high-angle ramp skill."""

    class terrain(Solo12V31RS06Thigh2Payload100SlopeHighConfig.terrain):
        max_init_terrain_level = 9
        terrain_length = 20.0
        spawn_offset_x = 2.0
        slope_ramp_start_distance = 3.5
        # Actual sequential probabilities: plane 10%, stairs 35%, slope 25%,
        # and continuous uneven 30%.
        mixed_plane_probability = 0.10
        mixed_stair_probability = 0.3888888888888889
        mixed_slope_probability = 0.45454545454545453
        mixed_continuous_uneven_probability = 1.0
        mixed_stair_height_range = [0.12, 0.15]
        mixed_slope_angle_range_deg = [35.0, 45.0]

    class commands(Solo12V31RS06Thigh2Payload100SlopeHighConfig.commands):
        class ranges(Solo12V31RS06Thigh2Payload100SlopeHighConfig.commands.ranges):
            lin_vel_x = [0.65, 1.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]


class Solo12V31RS06Thigh2Payload100UnevenTorqueRecoveryConfig(
    Solo12V31RS06Thigh2Payload100FinalReplayConfig
):
    """Recover rated-torque margin on level-9 uneven terrain without forgetting."""

    class terrain(Solo12V31RS06Thigh2Payload100FinalReplayConfig.terrain):
        # Actual sequential probabilities: plane 5%, stairs 15%, slope 15%,
        # and continuous uneven 65%.
        mixed_plane_probability = 0.05
        mixed_stair_probability = 0.15789473684210525
        mixed_slope_probability = 0.1875
        mixed_continuous_uneven_probability = 1.0

    class commands(Solo12V31RS06Thigh2Payload100FinalReplayConfig.commands):
        class ranges(Solo12V31RS06Thigh2Payload100FinalReplayConfig.commands.ranges):
            lin_vel_x = [0.85, 1.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]

    class rewards(Solo12V31RS06Thigh2Payload100FinalReplayConfig.rewards):
        class scales(Solo12V31RS06Thigh2Payload100FinalReplayConfig.rewards.scales):
            rated_torque_excess = -0.20
            rated_torque_incidence = -0.50


class Solo12V31RS06Thigh2StraightObstacleAConfig(
    Solo12V31RS06Thigh2Payload100AdaptConfig
):
    """Relearn facing-direction stairs/ramps without sideways escape."""

    class terrain(Solo12V31RS06Thigh2Payload100AdaptConfig.terrain):
        curriculum = True
        curriculum_use_command_progress = False
        curriculum_use_world_x_progress = True
        max_init_terrain_level = 2
        terrain_length = 14.0
        terrain_width = 8.0
        spawn_offset_x = 2.0
        slope_ramp_start_distance = 3.5
        mixed_plane_probability = 0.05
        mixed_stair_probability = 0.4736842105263158
        mixed_slope_probability = 0.90
        mixed_continuous_uneven_probability = 1.0
        mixed_stair_height_range = [0.06, 0.15]
        mixed_slope_angle_range_deg = [15.0, 45.0]
        slope_treshold = 1.05

    class commands(Solo12V31RS06Thigh2Payload100AdaptConfig.commands):
        heading_command = False
        forward_only_fraction = 0.0

        class ranges(Solo12V31RS06Thigh2Payload100AdaptConfig.commands.ranges):
            lin_vel_x = [0.55, 0.80]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class domain_rand(Solo12V31RS06Thigh2Payload100AdaptConfig.domain_rand):
        payload_min_endpoint_probability = 0.10
        payload_max_endpoint_probability = 0.50

    class rewards(Solo12V31RS06Thigh2Payload100AdaptConfig.rewards):
        class scales(Solo12V31RS06Thigh2Payload100AdaptConfig.rewards.scales):
            world_forward_tracking = 3.0
            world_heading_alignment = 2.0
            world_lateral_velocity = -1.0


class Solo12V31RS06Thigh2StraightObstacleBConfig(
    Solo12V31RS06Thigh2StraightObstacleAConfig
):
    """Bridge the measured 25--30 degree capability toward 35 degrees."""

    class terrain(Solo12V31RS06Thigh2StraightObstacleAConfig.terrain):
        max_init_terrain_level = 5
        mixed_plane_probability = 0.0
        mixed_stair_probability = 0.0
        mixed_slope_probability = 1.0
        mixed_continuous_uneven_probability = 0.0
        mixed_slope_angle_range_deg = [20.0, 35.0]

    class rewards(Solo12V31RS06Thigh2StraightObstacleAConfig.rewards):
        class scales(Solo12V31RS06Thigh2StraightObstacleAConfig.rewards.scales):
            # A flat-body penalty conflicts with the pitch needed to follow a
            # steep ramp. Height-relative posture terms remain active.
            orientation = 0.0


class Solo12V31RS06Thigh2StraightObstacleFinalConfig(
    Solo12V31RS06Thigh2StraightObstacleBConfig
):
    """Continue the pure-ramp curriculum through roughly 42 degrees."""

    class terrain(Solo12V31RS06Thigh2StraightObstacleBConfig.terrain):
        max_init_terrain_level = 5
        mixed_slope_angle_range_deg = [28.0, 42.0]

    class commands(Solo12V31RS06Thigh2StraightObstacleBConfig.commands):
        class ranges(Solo12V31RS06Thigh2StraightObstacleBConfig.commands.ranges):
            lin_vel_x = [0.65, 0.75]

    class rewards(Solo12V31RS06Thigh2StraightObstacleBConfig.rewards):
        class scales(Solo12V31RS06Thigh2StraightObstacleBConfig.rewards.scales):
            rated_torque_excess = -0.20
            rated_torque_incidence = -0.50


class Solo12V31RS06Thigh2StraightSlope45Config(
    Solo12V31RS06Thigh2StraightObstacleFinalConfig
):
    """Final pure-ramp polish over 35--45 degrees."""

    class terrain(Solo12V31RS06Thigh2StraightObstacleFinalConfig.terrain):
        max_init_terrain_level = 5
        mixed_slope_angle_range_deg = [35.0, 45.0]

    class rewards(Solo12V31RS06Thigh2StraightObstacleFinalConfig.rewards):
        class scales(Solo12V31RS06Thigh2StraightObstacleFinalConfig.rewards.scales):
            rated_torque_excess = -0.30
            rated_torque_incidence = -0.80


class Solo12V31RS06Thigh2StraightSlope45CapabilityConfig(
    Solo12V31RS06Thigh2StraightSlope45Config
):
    """Peak-envelope capability probe without the rated-duty objective."""

    class rewards(Solo12V31RS06Thigh2StraightSlope45Config.rewards):
        class scales(Solo12V31RS06Thigh2StraightSlope45Config.rewards.scales):
            rated_torque_excess = 0.0
            rated_torque_incidence = 0.0


class Solo12V31RS06Thigh3StraightObstacleAdaptConfig(
    Solo12V31RS06Thigh2StraightObstacleAConfig
):
    """Adapt the mixed-obstacle policy to RS02 thigh 3:1 and RS06 calf 3:1."""

    class control(Solo12V31RS06Thigh2StraightObstacleAConfig.control):
        joint_gear_ratios = {"hip": 1.0, "thigh": 3.0, "calf": 3.0}

    class asset(Solo12V31RS06Thigh2StraightObstacleAConfig.asset):
        # RS02 reaches its modeled 420 rpm endpoint at 14.6608 rad/s joint-side.
        dof_velocity_limits = {
            "thigh": 14.660765716752367,
            "calf": 16.755160819145562,
        }
        dof_torque_limits = {"thigh": 51.0, "calf": 108.0}

    class domain_rand(Solo12V31RS06Thigh2StraightObstacleAConfig.domain_rand):
        # RS02 reflected inertia scales with the square of the external ratio.
        armature_ranges = {
            "hip": [0.0, 0.00504],
            "thigh": [0.0, 0.04536],
            "calf": [0.045, 0.095],
        }


class Solo12V31RS06Thigh3StraightObstacleReplayConfig(
    Solo12V31RS06Thigh3StraightObstacleAdaptConfig
):
    """Recover 1 m/s plane/uneven and 15 cm stair performance after handoff."""

    class terrain(Solo12V31RS06Thigh3StraightObstacleAdaptConfig.terrain):
        max_init_terrain_level = 9
        mixed_plane_probability = 0.15
        mixed_stair_probability = 0.4117647058823529
        mixed_slope_probability = 0.0
        mixed_continuous_uneven_probability = 1.0
        mixed_stair_height_range = [0.10, 0.15]

    class commands(Solo12V31RS06Thigh3StraightObstacleAdaptConfig.commands):
        class ranges(Solo12V31RS06Thigh3StraightObstacleAdaptConfig.commands.ranges):
            lin_vel_x = [0.55, 1.0]

    class rewards(Solo12V31RS06Thigh3StraightObstacleAdaptConfig.rewards):
        class scales(Solo12V31RS06Thigh3StraightObstacleAdaptConfig.rewards.scales):
            rated_torque_excess = -0.20
            rated_torque_incidence = -0.50


class Solo12V31RS06Thigh3Speed15AdaptConfig(
    Solo12V31RS06Thigh3StraightObstacleReplayConfig
):
    """First high-speed handoff on plane and continuous uneven terrain."""

    class terrain(Solo12V31RS06Thigh3StraightObstacleReplayConfig.terrain):
        max_init_terrain_level = 9
        mixed_plane_probability = 0.30
        mixed_stair_probability = 0.0
        mixed_slope_probability = 0.0
        mixed_continuous_uneven_probability = 1.0

    class commands(Solo12V31RS06Thigh3StraightObstacleReplayConfig.commands):
        class ranges(Solo12V31RS06Thigh3StraightObstacleReplayConfig.commands.ranges):
            lin_vel_x = [0.90, 1.50]

    class domain_rand(Solo12V31RS06Thigh3StraightObstacleReplayConfig.domain_rand):
        payload_min_endpoint_probability = 0.10
        payload_max_endpoint_probability = 0.50


class Solo12V31RS06Thigh3Speed20AdaptConfig(
    Solo12V31RS06Thigh3Speed15AdaptConfig
):
    """Extend the +10 kg-capable plane/uneven policy to 2.0 m/s."""

    class terrain(Solo12V31RS06Thigh3Speed15AdaptConfig.terrain):
        mixed_plane_probability = 0.35

    class commands(Solo12V31RS06Thigh3Speed15AdaptConfig.commands):
        class ranges(Solo12V31RS06Thigh3Speed15AdaptConfig.commands.ranges):
            lin_vel_x = [1.30, 2.00]


class Solo12V31RS06Thigh3StraightSlopeAdaptConfig(
    Solo12V31RS06Thigh2StraightSlope45Config
):
    """Adapt the dedicated ramp policy to RS02 thigh 3:1 at 0.25--0.55 m/s."""

    class control(Solo12V31RS06Thigh2StraightSlope45Config.control):
        joint_gear_ratios = {"hip": 1.0, "thigh": 3.0, "calf": 3.0}

    class asset(Solo12V31RS06Thigh2StraightSlope45Config.asset):
        dof_velocity_limits = {
            "thigh": 14.660765716752367,
            "calf": 16.755160819145562,
        }
        dof_torque_limits = {"thigh": 51.0, "calf": 108.0}

    class domain_rand(Solo12V31RS06Thigh2StraightSlope45Config.domain_rand):
        friction_range = [1.0, 1.4]
        armature_ranges = {
            "hip": [0.0, 0.00504],
            "thigh": [0.0, 0.04536],
            "calf": [0.045, 0.095],
        }

    class commands(Solo12V31RS06Thigh2StraightSlope45Config.commands):
        class ranges(Solo12V31RS06Thigh2StraightSlope45Config.commands.ranges):
            lin_vel_x = [0.25, 0.55]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]

    class rewards(Solo12V31RS06Thigh2StraightSlope45Config.rewards):
        class scales(Solo12V31RS06Thigh2StraightSlope45Config.rewards.scales):
            rated_torque_excess = -0.30
            rated_torque_incidence = -0.80


RS03_THIGH_TORQUE_SPEED_CURVE = {
    # Peak-envelope samples digitized from the official 2026-07-13 RS03 48 V
    # T-N plot.  The documented 195 rpm no-load point closes at zero torque.
    "rpm": [
        0.0, 60.0, 100.0, 120.0, 135.0, 150.0,
        165.0, 180.0, 185.0, 188.0, 190.0, 195.0,
    ],
    "nm": [
        60.0, 60.0, 60.0, 60.0, 54.5, 47.0,
        39.0, 28.0, 23.0, 18.0, 10.0, 0.0,
    ],
}


class Solo12V31RS03ThighPayload100ObstacleAdaptConfig(
    Solo12V31RS06Thigh3StraightObstacleReplayConfig
):
    """Common RS03 +10 kg adaptation base for later capability branches."""

    class control(Solo12V31RS06Thigh3StraightObstacleReplayConfig.control):
        joint_gear_ratios = {"hip": 1.0, "thigh": 1.0, "calf": 3.0}
        torque_speed_curves = {
            **Solo12V31RS06Thigh3StraightObstacleReplayConfig.control.torque_speed_curves,
            "thigh": RS03_THIGH_TORQUE_SPEED_CURVE,
        }
        rated_motor_torques_nm = {"hip": 6.0, "thigh": 20.0, "calf": 11.0}
        # Rated rows in the official 2026-07-13 manual stall-overload tables.
        stall_rated_motor_torques_nm = {"hip": 6.0, "thigh": 13.0, "calf": 8.0}

    class asset(Solo12V31RS06Thigh3StraightObstacleReplayConfig.asset):
        dof_velocity_limits = {
            "thigh": 20.420352248333657,
            "calf": 16.755160819145562,
        }
        dof_torque_limits = {"thigh": 60.0, "calf": 108.0}
        body_mass_additions = {}
        base_mass_addition_kg = 3.044

    class domain_rand(Solo12V31RS06Thigh3StraightObstacleReplayConfig.domain_rand):
        armature_ranges = {
            "hip": [0.0, 0.00504],
            "thigh": [0.0, 0.00504],
            "calf": [0.045, 0.095],
        }

    class commands(Solo12V31RS06Thigh3StraightObstacleReplayConfig.commands):
        class ranges(Solo12V31RS06Thigh3StraightObstacleReplayConfig.commands.ranges):
            lin_vel_x = [0.0, 1.0]


class Solo12V31RS03ThighPayload100Speed20AdaptConfig(
    Solo12V31RS06Thigh3Speed20AdaptConfig
):
    """Specialize the common RS03 base for 1--2 m/s plane and uneven motion."""

    class control(Solo12V31RS06Thigh3Speed20AdaptConfig.control):
        joint_gear_ratios = {"hip": 1.0, "thigh": 1.0, "calf": 3.0}
        torque_speed_curves = {
            **Solo12V31RS06Thigh3Speed20AdaptConfig.control.torque_speed_curves,
            "thigh": RS03_THIGH_TORQUE_SPEED_CURVE,
        }
        rated_motor_torques_nm = {"hip": 6.0, "thigh": 20.0, "calf": 11.0}
        stall_rated_motor_torques_nm = {"hip": 6.0, "thigh": 13.0, "calf": 8.0}

    class asset(Solo12V31RS06Thigh3Speed20AdaptConfig.asset):
        dof_velocity_limits = {
            "thigh": 20.420352248333657,
            "calf": 16.755160819145562,
        }
        dof_torque_limits = {"thigh": 60.0, "calf": 108.0}
        body_mass_additions = {}
        base_mass_addition_kg = 3.044

    class domain_rand(Solo12V31RS06Thigh3Speed20AdaptConfig.domain_rand):
        armature_ranges = {
            "hip": [0.0, 0.00504],
            "thigh": [0.0, 0.00504],
            "calf": [0.045, 0.095],
        }

    class commands(Solo12V31RS06Thigh3Speed20AdaptConfig.commands):
        class ranges(Solo12V31RS06Thigh3Speed20AdaptConfig.commands.ranges):
            lin_vel_x = [1.0, 2.0]


class Solo12V31RS03ThighPayload100SpeedLimitPolishConfig(
    Solo12V31RS03ThighPayload100Speed20AdaptConfig
):
    """Reduce RS03 thigh overspeed impacts on high-speed uneven terrain."""

    class rewards(Solo12V31RS03ThighPayload100Speed20AdaptConfig.rewards):
        soft_dof_vel_limit = 0.85

        class scales(Solo12V31RS03ThighPayload100Speed20AdaptConfig.rewards.scales):
            dof_vel_limits = -1.0


class Solo12V31RS03ThighPayload100SlopeAdaptConfig(
    Solo12V31RS06Thigh3StraightSlopeAdaptConfig
):
    """Specialize the common RS03 base for 35--45 degree ramp motion."""

    class control(Solo12V31RS06Thigh3StraightSlopeAdaptConfig.control):
        joint_gear_ratios = {"hip": 1.0, "thigh": 1.0, "calf": 3.0}
        torque_speed_curves = {
            **Solo12V31RS06Thigh3StraightSlopeAdaptConfig.control.torque_speed_curves,
            "thigh": RS03_THIGH_TORQUE_SPEED_CURVE,
        }
        rated_motor_torques_nm = {"hip": 6.0, "thigh": 20.0, "calf": 11.0}
        stall_rated_motor_torques_nm = {"hip": 6.0, "thigh": 13.0, "calf": 8.0}

    class asset(Solo12V31RS06Thigh3StraightSlopeAdaptConfig.asset):
        dof_velocity_limits = {
            "thigh": 20.420352248333657,
            "calf": 16.755160819145562,
        }
        dof_torque_limits = {"thigh": 60.0, "calf": 108.0}
        body_mass_additions = {}
        base_mass_addition_kg = 3.044

    class domain_rand(Solo12V31RS06Thigh3StraightSlopeAdaptConfig.domain_rand):
        armature_ranges = {
            "hip": [0.0, 0.00504],
            "thigh": [0.0, 0.00504],
            "calf": [0.045, 0.095],
        }

    class commands(Solo12V31RS06Thigh3StraightSlopeAdaptConfig.commands):
        class ranges(Solo12V31RS06Thigh3StraightSlopeAdaptConfig.commands.ranges):
            # Keep the training command moving so world-X curriculum progress
            # measures ramp traversal. Static hold is tested by switching the
            # command to zero only after verified obstacle entry.
            lin_vel_x = [0.45, 0.7]


class Solo12V31RS03ThighPayload100SlopeWarmupConfig(
    Solo12V31RS03ThighPayload100SlopeAdaptConfig
):
    """First slope-branch stage after the common base: 20--35 degree ramps."""

    class terrain(Solo12V31RS03ThighPayload100SlopeAdaptConfig.terrain):
        max_init_terrain_level = 3
        mixed_slope_angle_range_deg = [20.0, 35.0]

    class commands(Solo12V31RS03ThighPayload100SlopeAdaptConfig.commands):
        class ranges(Solo12V31RS03ThighPayload100SlopeAdaptConfig.commands.ranges):
            lin_vel_x = [0.25, 0.55]


class Solo12V31RS03ThighPayload100SlopeBridgeConfig(
    Solo12V31RS03ThighPayload100SlopeAdaptConfig
):
    """Second slope-branch stage: bridge 28--42 degrees before final polish."""

    class terrain(Solo12V31RS03ThighPayload100SlopeAdaptConfig.terrain):
        max_init_terrain_level = 4
        mixed_slope_angle_range_deg = [28.0, 42.0]

    class commands(Solo12V31RS03ThighPayload100SlopeAdaptConfig.commands):
        class ranges(Solo12V31RS03ThighPayload100SlopeAdaptConfig.commands.ranges):
            lin_vel_x = [0.35, 0.65]


class Solo12V31RS06Payload100HardReplayConfig(
    Solo12V31RS06Payload100TorqueRecoveryStrongConfig
):
    """Replay hard continuous uneven terrain while retaining stairs and plane."""

    class terrain(Solo12V31RS06Payload100TorqueRecoveryStrongConfig.terrain):
        max_init_terrain_level = 9
        mixed_plane_probability = 0.10
        mixed_stair_probability = 0.25
        mixed_continuous_uneven_probability = 0.50
        mixed_stair_height_range = [0.115, 0.15]

    class commands(Solo12V31RS06Payload100TorqueRecoveryStrongConfig.commands):
        class ranges(Solo12V31RS06Payload100TorqueRecoveryStrongConfig.commands.ranges):
            lin_vel_x = [0.1, 1.0]


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
