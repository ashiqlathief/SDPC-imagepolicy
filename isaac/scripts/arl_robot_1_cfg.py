from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg

from diffuser.utils.path import project_path

ARL_ROBOT_1_USD_PATH = project_path("robot_model", "arl_robot_1", "arl_robot_1.usd")

ARL_ROBOT_1_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=ARL_ROBOT_1_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={},
        joint_vel={},
    ),
    actuators={},  # no free joints -- all 4 rotor mounts are PhysicsFixedJoints
)
"""Configuration for the NTNU-ARL 'lmf2' quadcopter (arl_robot_1.usd)."""