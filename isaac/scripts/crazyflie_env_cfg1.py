from __future__ import annotations
import importlib
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
from isaaclab.sim.spawners.materials import MdlFileCfg

cfg = importlib.import_module("config.avoiding-crazyflie")

_CYLINDERS_XY = cfg.CYLINDERS
_SPHERES_XYZ = getattr(cfg, 'SPHERES', [])
SPHERE_RADIUS = getattr(cfg, 'SPHERE_RADIUS', 0.10)
USE_DEPTH = cfg.USE_DEPTH
DEPTH_NEAR = cfg.DEPTH_NEAR
DEPTH_FAR = cfg.DEPTH_FAR
CORRIDOR_LENGTH = 11     # x direction
CORRIDOR_WIDTH = 5.0       # y direction (clearance between walls)
WALL_THICKNESS = 0.10
WALL_HEIGHT = 2.0
CEILING_HEIGHT  = WALL_HEIGHT          # = 1.0  (flush with obstacle tops)
CEILING_Z_CENTER = CEILING_HEIGHT + WALL_THICKNESS / 2.0   # centre of roof slab
CORRIDOR_X_OFFSET = -CORRIDOR_LENGTH / 2.0
FPV_REAL_WIDTH, FPV_REAL_HEIGHT = 96, 96
FPV_REAL_K = [212.49798583984375, 0.0, 215.16233825683594,
              0.0, 212.30624389648438, 121.23411560058594,
              0.0, 0.0, 1.0]

CYLINDERS = [(x , y, 1.0) for (x, y) in _CYLINDERS_XY]
RED_MAT   = PreviewSurfaceCfg(diffuse_color=(0.85, 0.10, 0.10))
BLUE_MAT  = PreviewSurfaceCfg(diffuse_color=(0.10, 0.30, 0.90))
GREEN_MAT   = PreviewSurfaceCfg(diffuse_color=(0.10, 0.75, 0.20))
ORANGE_MAT  = PreviewSurfaceCfg(diffuse_color=(0.90, 0.50, 0.10))
YELLOW_MAT  = PreviewSurfaceCfg(diffuse_color=(0.90, 0.85, 0.10))
PURPLE_MAT  = PreviewSurfaceCfg(diffuse_color=(0.55, 0.15, 0.75))
CYAN_MAT    = PreviewSurfaceCfg(diffuse_color=(0.10, 0.75, 0.80))
GRAY_MAT    = PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.45))
WHITE_MAT   = PreviewSurfaceCfg(diffuse_color=(0.90, 0.90, 0.90))
BLACK_MAT   = PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.05))
_WH_MAT_DIR = "Environments/Simple_Warehouse/Materials"
WALL_MAT    = MdlFileCfg(mdl_path=f"{ISAAC_NUCLEUS_DIR}/{_WH_MAT_DIR}/MI_WallA_01.mdl", project_uvw=True)
GROUND_MAT  = MdlFileCfg(mdl_path=f"{ISAAC_NUCLEUS_DIR}/{_WH_MAT_DIR}/MI_Floor_01.mdl", project_uvw=True)
CEILING_MAT = MdlFileCfg(mdl_path=f"{ISAAC_NUCLEUS_DIR}/{_WH_MAT_DIR}/MI_CeilingA_06b.mdl", project_uvw=True)
WAREHOUSE_ENV_USD = "Environments/Simple_Warehouse/warehouse.usd"

CRAZYFLIE = ArticulationCfg(
        spawn=sim_utils.MultiUsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Bitcraze/Crazyflie/cf2x.usd",
            # usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Bitcraze/Crazyflie/cf2x.usd",

            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
            copy_from_source=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.15),
            joint_pos={
                ".*": 0.0,
            },
            joint_vel={
                "m1_joint": 200.0,
                "m2_joint": -200.0,
                "m3_joint": 200.0,
                "m4_joint": -200.0,
            },
        ),
        actuators={
            "dummy": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )

@configclass
class CrazyflieSceneCfg(InteractiveSceneCfg):
    # -------------------------
    # Robot
    # -------------------------
    crazyflie = CRAZYFLIE.replace(
        prim_path="/World/envs/env_.*/Crazyflie",
        init_state=CRAZYFLIE.init_state.replace(pos=(CORRIDOR_X_OFFSET, 0.0, 0.5))
    )

    FPV_CAMERA_CFG = CameraCfg(
        prim_path="/World/envs/env_.*/Crazyflie/body/fpv",
        update_period=1.0 / 10.0,       # update every physics step (matches sim dt)
        height=FPV_REAL_HEIGHT,
        width=FPV_REAL_WIDTH,
        data_types=["rgb", "distance_to_camera"] if USE_DEPTH else ["rgb"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=FPV_REAL_K,
            width=FPV_REAL_WIDTH,
            height=FPV_REAL_HEIGHT,
            focus_distance=0.6,           # m   (typical focus plane)
            clipping_range=(0.4, 10.0),   # m   (D455 recommended range)
        ),
        # Pose relative to the Crazyflie
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.02),     # move camera slightly forward and up
            rot=(0.5, -0.5, 0.5, -0.5),   # quaternion (w,x,y,z); pointing forward in ROS convention
            convention="ros",
        ),
    )

    # -------------------------
    # Environment geometry
    # -------------------------
    warehouse_env = AssetBaseCfg(
        prim_path="/World/envs/env_.*/WarehouseEnv",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/{WAREHOUSE_ENV_USD}",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),  # 180 deg yaw about Z (w,x,y,z quaternion)
        ),
    )


    for _i, _pos in enumerate(CYLINDERS):
        vars()[f"cyl_{_i:02d}"] = RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/Cyl{_i:02d}",
            spawn=sim_utils.CylinderCfg(
                visual_material=RED_MAT,
                radius=0.15,
                height=2.0,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=_pos, rot=(1, 0, 0, 0)),
        )
    if CYLINDERS:
        del _i, _pos
