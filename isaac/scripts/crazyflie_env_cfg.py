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

_BOXES_XY = cfg.BOXES
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

FPV_REAL_WIDTH, FPV_REAL_HEIGHT = 424, 240
FPV_REAL_K = [212.49798583984375, 0.0, 215.16233825683594,
              0.0, 212.30624389648438, 121.23411560058594,
              0.0, 0.0, 1.0]

BOXES = [(x , y, 1.0 / 2.0) for (x, y) in _BOXES_XY]
CYLINDERS = [(x , y, 1.0 / 2.0) for (x, y) in _CYLINDERS_XY]
SPHERES_XYZ = [(x, y, z) for (x, y, z) in _SPHERES_XYZ]
RED_MAT   = PreviewSurfaceCfg(diffuse_color=(0.85, 0.10, 0.10))
BLUE_MAT  = PreviewSurfaceCfg(diffuse_color=(0.10, 0.30, 0.90))
_WH_MAT_DIR = "Environments/Simple_Warehouse/Materials"
WALL_MAT    = MdlFileCfg(mdl_path=f"{ISAAC_NUCLEUS_DIR}/{_WH_MAT_DIR}/MI_WallA_01.mdl", project_uvw=True)
GROUND_MAT  = MdlFileCfg(mdl_path=f"{ISAAC_NUCLEUS_DIR}/{_WH_MAT_DIR}/MI_Floor_01.mdl", project_uvw=True)
CEILING_MAT = MdlFileCfg(mdl_path=f"{ISAAC_NUCLEUS_DIR}/{_WH_MAT_DIR}/MI_CeilingA_06b.mdl", project_uvw=True)
WAREHOUSE_PROPS = [
    # (usd path relative to ISAAC_NUCLEUS_DIR, x, y, z, yaw quat (w,x,y,z))
    ("Environments/Simple_Warehouse/Props/SM_CardBoxA_01.usd",    1.0,  1.6, 0.0, (1.0, 0.0, 0.0, 0.0)),
    ("Props/Pallet/pallet.usd",                                   3.0, -1.6, 0.0, (0.92, 0.0, 0.0, 0.38)),
    ("Environments/Simple_Warehouse/Props/SM_BarelPlastic_A_01.usd", -2.0, 1.7, 0.0, (1.0, 0.0, 0.0, 0.0)),
    ("Environments/Simple_Warehouse/Props/SM_RackShelf_01.usd",  -4.5,  0.0, 0.0, (0.707, 0.0, 0.0, 0.707)),

    # NOTE: placeholder placements below (x/y/rot not verified in-sim yet) --
    # spread along both walls, facing into the corridor like SM_RackShelf_01 above.
    ("Environments/Simple_Warehouse/Props/SM_RackFrame_03_227.usd", -3.0,  2.1, 0.0, (0.707, 0.0, 0.0,  0.707)),
    # SM_RackFrame_03.usd moved to WALLS_MODULAR below (mounted flush on the
    # right wall as a stage truss, not freestanding).
    ("Environments/Simple_Warehouse/Props/SM_RackPile_03.usd",       0.0,  2.1, 0.0, (0.707, 0.0, 0.0,  0.707)),
    ("Environments/Simple_Warehouse/Props/SM_RackPile_04.usd",       0.0, -2.1, 0.0, (0.707, 0.0, 0.0, -0.707)),
    ("Environments/Simple_Warehouse/Props/SM_RackPile_06_234.usd",   2.0,  2.1, 0.0, (0.707, 0.0, 0.0,  0.707)),
    ("Environments/Simple_Warehouse/Props/SM_RackPile_06.usd",       2.0, -2.1, 0.0, (0.707, 0.0, 0.0, -0.707)),
    ("Environments/Simple_Warehouse/Props/SM_RackShelf_01_226.usd",  4.0,  2.1, 0.0, (0.707, 0.0, 0.0,  0.707)),
]

_WALL_ASSET_DIR    = "Environments/Simple_Warehouse/Props"
_WALL_6M           = f"{_WALL_ASSET_DIR}/SM_WallA_6M.usd"
_WALL_3M           = f"{_WALL_ASSET_DIR}/SM_WallA_3M.usd"
_WALL_CORNER       = f"{_WALL_ASSET_DIR}/SM_WallA_InnerCorner.usd"
_WALL_NATIVE_HEIGHT = 3.1
_WALL_Z_SCALE      = WALL_HEIGHT / _WALL_NATIVE_HEIGHT   # squash 3.1 m -> WALL_HEIGHT

_ROT_YAW_P90 = (0.7071, 0.0, 0.0,  0.7071)   # +90 deg about Z: local +X -> world +Y
_ROT_YAW_N90 = (0.7071, 0.0, 0.0, -0.7071)   # -90 deg about Z: local +X -> world -Y
_ROT_YAW_180 = (0.0,    0.0, 0.0,  1.0)      # 180 deg about Z

_CORR_X0 = CORRIDOR_X_OFFSET                       # open end of the corridor
_CORR_X1 = CORRIDOR_LENGTH + CORRIDOR_X_OFFSET      # closed end / corner vertex x
_CORR_Y  = CORRIDOR_WIDTH / 2.0                     # inner-face y of each side wall
_SEG_6M_LEN       = 6.0
_SEG_FILL_LEN     = CORRIDOR_LENGTH - 3.0 - _SEG_6M_LEN   # = 2.0 for CORRIDOR_LENGTH = 11
_SEG_FILL_SCALE_Y = _SEG_FILL_LEN / 3.0

# (usd, x, y, z, yaw quat (w,x,y,z), scale (sx,sy,sz))
WALLS_MODULAR = [
    # ---- left wall (y = +CORR_Y, outward = +Y) ----
    (_WALL_6M,     _CORR_X0 + _SEG_6M_LEN / 2.0,                -_CORR_Y, 0.0, _ROT_YAW_P90, (1.0, 1.0, _WALL_Z_SCALE)),
    (_WALL_3M,     _CORR_X0 + _SEG_6M_LEN + _SEG_FILL_LEN / 2.0,-_CORR_Y, 0.0, _ROT_YAW_P90, (1.0, _SEG_FILL_SCALE_Y, _WALL_Z_SCALE)),
    # mirrored (scale_y = -1) so its Y-leg points toward the corridor centreline, not outward
    (_WALL_CORNER, _CORR_X1,                                    _CORR_Y, 0.0, _ROT_YAW_180, (1.0, -1.0, _WALL_Z_SCALE)),

    # ---- right wall (y = -CORR_Y, outward = -Y) ----
    (_WALL_6M,     _CORR_X0 + _SEG_6M_LEN / 2.0,                _CORR_Y, 0.0, _ROT_YAW_N90, (1.0, 1.0, _WALL_Z_SCALE)),
    (_WALL_3M,     _CORR_X0 + _SEG_6M_LEN + _SEG_FILL_LEN / 2.0,_CORR_Y, 0.0, _ROT_YAW_N90, (1.0, _SEG_FILL_SCALE_Y, _WALL_Z_SCALE)),
    (_WALL_CORNER, _CORR_X1,                                    -_CORR_Y, 0.0, _ROT_YAW_180, (1.0, 1.0, _WALL_Z_SCALE)),

    # ---- stage truss, flush on the right wall (y = +CORR_Y) ----
    # placeholder placement/scale (native truss usd size unverified) -- check
    # in-sim and adjust z (mount height) / scale (span length) as needed.
    (f"{_WALL_ASSET_DIR}/SM_RackFrame_03.usd", _CORR_X0 + CORRIDOR_LENGTH / 2.0, _CORR_Y, 1.6, _ROT_YAW_N90, (1.0, 1.0, 1.0)),
]

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
        update_period=1.0 / 30.0,       # update every physics step (matches sim dt)
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
    ground = AssetBaseCfg(
        prim_path="/World/envs/env_.*/GroundPlane",
        spawn=sim_utils.CuboidCfg(
            size=(CORRIDOR_LENGTH + 2 * WALL_THICKNESS, CORRIDOR_WIDTH + 2 * WALL_THICKNESS, WALL_THICKNESS),
            visual_material=GROUND_MAT,
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(CORRIDOR_LENGTH / 2.0 + CORRIDOR_X_OFFSET, 0.0, -WALL_THICKNESS / 2.0),  # top face at z=0
        ),
    )

    # Left/right/end walls: see the WALLS_MODULAR spawn loop below.

    ceiling = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Ceiling",
        spawn=sim_utils.CuboidCfg(
            size=(CORRIDOR_LENGTH + 2 * WALL_THICKNESS, CORRIDOR_WIDTH + 2 * WALL_THICKNESS, WALL_THICKNESS),
            visual_material=CEILING_MAT,
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(CORRIDOR_LENGTH / 2.0 + CORRIDOR_X_OFFSET, 0.0, CEILING_Z_CENTER),
        ),
    )

    ceiling_light = AssetBaseCfg(
        prim_path="/World/CeilingLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=2000.0,
            color=(1.0, 0.95, 0.85),      # warm white
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(CORRIDOR_LENGTH / 2.0 + CORRIDOR_X_OFFSET, 0.0, WALL_HEIGHT - 0.05),  # just under the ceiling
            rot=(0.0, 0.0, 0.0, 0.0),  # rotate cylinder's local Z axis to align with the corridor's x-axis
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

    for _wi, (_usd, _wx, _wy, _wz, _wrot) in enumerate(WAREHOUSE_PROPS):
        vars()[f"warehouse_prop_{_wi:02d}"] = AssetBaseCfg(
            prim_path=f"/World/envs/env_.*/WarehouseProp{_wi:02d}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/{_usd}",
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(_wx, _wy, _wz), rot=_wrot),
        )
    if WAREHOUSE_PROPS:
        del _wi, _usd, _wx, _wy, _wz, _wrot

    for _mi, (_usd, _mx, _my, _mz, _mrot, _mscale) in enumerate(WALLS_MODULAR):
        vars()[f"wall_modular_{_mi:02d}"] = AssetBaseCfg(
            prim_path=f"/World/envs/env_.*/WallModular{_mi:02d}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/{_usd}",
                scale=_mscale,
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(_mx, _my, _mz), rot=_mrot),
        )
    if WALLS_MODULAR:
        del _mi, _usd, _mx, _my, _mz, _mrot, _mscale
