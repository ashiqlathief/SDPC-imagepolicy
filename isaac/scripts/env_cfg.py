from __future__ import annotations
import importlib
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR
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
WALL_HEIGHT = 3.0
CEILING_HEIGHT  = WALL_HEIGHT          # = 1.0  (flush with obstacle tops)
CEILING_Z_CENTER = CEILING_HEIGHT + WALL_THICKNESS / 2.0   # centre of roof slab
CORRIDOR_X_OFFSET = -CORRIDOR_LENGTH / 2.0

BOXES = [(x , y, 1.0 / 2.0) for (x, y) in _BOXES_XY]
CYLINDERS = [(x , y, 1.0 / 2.0) for (x, y) in _CYLINDERS_XY]
SPHERES_XYZ = [(x, y, z) for (x, y, z) in _SPHERES_XYZ]
RED_MAT   = PreviewSurfaceCfg(diffuse_color=(0.85, 0.10, 0.10))
BLUE_MAT  = PreviewSurfaceCfg(diffuse_color=(0.10, 0.30, 0.90))
FLOOR_BASE_MAT = PreviewSurfaceCfg(diffuse_color=(0.22, 0.22, 0.22), roughness=0.9)   # visible through the tile seams

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
    # ---- left wall (y = -CORR_Y) ----
    (_WALL_6M,     _CORR_X0 + _SEG_6M_LEN / 2.0,                -_CORR_Y, 0.0, _ROT_YAW_P90, (1.0, 1.0, _WALL_Z_SCALE)),
    (_WALL_3M,     _CORR_X0 + _SEG_6M_LEN + _SEG_FILL_LEN / 2.0,-_CORR_Y, 0.0, _ROT_YAW_P90, (1.0, _SEG_FILL_SCALE_Y, _WALL_Z_SCALE)),
    # mirrored (scale_y = -1) so its Y-leg points toward the corridor centreline, not outward
    (_WALL_CORNER, _CORR_X1,                                    _CORR_Y, 0.0, _ROT_YAW_180, (1.0, -1.0, _WALL_Z_SCALE)),

    # ---- right wall (y = +CORR_Y) ----
    (_WALL_6M,     _CORR_X0 + _SEG_6M_LEN / 2.0,                _CORR_Y, 0.0, _ROT_YAW_N90, (1.0, 1.0, _WALL_Z_SCALE)),
    (_WALL_3M,     _CORR_X0 + _SEG_6M_LEN + _SEG_FILL_LEN / 2.0,_CORR_Y, 0.0, _ROT_YAW_N90, (1.0, _SEG_FILL_SCALE_Y, _WALL_Z_SCALE)),
    (_WALL_CORNER, _CORR_X1,                                    -_CORR_Y, 0.0, _ROT_YAW_180, (1.0, 1.0, _WALL_Z_SCALE)),
]

_TRUSS_USD         = f"{_WALL_ASSET_DIR}/SM_RackFrame_03.usd"
_TRUSS_UNIT_WIDTH  = 1.0   # m, assumed frame width -- VERIFY IN-SIM
_TRUSS_Z           = 0.0   # floor-pivot -- touches the ground
_TRUSS_COUNT       = 10
_TRUSS_SPAN        = _TRUSS_COUNT * _TRUSS_UNIT_WIDTH
_TRUSS_X0          = _CORR_X0 + (CORRIDOR_LENGTH - _TRUSS_SPAN) / 2.0 + _TRUSS_UNIT_WIDTH / 2.0  # centred run

# (wall y, rotation) -- right wall (+CORR_Y, faces -Y into the corridor) and
# left wall (-CORR_Y, faces +Y into the corridor), matching the wall panels above.
for _wall_y, _wall_rot in ((_CORR_Y, _ROT_YAW_N90), (-_CORR_Y, _ROT_YAW_P90)):
    for _ti in range(_TRUSS_COUNT):
        WALLS_MODULAR.append((
            _TRUSS_USD,
            _TRUSS_X0 + _ti * _TRUSS_UNIT_WIDTH,
            _wall_y,
            _TRUSS_Z,
            _wall_rot,
            (1.0, 1.0, 1.0),
        ))
del _wall_y, _wall_rot, _ti

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
        init_state=CRAZYFLIE.init_state.replace(pos=(CORRIDOR_X_OFFSET, -1.0, 1.0))
    )

    FPV_CAMERA_CFG = CameraCfg(
        prim_path="/World/envs/env_.*/Crazyflie/body/fpv",
        update_period=1.0 / 10.0,       # update every physics step (matches sim dt)
        height=96,
        width=96,
        data_types=["rgb", "distance_to_camera"] if USE_DEPTH else ["rgb"],
        spawn=sim_utils.PinholeCameraCfg(   # camera intrinsics
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1000.0),
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
    # Base floor slab (structural), with the thin foam-tile mat (FLOOR_TILES) laid on top.
    floor_base = AssetBaseCfg(
        prim_path="/World/envs/env_.*/FloorBase",
        spawn=sim_utils.CuboidCfg(
            size=(CORRIDOR_LENGTH + 2 * WALL_THICKNESS, CORRIDOR_WIDTH + 2 * WALL_THICKNESS, WALL_THICKNESS),
            visual_material=FLOOR_BASE_MAT,
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(CORRIDOR_LENGTH / 2.0 + CORRIDOR_X_OFFSET, 0.0, -WALL_THICKNESS / 2.0),  # top face at z=0
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