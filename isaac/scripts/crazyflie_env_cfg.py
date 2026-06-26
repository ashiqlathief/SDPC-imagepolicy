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

cfg = importlib.import_module("config.avoiding-crazyflie")

_BOXES_XY = cfg.BOXES
_CYLINDERS_XY = cfg.CYLINDERS
_SPHERES_XYZ = getattr(cfg, 'SPHERES', [])
SPHERE_RADIUS = getattr(cfg, 'SPHERE_RADIUS', 0.10)
USE_DEPTH = cfg.USE_DEPTH
DEPTH_NEAR = cfg.DEPTH_NEAR
DEPTH_FAR = cfg.DEPTH_FAR
CORRIDOR_LENGTH = 4.5     # x direction
CORRIDOR_WIDTH = 2.0       # y direction (clearance between walls)
WALL_THICKNESS = 0.10
WALL_HEIGHT = 2.0
CEILING_HEIGHT  = WALL_HEIGHT          # = 1.0  (flush with obstacle tops)
CEILING_Z_CENTER = CEILING_HEIGHT + WALL_THICKNESS / 2.0   # centre of roof slab
BOXES = [(x, y, 1.0 / 2.0) for (x, y) in _BOXES_XY]
CYLINDERS = [(x, y, 1.0 / 2.0) for (x, y) in _CYLINDERS_XY]
RED_MAT   = PreviewSurfaceCfg(diffuse_color=(0.85, 0.10, 0.10))
GREEN_MAT = PreviewSurfaceCfg(diffuse_color=(0.10, 0.85, 0.10))
BLUE_MAT  = PreviewSurfaceCfg(diffuse_color=(0.10, 0.30, 0.90))

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
        init_state=CRAZYFLIE.init_state.replace(pos=(0.0, 0.0, 0.2))#0.45219916 -0.2618473   0.12736732
    )

    FPV_CAMERA_CFG = CameraCfg(
        prim_path="/World/envs/env_.*/Crazyflie/body/fpv",
        update_period=0.005,       # update every physics step (matches sim dt)
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

    CHASE_CAMERA_CFG = CameraCfg(
        prim_path="/World/envs/env_.*/Crazyflie/body/chasecam",
        update_period=0.0,        # render only when explicitly queried
        height=480,
        width=854,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,            # wider FOV than the FPV (24.0)
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1000.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-0.6, 0.0, 0.35),    # behind and above the drone body
            rot=(0.5, -0.5, 0.5, -0.5),   # same forward-facing orientation as FPV
            convention="ros",
        ),
    )

    SPECTATOR_CAMERA_CFG = CameraCfg(
        prim_path="/World/envs/env_.*/SpectatorCam",
        update_period=0.0,
        height=540,
        width=960,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=10.0,            # wide FOV to fit the whole corridor
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1000.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(2.5, 0.0, 3.5),
            rot=(0.0, 1.0, 0.0, 0.0),
            convention="ros",
        ),
    )
    #SPECTATOR_CAMERA_CFG = CameraCfg(
    #     prim_path="/World/envs/env_.*/SpectatorCam",
    #     update_period=0.0,
    #     height=540,
    #     width=960,
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=10.0,            # wide FOV to fit the whole corridor
    #         focus_distance=400.0,
    #         horizontal_aperture=20.955,
    #         clipping_range=(0.1, 1000.0),
    #     ),
    #     offset=CameraCfg.OffsetCfg(
    #         pos=(-1.0, 0.0, 3.5),
    #         rot=(0.2549804, -0.6595339, 0.6595339, -0.2549804),
    #         convention="ros",
    #     ),
    # )

    # -------------------------
    # Environment geometry
    # -------------------------
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # Left wall
    wall_left = AssetBaseCfg(
        prim_path="/World/envs/env_.*/WallLeft",
        spawn=sim_utils.CuboidCfg(
            size=(CORRIDOR_LENGTH, WALL_THICKNESS, WALL_HEIGHT),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(CORRIDOR_LENGTH / 2.0, +(CORRIDOR_WIDTH / 2.0 + WALL_THICKNESS / 2.0), WALL_HEIGHT / 2.0), #2.5, 1.05
        ),
    )

    # Right wall
    wall_right = AssetBaseCfg(
        prim_path="/World/envs/env_.*/WallRight",
        spawn=sim_utils.CuboidCfg(
            size=(CORRIDOR_LENGTH, WALL_THICKNESS, WALL_HEIGHT),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(CORRIDOR_LENGTH / 2.0, -(CORRIDOR_WIDTH / 2.0 + WALL_THICKNESS / 2.0), WALL_HEIGHT / 2.0),  #2.5,
        ),
    )

    # End wall (optional, like the closed end in your screenshot)
    wall_end = AssetBaseCfg(
        prim_path="/World/envs/env_.*/WallEnd",
        spawn=sim_utils.CuboidCfg(
            visual_material=GREEN_MAT,
            size=(WALL_THICKNESS, CORRIDOR_WIDTH + 2 * WALL_THICKNESS, WALL_HEIGHT),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(CORRIDOR_LENGTH + WALL_THICKNESS / 2.0, 0.0, WALL_HEIGHT / 2.0),
        ),
    )

    # Ceiling: closes the corridor top so the FPV depth camera doesn't see
    # through to open space above the walls (was returning inf depth there).
    ceiling = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Ceiling",
        spawn=sim_utils.CuboidCfg(
            size=(CORRIDOR_LENGTH + WALL_THICKNESS, CORRIDOR_WIDTH + 2 * WALL_THICKNESS, WALL_THICKNESS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(CORRIDOR_LENGTH / 2.0, 0.0, CEILING_Z_CENTER),
        ),
    )

    goal = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Goal",
        spawn=sim_utils.CuboidCfg(
            visual_material=GREEN_MAT,
            size=(WALL_THICKNESS, CORRIDOR_WIDTH + 2 * WALL_THICKNESS, 0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(4.0, 0.0, 0.05),
        ),
    )

    ceiling_light = AssetBaseCfg(
        prim_path="/World/CeilingLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=2000.0,
            color=(1.0, 0.95, 0.85),      # warm white
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(CORRIDOR_LENGTH / 2.0, 0.0, 1.0),   # above corridor centre
            rot=(0.966, -0.259, 0.0, 0.0),            # angled slightly forward
        ),
    )

    # # -------------------------
    # # Obstacles (boxes + cylinders)
    # # -------------------------
    # # vars() in a class body returns the class namespace directly,
    # # so assignments here become real class attributes picked up by @configclass.
    # for _i, _pos in enumerate(BOXES):
    #     vars()[f"box_{_i:02d}"] = AssetBaseCfg(
    #         prim_path=f"/World/envs/env_.*/Box{_i:02d}",
    #         spawn=sim_utils.CuboidCfg(
    #             visual_material=RED_MAT,
    #             size=(0.20, 0.20, WALL_HEIGHT),
    #             collision_props=sim_utils.CollisionPropertiesCfg(),
    #         ),
    #         init_state=AssetBaseCfg.InitialStateCfg(pos=_pos),
    #     )
    # del _i, _pos  # prevent loop vars from leaking into the class namespace

    for _i, _pos in enumerate(CYLINDERS):
        vars()[f"cyl_{_i:02d}"] = RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/Cyl{_i:02d}",
            spawn=sim_utils.CylinderCfg(
                visual_material=RED_MAT,
                radius=0.06,
                height=1.0,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=_pos, rot=(1, 0, 0, 0)),
        )
    if CYLINDERS:
        del _i, _pos

    # ── Floating sphere obstacles (planet mode) ────────────────────────────────
    for _i, _pos in enumerate(_SPHERES_XYZ):
        vars()[f"sph_{_i:02d}"] = RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/Sph{_i:02d}",
            spawn=sim_utils.SphereCfg(
                visual_material=RED_MAT,
                radius=SPHERE_RADIUS,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=_pos, rot=(1, 0, 0, 0)),
        )
    if _SPHERES_XYZ:
        del _i, _pos
