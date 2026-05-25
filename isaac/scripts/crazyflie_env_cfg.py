from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg

RED_MAT = PreviewSurfaceCfg(diffuse_color=(0.85, 0.10, 0.10))
GREEN_MAT = PreviewSurfaceCfg(diffuse_color=(0.10, 0.85, 0.10))

CRAZYFLIE = ArticulationCfg(
        spawn=sim_utils.MultiUsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Bitcraze/Crazyflie/cf2x.usd",
            # usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Bitcraze/Crazyflie/cf2x.usd",
            
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
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

CORRIDOR_LENGTH = 5.0     # x direction
CORRIDOR_WIDTH = 2.0       # y direction (clearance between walls)
WALL_THICKNESS = 0.10
WALL_HEIGHT = 1.0
CEILING_HEIGHT  = WALL_HEIGHT          # = 1.0  (flush with obstacle tops)
CEILING_Z_CENTER = CEILING_HEIGHT + WALL_THICKNESS / 2.0   # centre of roof slab

# Obstacles: (x, y, z) in corridor frame (z is center height of the primitive)
BOXES = [
    # (3.5,  0.2, WALL_HEIGHT / 2.0),
    # (2.2, -0.2, WALL_HEIGHT / 2.0),
    # (1.6,  0.2, WALL_HEIGHT / 2.0),
    # (3.7, -0.3, WALL_HEIGHT / 2.0), #(3.7, -0.3, WALL_HEIGHT / 2.0),
    # (2.8, -0.2, WALL_HEIGHT / 2.0),
    # (1.0, -0.3, WALL_HEIGHT / 2.0),
    (3.0,  0.2, WALL_HEIGHT / 2.0),
    (2.0, -0.2, WALL_HEIGHT / 2.0),
    (1.1,  0.2, WALL_HEIGHT / 2.0),
    (3.2, -0.3, WALL_HEIGHT / 2.0), #(3.7, -0.3, WALL_HEIGHT / 2.0),
    (2.3, -0.2, WALL_HEIGHT / 2.0),
    (0.6, -0.3, WALL_HEIGHT / 2.0),
    # (3.5, -0.4, WALL_HEIGHT / 2.0),
    # (2.2, -0.4, WALL_HEIGHT / 2.0),
    # (1.6, -0.4, WALL_HEIGHT / 2.0),
    # (3.7, -0.4, WALL_HEIGHT / 2.0), #(3.7, -0.3, WALL_HEIGHT / 2.0),
    # (2.8, -0.4,  WALL_HEIGHT / 2.0),
    # (1.0, -0.4, WALL_HEIGHT / 2.0),

    # (3.5, 0.7, WALL_HEIGHT / 2.0),
    # (2.2, 0.7, WALL_HEIGHT / 2.0),
    # (1.6, 0.7, WALL_HEIGHT / 2.0),
    # (3.7, 0.7, WALL_HEIGHT / 2.0), #(3.7, -0.3, WALL_HEIGHT / 2.0),
    # (2.8, 0.7,  WALL_HEIGHT / 2.0),
    # (1.0, 0.7, WALL_HEIGHT / 2.0),
]
CYLINDERS = [
    (1.0, -0.2, WALL_HEIGHT / 2.0),
    (2.3,  0.3, WALL_HEIGHT / 2.0),
    (0.7,  0.4, WALL_HEIGHT / 2.0),
    (3.0,  0.4, WALL_HEIGHT / 2.0),
    (1.5, 0.4, WALL_HEIGHT / 2.0),#(2.0,  0.4, WALL_HEIGHT / 2.0), --- IGNORE ---
    (2.0,  0.3, WALL_HEIGHT / 2.0),

    # (1.4, -0.2, WALL_HEIGHT / 2.0),
    # (2.8,  0.3, WALL_HEIGHT / 2.0),
    # (1.2,  0.4, WALL_HEIGHT / 2.0),
    # (3.5,  0.4, WALL_HEIGHT / 2.0),
    # (2.0, 0.4, WALL_HEIGHT / 2.0),#(2.0,  0.4, WALL_HEIGHT / 2.0), --- IGNORE ---
    # (2.5,  0.3, WALL_HEIGHT / 2.0),

    # (1.4, 0.4, WALL_HEIGHT / 2.0),
    # (2.8, 0.3, WALL_HEIGHT / 2.0),
    # (1.2, 0.4, WALL_HEIGHT / 2.0),
    # (3.5, 0.3, WALL_HEIGHT / 2.0),
    # (2.0, 0.4, WALL_HEIGHT / 2.0),#(2.0,  0.4, WALL_HEIGHT / 2.0),
    # (2.5, 0.4, WALL_HEIGHT / 2.0),
    
    # (1.4, -0.7, WALL_HEIGHT / 2.0),
    # (2.8, -0.8, WALL_HEIGHT / 2.0),
    # (1.2, -0.7, WALL_HEIGHT / 2.0),
    # (3.5, -0.8, WALL_HEIGHT / 2.0),
    # (3.2, -0.7, WALL_HEIGHT / 2.0),#(2.0,  0.4, WALL_HEIGHT / 2.0),
    # (2.5, -0.7, WALL_HEIGHT / 2.0),
    ]

@configclass
class CrazyflieSceneCfg(InteractiveSceneCfg):
    """Configuration for the Crazyflie quadcopter."""

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
        data_types=["rgb"],
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


    # -------------------------
    # Obstacles (boxes + cylinders)
    # -------------------------

    # # Boxes
    # box_00 = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/Box00",
    #     spawn=sim_utils.CuboidCfg(
    #         visual_material=RED_MAT,
    #         size=(0.20, 0.20, WALL_HEIGHT),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=BOXES[0]),
    # )
    # box_01 = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/Box01",
    #     spawn=sim_utils.CuboidCfg(visual_material=RED_MAT,
    #                               size=(0.20, 0.20, WALL_HEIGHT),
    #                               collision_props=sim_utils.CollisionPropertiesCfg()),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=BOXES[1]),
    # )
    # box_02 = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/Box02",
    #     spawn=sim_utils.CuboidCfg(visual_material=RED_MAT,
    #                               size=(0.20, 0.20, WALL_HEIGHT),
    #                               collision_props=sim_utils.CollisionPropertiesCfg()),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=BOXES[2]),
    # )
    # box_03 = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/Box03",
    #     spawn=sim_utils.CuboidCfg(visual_material=RED_MAT,
    #                               size=(0.20, 0.20,WALL_HEIGHT),
    #                               collision_props=sim_utils.CollisionPropertiesCfg()),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=BOXES[3]),
    # )
    # box_04 = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/Box04",
    #     spawn=sim_utils.CuboidCfg(visual_material=RED_MAT,
    #                               size=(0.20, 0.20, WALL_HEIGHT),
    #                               collision_props=sim_utils.CollisionPropertiesCfg()),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=BOXES[4]),
    # )
    # box_05 = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/Box05",
    #     spawn=sim_utils.CuboidCfg(visual_material=RED_MAT,
    #                               size=(0.20, 0.20, WALL_HEIGHT),
    #                               collision_props=sim_utils.CollisionPropertiesCfg()),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=BOXES[5]),
    # )

    # # Boxes on the other side of the corridor
    # box_06 = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/Box06",
    #     spawn=sim_utils.CuboidCfg(
    #         visual_material=RED_MAT,
    #         size=(0.20, 0.20, WALL_HEIGHT),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=BOXES[6]),
    # )
    # box_07 = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/Box07",
    #     spawn=sim_utils.CuboidCfg(visual_material=RED_MAT,
    #                               size=(0.20, 0.20, WALL_HEIGHT),
    #                               collision_props=sim_utils.CollisionPropertiesCfg()),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=BOXES[7]),
    # )
    # box_08 = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/Box08",
    #     spawn=sim_utils.CuboidCfg(visual_material=RED_MAT,
    #                               size=(0.20, 0.20, WALL_HEIGHT),
    #                               collision_props=sim_utils.CollisionPropertiesCfg()),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=BOXES[8]),
    # )
    # box_09 = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/Box09",
    #     spawn=sim_utils.CuboidCfg(visual_material=RED_MAT,
    #                               size=(0.20, 0.20,WALL_HEIGHT),
    #                               collision_props=sim_utils.CollisionPropertiesCfg()),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=BOXES[9]),
    # )
    # box_10 = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/Box10",
    #     spawn=sim_utils.CuboidCfg(visual_material=RED_MAT,
    #                               size=(0.20, 0.20, WALL_HEIGHT),
    #                               collision_props=sim_utils.CollisionPropertiesCfg()),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=BOXES[10]),
    # )
    # box_11 = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/Box11",
    #     spawn=sim_utils.CuboidCfg(visual_material=RED_MAT,
    #                               size=(0.20, 0.20, WALL_HEIGHT),
    #                               collision_props=sim_utils.CollisionPropertiesCfg()),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=BOXES[11]),
    # )


    # # Cylinders (stand-ins for cones)
    # cyl_00 = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/Cyl00",
    #     spawn=sim_utils.CylinderCfg(visual_material=RED_MAT,
    #         radius=0.06,
    #         height=WALL_HEIGHT,
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=CYLINDERS[0]),
    # )
    # cyl_01 = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/Cyl01",
    #     spawn=sim_utils.CylinderCfg(visual_material=RED_MAT,
    #                                 radius=0.06, height=WALL_HEIGHT,
    #                                 collision_props=sim_utils.CollisionPropertiesCfg()),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=CYLINDERS[1]),
    # )
    # cyl_02 = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/Cyl02",
    #     spawn=sim_utils.CylinderCfg(visual_material=RED_MAT,
    #                                 radius=0.06, height=WALL_HEIGHT,
    #                                 collision_props=sim_utils.CollisionPropertiesCfg()),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=CYLINDERS[2]),
    # )
    # cyl_03 = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/Cyl03",
    #     spawn=sim_utils.CylinderCfg(visual_material=RED_MAT,
    #                                 radius=0.06, height=WALL_HEIGHT,
    #                                 collision_props=sim_utils.CollisionPropertiesCfg()),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=CYLINDERS[3]),
    # )
    # cyl_04 = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/Cyl04",
    #     spawn=sim_utils.CylinderCfg(visual_material=RED_MAT,
    #                                 radius=0.06, height=WALL_HEIGHT,
    #                                 collision_props=sim_utils.CollisionPropertiesCfg()),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=CYLINDERS[4]),
    # )
    # cyl_05 = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/Cyl05",
    #     spawn=sim_utils.CylinderCfg(visual_material=RED_MAT,
    #                                 radius=0.06, height=WALL_HEIGHT,
    #                                 collision_props=sim_utils.CollisionPropertiesCfg()),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=CYLINDERS[5]),
    # )






    # cyl_06 = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/Cyl06",
    #     spawn=sim_utils.CylinderCfg(visual_material=RED_MAT,
    #         radius=0.06,
    #         height=WALL_HEIGHT,
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=CYLINDERS[6]),
    # )
    # cyl_07 = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/Cyl07",
    #     spawn=sim_utils.CylinderCfg(visual_material=RED_MAT,
    #                                 radius=0.06, height=WALL_HEIGHT,
    #                                 collision_props=sim_utils.CollisionPropertiesCfg()),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=CYLINDERS[7]),
    # )
    # cyl_08 = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/Cyl08",
    #     spawn=sim_utils.CylinderCfg(visual_material=RED_MAT,
    #                                 radius=0.06, height=WALL_HEIGHT,
    #                                 collision_props=sim_utils.CollisionPropertiesCfg()),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=CYLINDERS[8]),
    # )
    # cyl_09 = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/Cyl09",
    #     spawn=sim_utils.CylinderCfg(visual_material=RED_MAT,
    #                                 radius=0.06, height=WALL_HEIGHT,
    #                                 collision_props=sim_utils.CollisionPropertiesCfg()),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=CYLINDERS[9]),
    # )
    # cyl_10 = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/Cyl10",
    #     spawn=sim_utils.CylinderCfg(visual_material=RED_MAT,
    #                                 radius=0.06, height=WALL_HEIGHT,
    #                                 collision_props=sim_utils.CollisionPropertiesCfg()),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=CYLINDERS[10]),
    # )
    # cyl_11 = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/Cyl11",
    #     spawn=sim_utils.CylinderCfg(visual_material=RED_MAT,
    #                                 radius=0.06, height=WALL_HEIGHT,
    #                                 collision_props=sim_utils.CollisionPropertiesCfg()),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=CYLINDERS[11]),
    # )
    # cyl_12 = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/Cyl12",
    #     spawn=sim_utils.CylinderCfg(visual_material=RED_MAT,
    #                                 radius=0.06, height=WALL_HEIGHT,
    #                                 collision_props=sim_utils.CollisionPropertiesCfg()),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=CYLINDERS[12]),
    # )
    # cyl_13 = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/Cyl13",
    #     spawn=sim_utils.CylinderCfg(visual_material=RED_MAT,
    #                                 radius=0.06, height=WALL_HEIGHT,
    #                                 collision_props=sim_utils.CollisionPropertiesCfg()),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=CYLINDERS[13]),
    # )
    # cyl_14 = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/Cyl14",
    #     spawn=sim_utils.CylinderCfg(visual_material=RED_MAT,
    #                                 radius=0.06, height=WALL_HEIGHT,
    #                                 collision_props=sim_utils.CollisionPropertiesCfg()),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=CYLINDERS[14]),
    # )
