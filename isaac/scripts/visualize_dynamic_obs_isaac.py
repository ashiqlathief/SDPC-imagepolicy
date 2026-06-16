import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualise dynamic obstacles in Isaac Sim.")
parser.add_argument("--num_envs",  type=int,   default=1)
parser.add_argument("--amplitude", type=float, default=0.25,
                    help="Cylinder oscillation amplitude in metres")
parser.add_argument("--frequency", type=float, default=0.25,
                    help="Cylinder oscillation frequency in Hz")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless        = False   # we need the viewport
args_cli.device          = "cpu"

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── post-launch imports ───────────────────────────────────────────────────────
import math
import carb
import omni
import torch

import isaaclab.sim as sim_utils
from isaaclab.utils      import configclass
from isaaclab.assets     import AssetBaseCfg, RigidObjectCfg, RigidObject
from isaaclab.scene      import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim        import SimulationContext
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg

from isaac.scripts.crazyflie_env_cfg import (
    BOXES, CYLINDERS, WALL_HEIGHT, CORRIDOR_LENGTH, CORRIDOR_WIDTH,
    WALL_THICKNESS, RED_MAT, GREEN_MAT,
)

# ── keyboard state ────────────────────────────────────────────────────────────
paused = False

def _on_key(event, *args, **kwargs):
    global paused
    if event.type in (carb.input.KeyboardEventType.KEY_PRESS,
                      carb.input.KeyboardEventType.KEY_REPEAT):
        if event.input == carb.input.KeyboardInput.SPACE:
            paused = not paused
            print(f"[INFO] {'PAUSED' if paused else 'RESUMED'}")
        elif event.input == carb.input.KeyboardInput.Q:
            simulation_app.close()

appwindow = omni.appwindow.get_default_app_window()
_input    = carb.input.acquire_input_interface()
_input.subscribe_to_keyboard_events(appwindow.get_keyboard(), _on_key)

# ── scene config ─────────────────────────────────────────────────────────────
_CYL_KEYS = [f"cyl_{i:02d}" for i in range(len(CYLINDERS))]

def _build_scene_cfg() -> type:
    """
    Dynamically construct DynSceneCfg.

    @configclass requires named class attributes (dataclass fields), so we
    can't use a loop inside the class body.  Instead we build the namespace
    dict manually, then call type() + configclass() — equivalent to writing
    the decorator over a hand-typed class, just without the repetition.
    """
    ns: dict = {"__annotations__": {}}

    def _add(name: str, cfg, cfg_type: type) -> None:
        ns[name] = cfg
        ns["__annotations__"][name] = cfg_type

    # ground + walls + light (unique, so written out once each)
    _add("ground", AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    ), AssetBaseCfg)

    _add("wall_left", AssetBaseCfg(
        prim_path="/World/envs/env_.*/WallLeft",
        spawn=sim_utils.CuboidCfg(
            size=(CORRIDOR_LENGTH, WALL_THICKNESS, WALL_HEIGHT),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(CORRIDOR_LENGTH/2, +(CORRIDOR_WIDTH/2 + WALL_THICKNESS/2), WALL_HEIGHT/2),
        ),
    ), AssetBaseCfg)

    _add("wall_right", AssetBaseCfg(
        prim_path="/World/envs/env_.*/WallRight",
        spawn=sim_utils.CuboidCfg(
            size=(CORRIDOR_LENGTH, WALL_THICKNESS, WALL_HEIGHT),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(CORRIDOR_LENGTH/2, -(CORRIDOR_WIDTH/2 + WALL_THICKNESS/2), WALL_HEIGHT/2),
        ),
    ), AssetBaseCfg)

    _add("wall_end", AssetBaseCfg(
        prim_path="/World/envs/env_.*/WallEnd",
        spawn=sim_utils.CuboidCfg(
            visual_material=GREEN_MAT,
            size=(WALL_THICKNESS, CORRIDOR_WIDTH + 2*WALL_THICKNESS, WALL_HEIGHT),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(CORRIDOR_LENGTH + WALL_THICKNESS/2, 0.0, WALL_HEIGHT/2),
        ),
    ), AssetBaseCfg)

    _add("ceiling_light", AssetBaseCfg(
        prim_path="/World/CeilingLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(1.0, 0.95, 0.85)),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(CORRIDOR_LENGTH/2, 0.0, 1.0),
            rot=(0.966, -0.259, 0.0, 0.0),
        ),
    ), AssetBaseCfg)

    # static boxes — one loop replaces 12 repeated blocks
    _box_spawn = sim_utils.CuboidCfg(
        visual_material=RED_MAT,
        size=(0.20, 0.20, WALL_HEIGHT),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    for i, pos in enumerate(BOXES):
        _add(f"box_{i:02d}", AssetBaseCfg(
            prim_path=f"/World/envs/env_.*/Box{i:02d}",
            spawn=_box_spawn,
            init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
        ), AssetBaseCfg)

    # kinematic cylinders — one loop replaces 12 repeated blocks
    for i, pos in enumerate(CYLINDERS):
        _add(f"cyl_{i:02d}", RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/DynCyl{i:02d}",
            spawn=sim_utils.CylinderCfg(
                visual_material=RED_MAT,
                radius=0.06,
                height=WALL_HEIGHT,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=(1, 0, 0, 0)),
        ), RigidObjectCfg)

    cls = type("DynSceneCfg", (InteractiveSceneCfg,), ns)
    return configclass(cls)


DynSceneCfg = _build_scene_cfg()


# ── simulation loop ───────────────────────────────────────────────────────────

def main():
    device = "cpu"  # AppLauncher was started with device="cpu" above

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=device)
    sim     = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.5, -3.5, 2.5], target=[2.5, 0.0, 0.5])

    scene_cfg = DynSceneCfg(
        num_envs=args_cli.num_envs, env_spacing=2.0
    )
    scene = InteractiveScene(scene_cfg)

    # grab kinematic cylinder handles
    dyn_cyls: list[RigidObject] = [scene[k] for k in _CYL_KEYS]

    sim.reset()
    scene.update(sim.get_physics_dt())

    # rest positions
    cyl_x0 = [float(CYLINDERS[i][0]) for i in range(12)]
    cyl_y0 = [float(CYLINDERS[i][1]) for i in range(12)]
    cyl_z0 = [float(CYLINDERS[i][2]) for i in range(12)]

    # evenly spaced phases so cylinders don't all move in sync
    phases = [2.0 * math.pi * i / 12 for i in range(12)]

    N   = args_cli.num_envs
    dt  = float(sim.get_physics_dt())
    t   = 0.0

    print("[INFO] Scene ready. SPACE = pause/resume, Q = quit")
    print(f"[INFO] Amplitude={args_cli.amplitude} m   Frequency={args_cli.frequency} Hz")

    while simulation_app.is_running():
        if not paused:
            # move each kinematic cylinder
            for i, cyl_obj in enumerate(dyn_cyls):
                new_y = cyl_y0[i] + args_cli.amplitude * math.sin(
                    2.0 * math.pi * args_cli.frequency * t + phases[i]
                )
                new_y = max(-0.85, min(0.85, new_y))

                pose = torch.zeros(N, 7, device=device)
                pose[:, 0] = cyl_x0[i]
                pose[:, 1] = new_y
                pose[:, 2] = cyl_z0[i]
                pose[:, 3] = 1.0          # quaternion w=1 (no rotation)

                cyl_obj.write_root_pose_to_sim(pose)

            sim.step()
            t += dt

            for cyl_obj in dyn_cyls:
                cyl_obj.update(dt)
            scene.update(dt)

        else:
            # paused — still render, just don't advance time
            sim.render()


if __name__ == "__main__":
    main()
