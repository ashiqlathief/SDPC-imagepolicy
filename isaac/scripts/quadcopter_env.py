import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="This script demonstrates how to simulate a quadcopter.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import omni.appwindow
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.scene import InteractiveScene

# from crazyflie_env_cfg1 import CrazyflieSceneCfg, CORRIDOR_X_OFFSET, CORRIDOR_LENGTH, WALL_HEIGHT
from env_cfg import CrazyflieSceneCfg, CORRIDOR_X_OFFSET, CORRIDOR_LENGTH, WALL_HEIGHT

_capture_count = 0


def _capture_screenshot():
    """Fire-and-forget capture of whatever the viewport shows right now."""
    global _capture_count
    viewport = get_active_viewport()
    out_path = f"corridor_topdown_{_capture_count:02d}.png"
    capture_viewport_to_file(viewport, file_path=out_path)
    print(f"[INFO]: capturing -> {out_path}")
    _capture_count += 1


def _on_keyboard_event(event, *args, **kwargs):
    if event.type == carb.input.KeyboardEventType.KEY_PRESS and event.input.name == "SPACE":
        _capture_screenshot()
    return True


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    # Elevated view above the drone's spawn point, tilted to look down the
    # corridor's length toward the closed end wall (instead of straight down,
    # which just looks like an XY plot). This script doesn't step physics, so
    # the drone just sits there -- see CrazyflieSceneCfg's crazyflie.init_state
    # in env_cfg.py.
    _drone_x, _drone_y = CORRIDOR_X_OFFSET, 0.5
    _endwall_x = CORRIDOR_LENGTH + CORRIDOR_X_OFFSET   # closed end of the corridor
    sim.set_camera_view(
        [_drone_x, _drone_y, WALL_HEIGHT - 0.1],   # eye: above the drone
        [_endwall_x, _drone_y, 0.3],               # target: down the corridor, low, at the end wall
    )

    scene_cfg = CrazyflieSceneCfg(num_envs=args_cli.num_envs, env_spacing= 2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    appwindow = omni.appwindow.get_default_app_window()
    input_iface = carb.input.acquire_input_interface()
    keyboard = appwindow.get_keyboard()
    keyboard_sub = input_iface.subscribe_to_keyboard_events(keyboard, _on_keyboard_event)

    print("[INFO]: Setup complete. Press SPACE in the viewport to save a screenshot, close the window to quit.")

    while simulation_app.is_running():
        sim.render()

    input_iface.unsubscribe_to_keyboard_events(keyboard, keyboard_sub)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
