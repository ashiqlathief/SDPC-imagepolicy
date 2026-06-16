import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="This script demonstrates how to simulate a quadcopter.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.scene import InteractiveScene

from crazyflie_env_cfg import CrazyflieSceneCfg, BOXES, CYLINDERS, WALL_HEIGHT

def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([-2.5, 2.0, 2.0],  [0.0, 0.0, 0.0] )

    scene_cfg = CrazyflieSceneCfg(num_envs=args_cli.num_envs, env_spacing= 2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete...")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
