import os
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import argparse
import yaml
import time
from cv_bridge import CvBridge
import cv2
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import re

# ROS2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray
from utils import msg_to_pil, to_numpy, transform_images, load_model

from vint_train.training.train_utils import get_action

# UTILS
from topic_names import IMAGE_TOPIC, WAYPOINT_TOPIC, SAMPLED_ACTIONS_TOPIC

# CONSTANTS
TOPOMAP_IMAGES_DIR = "../topomaps/images"
MODEL_WEIGHTS_PATH = "../model_weights"
ROBOT_CONFIG_PATH = "../config/robot.yaml"
MODEL_CONFIG_PATH = "../config/models.yaml"
with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"]

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class ExplorationNode(Node):
    def __init__(self, args):
        super().__init__("exploration_node")

        # Initialize class variables
        self.args = args
        self.context_queue = []
        self.context_size = None
        self.subgoal = []
        self.closest_node = 0
        self.reached_goal = False
        self.offline_dir = f"{TOPOMAP_IMAGES_DIR}/{args.offline_dir}"
        self.offline_images = []
        self.offline_index = 0

        if self.offline_dir is not None:
            self.get_logger().info(f"Running in OFFLINE mode from {self.offline_dir}")
            image_files = sorted(os.listdir(self.offline_dir))
            for f in image_files:
                path = os.path.join(self.offline_dir, f)
                try:
                    img = PILImage.open(path)
                    self.offline_images.append(img)
                except Exception as e:
                    self.get_logger().warning(f"Skipping {path}: {e}")
        else:
            self.get_logger().info(
                "Running in ONLINE mode (ROS image subscriber only)."
            )

        # Load model parameters
        with open(MODEL_CONFIG_PATH, "r") as f:
            model_paths = yaml.safe_load(f)

        model_config_path = model_paths[args.model]["config_path"]
        with open(model_config_path, "r") as f:
            self.model_params = yaml.safe_load(f)

        self.context_size = self.model_params["context_size"]

        # Load model weights
        ckpth_path = model_paths[args.model]["ckpt_path"]
        if os.path.exists(ckpth_path):
            self.get_logger().info(f"Loading model from {ckpth_path}")
        else:
            raise FileNotFoundError(f"Model weights not found at {ckpth_path}")

        self.model = load_model(ckpth_path, self.model_params, device)
        self.model = self.model.to(device)
        self.model.eval()

        # Load topomap
        topomap_dir = f"{TOPOMAP_IMAGES_DIR}/{args.dir}"

        def timestamp_key(filename):
            # Extract YYYYMMDD_HHMMSS from filename
            match = re.match(r"(\d{8}_\d{6})", filename)
            return match.group(1) if match else filename

        topomap_filenames = sorted(os.listdir(topomap_dir), key=timestamp_key)

        num_nodes = len(topomap_filenames)
        self.topomap = []
        for i in range(num_nodes):
            image_path = os.path.join(topomap_dir, topomap_filenames[i])
            self.topomap.append(PILImage.open(image_path))

        # topomap_filenames = sorted(os.listdir(os.path.join(
        #     TOPOMAP_IMAGES_DIR, args.dir)), key=lambda x: int(x.split(".")[0]))
        # num_nodes = len(os.listdir(topomap_dir))
        # self.topomap = []
        # for i in range(num_nodes):
        #     image_path = os.path.join(topomap_dir, topomap_filenames[i])
        #     self.topomap.append(PILImage.open(image_path))

        assert -1 <= args.goal_node < len(self.topomap), "Invalid goal index"
        if args.goal_node == -1:
            self.goal_node = len(self.topomap) - 1
        else:
            self.goal_node = args.goal_node

        # Setup diffusion scheduler if using nomad model
        if self.model_params["model_type"] == "nomad":
            self.num_diffusion_iters = self.model_params["num_diffusion_iters"]
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.model_params["num_diffusion_iters"],
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                prediction_type="epsilon",
            )

        # ROS2 Publishers and Subscribers
        self.image_subscription = self.create_subscription(
            Image, IMAGE_TOPIC, self.callback_obs, 1  # QoS depth
        )

        self.waypoint_publisher = self.create_publisher(
            Float32MultiArray, WAYPOINT_TOPIC, 1
        )

        self.sampled_actions_publisher = self.create_publisher(
            Float32MultiArray, SAMPLED_ACTIONS_TOPIC, 1
        )

        self.goal_publisher = self.create_publisher(Bool, "/topoplan/reached_goal", 1)

        # Create timer for navigation loop
        timer_period = 1.0 / RATE  # seconds
        self.timer = self.create_timer(timer_period, self.navigation_loop)

        if self.offline_dir is not None:
            self.get_logger().info("Running OFFLINE mode — consuming images from disk")
        else:
            self.get_logger().info(
                "Running ONLINE mode — waiting for ROS2 image messages"
            )

    def callback_obs(self, msg):
        obs_img = msg_to_pil(msg)
        if self.context_size is not None:
            if len(self.context_queue) < self.context_size + 1:
                self.context_queue.append(obs_img)
            else:
                self.context_queue.pop(0)
                self.context_queue.append(obs_img)

    def navigation_loop(self):
        """Main navigation loop called by timer"""
        # EXPLORATION MODE
        chosen_waypoint = np.zeros(4)

        # debug: show tick
        self.get_logger().debug(
            f"[TICK] offline_index={self.offline_index}/{len(self.offline_images)} "
            f"context_len={len(self.context_queue)}"
        )

        if self.offline_dir is not None:
            if self.offline_index < len(self.offline_images):
                obs_img = self.offline_images[self.offline_index]
                self.offline_index += 1
                if len(self.context_queue) < self.context_size + 1:
                    self.context_queue.append(obs_img)
                else:
                    self.context_queue.pop(0)
                    self.context_queue.append(obs_img)
            else:
                self.get_logger().info("All offline images consumed. Stopping...")
                rclpy.shutdown()
                return

        if len(self.context_queue) > self.model_params["context_size"]:
            self.get_logger().info(
                f"Invoking model: context_len={len(self.context_queue)}, model_type={self.model_params['model_type']}"
            )
            start_model = time.time()
            if self.model_params["model_type"] == "nomad":
                chosen_waypoint = self._process_nomad_model()
            else:
                chosen_waypoint = self._process_other_model()

            self.get_logger().info(
                f"Model returned in {time.time() - start_model:.2f}s"
            )
        else:
            self.get_logger().debug(
                f"Waiting for context warm-up: {len(self.context_queue)}/{self.model_params['context_size']+1}"
            )

        # RECOVERY MODE
        if self.model_params["normalize"]:
            chosen_waypoint[:2] *= MAX_V / RATE

        waypoint_msg = Float32MultiArray()
        waypoint_msg.data = chosen_waypoint.tolist()
        self.waypoint_publisher.publish(waypoint_msg)

        self.reached_goal = self.closest_node == self.goal_node
        goal_msg = Bool()
        goal_msg.data = bool(self.reached_goal)
        self.goal_publisher.publish(goal_msg)

        if self.reached_goal:
            self.get_logger().info("Reached goal! Stopping...")

    def _process_nomad_model(self):
        """Process using nomad diffusion model"""
        obs_images = transform_images(
            self.context_queue, self.model_params["image_size"], center_crop=False
        )
        obs_images = torch.split(obs_images, 3, dim=1)
        obs_images = torch.cat(obs_images, dim=1)
        obs_images = obs_images.to(device)
        mask = torch.zeros(1).long().to(device)

        start = max(self.closest_node - self.args.radius, 0)
        end = min(self.closest_node + self.args.radius + 1, self.goal_node)
        goal_image = [
            transform_images(
                g_img, self.model_params["image_size"], center_crop=False
            ).to(device)
            for g_img in self.topomap[start : end + 1]
        ]
        goal_image = torch.concat(goal_image, dim=0)

        obsgoal_cond = self.model(
            "vision_encoder",
            obs_img=obs_images.repeat(len(goal_image), 1, 1, 1),
            goal_img=goal_image,
            input_goal_mask=mask.repeat(len(goal_image)),
        )
        dists = self.model("dist_pred_net", obsgoal_cond=obsgoal_cond)
        dists = to_numpy(dists.flatten())
        min_idx = np.argmin(dists)
        self.closest_node = min_idx + start
        self.get_logger().info(f"closest node: {self.closest_node}")

        sg_idx = min(
            min_idx + int(dists[min_idx] < self.args.close_threshold),
            len(obsgoal_cond) - 1,
        )
        obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)

        # infer action
        with torch.no_grad():
            # encoder vision features
            if len(obs_cond.shape) == 2:
                obs_cond = obs_cond.repeat(self.args.num_samples, 1)
            else:
                obs_cond = obs_cond.repeat(self.args.num_samples, 1, 1)

            # initialize action from Gaussian noise
            noisy_action = torch.randn(
                (self.args.num_samples, self.model_params["len_traj_pred"], 2),
                device=device,
            )
            naction = noisy_action

            # init scheduler
            self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

            start_time = time.time()
            for k in self.noise_scheduler.timesteps[:]:
                # predict noise
                noise_pred = self.model(
                    "noise_pred_net", sample=naction, timestep=k, global_cond=obs_cond
                )
                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred, timestep=k, sample=naction
                ).prev_sample
            self.get_logger().info(f"time elapsed: {time.time() - start_time}")

        naction = to_numpy(get_action(naction))
        sampled_actions_msg = Float32MultiArray()
        sampled_actions_msg.data = np.concatenate(
            (np.array([0]), naction.flatten())
        ).tolist()
        self.get_logger().info("published sampled actions")
        self.sampled_actions_publisher.publish(sampled_actions_msg)
        naction = naction[0]
        chosen_waypoint = naction[self.args.waypoint]

        return chosen_waypoint

    def _process_other_model(self):
        """Process using non-nomad models"""
        start = max(self.closest_node - self.args.radius, 0)
        end = min(self.closest_node + self.args.radius + 1, self.goal_node)
        distances = []
        waypoints = []
        batch_obs_imgs = []
        batch_goal_data = []

        for i, sg_img in enumerate(self.topomap[start : end + 1]):
            transf_obs_img = transform_images(
                self.context_queue, self.model_params["image_size"]
            )
            goal_data = transform_images(sg_img, self.model_params["image_size"])
            batch_obs_imgs.append(transf_obs_img)
            batch_goal_data.append(goal_data)

        # predict distances and waypoints
        batch_obs_imgs = torch.cat(batch_obs_imgs, dim=0).to(device)
        batch_goal_data = torch.cat(batch_goal_data, dim=0).to(device)

        distances, waypoints = self.model(batch_obs_imgs, batch_goal_data)
        distances = to_numpy(distances)
        waypoints = to_numpy(waypoints)

        # look for closest node
        min_dist_idx = np.argmin(distances)

        # chose subgoal and output waypoints
        if distances[min_dist_idx] > self.args.close_threshold:
            chosen_waypoint = waypoints[min_dist_idx][self.args.waypoint]
            self.closest_node = start + min_dist_idx
        else:
            chosen_waypoint = waypoints[min(min_dist_idx + 1, len(waypoints) - 1)][
                self.args.waypoint
            ]
            self.closest_node = min(start + min_dist_idx + 1, self.goal_node)

        self.get_logger().info(f"Distances: {distances}")
        self.get_logger().info(f"Chosen waypoint: {chosen_waypoint}")
        self.get_logger().info(f"Closest node: {self.closest_node}")

        return chosen_waypoint


def main(args: argparse.Namespace):
    # Initialize ROS2
    rclpy.init()

    # Create node
    exploration_node = ExplorationNode(args)

    try:
        # Spin the node
        rclpy.spin(exploration_node)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean shutdown
        exploration_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ViNT navigation with ROS2 or offline images"
    )
    parser.add_argument(
        "--model",
        "-m",
        default="nomad",
        type=str,
        help="model name (only nomad is supported) (hint: check ../config/models.yaml) (default: nomad)",
    )
    parser.add_argument(
        "--waypoint",
        "-w",
        default=2,  # close waypoints exhibit straight line motion (the middle waypoint is a good default)
        type=int,
        help=f"""index of the waypoint used for navigation (between 0 and 4 or 
        how many waypoints your model predicts) (default: 2)""",
    )
    parser.add_argument(
        "--dir",
        "-d",
        default="topomap",
        type=str,
        help="path to topomap images",
    )

    parser.add_argument(
        "--offline-dir",
        type=str,
        default=None,
        help="If set, load iamges from this directory instead of ROS topic",
    )

    parser.add_argument(
        "--goal-node",
        "-g",
        default=-1,
        type=int,
        help="""goal node index in the topomap (if -1, then the goal node is 
        the last node in the topomap) (default: -1)""",
    )
    parser.add_argument(
        "--close-threshold",
        "-t",
        default=3,
        type=int,
        help="""temporal distance within the next node in the topomap before 
        localizing to it (default: 3)""",
    )
    parser.add_argument(
        "--radius",
        "-r",
        default=4,
        type=int,
        help="""temporal number of local nodes to look at in the topomap for
        localization (default: 4)""",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        default=8,
        type=int,
        help=f"Number of actions sampled from the exploration model (default: 8)",
    )
    args = parser.parse_args()
    print(f"Using {device}")
    main(args)
