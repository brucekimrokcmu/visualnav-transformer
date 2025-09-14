#!/bin/bash

# Create a new tmux session
session_name="vint_locobot_$(date +%s)"
tmux new-session -d -s $session_name

# Split the window into four panes
tmux selectp -t 0 # select the first (0) pane
tmux splitw -h -p 50 # split it into two halves
tmux selectp -t 0 # select the first (0) pane
tmux splitw -v -p 50 # split it into two halves
tmux selectp -t 2 # select the new, second (2) pane
tmux splitw -v -p 50 # split it into two halves
tmux selectp -t 0 # go back to the first pane

# Function to setup environment in each pane
setup_env_commands="source /opt/ros/jazzy/setup.bash && \
source ~/tools/miniconda3/etc/profile.d/conda.sh && \
conda activate vint_deployment && \
export PYTHONPATH=~/projects/visualnav-transformer:~/projects/visualnav-transformer/diffusion_policy:\$CONDA_PREFIX/lib/python3.12/site-packages:\$PYTHONPATH"

# Pane 0: Launch file (ROS2 only)
tmux select-pane -t 0
tmux send-keys "source /opt/ros/jazzy/setup.bash" Enter
# Uncomment and modify if you have a ROS2 launch file
# tmux send-keys "ros2 launch your_package your_launch_file.py" Enter
tmux send-keys "echo 'ROS2 Launch pane ready. Start your launch files here.'" Enter

# Pane 1: Navigation script
tmux select-pane -t 1
tmux send-keys "$setup_env_commands && python ~/projects/visualnav-transformer/deployment/src/navigate.py --model vint --dir escuela --offline-dir escuela $@" Enter

# # Pane 2: Teleop script  
# tmux select-pane -t 2
# tmux send-keys "$setup_env_commands && python joy_teleop.py" Enter

# # Pane 3: PD Controller script
# tmux select-pane -t 3
# tmux send-keys "$setup_env_commands && python pd_controller.py" Enter

# Attach to the tmux session
tmux -2 attach-session -t $session_name


# source /opt/ros/jazzy/setup.bash
# source ~/tools/miniconda3/etc/profile.d/conda.sh
# conda activate vint_deployment
# export PYTHONPATH=~/projects/visualnav-transformer:~/projects/visualnav-transformer/diffusion_policy:$CONDA_PREFIX/lib/python3.12/site-packages:$PYTHONPATH
# python ~/projects/visualnav-transformer/deployment/src/navigate.py --model vint --dir escuela
