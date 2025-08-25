from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import _init_rerun
from lerobot.record import record_loop

NUM_EPISODES = 10    # 录制次数（5次）
FPS = 30            # 数据采集帧率（每秒30帧） 
EPISODE_TIME_SEC = 20   # 单次录制持续时间（60秒）
RESET_TIME_SEC = 5     # 环境重置时间（5秒）
TASK_DESCRIPTION = "Grab a pack of paper"

# Create the robot and teleoperator configurations
camera_config = {
    "top": RealSenseCameraConfig(
        serial_number_or_name="146222252404",
        fps=FPS,
        width=640,
        height=480,
        color_mode=ColorMode.RGB,
        use_depth=True,
        rotation=Cv2Rotation.NO_ROTATION
    ),
    "front": RealSenseCameraConfig(
        serial_number_or_name="146222254779",
        fps=FPS,
        width=640,
        height=480,
        color_mode=ColorMode.RGB,
        use_depth=True,
        rotation=Cv2Rotation.NO_ROTATION
    ),
}  # 图像旋转设置（无旋转）

robot_config = SO101FollowerConfig(
    port="/dev/ttyACM1", 
    id="my_awesome_follower_arm", 
    cameras=camera_config
)
teleop_config = SO101LeaderConfig(
    port="/dev/ttyACM0", 
    id="my_awesome_leader_arm"
)

# Initialize the robot and teleoperator
robot = SO101Follower(robot_config)
teleop = SO101Leader(teleop_config)

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# Create the dataset
dataset = LeRobotDataset.create(
    repo_id="Gutilence/demo14",
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,            # 存储为视频而非独立图片
    image_writer_threads=4,
)

# Initialize the keyboard listener and rerun visualization
_, events = init_keyboard_listener()
_init_rerun(session_name="recording")

# Connect the robot and teleoperator
robot.connect(calibrate=False)
teleop.connect(calibrate=False)

episode_idx = 0
while episode_idx < NUM_EPISODES and not events["stop_recording"]:
    print(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        teleop=teleop,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
    )

    # Reset the environment if not stopping or re-recording
    if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
        print("Reset the environment")
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop=teleop,
            control_time_s=RESET_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
        )

    if events["rerecord_episode"]:
        log_say("Re-recording episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        continue

    dataset.save_episode()
    episode_idx += 1

# Clean up
log_say("Stop recording")
robot.disconnect()
teleop.disconnect()
# dataset.push_to_hub()