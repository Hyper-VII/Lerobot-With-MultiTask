from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation

# Create a `RealSenseCameraConfig` specifying your camera’s serial number and enabling depth.
config = RealSenseCameraConfig(
    serial_number_or_name="146222252404",
    fps=15,
    width=640,
    height=480,
    color_mode=ColorMode.RGB,
    use_depth=True,
    rotation=Cv2Rotation.NO_ROTATION
)

# Instantiate and connect a `RealSenseCamera` with warm-up read (default).
camera = RealSenseCamera(config)
camera.connect()

# Capture a color frame via `read()` and a depth map via `read_depth()`.
try:
    color_frame = camera.read()
    depth_map = camera.read_depth()
    print("Color frame shape:", color_frame.shape)
    print("Depth map shape:", depth_map.shape)
finally:
    camera.disconnect()