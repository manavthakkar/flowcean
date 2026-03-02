from custom_transforms.scan_map_statistics import ScanMapStatistics

import flowcean.cli
from flowcean.ros import load_rosbag

config = flowcean.cli.initialize()

topics = {
    "/amcl_pose": [
        "pose.pose.position.x",
        "pose.pose.position.y",
        "pose.pose.orientation.x",
        "pose.pose.orientation.y",
        "pose.pose.orientation.z",
        "pose.pose.orientation.w",
    ],
    "/momo/pose": [
        "pose.position.x",
        "pose.position.y",
        "pose.orientation.x",
        "pose.orientation.y",
        "pose.orientation.z",
        "pose.orientation.w",
    ],
    "/scan": [
        "ranges",
        "angle_min",
        "angle_max",
        "angle_increment",
        "range_min",
        "range_max",
    ],
    "/map": [
        "data",
        "info.resolution",
        "info.width",
        "info.height",
        "info.origin.position.x",
        "info.origin.position.y",
        "info.origin.position.z",
        "info.origin.orientation.x",
        "info.origin.orientation.y",
        "info.origin.orientation.z",
        "info.origin.orientation.w",
    ],
    "/delocalizations": ["data"],
    "/particle_cloud": ["particles"],
}

# message_paths = config.rosbag.message_pathss
# bag_file_path = config.rosbag.training_paths[0]

message_paths = [
    "ros_msgs/sensor_msgs/msg/LaserScan.msg",
    "ros_msgs/nav2_msgs/msg/Particle.msg",
    "ros_msgs/nav2_msgs/msg/ParticleCloud.msg",
]

# bag_file_path = "/home/workstation/ros2_ws/src/flowcean/examples/robot_localization_failure/recordings/test_data/test_1/training_data_half/rec_20250923_135805_id_01" # works with this bag file

bag_file_path = "/home/workstation/ros2_ws/src/flowcean/examples/robot_localization_failure/recordings/test_data/colloseum/data_half/rec_20251202_160624_id_01"  # throws an error with this bag file


# Load raw lazy frame
raw_lf = load_rosbag(bag_file_path, topics, message_paths=message_paths)

print(raw_lf.collect())
# print(raw_lf.collect().schema)

# -----------------------------
# Extract occupancy map
# -----------------------------
map_df = raw_lf.select("/map").collect()
map_ts = map_df["/map"][0]
map_value = map_ts[0]["value"]

occupancy_map = {
    "data": map_value["data"],
    "info.width": map_value["info.width"],
    "info.height": map_value["info.height"],
    "info.resolution": map_value["info.resolution"],
    "info.origin.position.x": map_value["info.origin.position.x"],
    "info.origin.position.y": map_value["info.origin.position.y"],
    "info.origin.position.z": map_value["info.origin.position.z"],
    "info.origin.orientation.x": map_value["info.origin.orientation.x"],
    "info.origin.orientation.y": map_value["info.origin.orientation.y"],
    "info.origin.orientation.z": map_value["info.origin.orientation.z"],
    "info.origin.orientation.w": map_value["info.origin.orientation.w"],
}

# -----------------------------
# ScanMapStatistics
# -----------------------------
sms = ScanMapStatistics(
    occupancy_map=occupancy_map,
    scan_topic="/scan",
    sensor_pose_topic="/amcl_pose",
)
with_sms = sms.apply(raw_lf)

print(with_sms.collect())
