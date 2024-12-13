"""
data (polars.DataFrame)
  ├── particle_cloud (polars.Series)
  │      ├── message_data (dictionary)
  │      │      ├── "time" (int)
  │      │      └── "value" (inner_dict)
  │      │             └── "particle_dict" ("particles <str>": list_of_dict_of_particles)
  │      │                    ├── particle_data (dictionary)
  │      │                    │      ├── "pose" (dictionary)
  │      │                    │      │      ├── "position" (dictionary)
  │      │                    │      │      │      ├── "x" (float)
  │      │                    │      │      │      ├── "y" (float)
  │      │                    │      │      │      ├── "z" (float)
  │      │                    │      │      │      └── "__msgtype__" (string: "geometry_msgs/msg/Point")
  │      │                    │      │      ├── "orientation" (dictionary)
  │      │                    │      │      │      ├── "x" (float)
  │      │                    │      │      │      ├── "y" (float)
  │      │                    │      │      │      ├── "z" (float)
  │      │                    │      │      │      ├── "w" (float)
  │      │                    │      │      │      └── "__msgtype__" (string: "geometry_msgs/msg/Quaternion")
  │      │                    │      │      └── "__msgtype__" (string: "geometry_msgs/msg/Pose")
  │      │                    │      ├── "weight" (float)
  │      │                    │      └── "__msgtype__" (string: "nav2_msgs/msg/Particle")

"""

import polars as pl

data = pl.read_json("cached_ros_data.json") # this includes all the columns (topics) that were specified in the RosbagLoader

particle_cloud = data[0,3] # polars series of dictionaries (number of dictionaries = number of messages)

message_number = 0 # first message

message_data = particle_cloud[message_number] # dictionary of two keys: "time" and "value" (one message) i.e. first message in this case

time = message_data["time"] # time of the message

print(f"Time of message {message_number} is {time}")

particles_dict = message_data["value"] # dictionary of one key: "particles" (list of particles)

list_of_dict_of_particles = particles_dict["particles"] # list of dictionaries (number of dictionaries = number of particles)

print(f"Number of particles in message {message_number} are {len(list_of_dict_of_particles)}")

particle_number = 0 # first particle

particle_data = list_of_dict_of_particles[particle_number] # dictionary of three keys: "pose", "weight" and "__msgtype__" (one particle) i.e. first particle in this case

pose = particle_data["pose"] # dictionary of three keys: "position", "orientation" and "__msgtype__"

position = pose["position"] # dictionary of four keys: "x", "y", "z" and "__msgtype__"

orientation = pose["orientation"] # dictionary of five keys: "x", "y", "z", "w" and "__msgtype__"

print(f"Position of particle {particle_number} in message {message_number} is x: {position['x']}, y: {position['y']}, z: {position['z']}")

print(f"Orientation of particle {particle_number} in message {message_number} is x: {orientation['x']}, y: {orientation['y']}, z: {orientation['z']}, w: {orientation['w']}")

print(f"Weight of particle {particle_number} in message {message_number} is {particle_data['weight']}")







