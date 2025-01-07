import sys

import polars as pl
from features import extract_features_from_message

import particle_features # cpp module
from particle_features import Particle # cpp 

import time

sys.setrecursionlimit(1000000)

data = pl.read_json("./examples/ros_offline/cached_ros_data.json")

particle_cloud = data[0, 3]

number_of_messages = len(particle_cloud)

all_features = []

start_time = time.time()

for i in range(number_of_messages):
    print(f"Processing message {i+1}/{number_of_messages}")
    message_dictionary = particle_cloud[i]

    message_time = message_dictionary["time"]
    particles_dict = message_dictionary["value"]
    list_of_particles = particles_dict["particles"]

    # Convert Python dictionary to C++ Particle objects
    particles = []
    for p in list_of_particles:
        particle = Particle()  # Create an empty Particle object
        particle.x = p["pose"]["position"]["x"]
        particle.y = p["pose"]["position"]["y"]
        particle.weight = p["weight"]
        particles.append(particle)

    # features = extract_features_from_message(       # Python function
    #     list_of_particles, eps=0.3, min_samples=5
    # )

    features = particle_features.extract_all_features_cpp(particles) # C++ function

    features["time"] = message_time
    all_features.append(features)

features_df = pl.DataFrame(all_features)
time_values = features_df["time"].to_list()

new_data = {}

for col in features_df.columns:
    if col == "time":
        continue

    # Extract the feature values
    feature_values = features_df[col].to_list()

    # Combine each feature value with its corresponding time into a dictionary
    dict_list = [
        {"time": t, "value": val}
        for t, val in zip(time_values, feature_values, strict=False)
    ]

    # Put the entire dict_list as a single entry - a list of all structs.
    new_data[col] = [dict_list]

final_df = pl.DataFrame(new_data)

end_time = time.time()

print(f"Time taken: {end_time - start_time} seconds")

print(final_df.head())
print(final_df.shape)
