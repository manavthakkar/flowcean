import polars as pl
from features import *
import sys
sys.setrecursionlimit(1000000)

data = pl.read_json("cached_ros_data.json") # polars dataframe

particle_cloud = data[0,3] # polars series of dictionaries (number of dictionaries = number of messages)

number_of_messages = len(particle_cloud)

all_features = []  # This will store a dictionary of features per message

for i in range(number_of_messages):        # change the range for testing
    # Extract the i-th message dictionary
    message_dictionary = particle_cloud[i]

    # Extract time and value information
    time = message_dictionary["time"]
    particles_dict = message_dictionary["value"]
    list_of_particles = particles_dict["particles"]

    # Extract features for this message
    features = extract_features_from_message(list_of_particles)

    # Add the time to the features dictionary 
    features["time"] = time

    # Append the features to the list
    all_features.append(features)

print(f"Features extracted from all messages are: \n{all_features}")

features_df = pl.DataFrame(all_features)

# print(features_df.head())
# print(features_df.shape)

# features_df.write_json("all_messages_features.json") 

time_values = features_df["time"].to_list()

new_data = {}

# Iterate over every column except "time"
for col in features_df.columns:
    if col == "time":
        continue

    # Extract the feature values
    feature_values = features_df[col].to_list()

    # Combine each feature value with its corresponding time into a dictionary
    dict_list = []
    for t, val in zip(time_values, feature_values):
        dict_list.append({"time": t, "value": val})

    # Create a polars Series from this list of dictionaries
    # This will be an object column since it's storing Python dictionaries.
    new_data[col] = pl.Series(col, dict_list)

final_df = pl.DataFrame(new_data)

print(final_df.head())
print(final_df.shape)


