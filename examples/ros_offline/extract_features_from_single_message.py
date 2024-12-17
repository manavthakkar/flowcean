"""
Extract features from individual messages in the particle cloud (for testing)
"""
import polars as pl
from features import *
import sys
sys.setrecursionlimit(2500)

data = pl.read_json("cached_ros_data.json") # polars dataframe

particle_cloud = data[0,3] # polars series of dictionaries (number of dictionaries = number of messages)

print(f"Number of messages in the particle cloud are {len(particle_cloud)}") # number of messages in the particle cloud

print(f"Type of particle cloud is {type(particle_cloud)}")

message_number = 2 # message number to extract features from 

message_dictionary = particle_cloud[message_number] # dictionary of two keys: "time" and "value" (one message) 

print(f"Type of message dictionary is {type(message_dictionary)}")

time = message_dictionary["time"] # time of the message

print(f"Time of message {message_number} is {time}")

particles_dict = message_dictionary["value"]

list_of_particles = particles_dict["particles"] # list of dictionaries (number of dictionaries = number of particles) i.e in this case, list of particles in the first message

features = extract_features_from_message(list_of_particles) # extract features from the list of particles

print(f"Features extracted from the message {message_number} are {features}")

