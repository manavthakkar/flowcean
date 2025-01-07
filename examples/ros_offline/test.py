import features  # Import the Python module
from particle_features import Particle
import particle_features
# from particle_features import cog_max_dist, cog_mean_dist, cog_mean_absolute_deviation, cog_median, Particle  # Import the C++ functions
# from particle_features import cog_median_absolute_deviation, cog_min_dist
# from particle_features import cog_standard_deviation, smallest_enclosing_circle, circle_mean, circle_mean_absolute_deviation
# from particle_features import circle_median, circle_median_absolute_deviation, circle_min_dist, circle_std_deviation

list_of_particles = [
    {
        "pose": {
            "position": {
                "x": 2.7026532109185792,
                "y": 1.3363095842400234,
                "z": 0.0,
                "__msgtype__": "geometry_msgs/msg/Point",
            },
            "orientation": {
                "x": 0.0,
                "y": 0.0,
                "z": -0.6206412601751722,
                "w": 0.7840946538321596,
                "__msgtype__": "geometry_msgs/msg/Quaternion",
            },
            "__msgtype__": "geometry_msgs/msg/Pose",
        },
        "weight": 0.0005980861244019139,
        "__msgtype__": "nav2_msgs/msg/Particle",
    },
    {
        "pose": {
            "position": {
                "x": 2.9070964865479705,
                "y": 3.0649213798266697,
                "z": 0.0,
                "__msgtype__": "geometry_msgs/msg/Point",
            },
            "orientation": {
                "x": 0.0,
                "y": 0.0,
                "z": -0.45132518076103845,
                "w": 0.8923595582560967,
                "__msgtype__": "geometry_msgs/msg/Quaternion",
            },
            "__msgtype__": "geometry_msgs/msg/Pose",
        },
        "weight": 0.0005980861244019139,
        "__msgtype__": "nav2_msgs/msg/Particle",
    },
    {
        "pose": {
            "position": {
                "x": 2.80871858542121,
                "y": 1.5363776884978138,
                "z": 0.0,
                "__msgtype__": "geometry_msgs/msg/Point",
            },
            "orientation": {
                "x": 0.0,
                "y": 0.0,
                "z": -0.36432616851598243,
                "w": 0.9312714120676442,
                "__msgtype__": "geometry_msgs/msg/Quaternion",
            },
            "__msgtype__": "geometry_msgs/msg/Pose",
        },
        "weight": 0.0005980861244019139,
        "__msgtype__": "nav2_msgs/msg/Particle",
    },
    {
        "pose": {
            "position": {
                "x": 1.8221955477463578,
                "y": 1.6169840054666116,
                "z": 0.0,
                "__msgtype__": "geometry_msgs/msg/Point",
            },
            "orientation": {
                "x": 0.0,
                "y": 0.0,
                "z": -0.584478714347991,
                "w": 0.8114090414052085,
                "__msgtype__": "geometry_msgs/msg/Quaternion",
            },
            "__msgtype__": "geometry_msgs/msg/Pose",
        },
        "weight": 0.0005980861244019139,
        "__msgtype__": "nav2_msgs/msg/Particle",
    },
    {
        "pose": {
            "position": {
                "x": 2.12472141189225,
                "y": 1.5361849999975508,
                "z": 0.0,
                "__msgtype__": "geometry_msgs/msg/Point",
            },
            "orientation": {
                "x": 0.0,
                "y": 0.0,
                "z": -0.4347883702383812,
                "w": 0.900532660765534,
                "__msgtype__": "geometry_msgs/msg/Quaternion",
            },
            "__msgtype__": "geometry_msgs/msg/Pose",
        },
        "weight": 0.0005980861244019139,
        "__msgtype__": "nav2_msgs/msg/Particle",
    },
]

# Convert Python dictionary to C++ Particle objects
particles = []
for p in list_of_particles:
    particle = Particle()  # Create an empty Particle object
    particle.x = p["pose"]["position"]["x"]
    particle.y = p["pose"]["position"]["y"]
    particle.weight = p["weight"]
    particles.append(particle)

"""
# Python - Feature 1
max_distance, furthest_particle = features.cog_max_dist(list_of_particles)
print(max_distance)      # 1.32
print(furthest_particle) # (2.907, 3.064)

# C++ - Feature 1
max_distance_cpp, furthest_particle_cpp = particle_features.cog_max_dist(particles)
print(f"Max Distance: {max_distance_cpp}")
print(f"Furthest Particle: {furthest_particle_cpp}")

# Python - Feature 2
mean_distance = features.cog_mean_dist(list_of_particles)
print(f"Mean Distance (Python): {mean_distance}")

# C++ - Feature 2
mean_distance_cpp = particle_features.cog_mean_dist(particles)
print(f"Mean Distance (C++): {mean_distance_cpp}")

# Python - Feature 3
mad_distance = features.cog_mean_absolute_deviation(list_of_particles)
print(f"Mean Absolute Deviation (Python): {mad_distance}")

# C++ - Feature 3
mad_distance_cpp = particle_features.cog_mean_absolute_deviation(particles)
print(f"Mean Absolute Deviation (C++): {mad_distance_cpp}")

# Python - Feature 4
median_distance = features.cog_median(list_of_particles)
print(f"Median Distance (Python): {median_distance}")

# C++ - Feature 4
median_distance_cpp = particle_features.cog_median(particles)
print(f"Median Distance (C++): {median_distance_cpp}")

# Python - Feature 5
mad_distance = features.cog_median_absolute_deviation(list_of_particles)
print(f"Median Absolute Deviation (Python): {mad_distance}")

# C++ - Feature 5
mad_distance_cpp = particle_features.cog_median_absolute_deviation(particles)
print(f"Median Absolute Deviation (C++): {mad_distance_cpp}")

# Python - Feature 6
min_distance, closest_particle = features.cog_min_dist(list_of_particles)
print(f"Min Distance (Python): {min_distance}")
print(f"Closest Particle (Python): {closest_particle}")

# C++ - Feature 6
min_distance_cpp, closest_particle_cpp = particle_features.cog_min_dist(particles)
print(f"Min Distance (C++): {min_distance_cpp}")
print(f"Closest Particle (C++): {closest_particle_cpp}")

# Python - Feature 7
std_dev_distance = features.cog_standard_deviation(list_of_particles)
print(f"Standard Deviation (Python): {std_dev_distance}")

# C++ - Feature 7
std_dev_distance_cpp = particle_features.cog_standard_deviation(particles)
print(f"Standard Deviation (C++): {std_dev_distance_cpp}")

# Python - Feature 8
points = [(p["pose"]["position"]["x"], p["pose"]["position"]["y"]) for p in list_of_particles]
circle_center, circle_radius = features.smallest_enclosing_circle(points)
print(f"Circle Center (Python): {circle_center}")
print(f"Circle Radius (Python): {circle_radius}")

# C++ - Feature 8
circle_center_cpp, circle_radius_cpp = particle_features.smallest_enclosing_circle(points)
print(f"Circle Center (C++): {circle_center_cpp}")
print(f"Circle Radius (C++): {circle_radius_cpp}")

# Python - Feature 9
mean_distance = features.circle_mean(list_of_particles)
print(f"Mean Distance (Python): {mean_distance}")

# C++ - Feature 9
points = [(p["pose"]["position"]["x"], p["pose"]["position"]["y"]) for p in list_of_particles]
mean_distance_cpp = particle_features.circle_mean(points)
print(f"Mean Distance (C++): {mean_distance_cpp}")

# Python - Feature 10
mad_distance = features.circle_mean_absolute_deviation(list_of_particles)
print(f"Mean Absolute Deviation (Python): {mad_distance}")

# C++ - Feature 10
points = [(p["pose"]["position"]["x"], p["pose"]["position"]["y"]) for p in list_of_particles]
mad_distance_cpp = particle_features.circle_mean_absolute_deviation(points)
print(f"Mean Absolute Deviation (C++): {mad_distance_cpp}")

# Python - Feature 11
median_distance = features.circle_median(list_of_particles)
print(f"Median Distance (Python): {median_distance}")

# C++ - Feature 11
points = [(p["pose"]["position"]["x"], p["pose"]["position"]["y"]) for p in list_of_particles]
median_distance_cpp = particle_features.circle_median(points)
print(f"Median Distance (C++): {median_distance_cpp}")

# Python - Feature 12
mad_distance = features.circle_median_absolute_deviation(list_of_particles)
print(f"Median Absolute Deviation (Python): {mad_distance}")

# C++ - Feature 12
points = [(p["pose"]["position"]["x"], p["pose"]["position"]["y"]) for p in list_of_particles]
mad_distance_cpp = particle_features.circle_median_absolute_deviation(points)
print(f"Median Absolute Deviation (C++): {mad_distance_cpp}")

# Python - Feature 13
min_distance = features.circle_min_dist(list_of_particles)
print(f"Min Distance (Python): {min_distance}")

# C++ - Feature 13
points = [(p["pose"]["position"]["x"], p["pose"]["position"]["y"]) for p in list_of_particles]
min_distance_cpp = particle_features.circle_min_dist(points)
print(f"Min Distance (C++): {min_distance_cpp}")


# Python - Feature 14
std_dev_distance = features.circle_std_deviation(list_of_particles)
print(f"Standard Deviation (Python): {std_dev_distance}")

# C++ - Feature 14
points = [(p["pose"]["position"]["x"], p["pose"]["position"]["y"]) for p in list_of_particles]
std_dev_distance_cpp = particle_features.circle_std_deviation(points)
print(f"Standard Deviation (C++): {std_dev_distance_cpp}")

# Extract all features at once in Python (uses the C++ functions)
def extract_all_features_cpp(particles):
    features = {}
    features["cog_max_dist"] = particle_features.cog_max_dist(particles)[0]
    features["cog_mean_dist"] = particle_features.cog_mean_dist(particles)
    features["cog_mean_absolute_deviation"] = particle_features.cog_mean_absolute_deviation(particles)
    features["cog_median"] = particle_features.cog_median(particles)
    features["cog_median_absolute_deviation"] = particle_features.cog_median_absolute_deviation(particles)
    features["cog_min_dist"] = particle_features.cog_min_dist(particles)[0]
    features["cog_standard_deviation"] = particle_features.cog_standard_deviation(particles)
    points = [(p.x, p.y) for p in particles]
    features["circle_radius"] = particle_features.smallest_enclosing_circle(points)[1]
    features["circle_mean"] = particle_features.circle_mean(points)
    features["circle_mean_absolute_deviation"] = particle_features.circle_mean_absolute_deviation(points)
    features["circle_median"] = particle_features.circle_median(points)
    features["circle_median_absolute_deviation"] = particle_features.circle_median_absolute_deviation(points)
    features["circle_min_dist"] = particle_features.circle_min_dist(points)
    features["circle_std_deviation"] = particle_features.circle_std_deviation(points)
    return features

features_cpp = extract_all_features_cpp(particles)
print(features_cpp)
"""

# Extract all features at once from Python (uses the Python functions)
features_py = features.extract_features_from_message(list_of_particles)
print(features_py)


# Extract all features at once from cpp (everything is done in C++)
extracted_features_pybind = particle_features.extract_all_features_cpp(particles)
print(extracted_features_pybind)