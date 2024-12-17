"""
This script calculates/extracts features 1-14 from a single message from the ROS2 bag file
"""

import numpy as np
import random
import math

################# Features #################

# Feature 1
def cog_max_dist(list_of_particles):
    """
    Calculates the maximum distance from any particle to the center of gravity.

    Parameters:
    list_of_particles (list): List of dictionaries representing particles.

    Returns:
    tuple: The maximum distance and the coordinates of the particle furthest from the COG.
    """
    cog = calculate_center_of_gravity(list_of_particles)
    cog_x, cog_y = cog['x'], cog['y']

    max_distance = 0
    furthest_particle = None
    for particle in list_of_particles:
        px = particle['pose']['position']['x']
        py = particle['pose']['position']['y']
        distance = np.sqrt((px - cog_x)**2 + (py - cog_y)**2)
        if distance > max_distance:
            max_distance = distance
            furthest_particle = (px, py)

    return max_distance, furthest_particle

# Feature 2
def calculate_cog_mean(list_of_particles):
    """
    Calculates the mean position (center of gravity mean) over all particles.

    Parameters:
    list_of_particles (list): List of dictionaries representing particles.

    Returns:
    dict: A dictionary with the x and y coordinates of the mean position.
    """
    num_particles = len(list_of_particles)
    if num_particles == 0:
        return {'x': 0, 'y': 0}  # Default to origin if no particles

    mean_x = sum(particle['pose']['position']['x'] for particle in list_of_particles) / num_particles
    mean_y = sum(particle['pose']['position']['y'] for particle in list_of_particles) / num_particles

    return {'x': mean_x, 'y': mean_y}

# Feature 3
def calculate_cog_mean_absolute_deviation(list_of_particles):
    """
    Calculates the mean absolute deviation of distances to the center of gravity mean.

    Parameters:
    list_of_particles (list): List of dictionaries representing particles.

    Returns:
    float: The mean absolute deviation of distances.
    """
    cog_mean = calculate_cog_mean(list_of_particles)
    mean_x, mean_y = cog_mean['x'], cog_mean['y']

    # Calculate distances to the mean
    distances = [
        np.sqrt((particle['pose']['position']['x'] - mean_x)**2 + (particle['pose']['position']['y'] - mean_y)**2)
        for particle in list_of_particles
    ]

    # Calculate mean absolute deviation
    mean_distance = sum(distances) / len(distances)
    mean_absolute_deviation = sum(abs(d - mean_distance) for d in distances) / len(distances)

    return mean_absolute_deviation

# Feature 4
def calculate_cog_median(list_of_particles):
    """
    Calculates the median of distances from all particles to the center of gravity mean.

    Parameters:
    list_of_particles (list): List of dictionaries representing particles.

    Returns:
    float: The median distance from particles to the COG mean.
    """
    cog_mean = calculate_cog_mean(list_of_particles)
    mean_x, mean_y = cog_mean['x'], cog_mean['y']

    # Calculate distances to the mean
    distances = [
        np.sqrt((particle['pose']['position']['x'] - mean_x)**2 + (particle['pose']['position']['y'] - mean_y)**2)
        for particle in list_of_particles
    ]

    # Sort distances and calculate median
    distances.sort()
    n = len(distances)
    if n == 0:
        return 0  # Default to 0 if no particles

    if n % 2 == 1:
        return distances[n // 2]  # Middle element for odd number of distances
    else:
        return (distances[n // 2 - 1] + distances[n // 2]) / 2  # Average of two middle elements
    
# Feature 5
def calculate_cog_median_absolute_deviation(list_of_particles):
    """
    Calculates the median absolute deviation (MAD) from the median distance of particles to the center of gravity mean.

    Parameters:
    list_of_particles (list): List of dictionaries representing particles.

    Returns:
    float: The median absolute deviation of distances.
    """
    # calculate_cog_median to get the median distance
    median_distance = calculate_cog_median(list_of_particles)

    cog_mean = calculate_cog_mean(list_of_particles)
    mean_x, mean_y = cog_mean['x'], cog_mean['y']

    # Calculate distances to the mean
    distances = [
        np.sqrt((particle['pose']['position']['x'] - mean_x)**2 + (particle['pose']['position']['y'] - mean_y)**2)
        for particle in list_of_particles
    ]

    # Calculate absolute deviations from the median distance
    absolute_deviations = [abs(d - median_distance) for d in distances]

    # Calculate the median of the absolute deviations
    absolute_deviations.sort()
    n_dev = len(absolute_deviations)
    if n_dev == 0:
        return 0  # Default to 0 if no deviations

    if n_dev % 2 == 1:
        return absolute_deviations[n_dev // 2]
    else:
        return (absolute_deviations[n_dev // 2 - 1] + absolute_deviations[n_dev // 2]) / 2

# Feature 6
def cog_min_dist(list_of_particles):
    """
    Calculates the minimum distance from any particle to the center of gravity.

    Parameters:
    list_of_particles (list): List of dictionaries representing particles.

    Returns:
    tuple: The minimum distance and the coordinates of the particle closest to the COG.
    """
    cog = calculate_center_of_gravity(list_of_particles)
    cog_x, cog_y = cog['x'], cog['y']

    min_distance = float('inf')
    closest_particle = None
    for particle in list_of_particles:
        px = particle['pose']['position']['x']
        py = particle['pose']['position']['y']
        distance = np.sqrt((px - cog_x)**2 + (py - cog_y)**2)
        if distance < min_distance:
            min_distance = distance
            closest_particle = (px, py)

    return min_distance, closest_particle

# Feature 7
def calculate_cog_standard_deviation(list_of_particles):
    """
    Calculates the standard deviation of distances to the center of gravity mean.

    Parameters:
    list_of_particles (list): List of dictionaries representing particles.

    Returns:
    float: The standard deviation of distances.
    """
    cog_mean = calculate_cog_mean(list_of_particles)
    mean_x, mean_y = cog_mean['x'], cog_mean['y']

    # Calculate distances to the mean
    distances = [
        np.sqrt((particle['pose']['position']['x'] - mean_x)**2 + (particle['pose']['position']['y'] - mean_y)**2)
        for particle in list_of_particles
    ]

    # Calculate variance
    variance = sum((d - np.mean(distances))**2 for d in distances) / len(distances)

    # Standard deviation is the square root of variance
    return np.sqrt(variance)

# Feature 8
def smallest_enclosing_circle(points):
    """Find the smallest enclosing circle for a set of points."""
    shuffled_points = points[:]
    random.shuffle(shuffled_points)
    return welzl(shuffled_points)

# Feature 9
def circle_mean(list_of_particles):
    """Calculate the mean of distances from the circle center to the points."""
    points = [(particle['pose']['position']['x'], particle['pose']['position']['y']) for particle in list_of_particles]
    circle = smallest_enclosing_circle(points)
    center, _ = circle

    total_distance = sum(dist(center, point) for point in points)
    mean_distance = total_distance / len(points)
    return mean_distance

# Feature 10
def circle_mean_absolute_deviation(list_of_particles):
    """Calculate the mean absolute deviation of distances from the circle center."""
    points = [(particle['pose']['position']['x'], particle['pose']['position']['y']) for particle in list_of_particles]
    circle = smallest_enclosing_circle(points)
    center, _ = circle

    # Compute mean distance first
    distances = [dist(center, point) for point in points]
    mean_distance = sum(distances) / len(points)

    # Compute mean absolute deviation
    mean_absolute_deviation = sum(abs(d - mean_distance) for d in distances) / len(points)
    return mean_absolute_deviation

# Feature 11
def circle_median(list_of_particles):
    """Calculate the median of distances from the circle center to the points."""
    points = [(particle['pose']['position']['x'], particle['pose']['position']['y']) for particle in list_of_particles]
    circle = smallest_enclosing_circle(points)
    center, _ = circle

    # Compute distances and sort them
    distances = sorted(dist(center, point) for point in points)
    n = len(distances)

    # Find median
    if n % 2 == 1:  # Odd number of points
        return distances[n // 2]
    else:  # Even number of points
        return (distances[n // 2 - 1] + distances[n // 2]) / 2
    
# Feature 12
def circle_median_absolute_deviation(list_of_particles):
    """Calculate the median absolute deviation (MAD) of distances from the circle center."""
    points = [(particle['pose']['position']['x'], particle['pose']['position']['y']) for particle in list_of_particles]
    circle = smallest_enclosing_circle(points)
    center, _ = circle

    # Compute distances and the median distance
    distances = [dist(center, point) for point in points]
    median_distance = circle_median(list_of_particles)  

    # Compute median absolute deviation
    abs_deviation = sorted(abs(d - median_distance) for d in distances)
    n = len(abs_deviation)
    mad = abs_deviation[n // 2] if n % 2 == 1 else (abs_deviation[n // 2 - 1] + abs_deviation[n // 2]) / 2
    return mad

# Feature 13
def circle_min_dist(list_of_particles):
    """Calculate the minimum distance between the circle center and its closest particle."""
    points = [(particle['pose']['position']['x'], particle['pose']['position']['y']) for particle in list_of_particles]
    circle = smallest_enclosing_circle(points)
    center, _ = circle

    # Compute the minimum distance
    min_distance = min(dist(center, point) for point in points)
    return min_distance

# Feature 14
def circle_std_deviation(list_of_particles):
    """Calculate the standard deviation of distances from the circle center to the points."""
    points = [(particle['pose']['position']['x'], particle['pose']['position']['y']) for particle in list_of_particles]
    circle = smallest_enclosing_circle(points)
    center, _ = circle

    # Compute distances
    distances = [dist(center, point) for point in points]
    mean_distance = sum(distances) / len(points)

    # Compute standard deviation
    variance = sum((d - mean_distance) ** 2 for d in distances) / len(points)
    std_deviation = math.sqrt(variance)
    return std_deviation

# Extract all features
def extract_features_from_message(list_of_particles):
    """
    Extracts all features from a list of particles.

    Parameters:
    list_of_particles (list): List of dictionaries representing particles.

    Returns:
    dict: A dictionary containing all extracted features.
    """
    features = {}

    # Feature 1
    max_distance, furthest_particle = cog_max_dist(list_of_particles)
    features['cog_max_distance'] = max_distance
    #features['cog_max_distance_particle'] = furthest_particle

    # Feature 2
    features['cog_mean'] = calculate_cog_mean(list_of_particles)

    # Feature 3
    features['cog_mean_absolute_deviation'] = calculate_cog_mean_absolute_deviation(list_of_particles)

    # Feature 4
    features['cog_median'] = calculate_cog_median(list_of_particles)

    # Feature 5
    features['cog_median_absolute_deviation'] = calculate_cog_median_absolute_deviation(list_of_particles)

    # Feature 6
    min_distance, closest_particle = cog_min_dist(list_of_particles)
    features['cog_min_distance'] = min_distance
    #features['cog_min_distance_particle'] = closest_particle

    # Feature 7
    features['cog_standard_deviation'] = calculate_cog_standard_deviation(list_of_particles)

    # Feature 8
    points = [(particle['pose']['position']['x'], particle['pose']['position']['y']) for particle in list_of_particles]
    circle = smallest_enclosing_circle(points)
    features['circle_radius'] = circle[1]
    #features['circle_center'] = circle[0]

    # Feature 9
    features['circle_mean'] = circle_mean(list_of_particles)

    # Feature 10
    features['circle_mean_absolute_deviation'] = circle_mean_absolute_deviation(list_of_particles)

    # Feature 11
    features['circle_median'] = circle_median(list_of_particles)

    # Feature 12
    features['circle_median_absolute_deviation'] = circle_median_absolute_deviation(list_of_particles)

    # Feature 13
    features['circle_min_distance'] = circle_min_dist(list_of_particles)

    # Feature 14
    features['circle_standard_deviation'] = circle_std_deviation(list_of_particles)

    return features

############### Helper Functions ################

def calculate_center_of_gravity(list_of_particles):
    """
    Calculates the center of gravity of the particles based on their weights.

    Parameters:
    list_of_particles (list): List of dictionaries representing particles.

    Returns:
    dict: A dictionary with the x and y coordinates of the center of gravity.
    """
    total_weight = sum(particle['weight'] for particle in list_of_particles)
    if total_weight == 0:
        return {'x': 0, 'y': 0}  # Default to origin if no weight

    cog_x = sum(particle['pose']['position']['x'] * particle['weight'] for particle in list_of_particles) / total_weight
    cog_y = sum(particle['pose']['position']['y'] * particle['weight'] for particle in list_of_particles) / total_weight

    return {'x': cog_x, 'y': cog_y}

def welzl(points, boundary=None):
    """Recursive Welzl's algorithm to find the minimum enclosing circle."""
    if boundary is None:
        boundary = []

    if not points or len(boundary) == 3:
        if len(boundary) == 0:
            return ((0, 0), 0)
        elif len(boundary) == 1:
            return (boundary[0], 0)
        elif len(boundary) == 2:
            return circle_from_two_points(boundary[0], boundary[1])
        elif len(boundary) == 3:
            try:
                return circle_from_three_points(boundary[0], boundary[1], boundary[2])
            except ValueError:
                return circle_from_two_points(boundary[0], boundary[1])

    p = points.pop()
    circle = welzl(points.copy(), boundary)

    if is_in_circle(p, circle):
        points.append(p)
        return circle

    boundary.append(p)
    circle = welzl(points.copy(), boundary)
    boundary.pop()
    points.append(p)
    return circle

        
def circle_from_two_points(p1, p2):
    """Return the smallest circle from two points."""
    center = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    radius = dist(p1, p2) / 2
    return (center, radius)

def circle_from_three_points(p1, p2, p3):
    """Return the smallest circle from three points."""
    ax, ay = p1
    bx, by = p2
    cx, cy = p3

    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if d == 0:
        raise ValueError("Collinear points")

    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
    center = (ux, uy)
    radius = dist(center, p1)
    return (center, radius)

def dist(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def is_in_circle(point, circle):
    """Check if a point is inside or on the boundary of a circle."""
    center, radius = circle
    return dist(point, center) <= radius

if __name__ == "__main__":
    list_of_particles = [
        {'pose': {'position': {'x': 2.7026532109185792, 'y': 1.3363095842400234, 'z': 0.0, '__msgtype__': 'geometry_msgs/msg/Point'}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': -0.6206412601751722, 'w': 0.7840946538321596, '__msgtype__': 'geometry_msgs/msg/Quaternion'}, '__msgtype__': 'geometry_msgs/msg/Pose'}, 'weight': 0.0005980861244019139, '__msgtype__': 'nav2_msgs/msg/Particle'},
        {'pose': {'position': {'x': 2.9070964865479705, 'y': 3.0649213798266697, 'z': 0.0, '__msgtype__': 'geometry_msgs/msg/Point'}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': -0.45132518076103845, 'w': 0.8923595582560967, '__msgtype__': 'geometry_msgs/msg/Quaternion'}, '__msgtype__': 'geometry_msgs/msg/Pose'}, 'weight': 0.0005980861244019139, '__msgtype__': 'nav2_msgs/msg/Particle'},
        {'pose': {'position': {'x': 2.80871858542121, 'y': 1.5363776884978138, 'z': 0.0, '__msgtype__': 'geometry_msgs/msg/Point'}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': -0.36432616851598243, 'w': 0.9312714120676442, '__msgtype__': 'geometry_msgs/msg/Quaternion'}, '__msgtype__': 'geometry_msgs/msg/Pose'}, 'weight': 0.0005980861244019139, '__msgtype__': 'nav2_msgs/msg/Particle'},
        {'pose': {'position': {'x': 1.8221955477463578, 'y': 1.6169840054666116, 'z': 0.0, '__msgtype__': 'geometry_msgs/msg/Point'}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': -0.584478714347991, 'w': 0.8114090414052085, '__msgtype__': 'geometry_msgs/msg/Quaternion'}, '__msgtype__': 'geometry_msgs/msg/Pose'}, 'weight': 0.0005980861244019139, '__msgtype__': 'nav2_msgs/msg/Particle'},
        {'pose': {'position': {'x': 2.12472141189225, 'y': 1.5361849999975508, 'z': 0.0, '__msgtype__': 'geometry_msgs/msg/Point'}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': -0.4347883702383812, 'w': 0.900532660765534, '__msgtype__': 'geometry_msgs/msg/Quaternion'}, '__msgtype__': 'geometry_msgs/msg/Pose'}, 'weight': 0.0005980861244019139, '__msgtype__': 'nav2_msgs/msg/Particle'}
    ]

    # Extract all features
    features = extract_features_from_message(list_of_particles)
    print(features)