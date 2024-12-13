import matplotlib.pyplot as plt
import numpy as np

def plot_arrow_2d(position, orientation):
    """
    Plots an arrow on a 2D plot using the given position and orientation.

    Parameters:
    position (dict): Dictionary containing x, y coordinates and z (unused here).
    orientation (dict): Dictionary containing quaternion values (x, y, z, w).
    """
    # Extract position
    x = position['x']
    y = position['y']

    # Extract orientation (quaternion to yaw conversion)
    qz = orientation['z']
    qw = orientation['w']

    # Calculate the yaw angle (orientation in 2D plane)
    yaw = 2 * np.arctan2(qz, qw)

    # Define arrow length
    arrow_length = 0.5

    # Compute the arrow end point
    end_x = x + arrow_length * np.cos(yaw)
    end_y = y + arrow_length * np.sin(yaw)

    # Create the plot
    plt.quiver(x, y, end_x - x, end_y - y, angles='xy', scale_units='xy', scale=1, color='blue')
    plt.scatter([x], [y], color='red')  # Mark the position

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

def plot_particles(list_of_particles):
    """
    Plots all particles in a 2D plot using their position and orientation and marks the center of gravity.

    Parameters:
    list_of_particles (list): List of dictionaries representing particles.
    """
    # Determine plot bounds to avoid cutting off arrows
    all_x = [particle['pose']['position']['x'] for particle in list_of_particles]
    all_y = [particle['pose']['position']['y'] for particle in list_of_particles]
    margin = 1.0  # Add some margin around the particles
    min_x, max_x = min(all_x) - margin, max(all_x) + margin
    min_y, max_y = min(all_y) - margin, max(all_y) + margin

    plt.figure()
    for particle in list_of_particles:
        position = particle['pose']['position']
        orientation = particle['pose']['orientation']
        plot_arrow_2d(position, orientation)

    # Calculate and plot center of gravity
    cog = calculate_center_of_gravity(list_of_particles)
    plt.scatter(cog['x'], cog['y'], color='green', edgecolors='black', s=200, label='Center of Gravity', zorder=5)

    # Calculate and plot furthest particle from COG
    max_distance, furthest_particle = cog_max_dist(list_of_particles)
    if furthest_particle:
        plt.plot([cog['x'], furthest_particle[0]], [cog['y'], furthest_particle[1]], linestyle='--', color='purple', label='Max Distance')
        plt.scatter(furthest_particle[0], furthest_particle[1], color='purple', edgecolors='black', s=150, label='Furthest Particle')

    # Calculate and plot center of gravity mean
    cog_mean = calculate_cog_mean(list_of_particles)
    plt.scatter(cog_mean['x'], cog_mean['y'], color='orange', edgecolors='black', s=200, label='COG Mean', zorder=6)

    # Calculate and plot mean absolute deviation circle
    mean_absolute_deviation = calculate_cog_mean_absolute_deviation(list_of_particles)
    mad_circle = plt.Circle((cog_mean['x'], cog_mean['y']), mean_absolute_deviation, color='cyan', fill=False, linestyle='--', linewidth=2, label='MAD Circle', zorder=7)
    plt.gca().add_patch(mad_circle)

    # Calculate and plot median distance circle
    cog_median = calculate_cog_median(list_of_particles)
    median_circle = plt.Circle((cog_mean['x'], cog_mean['y']), cog_median, color='magenta', fill=False, linestyle=':', linewidth=2, label='Median Circle', zorder=8)
    plt.gca().add_patch(median_circle)

    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.legend()
    plt.title("2D Particles Plot")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

# Example usage
list_of_particles = [
    {'pose': {'position': {'x': 2.7026532109185792, 'y': 1.3363095842400234, 'z': 0.0, '__msgtype__': 'geometry_msgs/msg/Point'}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': -0.6206412601751722, 'w': 0.7840946538321596, '__msgtype__': 'geometry_msgs/msg/Quaternion'}, '__msgtype__': 'geometry_msgs/msg/Pose'}, 'weight': 0.0005980861244019139, '__msgtype__': 'nav2_msgs/msg/Particle'},
    {'pose': {'position': {'x': 2.9070964865479705, 'y': 3.0649213798266697, 'z': 0.0, '__msgtype__': 'geometry_msgs/msg/Point'}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': -0.45132518076103845, 'w': 0.8923595582560967, '__msgtype__': 'geometry_msgs/msg/Quaternion'}, '__msgtype__': 'geometry_msgs/msg/Pose'}, 'weight': 0.0005980861244019139, '__msgtype__': 'nav2_msgs/msg/Particle'},
    {'pose': {'position': {'x': 2.80871858542121, 'y': 1.5363776884978138, 'z': 0.0, '__msgtype__': 'geometry_msgs/msg/Point'}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': -0.36432616851598243, 'w': 0.9312714120676442, '__msgtype__': 'geometry_msgs/msg/Quaternion'}, '__msgtype__': 'geometry_msgs/msg/Pose'}, 'weight': 0.0005980861244019139, '__msgtype__': 'nav2_msgs/msg/Particle'},
    {'pose': {'position': {'x': 1.8221955477463578, 'y': 1.6169840054666116, 'z': 0.0, '__msgtype__': 'geometry_msgs/msg/Point'}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': -0.584478714347991, 'w': 0.8114090414052085, '__msgtype__': 'geometry_msgs/msg/Quaternion'}, '__msgtype__': 'geometry_msgs/msg/Pose'}, 'weight': 0.0005980861244019139, '__msgtype__': 'nav2_msgs/msg/Particle'},
    {'pose': {'position': {'x': 2.12472141189225, 'y': 1.5361849999975508, 'z': 0.0, '__msgtype__': 'geometry_msgs/msg/Point'}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': -0.4347883702383812, 'w': 0.900532660765534, '__msgtype__': 'geometry_msgs/msg/Quaternion'}, '__msgtype__': 'geometry_msgs/msg/Pose'}, 'weight': 0.0005980861244019139, '__msgtype__': 'nav2_msgs/msg/Particle'}
]

plot_particles(list_of_particles)
max_distance = cog_max_dist(list_of_particles)[0]
print(f"The maximum distance from a particle to the center of gravity is: {max_distance}")
cog_mean = calculate_cog_mean(list_of_particles)
print(f"The mean position (COG mean) over all particles is: {cog_mean}")
mean_absolute_deviation = calculate_cog_mean_absolute_deviation(list_of_particles)
print(f"The mean absolute deviation of distances to the COG mean is: {mean_absolute_deviation}")
cog_median = calculate_cog_median(list_of_particles)
print(f"The median distance from particles to the COG mean is: {cog_median}")
median_absolute_deviation = calculate_cog_median_absolute_deviation(list_of_particles)
print(f"The median absolute deviation (MAD) from the median distance is: {median_absolute_deviation}")
