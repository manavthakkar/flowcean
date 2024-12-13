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

    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.legend()
    plt.title("2D Particles Plot")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

list_of_dict_of_particles = [
    {'pose': {'position': {'x': 2.7026532109185792, 'y': 1.3363095842400234, 'z': 0.0, '__msgtype__': 'geometry_msgs/msg/Point'}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': -0.6206412601751722, 'w': 0.7840946538321596, '__msgtype__': 'geometry_msgs/msg/Quaternion'}, '__msgtype__': 'geometry_msgs/msg/Pose'}, 'weight': 0.0005980861244019139, '__msgtype__': 'nav2_msgs/msg/Particle'},
    {'pose': {'position': {'x': 2.9070964865479705, 'y': 3.0649213798266697, 'z': 0.0, '__msgtype__': 'geometry_msgs/msg/Point'}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': -0.45132518076103845, 'w': 0.8923595582560967, '__msgtype__': 'geometry_msgs/msg/Quaternion'}, '__msgtype__': 'geometry_msgs/msg/Pose'}, 'weight': 0.0005980861244019139, '__msgtype__': 'nav2_msgs/msg/Particle'},
    {'pose': {'position': {'x': 2.80871858542121, 'y': 1.5363776884978138, 'z': 0.0, '__msgtype__': 'geometry_msgs/msg/Point'}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': -0.36432616851598243, 'w': 0.9312714120676442, '__msgtype__': 'geometry_msgs/msg/Quaternion'}, '__msgtype__': 'geometry_msgs/msg/Pose'}, 'weight': 0.0005980861244019139, '__msgtype__': 'nav2_msgs/msg/Particle'},
    {'pose': {'position': {'x': 1.8221955477463578, 'y': 1.6169840054666116, 'z': 0.0, '__msgtype__': 'geometry_msgs/msg/Point'}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': -0.584478714347991, 'w': 0.8114090414052085, '__msgtype__': 'geometry_msgs/msg/Quaternion'}, '__msgtype__': 'geometry_msgs/msg/Pose'}, 'weight': 0.0005980861244019139, '__msgtype__': 'nav2_msgs/msg/Particle'},
    {'pose': {'position': {'x': 2.12472141189225, 'y': 1.5361849999975508, 'z': 0.0, '__msgtype__': 'geometry_msgs/msg/Point'}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': -0.4347883702383812, 'w': 0.900532660765534, '__msgtype__': 'geometry_msgs/msg/Quaternion'}, '__msgtype__': 'geometry_msgs/msg/Pose'}, 'weight': 0.0005980861244019139, '__msgtype__': 'nav2_msgs/msg/Particle'},
    {'pose': {'position': {'x': 3.787455097217961, 'y': 1.1168038529127071, 'z': 0.0, '__msgtype__': 'geometry_msgs/msg/Point'}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': -0.3181781169217904, 'w': 0.9480309519799992, '__msgtype__': 'geometry_msgs/msg/Quaternion'}, '__msgtype__': 'geometry_msgs/msg/Pose'}, 'weight': 0.0005980861244019139, '__msgtype__': 'nav2_msgs/msg/Particle'},
    {'pose': {'position': {'x': 1.9120902696224997, 'y': 1.6669677342968452, 'z': 0.0, '__msgtype__': 'geometry_msgs/msg/Point'}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': -0.4197803804401139, 'w': 0.9076257115119388, '__msgtype__': 'geometry_msgs/msg/Quaternion'}, '__msgtype__': 'geometry_msgs/msg/Pose'}, 'weight': 0.0005980861244019139, '__msgtype__': 'nav2_msgs/msg/Particle'},
    {'pose': {'position': {'x': 2.9745974840891294, 'y': 1.9407465525606, 'z': 0.0, '__msgtype__': 'geometry_msgs/msg/Point'}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': -0.38842390508186725, 'w': 0.9214808028173743, '__msgtype__': 'geometry_msgs/msg/Quaternion'}, '__msgtype__': 'geometry_msgs/msg/Pose'}, 'weight': 0.0005980861244019139, '__msgtype__': 'nav2_msgs/msg/Particle'},
    {'pose': {'position': {'x': 1.992383683023339, 'y': 2.471452533516195, 'z': 0.0, '__msgtype__': 'geometry_msgs/msg/Point'}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': -0.48955682976404485, 'w': 0.8719713931267344, '__msgtype__': 'geometry_msgs/msg/Quaternion'}, '__msgtype__': 'geometry_msgs/msg/Pose'}, 'weight': 0.0005980861244019139, '__msgtype__': 'nav2_msgs/msg/Particle'},
    {'pose': {'position': {'x': 3.131766433245124, 'y': 1.3945479132423562, 'z': 0.0, '__msgtype__': 'geometry_msgs/msg/Point'}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': -0.41647224023795093, 'w': 0.9091484329366588, '__msgtype__': 'geometry_msgs/msg/Quaternion'}, '__msgtype__': 'geometry_msgs/msg/Pose'}, 'weight': 0.0005980861244019139, '__msgtype__': 'nav2_msgs/msg/Particle'}
]


plot_particles(list_of_dict_of_particles)
