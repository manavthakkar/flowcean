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

def plot_particles(list_of_particles):
    """
    Plots all particles in a 2D plot using their position and orientation.

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

    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
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
