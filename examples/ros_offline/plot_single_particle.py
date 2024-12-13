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
    plt.figure()
    plt.quiver(x, y, end_x - x, end_y - y, angles='xy', scale_units='xy', scale=1, color='blue')
    plt.scatter([x], [y], color='red', label='Position')  # Mark the position
    plt.xlim(x - 1, x + 1)
    plt.ylim(y - 1, y + 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.legend()
    plt.title("2D Arrow Plot")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

# Example usage
position = {'x': 2.95741417749249, 'y': 1.5826094649904345, 'z': 0.0, '__msgtype__': 'geometry_msgs/msg/Point'}
orientation = {'x': 0.0, 'y': 0.0, 'z': -0.4665565937450478, 'w': 0.8844913480826245, '__msgtype__': 'geometry_msgs/msg/Quaternion'}

plot_arrow_2d(position, orientation)
