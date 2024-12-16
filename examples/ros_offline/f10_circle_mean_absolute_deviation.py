import random
import math
import matplotlib.pyplot as plt
import time

def dist(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

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

def is_in_circle(point, circle):
    """Check if a point is inside or on the boundary of a circle."""
    center, radius = circle
    return dist(point, center) <= radius

def welzl(points, boundary=[]):
    """Recursive Welzl's algorithm to find the minimum enclosing circle."""
    if not points or len(boundary) == 3:
        if len(boundary) == 0:
            return ((0, 0), 0)
        elif len(boundary) == 1:
            return (boundary[0], 0)
        elif len(boundary) == 2:
            return circle_from_two_points(boundary[0], boundary[1])
        elif len(boundary) == 3:
            return circle_from_three_points(boundary[0], boundary[1], boundary[2])

    p = points.pop()
    circle = welzl(points, boundary)

    if is_in_circle(p, circle):
        points.append(p)
        return circle

    boundary.append(p)
    circle = welzl(points, boundary)
    boundary.pop()
    points.append(p)
    return circle

def smallest_enclosing_circle(points):
    """Find the smallest enclosing circle for a set of points."""
    shuffled_points = points[:]
    random.shuffle(shuffled_points)
    return welzl(shuffled_points)

def plot_circle(points, circle):
    """Plot the points and the smallest enclosing circle."""
    center, radius = circle
    fig, ax = plt.subplots()
    
    # Plot points
    x_coords, y_coords = zip(*points)
    ax.scatter(x_coords, y_coords, label="Points")

    # Plot circle
    circle_plot = plt.Circle(center, radius, color='blue', fill=False, label="Smallest Enclosing Circle")
    ax.add_artist(circle_plot)

    # Plot center
    ax.scatter(*center, color='red', label="Center")

    ax.set_xlim(min(x_coords) - radius, max(x_coords) + radius)
    ax.set_ylim(min(y_coords) - radius, max(y_coords) + radius)
    ax.set_aspect('equal', adjustable='datalim')
    plt.legend()
    plt.show()

def circle_mean(list_of_particles):
    """Calculate the mean of distances from the circle center to the points."""
    points = [(particle['pose']['position']['x'], particle['pose']['position']['y']) for particle in list_of_particles]
    circle = smallest_enclosing_circle(points)
    center, _ = circle

    total_distance = sum(dist(center, point) for point in points)
    mean_distance = total_distance / len(points)
    return mean_distance

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

# Example usage
if __name__ == "__main__":
    # Dictionary of particles
    list_of_particles = [
        {'pose': {'position': {'x': 2.7026532109185792, 'y': 1.3363095842400234, 'z': 0.0, '__msgtype__': 'geometry_msgs/msg/Point'}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': -0.6206412601751722, 'w': 0.7840946538321596, '__msgtype__': 'geometry_msgs/msg/Quaternion'}, '__msgtype__': 'geometry_msgs/msg/Pose'}, 'weight': 0.0005980861244019139, '__msgtype__': 'nav2_msgs/msg/Particle'},
        {'pose': {'position': {'x': 2.9070964865479705, 'y': 3.0649213798266697, 'z': 0.0, '__msgtype__': 'geometry_msgs/msg/Point'}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': -0.45132518076103845, 'w': 0.8923595582560967, '__msgtype__': 'geometry_msgs/msg/Quaternion'}, '__msgtype__': 'geometry_msgs/msg/Pose'}, 'weight': 0.0005980861244019139, '__msgtype__': 'nav2_msgs/msg/Particle'},
        {'pose': {'position': {'x': 2.80871858542121, 'y': 1.5363776884978138, 'z': 0.0, '__msgtype__': 'geometry_msgs/msg/Point'}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': -0.36432616851598243, 'w': 0.9312714120676442, '__msgtype__': 'geometry_msgs/msg/Quaternion'}, '__msgtype__': 'geometry_msgs/msg/Pose'}, 'weight': 0.0005980861244019139, '__msgtype__': 'nav2_msgs/msg/Particle'},
        {'pose': {'position': {'x': 1.8221955477463578, 'y': 1.6169840054666116, 'z': 0.0, '__msgtype__': 'geometry_msgs/msg/Point'}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': -0.584478714347991, 'w': 0.8114090414052085, '__msgtype__': 'geometry_msgs/msg/Quaternion'}, '__msgtype__': 'geometry_msgs/msg/Pose'}, 'weight': 0.0005980861244019139, '__msgtype__': 'nav2_msgs/msg/Particle'},
        {'pose': {'position': {'x': 2.12472141189225, 'y': 1.5361849999975508, 'z': 0.0, '__msgtype__': 'geometry_msgs/msg/Point'}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': -0.4347883702383812, 'w': 0.900532660765534, '__msgtype__': 'geometry_msgs/msg/Quaternion'}, '__msgtype__': 'geometry_msgs/msg/Pose'}, 'weight': 0.0005980861244019139, '__msgtype__': 'nav2_msgs/msg/Particle'}
    ]

    # Extract (x, y) positions
    points = [(particle['pose']['position']['x'], particle['pose']['position']['y']) for particle in list_of_particles]

    # Measure time to compute the circle
    start_time = time.time()
    circle = smallest_enclosing_circle(points)
    end_time = time.time()

    print("Center:", circle[0])
    print("Radius:", circle[1])
    print(f"Time taken: {end_time - start_time:.4f} seconds")

    # Calculate the mean distance
    mean_distance = circle_mean(list_of_particles)
    print("Mean distance from center to points:", mean_distance)

    # Calculate the mean absolute deviation
    mean_absolute_deviation = circle_mean_absolute_deviation(list_of_particles)
    print("Mean absolute deviation from center to points:", mean_absolute_deviation)

    # Plot the result
    plot_circle(points, circle)
