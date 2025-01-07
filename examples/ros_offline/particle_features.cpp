#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <utility>

namespace py = pybind11;

struct Particle {
    double x, y, weight;
};

struct Circle {
    std::pair<double, double> center;
    double radius;
};

std::pair<double, std::pair<double, double>> cog_max_dist(const std::vector<Particle>& particles) {
    if (particles.empty()) {
        return {0.0, {0.0, 0.0}};
    }

    double total_weight = 0.0, cog_x = 0.0, cog_y = 0.0;

    for (const auto& particle : particles) {
        total_weight += particle.weight;
        cog_x += particle.x * particle.weight;
        cog_y += particle.y * particle.weight;
    }

    if (total_weight == 0) {
        return {0.0, {0.0, 0.0}};
    }

    cog_x /= total_weight;
    cog_y /= total_weight;

    double max_distance = 0.0;
    std::pair<double, double> furthest_particle;

    for (const auto& particle : particles) {
        double dx = particle.x - cog_x;
        double dy = particle.y - cog_y;
        double distance = std::sqrt(dx * dx + dy * dy);

        if (distance > max_distance) {
            max_distance = distance;
            furthest_particle = {particle.x, particle.y};
        }
    }

    return {max_distance, furthest_particle};
}

double cog_mean_dist(const std::vector<Particle>& particles) {
    if (particles.empty()) {
        return 0.0;
    }

    double total_weight = 0.0, cog_x = 0.0, cog_y = 0.0;

    // Calculate the center of gravity (COG)
    for (const auto& particle : particles) {
        total_weight += particle.weight;
        cog_x += particle.x * particle.weight;
        cog_y += particle.y * particle.weight;
    }

    if (total_weight == 0) {
        return 0.0;
    }

    cog_x /= total_weight;
    cog_y /= total_weight;

    // Calculate mean distance from COG
    double total_distance = 0.0;
    for (const auto& particle : particles) {
        double dx = particle.x - cog_x;
        double dy = particle.y - cog_y;
        total_distance += std::sqrt(dx * dx + dy * dy);
    }

    return total_distance / particles.size();
}

double cog_mean_absolute_deviation(const std::vector<Particle>& particles) {
    if (particles.empty()) {
        return 0.0;
    }

    double total_weight = 0.0, cog_x = 0.0, cog_y = 0.0;

    // Calculate the center of gravity (COG)
    for (const auto& particle : particles) {
        total_weight += particle.weight;
        cog_x += particle.x * particle.weight;
        cog_y += particle.y * particle.weight;
    }

    if (total_weight == 0) {
        return 0.0;
    }

    cog_x /= total_weight;
    cog_y /= total_weight;

    // Calculate distances from COG
    std::vector<double> distances;
    double total_distance = 0.0;

    for (const auto& particle : particles) {
        double dx = particle.x - cog_x;
        double dy = particle.y - cog_y;
        double distance = std::sqrt(dx * dx + dy * dy);
        distances.push_back(distance);
        total_distance += distance;
    }

    // Calculate mean distance
    double mean_distance = total_distance / particles.size();

    // Calculate mean absolute deviation
    double mad = 0.0;
    for (const auto& distance : distances) {
        mad += std::abs(distance - mean_distance);
    }

    return mad / particles.size();
}

#include <algorithm>

double cog_median(const std::vector<Particle>& particles) {
    if (particles.empty()) {
        return 0.0;
    }

    double total_weight = 0.0, cog_x = 0.0, cog_y = 0.0;

    // Calculate the center of gravity (COG)
    for (const auto& particle : particles) {
        total_weight += particle.weight;
        cog_x += particle.x * particle.weight;
        cog_y += particle.y * particle.weight;
    }

    if (total_weight == 0) {
        return 0.0;
    }

    cog_x /= total_weight;
    cog_y /= total_weight;

    // Calculate distances from COG
    std::vector<double> distances;
    for (const auto& particle : particles) {
        double dx = particle.x - cog_x;
        double dy = particle.y - cog_y;
        double distance = std::sqrt(dx * dx + dy * dy);
        distances.push_back(distance);
    }

    // Sort the distances to find the median
    std::sort(distances.begin(), distances.end());

    size_t n = distances.size();
    if (n % 2 == 1) {
        return distances[n / 2];  // Return the middle element for odd length
    } else {
        // For even length, return the average of the two middle elements
        return (distances[n / 2 - 1] + distances[n / 2]) / 2;
    }
}

double cog_median_absolute_deviation(const std::vector<Particle>& particles) {
    if (particles.empty()) {
        return 0.0;
    }

    double total_weight = 0.0, cog_x = 0.0, cog_y = 0.0;

    // Calculate the center of gravity (COG)
    for (const auto& particle : particles) {
        total_weight += particle.weight;
        cog_x += particle.x * particle.weight;
        cog_y += particle.y * particle.weight;
    }

    if (total_weight == 0) {
        return 0.0;
    }

    cog_x /= total_weight;
    cog_y /= total_weight;

    // Calculate the median distance (same as cog_median)
    std::vector<double> distances;
    for (const auto& particle : particles) {
        double dx = particle.x - cog_x;
        double dy = particle.y - cog_y;
        double distance = std::sqrt(dx * dx + dy * dy);
        distances.push_back(distance);
    }

    // Calculate the median distance
    std::sort(distances.begin(), distances.end());
    size_t n = distances.size();
    double median_distance;
    if (n % 2 == 1) {
        median_distance = distances[n / 2];  // Middle element for odd size
    } else {
        median_distance = (distances[n / 2 - 1] + distances[n / 2]) / 2;  // Average of middle elements for even size
    }

    // Calculate the absolute deviations from the median distance
    std::vector<double> absolute_deviations;
    for (const auto& distance : distances) {
        absolute_deviations.push_back(std::abs(distance - median_distance));
    }

    // Sort the absolute deviations
    std::sort(absolute_deviations.begin(), absolute_deviations.end());

    // Calculate the median of absolute deviations
    n = absolute_deviations.size();
    if (n % 2 == 1) {
        return absolute_deviations[n / 2];  // Middle element for odd size
    } else {
        return (absolute_deviations[n / 2 - 1] + absolute_deviations[n / 2]) / 2;  // Average of middle elements for even size
    }
}

std::pair<double, std::pair<double, double>> cog_min_dist(const std::vector<Particle>& particles) {
    if (particles.empty()) {
        return {0.0, {0.0, 0.0}};
    }

    double total_weight = 0.0, cog_x = 0.0, cog_y = 0.0;

    // Calculate the center of gravity (COG)
    for (const auto& particle : particles) {
        total_weight += particle.weight;
        cog_x += particle.x * particle.weight;
        cog_y += particle.y * particle.weight;
    }

    if (total_weight == 0) {
        return {0.0, {0.0, 0.0}};
    }

    cog_x /= total_weight;
    cog_y /= total_weight;

    double min_distance = std::numeric_limits<double>::infinity();
    std::pair<double, double> closest_particle;

    // Calculate the minimum distance to the center of gravity
    for (const auto& particle : particles) {
        double dx = particle.x - cog_x;
        double dy = particle.y - cog_y;
        double distance = std::sqrt(dx * dx + dy * dy);

        if (distance < min_distance) {
            min_distance = distance;
            closest_particle = {particle.x, particle.y};
        }
    }

    return {min_distance, closest_particle};
}

double cog_standard_deviation(const std::vector<Particle>& particles) {
    if (particles.empty()) {
        return 0.0;
    }

    double total_weight = 0.0, cog_x = 0.0, cog_y = 0.0;

    // Calculate the center of gravity (COG)
    for (const auto& particle : particles) {
        total_weight += particle.weight;
        cog_x += particle.x * particle.weight;
        cog_y += particle.y * particle.weight;
    }

    if (total_weight == 0) {
        return 0.0;
    }

    cog_x /= total_weight;
    cog_y /= total_weight;

    // Calculate distances from COG
    double total_distance = 0.0;
    std::vector<double> distances;
    for (const auto& particle : particles) {
        double dx = particle.x - cog_x;
        double dy = particle.y - cog_y;
        double distance = std::sqrt(dx * dx + dy * dy);
        distances.push_back(distance);
        total_distance += distance;
    }

    // Calculate mean distance
    double mean_distance = total_distance / particles.size();

    // Calculate variance
    double variance = 0.0;
    for (const auto& distance : distances) {
        variance += (distance - mean_distance) * (distance - mean_distance);
    }

    variance /= particles.size();

    // Return the standard deviation
    return std::sqrt(variance);
}

// Utility function to calculate Euclidean distance between two points
double dist(const std::pair<double, double>& p1, const std::pair<double, double>& p2) {
    return std::sqrt((p1.first - p2.first) * (p1.first - p2.first) + (p1.second - p2.second) * (p1.second - p2.second));
}

// Welzl's algorithm to find the smallest enclosing circle
Circle welzl(std::vector<std::pair<double, double>>& points, std::vector<std::pair<double, double>>& boundary) {
    if (points.empty() || boundary.size() == 3) {
        if (boundary.size() == 0) {
            return {{0, 0}, 0};  // Default to origin if no points
        }
        if (boundary.size() == 1) {
            return {boundary[0], 0};  // Circle from one point
        }
        if (boundary.size() == 2) {
            // Circle from two points
            double radius = dist(boundary[0], boundary[1]) / 2;
            std::pair<double, double> center = {(boundary[0].first + boundary[1].first) / 2,
                                                (boundary[0].second + boundary[1].second) / 2};
            return {center, radius};
        }
        // Circle from three points
        double ax = boundary[0].first, ay = boundary[0].second;
        double bx = boundary[1].first, by = boundary[1].second;
        double cx = boundary[2].first, cy = boundary[2].second;
        double d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
        double ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) +
                     (cx * cx + cy * cy) * (ay - by)) / d;
        double uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) +
                     (cx * cx + cy * cy) * (bx - ax)) / d;
        double radius = dist({ux, uy}, boundary[0]);
        return {{ux, uy}, radius};
    }

    std::pair<double, double> p = points.back();
    points.pop_back();
    Circle circle = welzl(points, boundary);
    if (dist(circle.center, p) <= circle.radius) {
        points.push_back(p);
        return circle;
    }

    boundary.push_back(p);
    Circle result = welzl(points, boundary);
    boundary.pop_back();
    points.push_back(p);
    return result;
}

std::pair<std::pair<double, double>, double> smallest_enclosing_circle(std::vector<std::pair<double, double>>& points) {
    std::random_shuffle(points.begin(), points.end());  // Shuffle for randomness
    std::vector<std::pair<double, double>> empty_boundary;  // Create an empty vector for boundary
    Circle circle = welzl(points, empty_boundary);  // Pass it by reference
    return {circle.center, circle.radius};
}

double circle_mean(const std::vector<std::pair<double, double>>& points) {
    // Find the smallest enclosing circle
    std::vector<std::pair<double, double>> points_copy = points; // Copy the points as welzl modifies the vector
    auto circle = smallest_enclosing_circle(points_copy);

    // Calculate the mean distance from the circle center to the points
    double total_distance = 0.0;
    for (const auto& point : points) {
        total_distance += dist(circle.first, point);  // circle.first is the center
    }

    return total_distance / points.size();
}

double circle_mean_absolute_deviation(const std::vector<std::pair<double, double>>& points) {
    // Find the smallest enclosing circle
    std::vector<std::pair<double, double>> points_copy = points; // Copy the points as welzl modifies the vector
    auto circle = smallest_enclosing_circle(points_copy);

    // Calculate the mean distance from the circle center to the points
    double total_distance = 0.0;
    std::vector<double> distances;
    for (const auto& point : points) {
        double distance = dist(circle.first, point);  // circle.first is the center
        distances.push_back(distance);
        total_distance += distance;
    }

    double mean_distance = total_distance / points.size();

    // Calculate the mean absolute deviation from the mean distance
    double mad = 0.0;
    for (const auto& distance : distances) {
        mad += std::abs(distance - mean_distance);
    }

    return mad / points.size();
}

double circle_median(const std::vector<std::pair<double, double>>& points) {
    // Find the smallest enclosing circle
    std::vector<std::pair<double, double>> points_copy = points; // Copy the points as welzl modifies the vector
    auto circle = smallest_enclosing_circle(points_copy);

    // Calculate the distances from the circle center to the points
    std::vector<double> distances;
    for (const auto& point : points) {
        distances.push_back(dist(circle.first, point));  // circle.first is the center
    }

    // Sort the distances to find the median
    std::sort(distances.begin(), distances.end());

    size_t n = distances.size();
    if (n % 2 == 1) {
        return distances[n / 2];  // Return the middle element for odd size
    } else {
        // For even size, return the average of the two middle elements
        return (distances[n / 2 - 1] + distances[n / 2]) / 2;
    }
}

double circle_median_absolute_deviation(const std::vector<std::pair<double, double>>& points) {
    // Find the smallest enclosing circle
    std::vector<std::pair<double, double>> points_copy = points; // Copy the points as welzl modifies the vector
    auto circle = smallest_enclosing_circle(points_copy);

    // Calculate the distances from the circle center to the points
    std::vector<double> distances;
    for (const auto& point : points) {
        distances.push_back(dist(circle.first, point));  // circle.first is the center
    }

    // Calculate the median distance
    std::sort(distances.begin(), distances.end());
    size_t n = distances.size();
    double median_distance;
    if (n % 2 == 1) {
        median_distance = distances[n / 2];  // Return the middle element for odd length
    } else {
        // For even length, return the average of the two middle elements
        median_distance = (distances[n / 2 - 1] + distances[n / 2]) / 2;
    }

    // Calculate the absolute deviations from the median distance
    std::vector<double> absolute_deviations;
    for (const auto& distance : distances) {
        absolute_deviations.push_back(std::abs(distance - median_distance));
    }

    // Sort the absolute deviations
    std::sort(absolute_deviations.begin(), absolute_deviations.end());

    // Calculate the median of absolute deviations
    n = absolute_deviations.size();
    if (n % 2 == 1) {
        return absolute_deviations[n / 2];  // Middle element for odd size
    } else {
        return (absolute_deviations[n / 2 - 1] + absolute_deviations[n / 2]) / 2;  // Average of middle elements for even size
    }
}

double circle_min_dist(const std::vector<std::pair<double, double>>& points) {
    // Find the smallest enclosing circle
    std::vector<std::pair<double, double>> points_copy = points; // Copy the points as welzl modifies the vector
    auto circle = smallest_enclosing_circle(points_copy);

    // Calculate the minimum distance from the circle center to the points
    double min_distance = std::numeric_limits<double>::infinity();

    for (const auto& point : points) {
        double distance = dist(circle.first, point);  // circle.first is the center
        if (distance < min_distance) {
            min_distance = distance;
        }
    }

    return min_distance;
}

double circle_std_deviation(const std::vector<std::pair<double, double>>& points) {
    // Find the smallest enclosing circle
    std::vector<std::pair<double, double>> points_copy = points; // Copy the points as welzl modifies the vector
    auto circle = smallest_enclosing_circle(points_copy);

    // Calculate the distances from the circle center to the points
    double total_distance = 0.0;
    std::vector<double> distances;
    for (const auto& point : points) {
        double distance = dist(circle.first, point);  // circle.first is the center
        distances.push_back(distance);
        total_distance += distance;
    }

    double mean_distance = total_distance / points.size();

    // Calculate variance
    double variance = 0.0;
    for (const auto& distance : distances) {
        variance += (distance - mean_distance) * (distance - mean_distance);
    }

    variance /= points.size();

    // Return the standard deviation (sqrt of variance)
    return std::sqrt(variance);
}

// Function to extract all features
std::map<std::string, double> extract_all_features_cpp(const std::vector<Particle>& particles) {
    std::map<std::string, double> features;

    features["cog_max_dist"] = cog_max_dist(particles).first;
    features["cog_mean_dist"] = cog_mean_dist(particles);
    features["cog_mean_absolute_deviation"] = cog_mean_absolute_deviation(particles);
    features["cog_median"] = cog_median(particles);
    features["cog_median_absolute_deviation"] = cog_median_absolute_deviation(particles);
    features["cog_min_dist"] = cog_min_dist(particles).first;
    features["cog_standard_deviation"] = cog_standard_deviation(particles);

    // For circle-related features, convert particles to points
    std::vector<std::pair<double, double>> points;
    for (const auto& particle : particles) {
        points.push_back({particle.x, particle.y});
    }

    features["circle_radius"] = smallest_enclosing_circle(points).second;
    features["circle_mean"] = circle_mean(points);
    features["circle_mean_absolute_deviation"] = circle_mean_absolute_deviation(points);
    features["circle_median"] = circle_median(points);
    features["circle_median_absolute_deviation"] = circle_median_absolute_deviation(points);
    features["circle_min_dist"] = circle_min_dist(points);
    features["circle_std_deviation"] = circle_std_deviation(points);

    return features;
}


PYBIND11_MODULE(particle_features, m) {
    py::class_<Particle>(m, "Particle")
        .def(py::init<>())
        .def_readwrite("x", &Particle::x)
        .def_readwrite("y", &Particle::y)
        .def_readwrite("weight", &Particle::weight);

    m.def("cog_max_dist", &cog_max_dist, "Calculate max distance to COG");
    m.def("cog_mean_dist", &cog_mean_dist, "Calculate mean distance to COG");
    m.def("cog_mean_absolute_deviation", &cog_mean_absolute_deviation, "Calculate mean absolute deviation from COG");
    m.def("cog_median", &cog_median, "Calculate median distance from COG");
    m.def("cog_median_absolute_deviation", &cog_median_absolute_deviation, "Calculate median absolute deviation from COG");
    m.def("cog_min_dist", &cog_min_dist, "Calculate minimum distance to COG");
    m.def("cog_standard_deviation", &cog_standard_deviation, "Calculate standard deviation of distances to COG");
    m.def("smallest_enclosing_circle", &smallest_enclosing_circle, "Calculate smallest enclosing circle");
    m.def("circle_mean", &circle_mean, "Calculate mean distance from circle center");
    m.def("circle_mean_absolute_deviation", &circle_mean_absolute_deviation, "Calculate mean absolute deviation from circle center");
    m.def("circle_median", &circle_median, "Calculate median distance from circle center");
    m.def("circle_median_absolute_deviation", &circle_median_absolute_deviation, "Calculate median absolute deviation from circle center");
    m.def("circle_min_dist", &circle_min_dist, "Calculate minimum distance to circle center");
    m.def("circle_std_deviation", &circle_std_deviation, "Calculate standard deviation of distances to circle center");
    m.def("extract_all_features_cpp", &extract_all_features_cpp, "Extract all features from particles");
}
