#include "graph_planning_node/path.hpp"
#include <queue>
#include <unordered_map>

#include <pcl/kdtree/kdtree_flann.h>

//─────────────────────────────
// Wrapper function: compute_path chooses between plain A* and dynamic A*.
std::vector<int> compute_path(const Graph &graph, int start, int goal, const std::string &planner_id,
                              const visualization_msgs::msg::MarkerArray &dynamic_obstacles_markers)
{
    if (planner_id == "dynamic_a_star") {
        float inflation_radius = 2.0f;
        float inflation_weight = 200.0f;
        float cost_scaling_factor = 1.0f;
        float inscribed_radius = 0.4f;
        // Compute dynamic cost map and pass it to a_star.
        std::vector<float> dynamic_cost_map = compute_dynamic_cost_map(graph, dynamic_obstacles_markers,
                                                 inflation_radius, cost_scaling_factor, inscribed_radius, inflation_weight);
        return a_star(graph, start, goal, &dynamic_cost_map);
    } else {
        return a_star(graph, start, goal);
    }
}

/////////////////////

float heuristic_cost_estimate(const TraversablePoint &a, const TraversablePoint &b)
{
    return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2) + std::pow(a.z - b.z, 2));
}

std::vector<int> a_star(const Graph &graph, int start, int goal,
                        const std::vector<float>* extra_costs)
{
    const auto &nodes = graph.get_nodes_cloud();
    const auto &adjacency_list = graph.get_adjacency_list();

    // If there are no nodes or adjacency connections, return an empty path
    if (nodes->points.empty() || adjacency_list.empty())
    {
        return {};
    }

    // Initialize data structures for A* search
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<>> open_set;
    std::unordered_map<int, int> came_from;
    std::unordered_map<int, float> g_score;
    std::unordered_map<int, float> f_score;

    // Starting point setup
    g_score[start] = 0.0f;
    f_score[start] = heuristic_cost_estimate(nodes->points[start], nodes->points[goal]);
    open_set.emplace(f_score[start], start);

    while (!open_set.empty())
    {
        int current = open_set.top().second;
        open_set.pop();

        // Goal reached
        if (current == goal)
        {
            std::vector<int> path;
            while (current != start)
            {
                path.push_back(current);
                current = came_from[current];
            }
            path.push_back(start);
            std::reverse(path.begin(), path.end());
            return path;
        }

        // Explore neighbors
        auto it = adjacency_list.find(current);
        if (it == adjacency_list.end())
        {
            // Node has no neighbors in adjacency list
            continue;
        }

        for (const auto &neighbor : it->second)
        {
            int neighbor_idx = neighbor.first;
            float extra = (extra_costs) ? (*extra_costs)[neighbor_idx] : 0.0f;
            float tentative_g_score = g_score[current] + neighbor.second + extra;

            if (g_score.find(neighbor_idx) == g_score.end() || tentative_g_score < g_score[neighbor_idx])
            {
                // This path to neighbor is better
                came_from[neighbor_idx] = current;
                g_score[neighbor_idx] = tentative_g_score;
                f_score[neighbor_idx] = tentative_g_score + heuristic_cost_estimate(nodes->points[neighbor_idx], nodes->points[goal]);
                open_set.emplace(f_score[neighbor_idx], neighbor_idx);
            }
        }
    }

    // Return empty path if no path is found
    return {};
}

//─────────────────────────────
// Compute a dynamic cost map from dynamic obstacle markers.
std::vector<float> compute_dynamic_cost_map(const Graph &graph,
                                              const visualization_msgs::msg::MarkerArray &markers,
                                              float inflation_radius,
                                              float cost_scaling_factor,
                                              float inscribed_radius,
                                              float inflation_weight)
{
    const auto &nodes = graph.get_nodes_cloud();
    std::vector<float> cost_map(nodes->points.size(), 0.0f);
    if (markers.markers.empty())
        return cost_map;

    // Build a kd-tree for the graph nodes.
    pcl::KdTreeFLANN<TraversablePoint> nodes_tree;
    nodes_tree.setInputCloud(nodes);

    // For each marker, update the cost map.
    for (const auto &marker : markers.markers)
    {
        // Use the marker's pose as the center.
        TraversablePoint query;
        query.x = marker.pose.position.x;
        query.y = marker.pose.position.y;
        query.z = marker.pose.position.z;

        // use inflation_radius as the effective search radius.
        std::vector<int> indices;
        std::vector<float> sqr_dists;
        if (nodes_tree.radiusSearch(query, inflation_radius, indices, sqr_dists) > 0)
        {
            for (size_t i = 0; i < indices.size(); ++i)
            {
                float d = std::sqrt(sqr_dists[i]);
                float cost = 0.0f;
                if (d < inscribed_radius)
                    cost = inflation_weight;
                else if (d < inflation_radius)
                    cost = inflation_weight * std::exp(-cost_scaling_factor * (d - inscribed_radius));
                else
                    cost = 0.0f;
                cost_map[indices[i]] += cost;
            }
        }
    }
    return cost_map;
}

tf2::Quaternion compute_orientation(const Eigen::Vector3f &current_point,
                                    const Eigen::Vector3f &next_point,
                                    Eigen::Vector3f normal_vector)
{
    // Compute the movement direction (X-axis)
    Eigen::Vector3f x_axis = (next_point - current_point).normalized();
    Eigen::Vector3f z_axis = normal_vector.normalized();

    if (std::fabs(x_axis.dot(z_axis)) > 1e-6)
    {
        z_axis -= (x_axis.dot(z_axis)) * x_axis;
        z_axis.normalize();
    }

    Eigen::Vector3f y_axis = z_axis.cross(x_axis).normalized();
    tf2::Matrix3x3 rotation_matrix(
        x_axis.x(), y_axis.x(), z_axis.x(),
        x_axis.y(), y_axis.y(), z_axis.y(),
        x_axis.z(), y_axis.z(), z_axis.z());

    tf2::Quaternion quaternion;
    rotation_matrix.getRotation(quaternion);

    return quaternion;
}

nav_msgs::msg::Path convert_to_ros_path(const Graph &graph, const std::vector<int> &path, const std::string &frame_id)
{
    nav_msgs::msg::Path path_msg;
    path_msg.header.frame_id = frame_id;

    if (path.size() < 2)
    {
        return path_msg;
    }

    const auto &nodes = graph.get_nodes_cloud();

    for (size_t i = 0; i < path.size(); ++i)
    {
        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header.frame_id = frame_id;

        const auto &point = nodes->points[path[i]];
        pose_stamped.pose.position.x = point.x;
        pose_stamped.pose.position.y = point.y;
        pose_stamped.pose.position.z = point.z;

        if (i < path.size() - 1)
        {
            Eigen::Vector3f current_point(point.x, point.y, point.z);
            Eigen::Vector3f next_point(nodes->points[path[i + 1]].x, nodes->points[path[i + 1]].y, nodes->points[path[i + 1]].z);
            Eigen::Vector3f normal_vector(point.normal_x, point.normal_y, point.normal_z);

            tf2::Quaternion orientation = compute_orientation(current_point, next_point, normal_vector);
            pose_stamped.pose.orientation.x = orientation.x();
            pose_stamped.pose.orientation.y = orientation.y();
            pose_stamped.pose.orientation.z = orientation.z();
            pose_stamped.pose.orientation.w = orientation.w();
        }
        else
        {
            pose_stamped.pose.orientation = path_msg.poses.back().pose.orientation;
        }

        path_msg.poses.push_back(pose_stamped);
    }

    return path_msg;
}

void recompute_orientation(nav_msgs::msg::Path &path)
{
    int path_size = path.poses.size();

    // Iterate through the smoothed path to update orientations
    for (int i = 0; i < path_size - 1; ++i)
    {
        geometry_msgs::msg::PoseStamped &current_pose = path.poses[i];
        geometry_msgs::msg::PoseStamped &next_pose = path.poses[i + 1];

        Eigen::Vector3f current_point(current_pose.pose.position.x, current_pose.pose.position.y, current_pose.pose.position.z);
        Eigen::Vector3f next_point(next_pose.pose.position.x, next_pose.pose.position.y, next_pose.pose.position.z);
        Eigen::Vector3f normal_vector(0, 0, 1); // Assuming upward normal

        tf2::Quaternion quaternion = compute_orientation(current_point, next_point, normal_vector);

        // Assign the new orientation to the current pose
        current_pose.pose.orientation.x = quaternion.x();
        current_pose.pose.orientation.y = quaternion.y();
        current_pose.pose.orientation.z = quaternion.z();
        current_pose.pose.orientation.w = quaternion.w();
    }

    // For the last point, use the same orientation as the second-to-last point
    if (path_size > 1)
    {
        path.poses.back().pose.orientation = path.poses[path_size - 2].pose.orientation;
    }
}

nav_msgs::msg::Path smooth_path_moving_average(const nav_msgs::msg::Path &path,
                                               int window_size)
{
    nav_msgs::msg::Path smoothed_path;
    smoothed_path.header = path.header;
    int path_size = path.poses.size();

    if (window_size < 2)
    {
        return path; // No need to smooth
    }

    if (path_size < 2)
    {
        return path; // No need to smooth if path has less than 2 points
    }

    // Add the first point without modification
    smoothed_path.poses.push_back(path.poses[0]);

    // Smooth the intermediate points (positions only)
    for (int i = 1; i < path_size - 1; ++i)
    {
        double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
        int count = 0;

        // Compute the average over the window
        int start = std::max(1, i - window_size);
        int end = std::min(i + window_size, path_size - 2);

        for (int j = start; j <= end; ++j)
        {
            const auto &point = path.poses[j].pose.position;
            sum_x += point.x;
            sum_y += point.y;
            sum_z += point.z;
            ++count;
        }

        if (count > 0)
        {
            geometry_msgs::msg::PoseStamped smoothed_pose;
            smoothed_pose.header = path.poses[i].header;
            smoothed_pose.pose.position.x = sum_x / count;
            smoothed_pose.pose.position.y = sum_y / count;
            smoothed_pose.pose.position.z = sum_z / count;

            smoothed_path.poses.push_back(smoothed_pose);
        }
    }

    // Add the last point without modification
    smoothed_path.poses.push_back(path.poses.back());

    // Recompute orientation after smoothing
    recompute_orientation(smoothed_path);

    return smoothed_path;
}