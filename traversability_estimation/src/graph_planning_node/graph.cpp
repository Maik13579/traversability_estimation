#include "graph_planning_node/graph.hpp"
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <omp.h>
#include <algorithm>

Graph::Graph()
    : traversable_cloud_(new pcl::PointCloud<TraversablePoint>),
      nodes_cloud_(new pcl::PointCloud<TraversablePoint>)
{
}

Graph::~Graph() = default;

void Graph::build_graph(pcl::PointCloud<TraversablePoint>::Ptr traversable_cloud, const GraphPlanningNodeConfig &config)
{
    const float max_distance = config.graph.max_distance;
    const int max_neighbors = config.graph.max_neighbors;
    const float lethal_cost = config.costs.lethal_cost;
    const float distance_weight = config.costs.distance_weight;
    const float alignment_cost_weight = config.costs.alignment_cost_weight;
    const float traversability_weight = config.costs.traversability_weight;
    const int n_threads = config.common.n_threads;

    // Store the original traversable cloud
    *traversable_cloud_ = *traversable_cloud;

    // Downsample the cloud to create nodes
    pcl::VoxelGrid<TraversablePoint> voxel_grid;
    voxel_grid.setInputCloud(traversable_cloud);
    voxel_grid.setLeafSize(config.graph.voxel_size, config.graph.voxel_size, config.graph.voxel_size);
    voxel_grid.filter(*nodes_cloud_);

    // Set up k-d tree for traversable cloud
    kdtree_.setInputCloud(traversable_cloud_);

    for (int i = 0; i < nodes_cloud_->points.size(); ++i)
    {
        std::vector<int> nearest_indices;
        std::vector<float> nearest_distances;

        // Find points in the original cloud near the downsampled node
        if (kdtree_.radiusSearch(nodes_cloud_->points[i], config.graph.voxel_size * 1.5, nearest_indices, nearest_distances) > 0)
        {
            // Use values from the closest point for `normal` and `cluster_id`
            int closest_point_idx = nearest_indices[0];
            const auto &closest_point = traversable_cloud_->points[closest_point_idx];

            nodes_cloud_->points[i].normal_x = closest_point.normal_x;
            nodes_cloud_->points[i].normal_y = closest_point.normal_y;
            nodes_cloud_->points[i].normal_z = closest_point.normal_z;
            nodes_cloud_->points[i].cluster_id = closest_point.cluster_id;

            // Initialize accumulators for averaging other attributes
            float total_slope = 0.0f, total_curvature = 0.0f, total_intensity = 0.0f;
            float total_inflation_cost = 0.0f, total_slope_cost = 0.0f, total_curvature_cost = 0.0f, total_final_cost = 0.0f;

            for (int idx : nearest_indices)
            {
                const auto &point = traversable_cloud_->points[idx];
                total_slope += point.slope;
                total_curvature += point.curvature;
                total_intensity += point.intensity;
                total_inflation_cost += point.inflation_cost;
                total_slope_cost += point.slope_cost;
                total_curvature_cost += point.curvature_cost;
                total_final_cost += point.final_cost;
            }

            // Compute averages for the remaining fields
            const size_t num_points = nearest_indices.size();
            nodes_cloud_->points[i].slope = total_slope / num_points;
            nodes_cloud_->points[i].curvature = total_curvature / num_points;
            nodes_cloud_->points[i].intensity = total_intensity / num_points;
            nodes_cloud_->points[i].inflation_cost = total_inflation_cost / num_points;
            nodes_cloud_->points[i].slope_cost = total_slope_cost / num_points;
            nodes_cloud_->points[i].curvature_cost = total_curvature_cost / num_points;
            nodes_cloud_->points[i].final_cost = total_final_cost / num_points;
        }
    }

    // Set up k-d tree for nodes cloud
    nodes_kdtree_.setInputCloud(nodes_cloud_);

    #pragma omp parallel for num_threads(n_threads)
    for (int i = 0; i < static_cast<int>(nodes_cloud_->points.size()); ++i)
    {
        std::vector<int> neighbor_indices;
        std::vector<float> neighbor_squared_distances;

        if (nodes_kdtree_.radiusSearch(nodes_cloud_->points[i], max_distance, neighbor_indices, neighbor_squared_distances) > 0)
        {
            int neighbors_connected = 0;
            std::vector<std::pair<int, float>> neighbors;
            for (size_t j = 0; j < neighbor_indices.size(); ++j)
            {
                neighbors.emplace_back(neighbor_indices[j], neighbor_squared_distances[j]);
            }
            std::sort(neighbors.begin(), neighbors.end(), [](const auto &a, const auto &b)
                      { return a.second < b.second; });

            for (const auto &neighbor : neighbors)
            {
                int neighbor_idx = neighbor.first;
                if (neighbor_idx == i)
                    continue;

                float distance = std::sqrt(neighbor.second);
                float alignment_cost = compute_alignment_cost(nodes_cloud_->points[i], nodes_cloud_->points[neighbor_idx], config);
                float edge_cost = distance * distance_weight \
                                + alignment_cost * alignment_cost_weight \
                                + nodes_cloud_->points[neighbor_idx].final_cost * traversability_weight;

                if (edge_cost >= lethal_cost)
                    continue;

                #pragma omp critical
                adjacency_list_[i].emplace_back(neighbor_idx, edge_cost);

                if (++neighbors_connected >= max_neighbors)
                    break;
            }
        }
    }
}

float Graph::compute_alignment_cost(const TraversablePoint &from_point, const TraversablePoint &to_point, const GraphPlanningNodeConfig &config) const
{
    const float alignment_slope_threshold = config.costs.alignment_slope_threshold;

    if (from_point.slope < alignment_slope_threshold)
        return 0.0f;

    Eigen::Vector2f movement_vector(to_point.x - from_point.x, to_point.y - from_point.y);
    float movement_norm = movement_vector.norm();
    if (movement_norm > std::numeric_limits<float>::epsilon())
        movement_vector /= movement_norm;
    else
        return 1.0f;

    Eigen::Vector2f slope_direction(from_point.normal_x, from_point.normal_y);
    float slope_norm = slope_direction.norm();
    if (slope_norm > std::numeric_limits<float>::epsilon())
        slope_direction /= slope_norm;
    else
        slope_direction.setZero();

    float dot_product = movement_vector.dot(slope_direction);
    dot_product = std::clamp(dot_product, -1.0f, 1.0f);
    return 1.0f - std::abs(dot_product);
}

int Graph::find_closest_point(const Eigen::Vector3f &point, double max_search_radius) const
{
    TraversablePoint search_point;
    search_point.x = point.x();
    search_point.y = point.y();
    search_point.z = point.z();

    std::vector<int> nearest_indices(1);
    std::vector<float> nearest_distances(1);

    if (kdtree_.radiusSearch(search_point, max_search_radius, nearest_indices, nearest_distances) > 0)
    {
        return nearest_indices[0];
    }
    else
    {
        return -1; // No point found within search radius
    }
}

int Graph::find_closest_node(const Eigen::Vector3f &point, double max_search_radius) const
{
    TraversablePoint search_point;
    search_point.x = point.x();
    search_point.y = point.y();
    search_point.z = point.z();

    std::vector<int> nearest_indices(1);
    std::vector<float> nearest_distances(1);

    if (nodes_kdtree_.radiusSearch(search_point, max_search_radius, nearest_indices, nearest_distances) > 0)
    {
        return nearest_indices[0];
    }
    else
    {
        return -1; // No node found within search radius
    }
}

traversability_estimation_interfaces::msg::Graph Graph::to_ros_message() const
{
    traversability_estimation_interfaces::msg::Graph graph_msg;

    // Convert both clouds to ROS messages
    pcl::toROSMsg(*traversable_cloud_, graph_msg.traversable_cloud);
    pcl::toROSMsg(*nodes_cloud_, graph_msg.nodes);

    // Populate the edges in the message
    for (const auto &entry : adjacency_list_)
    {
        int from_node = entry.first;
        for (const auto &neighbor : entry.second)
        {
            int to_node = neighbor.first;
            float cost = neighbor.second;

            graph_msg.edge_start_indices.push_back(from_node);
            graph_msg.edge_end_indices.push_back(to_node);
            graph_msg.edge_costs.push_back(cost);
        }
    }

    return graph_msg;
}
