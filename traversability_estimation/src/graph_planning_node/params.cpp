#include "graph_planning_node/params.hpp"

void load_parameters(GraphPlanningNodeConfig &config, rclcpp::Node *node)
{
    // Load the common parameters
    config.common.n_threads = declare_and_get_parameter<int>("common.n_threads", 4, node, "n_threads");
    config.common.map_frame = declare_and_get_parameter<std::string>("common.map_frame", "map", node, "map_frame");
    config.common.robot_frame = declare_and_get_parameter<std::string>("common.robot_frame", "base_link", node, "robot_frame");

    // Load the costs parameters
    config.costs.lethal_cost = declare_and_get_parameter<double>("costs.lethal_cost", 254.0, node, "lethal_cost");
    config.costs.distance_weight = declare_and_get_parameter<double>("costs.distance_weight", 50.0, node, "distance_weight");
    config.costs.alignment_cost_weight = declare_and_get_parameter<double>("costs.alignment_cost_weight", 1.0, node, "alignment_cost_weight");
    config.costs.alignment_slope_threshold = declare_and_get_parameter<double>("costs.alignment_slope_threshold", 10.0, node, "alignment_slope_threshold");
    config.costs.traversability_weight = declare_and_get_parameter<double>("costs.traversability_weight", 1.0, node, "traversability_weight");

    // Load the graph parameters
    config.graph.voxel_size = declare_and_get_parameter<double>("graph.voxel_size", 0.05, node, "voxel_size");
    config.graph.max_distance = declare_and_get_parameter<double>("graph.max_distance", 0.25, node, "max_distance");
    config.graph.max_neighbors = declare_and_get_parameter<int>("graph.max_neighbors", 100, node, "max_neighbors");

    // Load the path parameters
    config.path.smoothing_window_size = declare_and_get_parameter<int>("path.smoothing_window_size", 5, node, "smoothing_window_size");
}