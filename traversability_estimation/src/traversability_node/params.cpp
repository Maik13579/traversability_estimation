#include "traversability_node/params.hpp"

void load_parameters(TraversabilityNodeConfig &config, rclcpp::Node *node)
{
    // Load common parameters
    config.common.input_cloud = declare_and_get_parameter<std::string>("common.input_cloud", "/input_cloud", node, "input_cloud");
    config.common.n_threads = declare_and_get_parameter<int>("common.n_threads", 1, node, "n_threads");
    config.common.publish_rate = declare_and_get_parameter<double>("common.publish_rate", 1, node, "publish_rate");
    config.common.map_frame = declare_and_get_parameter<std::string>("common.map_frame", "map", node, "map_frame");

    // Load filter parameters
    config.filters.voxel_down_sampling.voxel_size = declare_and_get_parameter<double>("filters.voxel_down_sampling.voxel_size", 0.05, node, "filters.voxel_down_sampling.voxel_size");
    config.filters.voxel_down_sampling.downsample_all_data = declare_and_get_parameter<bool>("filters.voxel_down_sampling.downsample_all_data", true, node, "filters.voxel_down_sampling.downsample_all_data");
    config.filters.outlier_removal.radius = declare_and_get_parameter<double>("filters.outlier_removal.radius", 0.2, node, "filters.outlier_removal.radius");
    config.filters.outlier_removal.neighbors = declare_and_get_parameter<int>("filters.outlier_removal.neighbors", 5, node, "filters.outlier_removal.neighbors");
    config.filters.clustering.eps = declare_and_get_parameter<double>("filters.clustering.eps", 0.25, node, "filters.clustering.eps");
    config.filters.clustering.min_points = declare_and_get_parameter<int>("filters.clustering.min_points", 25, node, "filters.clustering.min_points");

    // Traversability estimation parameters
    config.traversability_estimation.normal_vector_estimation.radius = declare_and_get_parameter<double>("traversability_estimation.normal_vector_estimation.radius", 0.25, node, "traversability_estimation.normal_vector_estimation.radius");
    config.traversability_estimation.boundary_estimation.radius = declare_and_get_parameter<double>("traversability_estimation.boundary_estimation.radius", 0.25, node, "boundary_estimation.radius");
    config.traversability_estimation.boundary_estimation.angle_threshold = declare_and_get_parameter<double>("traversability_estimation.boundary_estimation.angle_threshold", 2.0, node, "boundary_estimation.angle_threshold");

    // Robot parameters
    config.robot.inscribed_radius = declare_and_get_parameter<double>("robot.inscribed_radius", 0.4, node, "robot.inscribed_radius");
    config.robot.normal_offset = declare_and_get_parameter<double>("robot.normal_offset", 0.25, node, "robot.normal_offset");
    config.robot.allowed_slope_angle = declare_and_get_parameter<double>("robot.allowed_slope_angle", 30.0, node, "robot.allowed_slope_angle");

    // Costs parameters
    config.costs.max_cost = declare_and_get_parameter<double>("costs.max_cost", 254.0, node, "costs.max_cost");
    config.costs.slope_weight = declare_and_get_parameter<double>("costs.slope_weight", 100.0, node, "costs.slope_weight");
    config.costs.inflation_weight = declare_and_get_parameter<double>("costs.inflation_weight", 254.0, node, "costs.inflation_weight");
    config.costs.curvature_weight = declare_and_get_parameter<double>("costs.curvature_weight", 100.0, node, "costs.curvature_weight");

    // Inflation parameters
    config.costs.inflation.radius = declare_and_get_parameter<double>("costs.inflation.radius", 1.5, node, "inflation.radius");
    config.costs.inflation.cost_scaling_factor = declare_and_get_parameter<double>("costs.inflation.cost_scaling_factor", 2.5, node, "inflation.cost_scaling_factor");
    config.costs.inflation.obstacle_normal_offset = declare_and_get_parameter<double>("costs.inflation.obstacle_normal_offset", 0.1, node, "inflation.obstacle_normal_offset");
    config.costs.inflation.boundary_normal_offset = declare_and_get_parameter<double>("costs.inflation.boundary_normal_offset", -1.0, node, "inflation.boundary_normal_offset");
}