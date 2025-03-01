#ifndef TRAVERSABILITY_ESTIMATION__TRAVERSABILITY_NODE_PARAMS_HPP_
#define TRAVERSABILITY_ESTIMATION__TRAVERSABILITY_NODE_PARAMS_HPP_

#include <rclcpp/rclcpp.hpp>
#include <type_traits>

template <typename T>
T declare_and_get_parameter(const std::string &param_name, const T &default_value, rclcpp::Node *node, const std::string &log_message)
{
    node->declare_parameter<T>(param_name, default_value);
    T param_value = node->get_parameter(param_name).get_value<T>();

    if constexpr (std::is_same<T, std::string>::value)
    {
        RCLCPP_INFO(node->get_logger(), "%s: %s", log_message.c_str(), param_value.c_str());
    }
    else
    {
        RCLCPP_INFO(node->get_logger(), "%s: %s", log_message.c_str(), std::to_string(param_value).c_str());
    }

    return param_value;
}

struct CommonParams
{
    std::string input_cloud;
    int n_threads;
    double publish_rate;
    std::string map_frame;
};

struct VoxelDownSamplingParams
{
    double voxel_size;
    bool downsample_all_data;
};

struct OutlierRemovalParams
{
    double radius;
    int neighbors;
};

struct ClusteringParams
{
    double eps;
    int min_points;
};

struct FiltersParams
{
    VoxelDownSamplingParams voxel_down_sampling;
    OutlierRemovalParams outlier_removal;
    ClusteringParams clustering;
};

struct NormalVectorEstimationParams
{
    double radius;
};

struct BoundaryEstimationParams
{
    double radius;
    double angle_threshold;
};

struct TraversabilityEstimationParams
{
    NormalVectorEstimationParams normal_vector_estimation;
    BoundaryEstimationParams boundary_estimation;
};

struct RobotParams
{
    double inscribed_radius;
    double normal_offset;
    double allowed_slope_angle;
};

struct InflationParams
{
    double radius;
    double cost_scaling_factor;
    double obstacle_normal_offset;
    double boundary_normal_offset;
};

struct CostsParams
{
    double max_cost;
    double slope_weight;
    double curvature_weight;
    double inflation_weight;
    InflationParams inflation;
};

struct TraversabilityNodeConfig
{
    CommonParams common;
    FiltersParams filters;
    TraversabilityEstimationParams traversability_estimation;
    RobotParams robot;
    CostsParams costs;
};

void load_parameters(TraversabilityNodeConfig &config, rclcpp::Node *node);

#endif // TRAVERSABILITY_ESTIMATION__TRAVERSABILITY_NODE_PARAMS_HPP_
