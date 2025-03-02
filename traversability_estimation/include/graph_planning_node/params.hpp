#ifndef TRAVERSABILITY_ESTIMATION__GRAPH_PLANNING_PARAMS_HPP_
#define TRAVERSABILITY_ESTIMATION__GRAPH_PLANNING_PARAMS_HPP_

#include <rclcpp/rclcpp.hpp>
#include <type_traits>

// Template function to declare and retrieve parameters
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
    int n_threads;
    std::string map_frame;
    std::string robot_frame;
};

struct CostsParams
{
    double lethal_cost;
    double distance_weight;
    double alignment_cost_weight;
    double alignment_slope_threshold;
    double traversability_weight;
};

struct GraphParams
{
    double voxel_size;
    double max_distance;
    int max_neighbors;
};

struct PathParams
{
    double dynamic_inflation_radius;
    double dynamic_inflation_weight;
    double cost_scaling_factor;
    int smoothing_window_size;
};

struct GraphPlanningNodeConfig
{
    CommonParams common;
    CostsParams costs;
    GraphParams graph;
    PathParams path;
};

void load_parameters(GraphPlanningNodeConfig &config, rclcpp::Node *node);

#endif // TRAVERSABILITY_ESTIMATION__GRAPH_PLANNING_PARAMS_HPP_
