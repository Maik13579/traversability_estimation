#ifndef TRAVERSABILITY_ESTIMATION__GRAPH_PLANNING_NODE_HPP_
#define TRAVERSABILITY_ESTIMATION__GRAPH_PLANNING_NODE_HPP_

// Other headers
#include "traversability_estimation/point_types.hpp"
#include "graph_planning_node/params.hpp"
#include "graph_planning_node/graph.hpp"
#include "graph_planning_node/path.hpp"

#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <nav_msgs/msg/path.hpp>

#include <rclcpp_action/rclcpp_action.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "nav2_msgs/action/compute_path_to_pose.hpp"
#include "traversability_estimation_interfaces/msg/graph.hpp"
#include "traversability_estimation_interfaces/srv/get_graph.hpp"

class GraphPlanningNode : public rclcpp::Node
{
public:
    GraphPlanningNode();
    ~GraphPlanningNode();

private:
    GraphPlanningNodeConfig config_;

    // Subscription for input point cloud and dynamic obstacle cloud
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_dynamic_obstacles_;

    // Services
    rclcpp::Service<traversability_estimation_interfaces::srv::GetGraph>::SharedPtr get_graph_service_;

    // Action server for computing paths
    rclcpp_action::Server<nav2_msgs::action::ComputePathToPose>::SharedPtr action_server_;

    // TF2 listener and buffer for transformations
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    Graph graph_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr dynamic_obstacles_;

    // Callbacks
    void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void callback_dynamic_obstacles(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    
    void handle_get_graph_request(
        const std::shared_ptr<traversability_estimation_interfaces::srv::GetGraph::Request> request,
        std::shared_ptr<traversability_estimation_interfaces::srv::GetGraph::Response> response);

    rclcpp_action::GoalResponse handle_goal(
        const rclcpp_action::GoalUUID &uuid,
        std::shared_ptr<const nav2_msgs::action::ComputePathToPose::Goal> goal);
    rclcpp_action::CancelResponse handle_cancel(
        const std::shared_ptr<rclcpp_action::ServerGoalHandle<nav2_msgs::action::ComputePathToPose>> goal_handle);
    void handle_accepted(
        const std::shared_ptr<rclcpp_action::ServerGoalHandle<nav2_msgs::action::ComputePathToPose>> goal_handle);
    void execute(
        const std::shared_ptr<rclcpp_action::ServerGoalHandle<nav2_msgs::action::ComputePathToPose>> goal_handle);
};

#endif // TRAVERSABILITY_ESTIMATION__GRAPH_PLANNING_NODE_HPP_
