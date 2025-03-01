#include "graph_planning_node/node.hpp"
#include <chrono>
#include <thread>
#include <Eigen/Dense>
#include <pcl_conversions/pcl_conversions.h>

GraphPlanningNode::GraphPlanningNode()
    : rclcpp::Node("graph_planning_node"),
      tf_buffer_(this->get_clock()),
      tf_listener_(tf_buffer_)
{
    load_parameters(config_, this);
    omp_set_num_threads(config_.common.n_threads);

    // Initialize subscriber
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "traversable", 1, std::bind(&GraphPlanningNode::callback, this, std::placeholders::_1));

    sub_dynamic_obstacles_ = this->create_subscription<visualization_msgs::msg::MarkerArray>(
        "obstacles/markers", 1, std::bind(&GraphPlanningNode::callback_dynamic_obstacles, this, std::placeholders::_1));

    // Initialize service
    get_graph_service_ = this->create_service<traversability_estimation_interfaces::srv::GetGraph>(
        "get_graph",
        std::bind(&GraphPlanningNode::handle_get_graph_request, this, std::placeholders::_1, std::placeholders::_2));

    // Initialize action
    action_server_ = rclcpp_action::create_server<nav2_msgs::action::ComputePathToPose>(
        this,
        "compute_path_to_pose",
        std::bind(&GraphPlanningNode::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
        std::bind(&GraphPlanningNode::handle_cancel, this, std::placeholders::_1),
        std::bind(&GraphPlanningNode::handle_accepted, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "Graph planning node initialized.");
}

GraphPlanningNode::~GraphPlanningNode() = default;

void GraphPlanningNode::callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    if (!graph_.get_nodes_cloud()->points.empty())
    {
        return; // Graph already built, skip
    }

    RCLCPP_INFO(this->get_logger(), "Received traversable cloud with %u points.", msg->width * msg->height);
    pcl::fromROSMsg(*msg, *graph_.get_nodes_cloud());

    graph_.build_graph(graph_.get_nodes_cloud(), config_);
    RCLCPP_INFO(this->get_logger(), "Graph built with %ld nodes and %ld edges.",
                graph_.get_nodes_cloud()->points.size(), graph_.get_adjacency_list().size());
}

void GraphPlanningNode::callback_dynamic_obstacles(const visualization_msgs::msg::MarkerArray::SharedPtr msg)
{
    dynamic_obstacles_markers_ = *msg;
}



void GraphPlanningNode::handle_get_graph_request(
    const std::shared_ptr<traversability_estimation_interfaces::srv::GetGraph::Request> /*request*/,
    std::shared_ptr<traversability_estimation_interfaces::srv::GetGraph::Response> response)
{
    // Convert the current graph to a ROS message format
    response->graph = graph_.to_ros_message();

    // Set the headers for the graph and nodes
    response->graph.header.frame_id = config_.common.map_frame;
    response->graph.header.stamp = this->now();
    response->graph.nodes.header.frame_id = config_.common.map_frame;
    response->graph.nodes.header.stamp = this->now();

    RCLCPP_INFO(this->get_logger(), "Graph service request handled.");
}

rclcpp_action::GoalResponse GraphPlanningNode::handle_goal(
    const rclcpp_action::GoalUUID &uuid,
    std::shared_ptr<const nav2_msgs::action::ComputePathToPose::Goal> goal)
{
    RCLCPP_INFO(this->get_logger(), "Received path computation request.");

    if (graph_.get_nodes_cloud()->points.empty())
    {
        RCLCPP_WARN(this->get_logger(), "Graph not built yet.");
        return rclcpp_action::GoalResponse::REJECT;
    }

    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse GraphPlanningNode::handle_cancel(
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<nav2_msgs::action::ComputePathToPose>> goal_handle)
{
    RCLCPP_INFO(this->get_logger(), "Path computation goal canceled.");
    return rclcpp_action::CancelResponse::ACCEPT;
}

void GraphPlanningNode::handle_accepted(
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<nav2_msgs::action::ComputePathToPose>> goal_handle)
{
    std::thread{std::bind(&GraphPlanningNode::execute, this, goal_handle)}.detach();
}

void GraphPlanningNode::execute(
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<nav2_msgs::action::ComputePathToPose>> goal_handle)
{
    auto result = std::make_shared<nav2_msgs::action::ComputePathToPose::Result>();
    auto start_time = std::chrono::steady_clock::now();
    const auto goal = goal_handle->get_goal();
    Eigen::Vector3f start_position, goal_position;
    //double max_search_radius = goal->max_search_radius;
    double max_search_radius = 1.0; //TODO add param
    if (max_search_radius <= 0.0)
    {
        max_search_radius = std::numeric_limits<double>::infinity();
    }

    // Determine start position
    if (!goal->use_start)
    {
        try
        {
            auto transform_stamped = tf_buffer_.lookupTransform(config_.common.map_frame, config_.common.robot_frame, tf2::TimePointZero);
            start_position = Eigen::Vector3f(
                transform_stamped.transform.translation.x,
                transform_stamped.transform.translation.y,
                transform_stamped.transform.translation.z);
        }
        catch (const tf2::TransformException &ex)
        {
            RCLCPP_ERROR(this->get_logger(), "Transform error: %s", ex.what());
            // result->error_code = nav2_msgs::action::ComputePathToPose::Result::TF_ERROR;
            // result->error_msg = "Failed to lookup the robot's current pose.";
            goal_handle->abort(result);
            return;
        }
    }
    else
    {
        start_position = Eigen::Vector3f(goal->start.pose.position.x, goal->start.pose.position.y, goal->start.pose.position.z);
    }

    // Determine goal position
    goal_position = Eigen::Vector3f(goal->goal.pose.position.x, goal->goal.pose.position.y, goal->goal.pose.position.z);

    // Find closest points in graph to start and goal positions
    int start_idx = graph_.find_closest_node(start_position, max_search_radius);
    int goal_idx = graph_.find_closest_node(goal_position, max_search_radius);

    if (start_idx == -1)
    {
        // result->error_code = nav2_msgs::action::ComputePathToPose::Result::START_OUTSIDE_MAP;
        // result->error_msg = "Start position is outside the map or invalid.";
        goal_handle->abort(result);
        return;
    }

    if (goal_idx == -1)
    {
        // result->error_code = nav2_msgs::action::ComputePathToPose::Result::GOAL_OUTSIDE_MAP;
        // result->error_msg = "Goal position is outside the map or invalid.";
        goal_handle->abort(result);
        return;
    }

    // Compute path using graph data
    auto path = compute_path(graph_, start_idx, goal_idx, goal->planner_id, dynamic_obstacles_markers_);
    if (path.empty())
    {
        // result->error_code = nav2_msgs::action::ComputePathToPose::Result::NO_VALID_PATH;
        // result->error_msg = "Failed to compute a valid path.";
        goal_handle->abort(result);
        return;
    }

    // End timing
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> planning_duration = end_time - start_time;

    // Populate result with computed path
    auto result_path = convert_to_ros_path(graph_, path, config_.common.map_frame);
    result->path = smooth_path_moving_average(result_path, config_.path.smoothing_window_size);
    result->planning_time = rclcpp::Duration::from_seconds(planning_duration.count());
    // result->error_code = nav2_msgs::action::ComputePathToPose::Result::NONE;
    // result->error_msg = "";
    goal_handle->succeed(result);
}
