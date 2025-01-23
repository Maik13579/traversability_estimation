#ifndef TRAVERSABILITY_ESTIMATION__GRAPH_PLANNING_PATH_HPP_
#define TRAVERSABILITY_ESTIMATION__GRAPH_PLANNING_PATH_HPP_

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <Eigen/Dense>
#include <pcl/kdtree/kdtree_flann.h>

#include "traversability_estimation/point_types.hpp"
#include "graph_planning_node/graph.hpp"
#include "graph_planning_node/params.hpp"

/**
 * @brief Estimates the cost of moving from one traversable point to another.
 */
float heuristic_cost_estimate(const TraversablePoint &a, const TraversablePoint &b);

/**
 * @brief Compute a path from a start node to a goal node using A* algorithm.
 */
std::vector<int> compute_path(const Graph &graph, int start, int goal);

/**
 * @brief Computes the orientation as a quaternion based on the movement direction
 * and a normal vector.
 */
tf2::Quaternion compute_orientation(const Eigen::Vector3f &current_point,
                                    const Eigen::Vector3f &next_point,
                                    Eigen::Vector3f normal_vector);

/**
 * @brief Convert a path represented as a sequence of point indices to a ROS Path message.
 */
nav_msgs::msg::Path convert_to_ros_path(const Graph &graph, const std::vector<int> &path, const std::string &frame_id = "map");

/**
 * @brief Recompute the orientation of each pose in the given path.
 */
void recompute_orientation(nav_msgs::msg::Path &path);

/**
 * @brief Smooth a path using a moving average.
 */
nav_msgs::msg::Path smooth_path_moving_average(const nav_msgs::msg::Path &path, int window_size);

#endif // TRAVERSABILITY_ESTIMATION__GRAPH_PLANNING_PATH_HPP_
