#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose as Nav2ComputePathToPose

class GraphPlanningClient(Node):

    def __init__(self):
        super().__init__('graph_planning_client')
        self._action_client = ActionClient(self, Nav2ComputePathToPose, 'compute_path_to_pose')

        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()

    def send_goal(self, start_pose: PoseStamped, goal_pose: PoseStamped, planner_id: str='a_star'):
        """
        Send goal to the action server and return the result synchronously.

        Parameters:
        - start_pose: PoseStamped, start pose of the path
        - goal_pose: PoseStamped, goal pose of the path

        Returns:
        - result: Result of the action, including path and smooth_path.
        """
        goal_msg = Nav2ComputePathToPose.Goal()
        goal_msg.start = start_pose
        goal_msg.goal = goal_pose
        goal_msg.use_start = True
        goal_msg.planner_id = planner_id

        self.get_logger().info('Sending goal...')
        send_goal_future = self._action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        
        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return None

        self.get_logger().info('Goal accepted')
        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, get_result_future)
        
        result = get_result_future.result().result
        return result
    

