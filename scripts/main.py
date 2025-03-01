#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from call_action import GraphPlanningClient
from generate_cloud import generate_cloud

from geometry_msgs.msg import PoseStamped, PointStamped
from visualization_msgs.msg import Marker
from nav_msgs.msg import Path

class DemoNode(Node):
    def __init__(self, path):
        super().__init__('demo_node')

        # Generate pointcloud
        generate_cloud(path=path, voxel_size=0.1)
        self.path = path

        # Initialize ROS subscriptions and publishers
        self.clicked_point_sub = self.create_subscription(PointStamped, 'clicked_point', self.clicked_point_callback, 10)
        self.world_pub = self.create_publisher(Marker, 'world', 10)
        self.start_marker_pub = self.create_publisher(Marker, 'start_marker', 10)
        self.end_marker_pub = self.create_publisher(Marker, 'end_marker', 10)

        self.planners = {
            'a_star': self.create_publisher(Path, 'path/a_star', 10),
            'dynamic_a_star': self.create_publisher(Path, 'path/dynamic_a_star', 10)
        }

        self.graph_client = GraphPlanningClient()

        # Track the number of clicks
        self.click_count = 1
        self.publish_world()

        # Timer callback to repeatedly call the action client
        self.action_timer = self.create_timer(0.25, self.timer_callback)


    def publish_world(self):
        """Publish a marker representing the world mesh."""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "world"
        marker.id = 0
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD
        marker.scale.x = marker.scale.y = marker.scale.z = 1.0
        marker.color.a = 1.0
        marker.color.r = marker.color.g = marker.color.b = 1.0
        marker.mesh_resource = "file://" + self.path
        marker.mesh_use_embedded_materials = False
        self.world_pub.publish(marker)

    def create_pose_stamped(self, msg):
        """Convert a PointStamped message to a PoseStamped message with a fixed orientation."""
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose.position = msg.point
        pose.pose.orientation.w = 1.0
        return pose

    def clicked_point_callback(self, msg):
        """Callback to handle clicked points and alternate between start and goal markers."""
        clicked_pose = self.create_pose_stamped(msg)
        self.click_count += 1

        if self.click_count % 2 == 0:
            # Store the clicked point as the start position
            self.start_pose = clicked_pose
            # Publish start marker and TF for base_link
            self.publish_start_marker(self.start_pose)
            self.get_logger().info("Start position set.")
        else:
            # Store the clicked point as the goal position
            self.goal_pose = clicked_pose
            # Publish end marker, TF for goal, and send goals to clients
            self.publish_end_marker(self.goal_pose)
            
           

    def timer_callback(self):
        """Timer callback that calls the action repeatedly if start and goal are available."""
        self.publish_world()
        if hasattr(self, 'start_pose') and hasattr(self, 'goal_pose'):

            for planner, pub in self.planners.items():
                result = self.graph_client.send_goal(self.start_pose, self.goal_pose, planner)
                if result is not None:
                    secconds = result.planning_time.sec + result.planning_time.nanosec * 1e-9
                    self.get_logger().info(planner+" planning time: "+str(secconds))
                    pub.publish(result.path)

    def publish_marker(self, pose, publisher, ns, marker_id, r, g, b):
        """Publish a marker with specified color to represent the start or end position."""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = pose.pose.position
        marker.scale.x = marker.scale.y = marker.scale.z = 1.0
        marker.color.a = 1.0
        marker.color.r, marker.color.g, marker.color.b = r, g, b
        publisher.publish(marker)

    def publish_start_marker(self, start_pose):
        """Publish the start marker."""
        self.publish_marker(start_pose, self.start_marker_pub, "start_marker", 0, 1.0, 0.0, 0.0)
        self.get_logger().info(f"Published start position and base_link frame at ({start_pose.pose.position.x}, {start_pose.pose.position.y}, {start_pose.pose.position.z})")

    def publish_end_marker(self, end_pose):
        """Publish the end marker."""
        self.publish_marker(end_pose, self.end_marker_pub, "end_marker", 1, 0.0, 1.0, 0.0)
        self.get_logger().info(f"Published goal position and goal frame at ({end_pose.pose.position.x}, {end_pose.pose.position.y}, {end_pose.pose.position.z})")


def main(args=None):
    """Initialize ROS and run the node."""
    rclpy.init(args=args)
    import sys
    node = DemoNode(path=sys.argv[1])
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down due to keyboard interrupt.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
