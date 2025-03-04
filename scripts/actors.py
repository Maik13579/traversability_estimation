#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
import random
import math
import sys
import copy

from call_action import GraphPlanningClient
import sensor_msgs_py.point_cloud2 as pc2

# Global variables
REPLANNING_RATE = 1.0  # Hz: actor replanning rate
SIMULATION_RATE = 10.0  # Hz: simulation update rate for smooth motion
ACTOR_SPEED = 1.0     # m/s: travel speed for each actor
ACTOR_MARKER_SIZE = (0.5, 0.5, 1.0)  # Actor cube dimensions (x,y,z)
POINT_CLOUD_TOPIC = '/traversable'
ACTOR_TOPIC = '/obstacles/markers'
DYNAMIC_PLANNER = 'dynamic_a_star'
INTERPOLATION_RESOLUTION = 0.1  # m: spacing for interpolated path
MAX_REPLAN_ATTEMPTS = 3  # Maximum planning attempts before selecting a new random goal

class Actor:
    def __init__(self, actor_id, start_point, node, graph_client, point_cloud):
        self.id = actor_id
        self.node = node
        self.graph_client = graph_client
        self.point_cloud = point_cloud  # list of (x, y, z) tuples
        self.current_pose = self.create_pose(start_point)
        self.goal_pose = self.choose_new_goal()
        self.interpolated_path = []  # List of PoseStamped
        self.path_index = 0
        self.plan()

    def create_pose(self, point):
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.node.get_clock().now().to_msg()
        pose.pose.position.x = float(point[0])
        pose.pose.position.y = float(point[1])
        pose.pose.position.z = float(point[2])
        pose.pose.orientation.w = 1.0
        return pose

    def choose_new_goal(self):
        while True:
            point = random.choice(self.point_cloud)
            dx = float(point[0]) - self.current_pose.pose.position.x
            dy = float(point[1]) - self.current_pose.pose.position.y
            dz = float(point[2]) - self.current_pose.pose.position.z
            if math.sqrt(dx*dx + dy*dy + dz*dz) > 0.1:
                return self.create_pose(point)

    def plan(self):
        attempt = 0
        while attempt < MAX_REPLAN_ATTEMPTS:
            result = self.graph_client.send_goal(self.current_pose, self.goal_pose, DYNAMIC_PLANNER)
            if result is not None and len(result.path.poses) > 0:
                seconds = result.planning_time.sec + result.planning_time.nanosec * 1e-9
                self.node.get_logger().info(f"Actor {self.id} plan received (t={seconds:.2f}s) on attempt {attempt+1}")
                self.interpolated_path = self.interpolate_path(result.path.poses)
                self.path_index = 0
                return
            else:
                self.node.get_logger().warn(f"Actor {self.id} planning failed on attempt {attempt+1}/{MAX_REPLAN_ATTEMPTS}")
                attempt += 1

        self.node.get_logger().warn(f"Actor {self.id} planning failed {MAX_REPLAN_ATTEMPTS} times, selecting a new random goal")
        self.goal_pose = self.choose_new_goal()
        result = self.graph_client.send_goal(self.current_pose, self.goal_pose, DYNAMIC_PLANNER)
        if result is not None and len(result.path.poses) > 0:
            seconds = result.planning_time.sec + result.planning_time.nanosec * 1e-9
            self.node.get_logger().info(f"Actor {self.id} new goal plan received (t={seconds:.2f}s)")
            self.interpolated_path = self.interpolate_path(result.path.poses)
            self.path_index = 0


    def interpolate_path(self, poses, resolution=INTERPOLATION_RESOLUTION):
        if not poses:
            return []
        interpolated = []
        for i in range(len(poses) - 1):
            start = poses[i].pose.position
            end = poses[i+1].pose.position
            dx = end.x - start.x
            dy = end.y - start.y
            dz = end.z - start.z
            seg_len = math.sqrt(dx*dx + dy*dy + dz*dz)
            steps = max(int(seg_len / resolution), 1)
            for j in range(steps):
                t = j / steps
                pose = PoseStamped()
                pose.header.frame_id = "map"
                pose.header.stamp = self.node.get_clock().now().to_msg()
                pose.pose.position.x = start.x + t * dx
                pose.pose.position.y = start.y + t * dy
                pose.pose.position.z = start.z + t * dz
                pose.pose.orientation.w = 1.0
                interpolated.append(pose)
        interpolated.append(poses[-1])
        return interpolated

    def update(self, dt):
        if self.interpolated_path and self.path_index < len(self.interpolated_path) - 1:
            current = self.current_pose.pose.position
            next_p = self.interpolated_path[self.path_index + 1].pose.position
            dx = next_p.x - current.x
            dy = next_p.y - current.y
            dz = next_p.z - current.z
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            travel = ACTOR_SPEED * dt
            if travel >= dist:
                self.current_pose = self.interpolated_path[self.path_index + 1]
                self.path_index += 1
            else:
                ratio = travel / dist
                new_pose = PoseStamped()
                new_pose.header.frame_id = "map"
                new_pose.header.stamp = self.node.get_clock().now().to_msg()
                new_pose.pose.position.x = current.x + ratio * dx
                new_pose.pose.position.y = current.y + ratio * dy
                new_pose.pose.position.z = current.z + ratio * dz
                new_pose.pose.orientation.w = 1.0
                self.current_pose = new_pose

        if self.path_index >= len(self.interpolated_path) - 1:
            self.current_pose = self.goal_pose
            self.goal_pose = self.choose_new_goal()
            self.plan()

class ActorsNode(Node):
    def __init__(self, num_actors):
        super().__init__('actors_node')
        self.num_actors = num_actors
        self.point_cloud = None
        self.cloud_received = False
        self.actors = []

        self.marker_array_pub = self.create_publisher(MarkerArray, ACTOR_TOPIC, 10)
        self.path_publishers = {}
        for i in range(self.num_actors):
            topic = f'actor_{i}_path'
            self.path_publishers[i] = self.create_publisher(Path, topic, 10)

        self.cloud_sub = self.create_subscription(PointCloud2, POINT_CLOUD_TOPIC, self.cloud_callback, 10)
        self.graph_client = GraphPlanningClient()

        self.sim_timer = self.create_timer(1.0 / SIMULATION_RATE, self.simulation_callback)
        self.replan_timer = self.create_timer(1.0 / REPLANNING_RATE, self.replanning_callback)

    def cloud_callback(self, msg):
        if not self.cloud_received:
            self.point_cloud = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
            self.cloud_received = True
            self.get_logger().info("Received traversable points cloud.")
            self.destroy_subscription(self.cloud_sub)
            self.spawn_actors()

    def spawn_actors(self):
        for i in range(self.num_actors):
            start = random.choice(self.point_cloud)
            actor = Actor(i, start, self, self.graph_client, self.point_cloud)
            self.actors.append(actor)
        self.get_logger().info(f"Spawned {self.num_actors} actors.")

    def simulation_callback(self):
        if not self.cloud_received:
            return
        dt = 1.0 / SIMULATION_RATE
        marker_array = MarkerArray()
        for actor in self.actors:
            actor.update(dt)
            path_msg = Path()
            path_msg.header.frame_id = "map"
            path_msg.header.stamp = self.get_clock().now().to_msg()
            path_msg.poses = actor.interpolated_path
            self.path_publishers[actor.id].publish(path_msg)
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "actor"
            marker.id = actor.id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose = actor.current_pose.pose
            marker.scale.x = ACTOR_MARKER_SIZE[0]
            marker.scale.y = ACTOR_MARKER_SIZE[1]
            marker.scale.z = ACTOR_MARKER_SIZE[2]
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker_array.markers.append(marker)
        self.marker_array_pub.publish(marker_array)

    def replanning_callback(self):
        if not self.cloud_received:
            return
        for actor in self.actors:
            actor.plan()

def main(args=None):
    rclpy.init(args=args)
    if len(sys.argv) < 2:
        print("Usage: python3 actors.py <number_of_actors>")
        return
    try:
        num_actors = int(sys.argv[1])
    except Exception as e:
        print("Invalid number of actors")
        return
    node = ActorsNode(num_actors)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
