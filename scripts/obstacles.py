#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import yaml
import numpy as np

from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
from rclpy.qos import QoSProfile, QoSHistoryPolicy

class Obstacle:
    def __init__(self, name, config, is_dynamic):
        self.name = name
        self.is_dynamic = is_dynamic

        # Parse the size string ("1.0, 1.0, 1.0") into a list of floats.
        size_str = config.get('size', "1.0,1.0,1.0")
        self.size = [float(x.strip()) for x in size_str.split(',')]
        # Compute an offset (half the size) so that the marker is centered.
        self.offset = np.array(self.size) / 2.0

        if is_dynamic:
            # For dynamic obstacles, store the path (list of waypoints) and speed.
            self.path = config.get('path', [])
            self.speed = config.get('speed', 0.1)  # m/s
            if len(self.path) < 2:
                raise ValueError(f"Dynamic obstacle '{name}' requires at least two waypoints in its path.")
            self.path = [np.array(pt, dtype=float) for pt in self.path]
            self.current_segment = 0
            self.current_pos = self.path[0].copy()
            self.target_pos = self.path[1].copy()
        else:
            pos_str = config.get('position', "0.0,0.0,0.0")
            self.position = np.array([float(x.strip()) for x in pos_str.split(',')], dtype=float)

    def update(self, dt):
        """Update the position of a dynamic obstacle along its path."""
        if not self.is_dynamic:
            return

        direction = self.target_pos - self.current_pos
        distance = np.linalg.norm(direction)
        if distance < 1e-6:
            self.current_segment = (self.current_segment + 1) % len(self.path)
            self.current_pos = self.path[self.current_segment].copy()
            next_index = (self.current_segment + 1) % len(self.path)
            self.target_pos = self.path[next_index].copy()
            return

        direction_norm = direction / distance
        step = self.speed * dt
        if step >= distance:
            self.current_pos = self.target_pos.copy()
            self.current_segment = (self.current_segment + 1) % len(self.path)
            next_index = (self.current_segment + 1) % len(self.path)
            self.target_pos = self.path[next_index].copy()
        else:
            self.current_pos = self.current_pos + direction_norm * step

    def get_current_position(self):
        return self.current_pos if self.is_dynamic else self.position

class ObstaclesNode(Node):
    def __init__(self, config_file):
        super().__init__('obstacles_node')

        # Load obstacle configuration from YAML file.
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        self.dynamic_obstacles = []
        self.static_obstacles = []

        if 'dynamic_obstacles' in config:
            for name, obs_config in config['dynamic_obstacles'].items():
                try:
                    obs = Obstacle(name, obs_config, is_dynamic=True)
                    self.dynamic_obstacles.append(obs)
                    self.get_logger().info(f"Loaded dynamic obstacle: {name}")
                except Exception as e:
                    self.get_logger().error(f"Error loading dynamic obstacle '{name}': {e}")

        if 'static_obstacles' in config:
            for name, obs_config in config['static_obstacles'].items():
                try:
                    obs = Obstacle(name, obs_config, is_dynamic=False)
                    self.static_obstacles.append(obs)
                    self.get_logger().info(f"Loaded static obstacle: {name}")
                except Exception as e:
                    self.get_logger().error(f"Error loading static obstacle '{name}': {e}")

        # Create a publisher for MarkerArray with a custom QoS.
        qos = QoSProfile(depth=1, history=QoSHistoryPolicy.KEEP_LAST)
        self.marker_array_pub = self.create_publisher(MarkerArray, 'obstacles/markers', qos)

        self.last_time = self.get_clock().now()
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time

        for obs in self.dynamic_obstacles:
            obs.update(dt)

        marker_array = MarkerArray()
        marker_id = 0
        for obs in self.dynamic_obstacles + self.static_obstacles:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = current_time.to_msg()
            marker.ns = "obstacles"
            marker.id = marker_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            pos = obs.get_current_position()
            marker.pose.position.x = pos[0]
            marker.pose.position.y = pos[1]
            marker.pose.position.z = pos[2]
            marker.pose.orientation.w = 1.0

            marker.scale.x = obs.size[0]
            marker.scale.y = obs.size[1]
            marker.scale.z = obs.size[2]

            if obs.is_dynamic:
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            else:
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0
            marker.color.a = 1.0

            marker_array.markers.append(marker)
            marker_id += 1

        self.marker_array_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    import sys
    if len(sys.argv) < 2:
        print("Usage: obstacles_node.py <config_file.yaml>")
        return
    node = ObstaclesNode(sys.argv[1])
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down obstacles node.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
