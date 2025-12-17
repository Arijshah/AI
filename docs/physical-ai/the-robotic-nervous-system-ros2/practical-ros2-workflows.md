---
sidebar_label: 'Practical ROS 2 Workflows'
title: 'Practical ROS 2 Workflows'
---

# Practical ROS 2 Workflows

## Overview

This lesson focuses on practical workflows and best practices for developing, testing, and deploying ROS 2 applications. We'll cover the complete development lifecycle from workspace setup to deployment, including debugging techniques, testing strategies, and tools that make ROS 2 development more efficient and robust.

Understanding these practical workflows is essential for building production-ready robotic applications that are maintainable, scalable, and reliable.

## Learning Outcomes

By the end of this lesson, you will be able to:
- Set up and organize ROS 2 workspaces effectively
- Use ROS 2 tools for debugging and monitoring systems
- Implement testing strategies for ROS 2 nodes
- Apply best practices for package organization and documentation
- Deploy ROS 2 applications in different environments

## Hands-on Steps

1. **Workspace Setup**: Create and organize a ROS 2 workspace
2. **Package Creation**: Build a complete ROS 2 package with multiple nodes
3. **Testing Implementation**: Add unit and integration tests
4. **Debugging Techniques**: Use ROS 2 tools for debugging
5. **Deployment Preparation**: Package for distribution

### Prerequisites

- Understanding of ROS 2 concepts (nodes, topics, services, parameters)
- Experience with Python and basic ROS 2 node development
- Familiarity with Linux command line

## Code Examples

Let's start by creating a complete ROS 2 package structure with multiple nodes and proper organization:

```python
# robot_control_package/robot_control_nodes/robot_control_nodes/motion_controller.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from rclpy.qos import QoSProfile, ReliabilityPolicy
import math

class MotionController(Node):
    """
    A comprehensive motion controller node that integrates
    navigation, obstacle avoidance, and state management
    """
    def __init__(self):
        super().__init__('motion_controller')

        # Declare parameters
        self.declare_parameter('linear_speed', 0.3)
        self.declare_parameter('angular_speed', 0.5)
        self.declare_parameter('obstacle_threshold', 1.0)
        self.declare_parameter('safety_margin', 0.5)

        # Get parameter values
        self.linear_speed = self.get_parameter('linear_speed').value
        self.angular_speed = self.get_parameter('angular_speed').value
        self.obstacle_threshold = self.get_parameter('obstacle_threshold').value
        self.safety_margin = self.get_parameter('safety_margin').value

        # Create QoS profiles
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/motion_status', 10)

        # Subscribers
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, sensor_qos)

        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

        # Timers
        self.control_timer = self.create_timer(0.1, self.control_loop)

        # Robot state
        self.current_pose = Point()
        self.current_twist = Twist()
        self.obstacle_detected = False
        self.obstacle_distance = float('inf')
        self.robot_state = "IDLE"
        self.goal_reached = False
        self.navigation_active = False

        # Goal position (will be set by external commands)
        self.goal_position = Point()
        self.goal_position.x = 0.0
        self.goal_position.y = 0.0

        # Logging
        self.get_logger().info(f"Motion Controller initialized with speed: {self.linear_speed}m/s")

    def laser_callback(self, msg):
        """Process laser scan data for obstacle detection"""
        # Get minimum distance in front (±30 degrees)
        front_ranges = msg.ranges[330:] + msg.ranges[:30]
        front_ranges = [r for r in front_ranges if not math.isnan(r) and 0.1 < r < 10.0]

        if front_ranges:
            self.obstacle_distance = min(front_ranges)
            self.obstacle_detected = self.obstacle_distance < (self.obstacle_threshold + self.safety_margin)
        else:
            self.obstacle_distance = float('inf')
            self.obstacle_detected = False

    def odom_callback(self, msg):
        """Process odometry data to get current pose"""
        self.current_pose.x = msg.pose.pose.position.x
        self.current_pose.y = msg.pose.pose.position.y
        self.current_pose.z = msg.pose.pose.position.z

        # Extract orientation (simplified - just using z component for 2D)
        self.current_twist.linear.x = msg.twist.twist.linear.x
        self.current_twist.linear.y = msg.twist.twist.linear.y
        self.current_twist.angular.z = msg.twist.twist.angular.z

    def set_goal(self, x, y):
        """Set navigation goal"""
        self.goal_position.x = x
        self.goal_position.y = y
        self.navigation_active = True
        self.goal_reached = False
        self.get_logger().info(f"Navigation goal set to: ({x}, {y})")

    def calculate_control_command(self):
        """Calculate control command based on current state"""
        cmd = Twist()

        if not self.navigation_active:
            return cmd

        # Calculate distance to goal
        dx = self.goal_position.x - self.current_pose.x
        dy = self.goal_position.y - self.current_pose.y
        distance_to_goal = math.sqrt(dx*dx + dy*dy)

        # Check if goal is reached
        if distance_to_goal < 0.2:  # 20cm tolerance
            self.goal_reached = True
            self.navigation_active = False
            self.robot_state = "GOAL_REACHED"
            return cmd

        # Calculate desired angle to goal
        desired_angle = math.atan2(dy, dx)

        # Simple proportional controller for angular velocity
        angle_diff = desired_angle
        cmd.angular.z = max(-self.angular_speed, min(self.angular_speed, 2.0 * angle_diff))

        # Move forward if not rotating significantly and no obstacle
        if abs(cmd.angular.z) < 0.2 and not self.obstacle_detected:
            cmd.linear.x = self.linear_speed
            self.robot_state = "MOVING_TO_GOAL"
        elif self.obstacle_detected:
            cmd.linear.x = 0.0
            cmd.angular.z = self.angular_speed  # Rotate to avoid
            self.robot_state = "OBSTACLE_AVOIDANCE"
        else:
            cmd.linear.x = 0.0
            self.robot_state = "ROTATING_TO_GOAL"

        return cmd

    def control_loop(self):
        """Main control loop"""
        cmd = self.calculate_control_command()

        # Publish command
        self.cmd_vel_pub.publish(cmd)

        # Publish status
        status_msg = String()
        status_msg.data = f"State: {self.robot_state}, Goal: ({self.goal_position.x:.2f}, {self.goal_position.y:.2f}), " \
                         f"Pos: ({self.current_pose.x:.2f}, {self.current_pose.y:.2f}), " \
                         f"Obstacle: {self.obstacle_detected}, Distance: {self.obstacle_distance:.2f}m"
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    controller = MotionController()

    # Example: Set a goal after startup
    import time
    time.sleep(1)  # Wait for connections
    controller.set_goal(2.0, 1.0)  # Move to (2, 1)

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info("Motion Controller stopped by user")
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Now let's create a diagnostic node that monitors the health of our ROS 2 system:

```python
# robot_control_package/robot_control_nodes/robot_control_nodes/system_monitor.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from rclpy.qos import QoSProfile
import psutil
import socket

class SystemMonitor(Node):
    """
    System monitoring node that publishes diagnostic information
    """
    def __init__(self):
        super().__init__('system_monitor')

        # Publishers
        self.diag_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 10)
        self.status_pub = self.create_publisher(String, '/system_status', 10)

        # Timer for diagnostics
        self.diag_timer = self.create_timer(1.0, self.publish_diagnostics)

        self.get_logger().info("System Monitor initialized")

    def get_cpu_usage(self):
        """Get CPU usage percentage"""
        return psutil.cpu_percent(interval=1)

    def get_memory_usage(self):
        """Get memory usage percentage"""
        memory = psutil.virtual_memory()
        return memory.percent

    def get_disk_usage(self):
        """Get disk usage percentage"""
        disk = psutil.disk_usage('/')
        return (disk.used / disk.total) * 100

    def get_network_status(self):
        """Get basic network status"""
        try:
            # Try to connect to an external server to check internet
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            return False

    def publish_diagnostics(self):
        """Publish system diagnostic information"""
        diag_array = DiagnosticArray()
        diag_array.header.stamp = self.get_clock().now().to_msg()

        # CPU Diagnostic
        cpu_diag = DiagnosticStatus()
        cpu_diag.name = "CPU Usage"
        cpu_usage = self.get_cpu_usage()
        cpu_diag.level = DiagnosticStatus.OK if cpu_usage < 80 else DiagnosticStatus.WARN if cpu_usage < 90 else DiagnosticStatus.ERROR
        cpu_diag.message = f"CPU usage: {cpu_usage}%"
        cpu_diag.hardware_id = "cpu"
        cpu_diag.values = [
            {"key": "usage_percent", "value": str(cpu_usage)},
            {"key": "status", "value": "OK" if cpu_usage < 80 else "HIGH"}
        ]
        diag_array.status.append(cpu_diag)

        # Memory Diagnostic
        mem_diag = DiagnosticStatus()
        mem_diag.name = "Memory Usage"
        mem_usage = self.get_memory_usage()
        mem_diag.level = DiagnosticStatus.OK if mem_usage < 80 else DiagnosticStatus.WARN if mem_usage < 90 else DiagnosticStatus.ERROR
        mem_diag.message = f"Memory usage: {mem_usage}%"
        mem_diag.hardware_id = "memory"
        mem_diag.values = [
            {"key": "usage_percent", "value": str(mem_usage)},
            {"key": "status", "value": "OK" if mem_usage < 80 else "HIGH"}
        ]
        diag_array.status.append(mem_diag)

        # Disk Diagnostic
        disk_diag = DiagnosticStatus()
        disk_diag.name = "Disk Usage"
        disk_usage = self.get_disk_usage()
        disk_diag.level = DiagnosticStatus.OK if disk_usage < 80 else DiagnosticStatus.WARN if disk_usage < 90 else DiagnosticStatus.ERROR
        disk_diag.message = f"Disk usage: {disk_usage:.1f}%"
        disk_diag.hardware_id = "disk"
        disk_diag.values = [
            {"key": "usage_percent", "value": f"{disk_usage:.1f}"},
            {"key": "status", "value": "OK" if disk_usage < 80 else "HIGH"}
        ]
        diag_array.status.append(disk_diag)

        # Network Diagnostic
        net_diag = DiagnosticStatus()
        net_diag.name = "Network Connectivity"
        net_connected = self.get_network_status()
        net_diag.level = DiagnosticStatus.OK if net_connected else DiagnosticStatus.ERROR
        net_diag.message = "Network: Connected" if net_connected else "Network: Disconnected"
        net_diag.hardware_id = "network"
        net_diag.values = [
            {"key": "connected", "value": str(net_connected)},
            {"key": "status", "value": "OK" if net_connected else "ERROR"}
        ]
        diag_array.status.append(net_diag)

        # Publish diagnostics
        self.diag_pub.publish(diag_array)

        # Publish simple status
        status_msg = String()
        status_msg.data = f"CPU: {cpu_usage:.1f}%, Mem: {mem_usage:.1f}%, Disk: {disk_usage:.1f}%, Net: {'UP' if net_connected else 'DOWN'}"
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    monitor = SystemMonitor()

    try:
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        monitor.get_logger().info("System Monitor stopped by user")
    finally:
        monitor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Let's also create a launch file to bring up our complete system:

```python
# robot_control_package/robot_control_nodes/launch/navigation_system.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    log_level = LaunchConfiguration('log_level', default='info')

    # Declare launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )

    declare_log_level = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Log level for nodes'
    )

    # Motion controller node
    motion_controller = Node(
        package='robot_control_nodes',
        executable='motion_controller',
        name='motion_controller',
        parameters=[
            {'linear_speed': 0.3},
            {'angular_speed': 0.5},
            {'obstacle_threshold': 1.0},
            {'safety_margin': 0.5}
        ],
        remappings=[
            ('/cmd_vel', '/cmd_vel'),
            ('/scan', '/scan'),
            ('/odom', '/odom'),
        ],
        arguments=['--ros-args', '--log-level', log_level],
        additional_env={'RCUTILS_LOGGING_USE_STDOUT': '1'},
    )

    # System monitor node
    system_monitor = Node(
        package='robot_control_nodes',
        executable='system_monitor',
        name='system_monitor',
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        arguments=['--ros-args', '--log-level', log_level],
    )

    # Joint state publisher (for visualization)
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
    )

    # Robot state publisher (for TF transforms)
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        arguments=[os.path.join(get_package_share_directory('robot_description'), 'urdf', 'robot.urdf')]
        if os.path.exists(os.path.join(get_package_share_directory('robot_description'), 'urdf', 'robot.urdf'))
        else []
    )

    # RViz2 for visualization
    rviz_config = os.path.join(
        get_package_share_directory('robot_control_nodes'),
        'rviz',
        'navigation_system.rviz'
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        condition='true'  # Only run if condition is met
    )

    # Return launch description
    return LaunchDescription([
        declare_use_sim_time,
        declare_log_level,
        motion_controller,
        system_monitor,
        joint_state_publisher,
        robot_state_publisher,
        # rviz  # Commented out as RViz config may not exist yet
    ])
```

## Small Simulation

Let's create a testing framework for our ROS 2 nodes to demonstrate proper testing practices:

```python
# robot_control_package/robot_control_nodes/test/test_motion_controller.py
import unittest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import threading
import time

class TestMotionController(unittest.TestCase):
    """
    Unit tests for the MotionController node
    """
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        # Create a test node to interact with the motion controller
        self.test_node = Node('test_motion_controller')
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.test_node)

        # Publishers for test inputs
        self.laser_pub = self.test_node.create_publisher(LaserScan, '/scan', 10)
        self.odom_pub = self.test_node.create_publisher(Odometry, '/odom', 10)

        # Subscribers for test outputs
        self.cmd_vel_sub = self.test_node.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.status_sub = self.test_node.create_subscription(
            String, '/motion_status', self.status_callback, 10)

        # Test variables
        self.last_cmd_vel = None
        self.last_status = None

    def cmd_vel_callback(self, msg):
        self.last_cmd_vel = msg

    def status_callback(self, msg):
        self.last_status = msg

    def create_laser_scan_msg(self, ranges):
        """Create a LaserScan message with specified ranges"""
        from sensor_msgs.msg import LaserScan
        msg = LaserScan()
        msg.header.stamp = self.test_node.get_clock().now().to_msg()
        msg.header.frame_id = 'laser_frame'
        msg.angle_min = -math.pi/2
        msg.angle_max = math.pi/2
        msg.angle_increment = math.pi / len(ranges)
        msg.time_increment = 0.0
        msg.scan_time = 0.0
        msg.range_min = 0.1
        msg.range_max = 10.0
        msg.ranges = ranges
        return msg

    def test_obstacle_detection(self):
        """Test that the controller stops when obstacles are detected"""
        # Publish a laser scan with an obstacle close in front
        obstacle_ranges = [0.5] * 360  # All ranges are 0.5m (close obstacle)
        obstacle_scan = self.create_laser_scan_msg(obstacle_ranges)

        self.laser_pub.publish(obstacle_scan)
        time.sleep(0.2)  # Wait for processing

        # Check that linear velocity is zero (stopped due to obstacle)
        if self.last_cmd_vel:
            self.assertEqual(self.last_cmd_vel.linear.x, 0.0)

    def test_clear_path_movement(self):
        """Test that the controller moves forward when path is clear"""
        # Publish a laser scan with no obstacles
        clear_ranges = [5.0] * 360  # All ranges are 5m (clear path)
        clear_scan = self.create_laser_scan_msg(clear_ranges)

        self.laser_pub.publish(clear_scan)
        time.sleep(0.2)  # Wait for processing

        # Check that robot is moving forward (linear velocity > 0)
        if self.last_cmd_vel:
            self.assertGreater(self.last_cmd_vel.linear.x, 0.0)

    def test_rotation_behavior(self):
        """Test rotation when obstacles are detected"""
        # Publish a laser scan with obstacle on the left
        ranges = [5.0] * 360
        ranges[0:90] = [0.3] * 90  # Obstacle on the left
        left_obstacle_scan = self.create_laser_scan_msg(ranges)

        self.laser_pub.publish(left_obstacle_scan)
        time.sleep(0.2)  # Wait for processing

        # Check that robot rotates (angular velocity != 0)
        if self.last_cmd_vel:
            self.assertNotEqual(self.last_cmd_vel.angular.z, 0.0)

def main():
    unittest.main()

if __name__ == '__main__':
    main()
```

Let's also create a command-line tool for managing our ROS 2 system:

```python
# robot_control_package/robot_control_nodes/robot_control_nodes/system_manager.py
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.qos import QoSProfile
import sys
import argparse

class SystemManager(Node):
    """
    Command-line interface for managing the robot system
    """
    def __init__(self):
        super().__init__('system_manager')

        # Publisher for system commands
        self.command_pub = self.create_publisher(String, '/system_command', 10)

        self.get_logger().info("System Manager ready")

    def send_command(self, command):
        """Send a command to the system"""
        msg = String()
        msg.data = command
        self.command_pub.publish(msg)
        self.get_logger().info(f"Sent command: {command}")

def main():
    parser = argparse.ArgumentParser(description='ROS 2 System Manager')
    parser.add_argument('command', choices=['start', 'stop', 'reset', 'calibrate', 'emergency_stop'],
                       help='System command to execute')
    parser.add_argument('--node', help='Specific node to target')

    args = parser.parse_args()

    rclpy.init()
    manager = SystemManager()

    # Wait for publisher to connect
    time.sleep(0.5)

    # Send the command
    if args.node:
        command = f"{args.command}:{args.node}"
    else:
        command = args.command

    manager.send_command(command)
    manager.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    import time
    main()
```

## Quick Recap

In this lesson, we've covered:

- **Complete Package Structure**: Creating well-organized ROS 2 packages with proper file hierarchy
- **System Monitoring**: Implementing diagnostic nodes to monitor system health
- **Launch Files**: Creating comprehensive launch files for system deployment
- **Testing Framework**: Developing unit tests for ROS 2 nodes
- **Command Line Tools**: Building management tools for system operation
- **Best Practices**: Proper error handling, logging, and parameter management

These practical workflows form the foundation of professional ROS 2 development. Proper organization, testing, and monitoring are essential for creating robust robotic applications that can be deployed in real-world scenarios.

The combination of well-structured packages, comprehensive testing, and effective monitoring tools ensures that your ROS 2 applications are maintainable, reliable, and ready for production use.

## Summary of Chapter 2

In Chapter 2: "The Robotic Nervous System (ROS 2)", we've covered:

1. **Introduction to ROS 2**: Core concepts including nodes, topics, services, and actions
2. **Advanced Nodes and Parameters**: Creating sophisticated nodes with dynamic configuration
3. **URDF and Humanoid Modeling**: Creating detailed robot models for simulation and control
4. **Practical Workflows**: Complete development, testing, and deployment practices

This chapter provides a solid foundation for understanding how ROS 2 serves as the "nervous system" of robotic applications, enabling complex systems to communicate and coordinate effectively.