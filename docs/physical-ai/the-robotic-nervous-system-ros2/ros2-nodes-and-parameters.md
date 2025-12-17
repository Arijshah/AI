---
sidebar_label: 'ROS 2 Nodes and Parameters'
title: 'ROS 2 Nodes and Parameters'
---

# ROS 2 Nodes and Parameters

## Overview

Nodes are the fundamental building blocks of ROS 2 applications, representing individual processes that perform specific functions within a robot system. In this lesson, we'll dive deeper into node creation, parameter management, and best practices for building robust ROS 2 nodes. We'll also explore how to create nodes that can be dynamically configured through parameters.

Understanding nodes and parameters is crucial for building flexible and maintainable robot systems that can adapt to different environments and requirements.

## Learning Outcomes

By the end of this lesson, you will be able to:
- Create complex ROS 2 nodes with multiple publishers, subscribers, and services
- Implement parameter handling in ROS 2 nodes
- Use ROS 2 launch files to manage multiple nodes
- Apply best practices for node design and error handling
- Configure nodes dynamically through parameters

## Hands-on Steps

1. **Advanced Node Creation**: Build a node with multiple communication patterns
2. **Parameter Implementation**: Add dynamic configuration to nodes
3. **Launch File Creation**: Create launch files to manage node deployment
4. **Node Testing**: Test nodes with different parameter configurations
5. **Error Handling**: Implement robust error handling in nodes

### Prerequisites

- Understanding of basic ROS 2 concepts (nodes, topics, services)
- Basic Python programming knowledge
- Experience with object-oriented programming

## Code Examples

Let's start by creating a more sophisticated node that demonstrates multiple communication patterns and parameter handling:

```python
# advanced_robot_node.py
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String, Float32
from example_interfaces.srv import SetBool
import math

class AdvancedRobotNode(Node):
    """
    Advanced robot node demonstrating multiple ROS 2 concepts:
    - Multiple publishers and subscribers
    - Parameter handling
    - Services
    - Dynamic reconfiguration
    """
    def __init__(self):
        super().__init__('advanced_robot_node')

        # Declare parameters with default values
        self.declare_parameter('robot_name', 'turtlebot')
        self.declare_parameter('linear_speed', 0.2)
        self.declare_parameter('angular_speed', 0.5)
        self.declare_parameter('obstacle_threshold', 1.0)
        self.declare_parameter('safety_mode', True)

        # Get parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.linear_speed = self.get_parameter('linear_speed').value
        self.angular_speed = self.get_parameter('angular_speed').value
        self.obstacle_threshold = self.get_parameter('obstacle_threshold').value
        self.safety_mode = self.get_parameter('safety_mode').value

        # Create QoS profile for sensor data (with reliability)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Publishers
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_publisher = self.create_publisher(String, '/robot_status', 10)
        self.distance_publisher = self.create_publisher(Float32, '/obstacle_distance', 10)

        # Subscribers
        self.laser_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            sensor_qos
        )

        # Services
        self.safety_service = self.create_service(
            SetBool,
            '/toggle_safety_mode',
            self.toggle_safety_mode_callback
        )

        # Timer for main control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)

        # Robot state
        self.obstacle_detected = False
        self.obstacle_distance = float('inf')
        self.robot_state = "IDLE"
        self.emergency_stop = False

        # Parameter callback
        self.add_on_set_parameters_callback(self.parameter_callback)

        self.get_logger().info(f"Advanced Robot Node initialized for {self.robot_name}")
        self.get_logger().info(f"Parameters - Speed: {self.linear_speed}m/s, Threshold: {self.obstacle_threshold}m")

    def parameter_callback(self, params):
        """
        Callback for parameter changes
        """
        for param in params:
            if param.name == 'linear_speed' and param.type_ == Parameter.Type.DOUBLE:
                self.linear_speed = param.value
                self.get_logger().info(f"Linear speed updated to: {self.linear_speed}")
            elif param.name == 'angular_speed' and param.type_ == Parameter.Type.DOUBLE:
                self.angular_speed = param.value
                self.get_logger().info(f"Angular speed updated to: {self.angular_speed}")
            elif param.name == 'obstacle_threshold' and param.type_ == Parameter.Type.DOUBLE:
                self.obstacle_threshold = param.value
                self.get_logger().info(f"Obstacle threshold updated to: {self.obstacle_threshold}")
            elif param.name == 'safety_mode' and param.type_ == Parameter.Type.BOOL:
                self.safety_mode = param.value
                self.get_logger().info(f"Safety mode updated to: {self.safety_mode}")

        return SetBool.Response(success=True, message="Parameters updated")

    def laser_callback(self, msg):
        """
        Process laser scan data to detect obstacles
        """
        # Get the minimum distance from the front sector (±30 degrees)
        front_ranges = msg.ranges[330:] + msg.ranges[:30]  # Front 60 degrees
        front_ranges = [r for r in front_ranges if not math.isnan(r) and r > 0 and r < 10.0]

        if front_ranges:
            self.obstacle_distance = min(front_ranges)
            self.obstacle_detected = self.obstacle_distance < self.obstacle_threshold
        else:
            self.obstacle_distance = float('inf')
            self.obstacle_detected = False

    def toggle_safety_mode_callback(self, request, response):
        """
        Service callback to toggle safety mode
        """
        self.safety_mode = request.data
        response.success = True
        response.message = f"Safety mode set to {self.safety_mode}"
        self.get_logger().info(f"Safety mode toggled: {self.safety_mode}")
        return response

    def control_loop(self):
        """
        Main control loop with parameter-based behavior
        """
        cmd_msg = Twist()

        if self.emergency_stop:
            # Emergency stop - zero all velocities
            cmd_msg.linear.x = 0.0
            cmd_msg.angular.z = 0.0
            self.robot_state = "EMERGENCY_STOP"
        elif self.obstacle_detected and self.safety_mode:
            # Obstacle detected in safety mode
            self.robot_state = "AVOIDING"
            cmd_msg.linear.x = 0.0  # Stop linear motion
            cmd_msg.angular.z = self.angular_speed  # Rotate to avoid
        else:
            # Normal operation
            self.robot_state = "MOVING"
            cmd_msg.linear.x = self.linear_speed
            cmd_msg.angular.z = 0.0

        # Publish command
        self.cmd_vel_publisher.publish(cmd_msg)

        # Publish status
        status_msg = String()
        status_msg.data = f"Robot: {self.robot_name}, State: {self.robot_state}, " \
                         f"Obstacle: {self.obstacle_detected}, Distance: {self.obstacle_distance:.2f}m"
        self.status_publisher.publish(status_msg)

        # Publish obstacle distance
        distance_msg = Float32()
        distance_msg.data = self.obstacle_distance
        self.distance_publisher.publish(distance_msg)

def main(args=None):
    rclpy.init(args=args)
    advanced_robot_node = AdvancedRobotNode()

    try:
        rclpy.spin(advanced_robot_node)
    except KeyboardInterrupt:
        advanced_robot_node.get_logger().info("Advanced Robot Node stopped by user")
    finally:
        advanced_robot_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Now let's create a launch file to manage this node with different configurations:

```python
# robot_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='turtlebot',
        description='Name of the robot'
    )

    linear_speed_arg = DeclareLaunchArgument(
        'linear_speed',
        default_value='0.2',
        description='Linear speed of the robot'
    )

    angular_speed_arg = DeclareLaunchArgument(
        'angular_speed',
        default_value='0.5',
        description='Angular speed of the robot'
    )

    obstacle_threshold_arg = DeclareLaunchArgument(
        'obstacle_threshold',
        default_value='1.0',
        description='Obstacle detection threshold'
    )

    # Get launch configurations
    robot_name = LaunchConfiguration('robot_name')
    linear_speed = LaunchConfiguration('linear_speed')
    angular_speed = LaunchConfiguration('angular_speed')
    obstacle_threshold = LaunchConfiguration('obstacle_threshold')

    # Create the robot node
    robot_node = Node(
        package='robot_package',
        executable='advanced_robot_node',
        name='advanced_robot_node',
        parameters=[
            {
                'robot_name': robot_name,
                'linear_speed': linear_speed,
                'angular_speed': angular_speed,
                'obstacle_threshold': obstacle_threshold,
                'safety_mode': True
            }
        ],
        remappings=[
            ('/cmd_vel', '/cmd_vel'),
            ('/scan', '/scan'),
        ]
    )

    return LaunchDescription([
        robot_name_arg,
        linear_speed_arg,
        angular_speed_arg,
        obstacle_threshold_arg,
        robot_node
    ])
```

Let's also create a parameter file for configuration:

```yaml
# robot_params.yaml
advanced_robot_node:
  ros__parameters:
    robot_name: "turtlebot"
    linear_speed: 0.3
    angular_speed: 0.6
    obstacle_threshold: 1.5
    safety_mode: true
```

## Small Simulation

Let's create a simulation environment that demonstrates parameter changes affecting robot behavior:

```python
# parameter_simulation.py
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String
from example_interfaces.srv import SetParameters
import time

class ParameterSimulationNode(Node):
    """
    Node to demonstrate parameter changes affecting robot behavior
    """
    def __init__(self):
        super().__init__('parameter_simulation_node')

        # Declare parameters
        self.declare_parameter('simulation_speed', 1.0)
        self.declare_parameter('environment_type', 'indoor')
        self.declare_parameter('difficulty_level', 1)

        # Get initial parameter values
        self.simulation_speed = self.get_parameter('simulation_speed').value
        self.environment_type = self.get_parameter('environment_type').value
        self.difficulty_level = self.get_parameter('difficulty_level').value

        # Publisher for simulation status
        self.status_publisher = self.create_publisher(String, '/simulation_status', 10)

        # Timer for simulation updates
        self.sim_timer = self.create_timer(1.0 / self.simulation_speed, self.simulation_step)

        # Parameter callback
        self.add_on_set_parameters_callback(self.parameter_callback)

        self.step_count = 0
        self.get_logger().info(f"Parameter Simulation started - Speed: {self.simulation_speed}, Environment: {self.environment_type}")

    def parameter_callback(self, params):
        """
        Handle parameter changes
        """
        for param in params:
            if param.name == 'simulation_speed':
                self.simulation_speed = param.value
                # Update timer rate
                self.sim_timer.timer_period_ns = int(1e9 / self.simulation_speed)
                self.get_logger().info(f"Simulation speed updated to: {self.simulation_speed}")
            elif param.name == 'environment_type':
                self.environment_type = param.value
                self.get_logger().info(f"Environment type updated to: {self.environment_type}")
            elif param.name == 'difficulty_level':
                self.difficulty_level = param.value
                self.get_logger().info(f"Difficulty level updated to: {self.difficulty_level}")

        return SetParameters.Response(results=[rclpy.Parameter.Type.DECLARE_SUCCESSFUL for p in params])

    def simulation_step(self):
        """
        Simulation step that demonstrates parameter effects
        """
        self.step_count += 1

        # Simulate different behaviors based on parameters
        if self.environment_type == 'indoor':
            obstacle_factor = 1.0
        elif self.environment_type == 'outdoor':
            obstacle_factor = 0.7
        else:
            obstacle_factor = 0.5

        difficulty_modifier = self.difficulty_level * 0.1

        # Simulate some "sensor" data
        simulated_data = f"Step {self.step_count}: Env={self.environment_type}, Diff={self.difficulty_level}, Factor={obstacle_factor:.1f}"

        # Publish status
        status_msg = String()
        status_msg.data = simulated_data
        self.status_publisher.publish(status_msg)

        self.get_logger().info(f"Simulation: {simulated_data}")

def main(args=None):
    rclpy.init(args=args)
    param_sim_node = ParameterSimulationNode()

    try:
        rclpy.spin(param_sim_node)
    except KeyboardInterrupt:
        param_sim_node.get_logger().info("Parameter Simulation stopped by user")
    finally:
        param_sim_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Quick Recap

In this lesson, we've covered:

- **Advanced Node Creation**: Building nodes with multiple publishers, subscribers, and services
- **Parameter Management**: Declaring, using, and dynamically updating parameters
- **Launch Files**: Managing multiple nodes and configurations
- **Parameter Callbacks**: Responding to parameter changes at runtime
- **Best Practices**: Proper error handling, QoS profiles, and node design patterns

Parameters provide a powerful way to configure robot behavior without recompiling code. They allow the same node to operate in different environments or under different conditions by simply changing parameter values. This flexibility is essential for deploying robots in various scenarios and for tuning performance during development.

The combination of nodes, parameters, and launch files creates a robust framework for building complex robot systems that can be easily configured and managed.

In the next lesson, we'll explore URDF (Unified Robot Description Format) for modeling humanoid robots and how to integrate these models with ROS 2.