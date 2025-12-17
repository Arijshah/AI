---
sidebar_label: 'Introduction to ROS 2'
title: 'Introduction to ROS 2'
---

# Introduction to ROS 2

## Overview

ROS 2 (Robot Operating System 2) serves as the "nervous system" of modern robotics, providing a flexible framework for developing robot applications. Unlike traditional software frameworks, ROS 2 offers a collection of libraries, tools, and conventions that enable communication between different parts of a robot system, from low-level hardware drivers to high-level AI algorithms.

This lesson introduces the core concepts of ROS 2, including nodes, topics, services, and actions - the fundamental building blocks for creating distributed robotic applications.

## Learning Outcomes

By the end of this lesson, you will be able to:
- Understand the architecture and components of ROS 2
- Explain the differences between ROS 1 and ROS 2
- Identify when to use topics, services, and actions
- Set up a basic ROS 2 development environment
- Create and run simple ROS 2 nodes

## Hands-on Steps

1. **Environment Setup**: Install ROS 2 and set up the development workspace
2. **Basic Node Creation**: Create a simple publisher and subscriber
3. **Topic Communication**: Implement message passing between nodes
4. **Service Implementation**: Create a request-response pattern
5. **System Monitoring**: Use ROS 2 tools to monitor the system

### Prerequisites

- Basic Python programming knowledge
- Understanding of distributed systems concepts
- Linux command line familiarity (or WSL for Windows users)

## Code Examples

Let's start by creating a simple ROS 2 publisher and subscriber to understand the node-topic communication pattern:

```python
# publisher_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

```python
# subscriber_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Now let's look at creating a service server and client:

```python
# service_server.py
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Incoming request\na: {request.a}, b: {request.b}\nsum: {response.sum}')
        return response

def main(args=None):
    rclpy.init(args=args)
    minimal_service = MinimalService()
    rclpy.spin(minimal_service)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

```python
# service_client.py
import sys
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalClient(Node):
    def __init__(self):
        super().__init__('minimal_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    minimal_client = MinimalClient()
    response = minimal_client.send_request(int(sys.argv[1]), int(sys.argv[2]))
    minimal_client.get_logger().info(f'Result of add_two_ints: {response.sum}')
    minimal_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Small Simulation

Let's create a more complex example that demonstrates a simple robot control system using ROS 2 patterns:

```python
# robot_controller.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import math

class RobotController(Node):
    """
    A simple robot controller that demonstrates ROS 2 concepts
    """
    def __init__(self):
        super().__init__('robot_controller')

        # Publishers
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_publisher = self.create_publisher(String, '/robot_status', 10)

        # Subscribers
        self.laser_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )

        # Timer for control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)

        # Robot state
        self.obstacle_detected = False
        self.obstacle_distance = float('inf')
        self.robot_state = "IDLE"  # IDLE, MOVING, AVOIDING

        self.get_logger().info("Robot Controller initialized")

    def laser_callback(self, msg):
        """Process laser scan data to detect obstacles"""
        # Get the minimum distance from the front sector (±30 degrees)
        front_ranges = msg.ranges[330:] + msg.ranges[:30]  # Front 60 degrees
        front_ranges = [r for r in front_ranges if not math.isnan(r) and r > 0]

        if front_ranges:
            self.obstacle_distance = min(front_ranges)
            self.obstacle_detected = self.obstacle_distance < 1.0  # 1 meter threshold
        else:
            self.obstacle_distance = float('inf')
            self.obstacle_detected = False

    def control_loop(self):
        """Main control loop"""
        cmd_msg = Twist()

        if self.obstacle_detected:
            # Stop and rotate to avoid obstacle
            self.robot_state = "AVOIDING"
            cmd_msg.linear.x = 0.0
            cmd_msg.angular.z = 0.5  # Rotate counter-clockwise
        else:
            # Move forward
            self.robot_state = "MOVING"
            cmd_msg.linear.x = 0.3  # Move forward at 0.3 m/s
            cmd_msg.angular.z = 0.0

        # Publish command
        self.cmd_vel_publisher.publish(cmd_msg)

        # Publish status
        status_msg = String()
        status_msg.data = f"State: {self.robot_state}, Obstacle: {self.obstacle_detected}, Distance: {self.obstacle_distance:.2f}m"
        self.status_publisher.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    robot_controller = RobotController()

    try:
        rclpy.spin(robot_controller)
    except KeyboardInterrupt:
        robot_controller.get_logger().info("Robot Controller stopped by user")
    finally:
        robot_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Quick Recap

In this lesson, we've covered:

- **ROS 2 Architecture**: The distributed system approach for robot software
- **Core Components**: Nodes (processes), Topics (data streams), Services (request-response), Actions (goal-oriented)
- **Communication Patterns**: Publisher-subscriber for asynchronous data flow and client-server for synchronous requests
- **Practical Implementation**: Created working examples of publishers, subscribers, services, and a robot controller

ROS 2 provides the communication backbone that allows different parts of a robot system to work together seamlessly. The publish-subscribe model enables loose coupling between components, while services provide synchronous request-response communication for critical operations.

In the next lesson, we'll explore creating more sophisticated ROS 2 nodes with parameters and advanced messaging patterns.