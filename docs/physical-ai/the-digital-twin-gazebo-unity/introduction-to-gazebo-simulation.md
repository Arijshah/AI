---
sidebar_label: 'Introduction to Gazebo Simulation'
title: 'Introduction to Gazebo Simulation'
---

# Introduction to Gazebo Simulation

## Overview

Gazebo is a powerful physics-based simulation environment that serves as a "digital twin" for robotic systems. It provides realistic physics simulation, sensor modeling, and visual rendering that allows developers to test and validate robot behaviors in a safe, controlled environment before deploying to real hardware. This lesson introduces the core concepts of Gazebo simulation and demonstrates how to create and interact with simulated robots.

Gazebo is particularly valuable for Physical AI development because it enables rapid prototyping, testing of complex scenarios, and safe experimentation with robot behaviors that would be risky or expensive to test on real hardware.

## Learning Outcomes

By the end of this lesson, you will be able to:
- Understand the architecture and components of Gazebo simulation
- Set up a basic Gazebo environment with ROS 2 integration
- Create simple simulation worlds with objects and obstacles
- Spawn and control robots in simulation
- Use Gazebo's physics engine and sensor models

## Hands-on Steps

1. **Gazebo Installation**: Set up Gazebo with ROS 2 integration
2. **Basic World Creation**: Create a simple simulation environment
3. **Robot Spawning**: Add a robot model to the simulation
4. **Sensor Integration**: Add and configure basic sensors
5. **Control Interface**: Connect to ROS 2 for robot control

### Prerequisites

- Understanding of ROS 2 concepts and basic node development
- Knowledge of URDF for robot modeling (from Chapter 2)
- Basic understanding of physics concepts (mass, friction, collision)

## Code Examples

Let's start by creating a basic Gazebo world file:

```xml
<!-- basic_world.world -->
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="basic_world">
    <!-- Include the default camera sensor -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Add ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Add a simple box obstacle -->
    <model name="box_obstacle">
      <pose>2 0 0.5 0 0 0</pose>
      <link name="box_link">
        <collision name="box_collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="box_visual">
          <geometry>
            <box>
              <size>1 1 1</size>
          </geometry>
          <material>
            <ambient>0.8 0.2 0.2 1</ambient>
            <diffuse>0.8 0.2 0.2 1</diffuse>
            <specular>0.8 0.2 0.2 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.1667</ixx>
            <ixy>0.0</ixy>
            <ixz>0.0</ixz>
            <iyy>0.1667</iyy>
            <iyz>0.0</iyz>
            <izz>0.1667</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <!-- Add a simple cylinder obstacle -->
    <model name="cylinder_obstacle">
      <pose>-2 1 0.5 0 0 0</pose>
      <link name="cylinder_link">
        <collision name="cylinder_collision">
          <geometry>
            <cylinder>
              <radius>0.3</radius>
              <length>1.0</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name="cylinder_visual">
          <geometry>
            <cylinder>
              <radius>0.3</radius>
              <length>1.0</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0.2 0.8 0.2 1</ambient>
            <diffuse>0.2 0.8 0.2 1</diffuse>
            <specular>0.2 0.8 0.2 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.125</ixx>
            <ixy>0.0</ixy>
            <ixz>0.0</ixz>
            <iyy>0.125</iyy>
            <iyz>0.0</iyz>
            <izz>0.09</izz>
          </inertia>
        </inertial>
      </link>
    </model>
  </world>
</sdf>
```

Now let's create a simple robot model that can be used in Gazebo:

```xml
<!-- simple_robot.urdf -->
<?xml version="1.0"?>
<robot name="simple_robot">
  <!-- Materials -->
  <material name="blue">
    <color rgba="0.0 0.0 0.8 1.0"/>
  </material>
  <material name="green">
    <color rgba="0.0 0.8 0.0 1.0"/>
  </material>
  <material name="red">
    <color rgba="0.8 0.0 0.0 1.0"/>
  </material>
  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>

  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.2" radius="0.2"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.2" radius="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Left Wheel -->
  <link name="left_wheel">
    <visual>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="0 0.15 -0.05" rpy="1.57 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <!-- Right Wheel -->
  <link name="right_wheel">
    <visual>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="0 -0.15 -0.05" rpy="1.57 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <!-- Camera Mount -->
  <link name="camera_mount">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="camera_joint" type="fixed">
    <parent link="base_link"/>
    <child link="camera_mount"/>
    <origin xyz="0.15 0 0.05" rpy="0 0 0"/>
  </joint>

  <!-- Gazebo-specific plugins and sensors -->
  <gazebo reference="base_link">
    <material>Gazebo/White</material>
  </gazebo>

  <gazebo reference="left_wheel">
    <material>Gazebo/Blue</material>
    <mu1>1.0</mu1>
    <mu2>1.0</mu2>
    <kp>1000000.0</kp>
    <kd>100.0</kd>
  </gazebo>

  <gazebo reference="right_wheel">
    <material>Gazebo/Blue</material>
    <mu1>1.0</mu1>
    <mu2>1.0</mu2>
    <kp>1000000.0</kp>
    <kd>100.0</kd>
  </gazebo>

  <!-- Differential drive plugin -->
  <gazebo>
    <plugin name="differential_drive" filename="libgazebo_ros_diff_drive.so">
      <ros>
        <namespace>/simple_robot</namespace>
        <remapping>cmd_vel:=cmd_vel</remapping>
        <remapping>odom:=odom</remapping>
      </ros>
      <left_joint>left_wheel_joint</left_joint>
      <right_joint>right_wheel_joint</right_joint>
      <wheel_separation>0.3</wheel_separation>
      <wheel_diameter>0.2</wheel_diameter>
      <max_wheel_torque>20</max_wheel_torque>
      <max_wheel_acceleration>1.0</max_wheel_acceleration>
      <publish_odom>true</publish_odom>
      <publish_odom_tf>true</publish_odom_tf>
      <publish_wheel_tf>true</publish_wheel_tf>
      <odometry_frame>odom</odometry_frame>
      <robot_base_frame>base_link</robot_base_frame>
    </plugin>
  </gazebo>

  <!-- Camera sensor plugin -->
  <gazebo reference="camera_mount">
    <sensor name="camera" type="camera">
      <pose>0 0 0 0 0 0</pose>
      <visualize>true</visualize>
      <update_rate>30</update_rate>
      <camera name="head">
        <horizontal_fov>1.3962634</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>100</far>
        </clip>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <ros>
          <namespace>/simple_robot</namespace>
          <remapping>image_raw:=camera/image_raw</remapping>
          <remapping>camera_info:=camera/camera_info</remapping>
        </ros>
        <camera_name>camera</camera_name>
        <image_topic_name>image_raw</image_topic_name>
        <camera_info_topic_name>camera_info</camera_info_topic_name>
        <frame_name>camera_mount</frame_name>
      </plugin>
    </sensor>
  </gazebo>
</robot>
```

Now let's create a ROS 2 node that can interact with our simulated robot:

```python
# gazebo_robot_controller.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np

class GazeboRobotController(Node):
    """
    Controller for interacting with a robot in Gazebo simulation
    """
    def __init__(self):
        super().__init__('gazebo_robot_controller')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/simple_robot/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/robot_status', 10)

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/simple_robot/odom', self.odom_callback, 10)

        self.scan_sub = self.create_subscription(
            LaserScan, '/simple_robot/scan', self.scan_callback, 10)

        self.image_sub = self.create_subscription(
            Image, '/simple_robot/camera/image_raw', self.image_callback, 10)

        # Timer for control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)

        # Robot state
        self.current_pose = Pose()
        self.current_twist = Twist()
        self.laser_ranges = []
        self.obstacle_detected = False
        self.obstacle_distance = float('inf')
        self.robot_state = "IDLE"

        # Image processing
        self.cv_bridge = CvBridge()
        self.latest_image = None

        # Control parameters
        self.linear_speed = 0.3
        self.angular_speed = 0.5
        self.obstacle_threshold = 1.0

        self.get_logger().info("Gazebo Robot Controller initialized")

    def odom_callback(self, msg):
        """Process odometry data"""
        self.current_pose = msg.pose.pose
        self.current_twist = msg.twist.twist

    def scan_callback(self, msg):
        """Process laser scan data"""
        self.laser_ranges = msg.ranges

        # Check for obstacles in front (±30 degrees)
        front_ranges = msg.ranges[330:] + msg.ranges[:30]
        front_ranges = [r for r in front_ranges if not np.isnan(r) and 0.1 < r < 10.0]

        if front_ranges:
            self.obstacle_distance = min(front_ranges)
            self.obstacle_detected = self.obstacle_distance < self.obstacle_threshold
        else:
            self.obstacle_distance = float('inf')
            self.obstacle_detected = False

    def image_callback(self, msg):
        """Process camera image data"""
        try:
            # Convert ROS Image message to OpenCV image
            self.latest_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Could not convert image: {e}")

    def control_loop(self):
        """Main control loop for the robot"""
        cmd = Twist()

        if self.obstacle_detected:
            # Stop and rotate to avoid obstacle
            self.robot_state = "AVOIDING_OBSTACLE"
            cmd.linear.x = 0.0
            cmd.angular.z = self.angular_speed
        else:
            # Move forward
            self.robot_state = "MOVING_FORWARD"
            cmd.linear.x = self.linear_speed
            cmd.angular.z = 0.0

        # Publish command
        self.cmd_vel_pub.publish(cmd)

        # Publish status
        status_msg = String()
        status_msg.data = f"State: {self.robot_state}, Pos: ({self.current_pose.position.x:.2f}, {self.current_pose.position.y:.2f}), " \
                         f"Obstacle: {self.obstacle_detected}, Distance: {self.obstacle_distance:.2f}m"
        self.status_pub.publish(status_msg)

        # Process image if available (simple color detection example)
        if self.latest_image is not None:
            # Simple color detection - find red objects
            hsv = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2HSV)
            lower_red = np.array([0, 100, 100])
            upper_red = np.array([10, 255, 255])
            mask = cv2.inRange(hsv, lower_red, upper_red)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Find the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 100:  # Only react to significant objects
                    # Calculate the centroid
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        image_center = self.latest_image.shape[1] / 2
                        error = cx - image_center

                        # Adjust angular velocity based on object position
                        cmd.angular.z = -0.005 * error  # Negative for correct direction
                        self.cmd_vel_pub.publish(cmd)
                        self.get_logger().info(f"Detected red object, adjusting direction: {error:.2f}")

def main(args=None):
    rclpy.init(args=args)
    controller = GazeboRobotController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info("Gazebo Robot Controller stopped by user")
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Small Simulation

Let's create a more advanced simulation example that demonstrates sensor fusion in Gazebo:

```python
# sensor_fusion_simulator.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan, Imu, MagneticField
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, String
import numpy as np
import math

class SensorFusionSimulator(Node):
    """
    Demonstrates sensor fusion in Gazebo simulation using multiple sensor types
    """
    def __init__(self):
        super().__init__('sensor_fusion_simulator')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/simple_robot/cmd_vel', 10)
        self.fused_pose_pub = self.create_publisher(Point, '/fused_pose', 10)
        self.status_pub = self.create_publisher(String, '/sensor_fusion_status', 10)

        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, '/simple_robot/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/simple_robot/scan', self.scan_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/simple_robot/imu', self.imu_callback, 10)

        # Timer for fusion loop
        self.fusion_timer = self.create_timer(0.05, self.fusion_loop)

        # Robot state estimation
        self.position = Point()
        self.velocity = Point()
        self.orientation = 0.0  # yaw angle
        self.angular_velocity = 0.0
        self.linear_velocity = 0.0

        # Sensor data storage
        self.odom_data = None
        self.scan_data = None
        self.imu_data = None

        # Kalman filter parameters (simplified)
        self.position_uncertainty = 0.1
        self.orientation_uncertainty = 0.1

        # Fusion weights for different sensors
        self.odom_weight = 0.7
        self.imu_weight = 0.3

        self.get_logger().info("Sensor Fusion Simulator initialized")

    def odom_callback(self, msg):
        """Process odometry data"""
        self.odom_data = msg

    def scan_callback(self, msg):
        """Process laser scan data for obstacle detection and localization"""
        self.scan_data = msg

        # Process scan to detect obstacles and estimate position relative to known landmarks
        # This is a simplified approach - in reality, this would involve more complex SLAM
        front_ranges = msg.ranges[330:] + msg.ranges[:30]
        front_ranges = [r for r in front_ranges if not np.isnan(r) and 0.1 < r < 10.0]

        if front_ranges:
            min_dist = min(front_ranges)
            # If we know the world layout, we could estimate position based on distances to known objects
            # For now, just log the information
            self.get_logger().info(f"Closest obstacle: {min_dist:.2f}m")

    def imu_callback(self, msg):
        """Process IMU data"""
        self.imu_data = msg

        # Extract orientation from quaternion
        # Simplified: just using z-axis rotation (yaw)
        w = msg.orientation.w
        z = msg.orientation.z
        self.orientation = math.atan2(2.0 * (w * z), 1.0 - 2.0 * (z * z))

        # Extract angular velocity
        self.angular_velocity = msg.angular_velocity.z
        self.linear_velocity = math.sqrt(
            msg.linear_acceleration.x**2 +
            msg.linear_acceleration.y**2
        )

    def fusion_loop(self):
        """Main sensor fusion loop"""
        fused_position = Point()

        # If we have both odom and IMU data, perform simple fusion
        if self.odom_data and self.imu_data:
            # Position fusion (weighted average)
            odom_x = self.odom_data.pose.pose.position.x
            odom_y = self.odom_data.pose.pose.position.y

            # Use IMU to predict position change since last odom
            dt = 0.05  # timer period
            dx = self.linear_velocity * math.cos(self.orientation) * dt
            dy = self.linear_velocity * math.sin(self.orientation) * dt

            # Fused position
            fused_x = self.odom_weight * odom_x + self.imu_weight * (self.position.x + dx)
            fused_y = self.odom_weight * odom_y + self.imu_weight * (self.position.y + dy)

            fused_position.x = fused_x
            fused_position.y = fused_y
            fused_position.z = self.odom_data.pose.pose.position.z  # Use odom z for now

            # Update our position estimate
            self.position = fused_position

            # Publish fused position
            self.fused_pose_pub.publish(fused_position)

            # Publish status
            status_msg = String()
            status_msg.data = f"Fused Pos: ({fused_position.x:.2f}, {fused_position.y:.2f}), " \
                             f"Ori: {self.orientation:.2f}, LinVel: {self.linear_velocity:.2f}m/s"
            self.status_pub.publish(status_msg)
        elif self.odom_data:
            # If only odom is available, use that
            self.position = self.odom_data.pose.pose.position
            self.fused_pose_pub.publish(self.position)
        elif self.imu_data:
            # If only IMU is available, integrate
            dt = 0.05
            dx = self.linear_velocity * math.cos(self.orientation) * dt
            dy = self.linear_velocity * math.sin(self.orientation) * dt

            self.position.x += dx
            self.position.y += dy

            fused_position = self.position
            self.fused_pose_pub.publish(fused_position)

def main(args=None):
    rclpy.init(args=args)
    fusion_sim = SensorFusionSimulator()

    try:
        rclpy.spin(fusion_sim)
    except KeyboardInterrupt:
        fusion_sim.get_logger().info("Sensor Fusion Simulator stopped by user")
    finally:
        fusion_sim.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Quick Recap

In this lesson, we've covered:

- **Gazebo Architecture**: Understanding the components of the Gazebo simulation environment
- **World Creation**: Creating simulation environments with objects and obstacles
- **Robot Modeling**: Extending URDF with Gazebo-specific plugins and sensors
- **Sensor Integration**: Adding cameras, LIDAR, and IMU sensors to simulated robots
- **Control Systems**: Connecting simulated robots to ROS 2 for control and interaction
- **Sensor Fusion**: Combining multiple sensor inputs for improved robot perception

Gazebo provides a realistic physics simulation environment that serves as an essential "digital twin" for robotics development. It allows for safe, rapid prototyping and testing of robot behaviors before deployment to real hardware.

In the next lesson, we'll explore physics and collision modeling in more detail, including advanced material properties and realistic interaction modeling.