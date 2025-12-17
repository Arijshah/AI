---
sidebar_label: 'Sensor Simulation'
title: 'Sensor Simulation'
---

# Sensor Simulation

## Overview

Sensor simulation is a critical component of digital twin technology, enabling robots to perceive their environment in virtual worlds just as they would with real sensors. This lesson covers the simulation of various sensor types including LIDAR, depth cameras, IMUs, and other perception systems. Accurate sensor simulation allows for realistic testing of perception algorithms, SLAM systems, and robot autonomy before deployment on physical hardware.

Understanding how to properly configure and validate sensor models is essential for creating believable digital twins that can effectively prepare robots for real-world operation.

## Learning Outcomes

By the end of this lesson, you will be able to:
- Configure and simulate LIDAR sensors with realistic noise and limitations
- Set up depth camera simulation with proper distortion and resolution
- Model IMU sensors with realistic drift and noise characteristics
- Simulate multiple sensor types for sensor fusion applications
- Validate sensor data against expected real-world behavior

## Hands-on Steps

1. **LIDAR Simulation**: Configure realistic LIDAR models with noise parameters
2. **Depth Camera Setup**: Create depth camera sensors with proper distortion
3. **IMU Modeling**: Implement realistic IMU sensors with drift characteristics
4. **Multi-Sensor Integration**: Combine multiple sensors for fusion applications
5. **Sensor Validation**: Compare simulated vs. expected sensor behavior

### Prerequisites

- Understanding of ROS 2 sensor message types
- Knowledge of sensor physics and characteristics
- Experience with Gazebo and Unity integration

## Code Examples

Let's start by creating a comprehensive robot model with various sensors:

```xml
<!-- sensor_robot.urdf -->
<?xml version="1.0"?>
<robot name="sensor_robot">
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
  <material name="black">
    <color rgba="0.1 0.1 0.1 1.0"/>
  </material>

  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <geometry>
        <box size="0.5 0.4 0.3"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <geometry>
        <box size="0.5 0.4 0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="15.0"/>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <inertia ixx="0.21875" ixy="0.0" ixz="0.0"
               iyy="0.31458" iyz="0.0"
               izz="0.40625"/>
    </inertial>
  </link>

  <!-- LIDAR Mount -->
  <link name="lidar_mount">
    <visual>
      <geometry>
        <cylinder length="0.05" radius="0.05"/>
      </geometry>
      <material name="black"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.05" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.00015" ixy="0.0" ixz="0.0"
               iyy="0.00015" iyz="0.0"
               izz="0.0001"/>
    </inertial>
  </link>

  <joint name="lidar_joint" type="fixed">
    <parent link="base_link"/>
    <child link="lidar_mount"/>
    <origin xyz="0.2 0 0.3" rpy="0 0 0"/>
  </joint>

  <!-- Camera Mount -->
  <link name="camera_mount">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
      <material name="black"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.000025" ixy="0.0" ixz="0.0"
               iyy="0.000025" iyz="0.0"
               izz="0.000025"/>
    </inertial>
  </link>

  <joint name="camera_joint" type="fixed">
    <parent link="base_link"/>
    <child link="camera_mount"/>
    <origin xyz="0.22 0 0.25" rpy="0 0 0"/>
  </joint>

  <!-- IMU Mount -->
  <link name="imu_mount">
    <visual>
      <geometry>
        <box size="0.02 0.02 0.02"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.02 0.02 0.02"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="0.000001" ixy="0.0" ixz="0.0"
               iyy="0.000001" iyz="0.0"
               izz="0.000001"/>
    </inertial>
  </link>

  <joint name="imu_joint" type="fixed">
    <parent link="base_link"/>
    <child link="imu_mount"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
  </joint>

  <!-- Wheel links -->
  <link name="left_wheel">
    <visual>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
      <geometry>
        <cylinder length="0.08" radius="0.15"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
      <geometry>
        <cylinder length="0.08" radius="0.15"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.008125" ixy="0.0" ixz="0.0"
               iyy="0.008125" iyz="0.0"
               izz="0.01125"/>
    </inertial>
  </link>

  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="0 0.25 0.15" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <link name="right_wheel">
    <visual>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
      <geometry>
        <cylinder length="0.08" radius="0.15"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
      <geometry>
        <cylinder length="0.08" radius="0.15"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.008125" ixy="0.0" ixz="0.0"
               iyy="0.008125" iyz="0.0"
               izz="0.01125"/>
    </inertial>
  </link>

  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="0 -0.25 0.15" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <!-- Gazebo-specific sensor plugins -->

  <!-- LIDAR Sensor (Hokuyo-like) -->
  <gazebo reference="lidar_mount">
    <sensor name="laser" type="ray">
      <pose>0 0 0 0 0 0</pose>
      <visualize>true</visualize>
      <update_rate>10</update_rate>
      <ray>
        <scan>
          <horizontal>
            <samples>720</samples>
            <resolution>1</resolution>
            <min_angle>-2.35619</min_angle>  <!-- -135 degrees -->
            <max_angle>2.35619</max_angle>   <!-- 135 degrees -->
          </horizontal>
        </scan>
        <range>
          <min>0.1</min>
          <max>30.0</max>
          <resolution>0.01</resolution>
        </range>
      </ray>
      <plugin name="laser_controller" filename="libgazebo_ros_ray_sensor.so">
        <ros>
          <namespace>/sensor_robot</namespace>
          <remapping>~/out:=scan</remapping>
        </ros>
        <output_type>sensor_msgs/LaserScan</output_type>
        <frame_name>lidar_mount</frame_name>
      </plugin>
      <!-- Add noise to make it more realistic -->
      <noise type="gaussian">
        <mean>0.0</mean>
        <stddev>0.01</stddev>
      </noise>
    </sensor>
  </gazebo>

  <!-- Depth Camera -->
  <gazebo reference="camera_mount">
    <sensor name="depth_camera" type="depth">
      <pose>0 0 0 0 0 0</pose>
      <visualize>true</visualize>
      <update_rate>30</update_rate>
      <camera name="head">
        <horizontal_fov>1.047</horizontal_fov>  <!-- 60 degrees -->
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>10</far>
        </clip>
        <noise>
          <type>gaussian</type>
          <mean>0.0</mean>
          <stddev>0.007</stddev>
        </noise>
      </camera>
      <plugin name="depth_camera_controller" filename="libgazebo_ros_openni_kinect.so">
        <ros>
          <namespace>/sensor_robot</namespace>
          <remapping>rgb/image_raw:=camera/color/image_raw</remapping>
          <remapping>depth/image_raw:=camera/depth/image_raw</remapping>
          <remapping>depth/camera_info:=camera/depth/camera_info</remapping>
        </ros>
        <camera_name>camera</camera_name>
        <frame_name>camera_mount</frame_name>
        <baseline>0.1</baseline>
        <distortion_k1>0.0</distortion_k1>
        <distortion_k2>0.0</distortion_k2>
        <distortion_k3>0.0</distortion_k3>
        <distortion_t1>0.0</distortion_t1>
        <distortion_t2>0.0</distortion_t2>
      </plugin>
    </sensor>
  </gazebo>

  <!-- IMU Sensor -->
  <gazebo reference="imu_mount">
    <sensor name="imu_sensor" type="imu">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <visualize>false</visualize>
      <imu>
        <angular_velocity>
          <x>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>0.0017</stddev>  <!-- ~0.1 deg/s (1 sigma) -->
              <bias_mean>0.0001</bias_mean>
              <bias_stddev>0.00001</bias_stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>0.0017</stddev>
              <bias_mean>0.0001</bias_mean>
              <bias_stddev>0.00001</bias_stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>0.0017</stddev>
              <bias_mean>0.0001</bias_mean>
              <bias_stddev>0.00001</bias_stddev>
            </noise>
          </z>
        </angular_velocity>
        <linear_acceleration>
          <x>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>  <!-- 17 mg (1 sigma) -->
              <bias_mean>0.01</bias_mean>
              <bias_stddev>0.001</bias_stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
              <bias_mean>0.01</bias_mean>
              <bias_stddev>0.001</bias_stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
              <bias_mean>0.01</bias_mean>
              <bias_stddev>0.001</bias_stddev>
            </noise>
          </z>
        </linear_acceleration>
      </imu>
      <plugin name="imu_controller" filename="libgazebo_ros_imu.so">
        <ros>
          <namespace>/sensor_robot</namespace>
          <remapping>imu:=imu</remapping>
        </ros>
        <frame_name>imu_mount</frame_name>
        <body_name>imu_mount</body_name>
        <update_rate>100</update_rate>
      </plugin>
    </sensor>
  </gazebo>

  <!-- GPS Sensor -->
  <gazebo reference="base_link">
    <sensor name="gps_sensor" type="gps">
      <always_on>true</always_on>
      <update_rate>10</update_rate>
      <plugin name="gps_controller" filename="libgazebo_ros_gps.so">
        <ros>
          <namespace>/sensor_robot</namespace>
          <remapping>fix:=gps/fix</remapping>
        </ros>
        <frame_name>base_link</frame_name>
        <update_rate>10</update_rate>
        <gaussian_noise>0.1</gaussian_noise>
        <velocity_gaussian_noise>0.1</velocity_gaussian_noise>
      </plugin>
    </sensor>
  </gazebo>

  <!-- Differential drive plugin -->
  <gazebo>
    <plugin name="differential_drive" filename="libgazebo_ros_diff_drive.so">
      <ros>
        <namespace>/sensor_robot</namespace>
        <remapping>cmd_vel:=cmd_vel</remapping>
        <remapping>odom:=odom</remapping>
      </ros>
      <left_joint>left_wheel_joint</left_joint>
      <right_joint>right_wheel_joint</right_joint>
      <wheel_separation>0.5</wheel_separation>
      <wheel_diameter>0.3</wheel_diameter>
      <max_wheel_torque>20</max_wheel_torque>
      <max_wheel_acceleration>1.0</max_wheel_acceleration>
      <publish_odom>true</publish_odom>
      <publish_odom_tf>true</publish_odom_tf>
      <publish_wheel_tf>true</publish_wheel_tf>
      <odometry_frame>odom</odometry_frame>
      <robot_base_frame>base_link</robot_base_frame>
    </plugin>
  </gazebo>

  <!-- Joint state publisher -->
  <gazebo>
    <plugin name="joint_state_publisher" filename="libgazebo_ros_joint_state_publisher.so">
      <ros>
        <namespace>/sensor_robot</namespace>
        <remapping>joint_states:=joint_states</remapping>
      </ros>
      <update_rate>30</update_rate>
      <joint_name>left_wheel_joint</joint_name>
      <joint_name>right_wheel_joint</joint_name>
    </plugin>
  </gazebo>
</robot>
```

Now let's create a ROS 2 node for sensor data processing and validation:

```python
# sensor_processor.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, Imu, NavSatFix
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import statistics

class SensorProcessor(Node):
    """
    Process and validate data from multiple simulated sensors
    """
    def __init__(self):
        super().__init__('sensor_processor')

        # Publishers
        self.status_pub = self.create_publisher(String, '/sensor_status', 10)
        self.laser_stats_pub = self.create_publisher(String, '/laser_stats', 10)
        self.imu_stats_pub = self.create_publisher(String, '/imu_stats', 10)

        # Subscribers
        self.scan_sub = self.create_subscription(LaserScan, '/sensor_robot/scan', self.scan_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/sensor_robot/imu', self.imu_callback, 10)
        self.gps_sub = self.create_subscription(NavSatFix, '/sensor_robot/gps/fix', self.gps_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/sensor_robot/odom', self.odom_callback, 10)
        self.image_sub = self.create_subscription(Image, '/sensor_robot/camera/color/image_raw', self.image_callback, 10)

        # Timers
        self.processing_timer = self.create_timer(1.0, self.process_sensor_data)

        # Sensor data storage
        self.scan_data = None
        self.imu_data = None
        self.gps_data = None
        self.odom_data = None
        self.image_data = None
        self.cv_bridge = CvBridge()

        # Statistics tracking
        self.laser_ranges_history = []
        self.imu_angular_velocity_history = []
        self.imu_linear_acceleration_history = []
        self.position_history = []

        # Sensor validation parameters
        self.laser_min_range = 0.1
        self.laser_max_range = 30.0
        self.imu_angular_velocity_limits = 10.0  # rad/s
        self.imu_linear_acceleration_limits = 50.0  # m/s²

        self.get_logger().info("Sensor Processor initialized")

    def scan_callback(self, msg):
        """Process LIDAR scan data"""
        self.scan_data = msg

        # Store for statistics
        valid_ranges = [r for r in msg.ranges if not math.isnan(r) and self.laser_min_range < r < self.laser_max_range]
        if valid_ranges:
            self.laser_ranges_history.extend(valid_ranges[-10:])  # Keep last 10 valid readings
            if len(self.laser_ranges_history) > 100:
                self.laser_ranges_history = self.laser_ranges_history[-100:]  # Limit history

    def imu_callback(self, msg):
        """Process IMU data"""
        self.imu_data = msg

        # Store for statistics
        self.imu_angular_velocity_history.append([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])
        self.imu_linear_acceleration_history.append([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])

        # Limit history size
        if len(self.imu_angular_velocity_history) > 1000:
            self.imu_angular_velocity_history = self.imu_angular_velocity_history[-1000:]
        if len(self.imu_linear_acceleration_history) > 1000:
            self.imu_linear_acceleration_history = self.imu_linear_acceleration_history[-1000:]

    def gps_callback(self, msg):
        """Process GPS data"""
        self.gps_data = msg

    def odom_callback(self, msg):
        """Process odometry data"""
        self.odom_data = msg

        # Store position for drift analysis
        pos = msg.pose.pose.position
        self.position_history.append((pos.x, pos.y, pos.z))
        if len(self.position_history) > 1000:
            self.position_history = self.position_history[-1000:]

    def image_callback(self, msg):
        """Process camera image data"""
        try:
            self.image_data = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Could not convert image: {e}")

    def analyze_laser_data(self):
        """Analyze LIDAR sensor data"""
        if not self.scan_data or not self.laser_ranges_history:
            return "No LIDAR data available"

        # Basic statistics
        avg_range = statistics.mean(self.laser_ranges_history) if self.laser_ranges_history else 0
        min_range = min(self.laser_ranges_history) if self.laser_ranges_history else 0
        max_range = max(self.laser_ranges_history) if self.laser_ranges_history else 0

        # Check for valid range values
        invalid_count = sum(1 for r in self.scan_data.ranges if math.isnan(r) or r <= 0)
        total_readings = len(self.scan_data.ranges)
        invalid_percentage = (invalid_count / total_readings) * 100 if total_readings > 0 else 0

        # Analyze for obstacles
        obstacle_distances = [r for r in self.scan_data.ranges if self.laser_min_range < r < 1.0]
        obstacle_count = len(obstacle_distances)

        analysis = f"LIDAR: Avg={avg_range:.2f}m, Min={min_range:.2f}m, Max={max_range:.2f}m, " \
                  f"Invalid={invalid_percentage:.1f}%, Obstacles={obstacle_count}"

        # Publish detailed stats
        stats_msg = String()
        stats_msg.data = analysis
        self.laser_stats_pub.publish(stats_msg)

        return analysis

    def analyze_imu_data(self):
        """Analyze IMU sensor data"""
        if not self.imu_data or not self.imu_angular_velocity_history:
            return "No IMU data available"

        # Calculate statistics for angular velocity
        if self.imu_angular_velocity_history:
            avg_ang_vel = np.mean(self.imu_angular_velocity_history, axis=0)
            std_ang_vel = np.std(self.imu_angular_velocity_history, axis=0)
            max_ang_vel = np.max(np.abs(self.imu_angular_velocity_history), axis=0)

        # Calculate statistics for linear acceleration
        if self.imu_linear_acceleration_history:
            avg_lin_acc = np.mean(self.imu_linear_acceleration_history, axis=0)
            std_lin_acc = np.std(self.imu_linear_acceleration_history, axis=0)
            max_lin_acc = np.max(np.abs(self.imu_linear_acceleration_history), axis=0)

        # Check for drift in linear acceleration (should average around gravity)
        gravity_drift = abs(avg_lin_acc[2] + 9.81) if len(avg_lin_acc) > 2 else 0  # Z should be ~-9.81 for gravity

        analysis = f"IMU: AngVel(Avg)[{avg_ang_vel[0]:.3f}, {avg_ang_vel[1]:.3f}, {avg_ang_vel[2]:.3f}], " \
                  f"LinAcc(Avg)[{avg_lin_acc[0]:.2f}, {avg_lin_acc[1]:.2f}, {avg_lin_acc[2]:.2f}], " \
                  f"GravityDrift: {gravity_drift:.2f}m/s²"

        # Publish detailed stats
        stats_msg = String()
        stats_msg.data = analysis
        self.imu_stats_pub.publish(stats_msg)

        return analysis

    def analyze_sensor_fusion(self):
        """Analyze data from multiple sensors together"""
        fusion_analysis = []

        # Check if robot is moving and if sensors agree
        if self.odom_data and self.imu_data:
            linear_speed = math.sqrt(
                self.odom_data.twist.twist.linear.x**2 +
                self.odom_data.twist.twist.linear.y**2
            )

            # IMU-based linear acceleration magnitude
            imu_lin_acc_mag = math.sqrt(
                self.imu_data.linear_acceleration.x**2 +
                self.imu_data.linear_acceleration.y**2 +
                self.imu_data.linear_acceleration.z**2
            )

            fusion_analysis.append(f"Motion: OdomSpeed={linear_speed:.3f}m/s, IMULinAcc={imu_lin_acc_mag:.3f}m/s²")

        # Check GPS consistency with odometry
        if self.gps_data and self.odom_data and self.position_history:
            if len(self.position_history) >= 2:
                # Calculate recent movement from odometry
                recent_pos = self.position_history[-1]
                prev_pos = self.position_history[-2]
                odom_displacement = math.sqrt(
                    (recent_pos[0] - prev_pos[0])**2 +
                    (recent_pos[1] - prev_pos[1])**2
                )

                fusion_analysis.append(f"Position: OdomDisplacement={odom_displacement:.3f}m")

        return "; ".join(fusion_analysis) if fusion_analysis else "No fusion data available"

    def process_sensor_data(self):
        """Main processing function that runs periodically"""
        # Analyze each sensor type
        laser_analysis = self.analyze_laser_data()
        imu_analysis = self.analyze_imu_data()
        fusion_analysis = self.analyze_sensor_fusion()

        # Overall status
        status_msg = String()
        status_msg.data = f"LIDAR: {laser_analysis} | IMU: {imu_analysis} | Fusion: {fusion_analysis}"
        self.status_pub.publish(status_msg)

        # Log for debugging
        self.get_logger().info(f"Sensor Status: {status_msg.data}")

def main(args=None):
    rclpy.init(args=args)
    processor = SensorProcessor()

    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        processor.get_logger().info("Sensor Processor stopped by user")
    finally:
        processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Now let's create a sensor validation node that compares simulated data with expected real-world characteristics:

```python
# sensor_validator.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, NavSatFix
from geometry_msgs.msg import Vector3
from std_msgs.msg import String, Bool
import numpy as np
import math
from collections import deque
import statistics

class SensorValidator(Node):
    """
    Validate simulated sensor data against expected real-world characteristics
    """
    def __init__(self):
        super().__init__('sensor_validator')

        # Publishers
        self.validation_pub = self.create_publisher(String, '/sensor_validation', 10)
        self.health_pub = self.create_publisher(String, '/sensor_health', 10)

        # Subscribers
        self.scan_sub = self.create_subscription(LaserScan, '/sensor_robot/scan', self.scan_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/sensor_robot/imu', self.imu_callback, 10)
        self.gps_sub = self.create_subscription(NavSatFix, '/sensor_robot/gps/fix', self.gps_callback, 10)

        # Timers
        self.validation_timer = self.create_timer(2.0, self.run_validation)

        # Sensor data with history
        self.scan_history = deque(maxlen=10)  # Keep last 10 scans
        self.imu_history = deque(maxlen=100)  # Keep last 100 IMU readings
        self.gps_history = deque(maxlen=20)   # Keep last 20 GPS readings

        # Validation parameters
        self.validation_results = {
            'lidar': {'status': 'unknown', 'details': ''},
            'imu': {'status': 'unknown', 'details': ''},
            'gps': {'status': 'unknown', 'details': ''}
        }

        # Expected characteristics for validation
        self.expected_lidar_specs = {
            'min_range': 0.1,
            'max_range': 30.0,
            'fov_horizontal': 270,  # degrees
            'resolution': 0.5  # degrees per sample
        }

        self.expected_imu_specs = {
            'angular_velocity_noise_std': 0.0017,  # rad/s
            'linear_acceleration_noise_std': 0.017,  # m/s²
            'gravity': 9.81  # m/s²
        }

        self.get_logger().info("Sensor Validator initialized")

    def scan_callback(self, msg):
        """Process and store LIDAR scan data"""
        self.scan_history.append({
            'header': msg.header,
            'ranges': list(msg.ranges),
            'intensities': list(msg.intensities) if msg.intensities else [],
            'angle_min': msg.angle_min,
            'angle_max': msg.angle_max,
            'angle_increment': msg.angle_increment,
            'time_increment': msg.time_increment,
            'scan_time': msg.scan_time,
            'range_min': msg.range_min,
            'range_max': msg.range_max
        })

    def imu_callback(self, msg):
        """Process and store IMU data"""
        self.imu_history.append({
            'header': msg.header,
            'orientation': {
                'x': msg.orientation.x,
                'y': msg.orientation.y,
                'z': msg.orientation.z,
                'w': msg.orientation.w
            },
            'angular_velocity': {
                'x': msg.angular_velocity.x,
                'y': msg.angular_velocity.y,
                'z': msg.angular_velocity.z
            },
            'linear_acceleration': {
                'x': msg.linear_acceleration.x,
                'y': msg.linear_acceleration.y,
                'z': msg.linear_acceleration.z
            }
        })

    def gps_callback(self, msg):
        """Process and store GPS data"""
        self.gps_history.append({
            'header': msg.header,
            'latitude': msg.latitude,
            'longitude': msg.longitude,
            'altitude': msg.altitude,
            'position_covariance': list(msg.position_covariance),
            'position_covariance_type': msg.position_covariance_type
        })

    def validate_lidar(self):
        """Validate LIDAR sensor characteristics"""
        if not self.scan_history:
            return {'status': 'no_data', 'details': 'No LIDAR data received'}

        latest_scan = self.scan_history[-1]
        issues = []

        # Check range limits
        valid_ranges = [r for r in latest_scan['ranges'] if not math.isnan(r)]
        if valid_ranges:
            min_range = min(valid_ranges)
            max_range = max(valid_ranges)

            if min_range < self.expected_lidar_specs['min_range']:
                issues.append(f"Range too low: {min_range:.2f}m < {self.expected_lidar_specs['min_range']}m")
            if max_range > self.expected_lidar_specs['max_range']:
                issues.append(f"Range too high: {max_range:.2f}m > {self.expected_lidar_specs['max_range']}m")

        # Check field of view
        fov_degrees = math.degrees(latest_scan['angle_max'] - latest_scan['angle_min'])
        expected_fov = self.expected_lidar_specs['fov_horizontal']
        if abs(fov_degrees - expected_fov) > 10:  # Allow 10 degree tolerance
            issues.append(f"FOV mismatch: {fov_degrees:.1f}° vs expected {expected_fov}°")

        # Check number of samples vs expected resolution
        expected_samples = (fov_degrees / self.expected_lidar_specs['resolution'])
        actual_samples = len(latest_scan['ranges'])
        if abs(actual_samples - expected_samples) > expected_samples * 0.1:  # 10% tolerance
            issues.append(f"Sample count mismatch: {actual_samples} vs expected ~{expected_samples}")

        # Check for realistic data patterns (not all same values)
        if len(set(valid_ranges)) < len(valid_ranges) * 0.1:  # Less than 10% unique values
            issues.append("Unrealistic: too many identical range values")

        status = 'valid' if not issues else 'issues'
        details = '; '.join(issues) if issues else 'All checks passed'

        return {'status': status, 'details': details}

    def validate_imu(self):
        """Validate IMU sensor characteristics"""
        if len(self.imu_history) < 10:  # Need some history for validation
            return {'status': 'insufficient_data', 'details': 'Need more IMU data for validation'}

        # Extract angular velocity and linear acceleration data
        ang_vel_data = []
        lin_acc_data = []

        for imu_reading in self.imu_history:
            ang_vel_data.append([
                imu_reading['angular_velocity']['x'],
                imu_reading['angular_velocity']['y'],
                imu_reading['angular_velocity']['z']
            ])
            lin_acc_data.append([
                imu_reading['linear_acceleration']['x'],
                imu_reading['linear_acceleration']['y'],
                imu_reading['linear_acceleration']['z']
            ])

        ang_vel_array = np.array(ang_vel_data)
        lin_acc_array = np.array(lin_acc_data)

        issues = []

        # Check angular velocity noise levels
        ang_vel_std = np.std(ang_vel_array, axis=0)
        expected_ang_vel_noise = self.expected_imu_specs['angular_velocity_noise_std']

        for i, std_val in enumerate(ang_vel_std):
            if std_val > expected_ang_vel_noise * 3:  # 3x tolerance
                axis = ['X', 'Y', 'Z'][i]
                issues.append(f"Angular velocity {axis} noise too high: {std_val:.6f} > {expected_ang_vel_noise:.6f}")

        # Check linear acceleration noise levels
        lin_acc_std = np.std(lin_acc_array, axis=0)
        expected_lin_acc_noise = self.expected_imu_specs['linear_acceleration_noise_std']

        for i, std_val in enumerate(lin_acc_std):
            if std_val > expected_lin_acc_noise * 3:  # 3x tolerance
                axis = ['X', 'Y', 'Z'][i]
                issues.append(f"Linear acceleration {axis} noise too high: {std_val:.6f} > {expected_lin_acc_noise:.6f}")

        # Check gravity in Z-axis (should be around 9.81 m/s² when robot is stationary)
        lin_acc_z_mean = np.mean(lin_acc_array[:, 2])
        gravity_diff = abs(abs(lin_acc_z_mean) - self.expected_imu_specs['gravity'])

        if gravity_diff > 1.0:  # Allow 1 m/s² tolerance
            issues.append(f"Gravity detection issue: Z-axis avg {lin_acc_z_mean:.2f} vs expected ~{self.expected_imu_specs['gravity']}")

        # Check for realistic IMU data patterns
        # (e.g., shouldn't have impossible acceleration values)
        max_acc_magnitude = np.max(np.linalg.norm(lin_acc_array, axis=1))
        if max_acc_magnitude > 100:  # Unlikely to have >100 m/s² acceleration
            issues.append(f"Unrealistic acceleration: {max_acc_magnitude:.2f} m/s²")

        status = 'valid' if not issues else 'issues'
        details = '; '.join(issues) if issues else 'All checks passed'

        return {'status': status, 'details': details}

    def validate_gps(self):
        """Validate GPS sensor characteristics"""
        if not self.gps_history:
            return {'status': 'no_data', 'details': 'No GPS data received'}

        latest_gps = self.gps_history[-1]
        issues = []

        # Check if coordinates are reasonable (not zero or invalid)
        if latest_gps['latitude'] == 0.0 and latest_gps['longitude'] == 0.0:
            issues.append("GPS at origin (0,0) - possibly not initialized")

        # Check position covariance (should be finite and reasonable)
        pos_cov = latest_gps['position_covariance']
        if any(c > 1000 for c in pos_cov if not math.isnan(c)):  # If covariance is too high
            issues.append("High position uncertainty")

        # Check for realistic coordinate changes between readings
        if len(self.gps_history) > 1:
            prev_gps = self.gps_history[-2]
            lat_diff = abs(latest_gps['latitude'] - prev_gps['latitude'])
            lon_diff = abs(latest_gps['longitude'] - prev_gps['longitude'])

            # Convert to approximate meters (roughly 111km per degree latitude)
            lat_meters = lat_diff * 111000
            lon_meters = lon_diff * 111000 * math.cos(math.radians(latest_gps['latitude']))
            distance = math.sqrt(lat_meters**2 + lon_meters**2)

            # If the robot is moving too fast (e.g., >10 m/s) for a ground robot, flag it
            # This assumes ~2Hz GPS update rate
            if distance > 20:  # More than 20m in ~0.5s (40 m/s average)
                issues.append(f"Unrealistic GPS movement: {distance:.2f}m in recent update")

        status = 'valid' if not issues else 'issues'
        details = '; '.join(issues) if issues else 'All checks passed'

        return {'status': status, 'details': details}

    def run_validation(self):
        """Run comprehensive sensor validation"""
        # Validate each sensor type
        self.validation_results['lidar'] = self.validate_lidar()
        self.validation_results['imu'] = self.validate_imu()
        self.validation_results['gps'] = self.validate_gps()

        # Overall validation status
        all_valid = all(result['status'] in ['valid', 'issues'] for result in self.validation_results.values())
        any_issues = any(result['status'] == 'issues' for result in self.validation_results.values())

        overall_status = 'valid' if all_valid and not any_issues else 'issues' if any_issues else 'invalid'

        # Publish validation results
        validation_msg = String()
        validation_msg.data = f"Overall: {overall_status} | " \
                             f"LIDAR: {self.validation_results['lidar']['status']} | " \
                             f"IMU: {self.validation_results['imu']['status']} | " \
                             f"GPS: {self.validation_results['gps']['status']}"
        self.validation_pub.publish(validation_msg)

        # Publish detailed health report
        health_msg = String()
        health_details = []
        for sensor, result in self.validation_results.items():
            health_details.append(f"{sensor.upper()}: {result['status']} - {result['details']}")

        health_msg.data = " | ".join(health_details)
        self.health_pub.publish(health_msg)

        # Log results
        self.get_logger().info(f"Sensor Validation: {validation_msg.data}")

def main(args=None):
    rclpy.init(args=args)
    validator = SensorValidator()

    try:
        rclpy.spin(validator)
    except KeyboardInterrupt:
        validator.get_logger().info("Sensor Validator stopped by user")
    finally:
        validator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Small Simulation

Let's create a sensor fusion simulation that demonstrates how multiple sensors work together:

```python
# sensor_fusion_demo.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, NavSatFix
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float64MultiArray
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from collections import deque

class SensorFusionDemo(Node):
    """
    Demonstrate sensor fusion using multiple simulated sensors
    """
    def __init__(self):
        super().__init__('sensor_fusion_demo')

        # Publishers
        self.fused_pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/sensor_fusion/pose', 10)
        self.fused_twist_pub = self.create_publisher(TwistWithCovarianceStamped, '/sensor_fusion/twist', 10)
        self.status_pub = self.create_publisher(String, '/sensor_fusion/status', 10)

        # Subscribers
        self.scan_sub = self.create_subscription(LaserScan, '/sensor_robot/scan', self.scan_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/sensor_robot/imu', self.imu_callback, 10)
        self.gps_sub = self.create_subscription(NavSatFix, '/sensor_robot/gps/fix', self.gps_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/sensor_robot/odom', self.odom_callback, 10)

        # Timers
        self.fusion_timer = self.create_timer(0.05, self.run_fusion)  # 20 Hz

        # Sensor data storage
        self.scan_data = None
        self.imu_data = None
        self.gps_data = None
        self.odom_data = None

        # Fusion state
        self.fused_position = np.array([0.0, 0.0, 0.0])  # x, y, z
        self.fused_orientation = np.array([0.0, 0.0, 0.0, 1.0])  # x, y, z, w quaternion
        self.fused_velocity = np.array([0.0, 0.0, 0.0])  # vx, vy, vz
        self.fused_angular_velocity = np.array([0.0, 0.0, 0.0])  # wx, wy, wz

        # Covariance matrices (diagonal for simplicity)
        self.position_covariance = np.array([0.1, 0.1, 0.1])  # Initial uncertainty
        self.orientation_covariance = np.array([0.05, 0.05, 0.05])  # Initial uncertainty
        self.velocity_covariance = np.array([0.2, 0.2, 0.2])  # Initial uncertainty

        # Sensor weights for fusion
        self.sensor_weights = {
            'odom': 0.6,   # Wheel odometry - good for short-term precision
            'imu': 0.3,    # IMU - good for orientation and acceleration
            'gps': 0.1,    # GPS - good for absolute position (when available)
            'lidar': 0.2   # LIDAR - good for relative positioning to landmarks
        }

        # History for smoothing
        self.position_history = deque(maxlen=10)
        self.velocity_history = deque(maxlen=10)

        # Flags to track sensor availability
        self.imu_available = False
        self.gps_available = False
        self.lidar_available = False

        self.get_logger().info("Sensor Fusion Demo initialized")

    def scan_callback(self, msg):
        """Process LIDAR scan data"""
        self.scan_data = msg
        self.lidar_available = True

    def imu_callback(self, msg):
        """Process IMU data"""
        self.imu_data = msg
        self.imu_available = True

    def gps_callback(self, msg):
        """Process GPS data"""
        self.gps_data = msg
        self.gps_available = True

    def odom_callback(self, msg):
        """Process odometry data"""
        self.odom_data = msg

    def predict_state(self, dt):
        """Predict state based on IMU and previous state"""
        if self.imu_data and dt > 0:
            # Update orientation from angular velocity
            ang_vel = np.array([
                self.imu_data.angular_velocity.x,
                self.imu_data.angular_velocity.y,
                self.imu_data.angular_velocity.z
            ])

            # Convert angular velocity to quaternion derivative
            q = self.fused_orientation
            omega_matrix = np.array([
                [0, -ang_vel[0], -ang_vel[1], -ang_vel[2]],
                [ang_vel[0], 0, ang_vel[2], -ang_vel[1]],
                [ang_vel[1], -ang_vel[2], 0, ang_vel[0]],
                [ang_vel[2], ang_vel[1], -ang_vel[0], 0]
            ])

            # Integrate quaternion
            q_dot = 0.5 * omega_matrix @ q
            self.fused_orientation += q_dot * dt
            # Normalize quaternion
            self.fused_orientation = self.fused_orientation / np.linalg.norm(self.fused_orientation)

            # Update velocity from linear acceleration (in world frame)
            acc = np.array([
                self.imu_data.linear_acceleration.x,
                self.imu_data.linear_acceleration.y,
                self.imu_data.linear_acceleration.z
            ])

            # Transform acceleration to world frame using current orientation
            r = R.from_quat(self.fused_orientation)
            world_acc = r.apply(acc)

            self.fused_velocity += world_acc * dt

            # Update position from velocity
            self.fused_position += self.fused_velocity * dt

    def update_from_odom(self):
        """Update state estimate from odometry"""
        if self.odom_data:
            odom_pos = self.odom_data.pose.pose.position
            odom_pos_array = np.array([odom_pos.x, odom_pos.y, odom_pos.z])

            # Simple weighted update
            self.fused_position = (self.sensor_weights['odom'] * odom_pos_array +
                                  (1 - self.sensor_weights['odom']) * self.fused_position)

            # Update velocity from odometry twist
            odom_vel = self.odom_data.twist.twist.linear
            odom_vel_array = np.array([odom_vel.x, odom_vel.y, odom_vel.z])
            self.fused_velocity = (self.sensor_weights['odom'] * odom_vel_array +
                                  (1 - self.sensor_weights['odom']) * self.fused_velocity)

    def update_from_gps(self):
        """Update state estimate from GPS (simplified)"""
        if self.gps_data:
            # Convert lat/lon to local coordinates (very simplified)
            # In a real system, you'd use a proper coordinate transformation
            gps_pos = np.array([self.gps_data.longitude * 1000,
                               self.gps_data.latitude * 1000,
                               self.gps_data.altitude])

            # Update position with GPS (low weight due to drift)
            self.fused_position = (self.sensor_weights['gps'] * gps_pos +
                                  (1 - self.sensor_weights['gps']) * self.fused_position)

    def update_from_lidar(self):
        """Update state from LIDAR landmark detection (simplified)"""
        if self.scan_data:
            # Simplified: detect "landmarks" as consistent LIDAR returns
            # In reality, you'd run SLAM or landmark detection algorithms
            valid_ranges = [i for i, r in enumerate(self.scan_data.ranges)
                           if not math.isnan(r) and self.scan_data.range_min < r < self.scan_data.range_max]

            if len(valid_ranges) > 10:  # If we have significant features
                # This is a placeholder - in reality you'd match to a map
                # For now, just reduce uncertainty if we have good LIDAR data
                self.position_covariance *= 0.9  # Reduce uncertainty

    def run_fusion(self):
        """Main sensor fusion loop"""
        current_time = self.get_clock().now()

        # Get time delta since last update
        dt = 0.05  # Fixed 20Hz rate

        # Prediction step: use IMU to predict state
        self.predict_state(dt)

        # Update steps: incorporate other sensor data
        self.update_from_odom()
        if self.gps_available:
            self.update_from_gps()
        if self.lidar_available:
            self.update_from_lidar()

        # Store in history for smoothing
        self.position_history.append(self.fused_position.copy())
        self.velocity_history.append(self.fused_velocity.copy())

        # Apply simple smoothing
        if len(self.position_history) > 1:
            smoothed_pos = np.mean(self.position_history, axis=0)
            self.fused_position = 0.7 * self.fused_position + 0.3 * smoothed_pos

        if len(self.velocity_history) > 1:
            smoothed_vel = np.mean(self.velocity_history, axis=0)
            self.fused_velocity = 0.7 * self.fused_velocity + 0.3 * smoothed_vel

        # Publish fused state
        self.publish_fused_state(current_time)

        # Publish status
        status_msg = String()
        status_msg.data = f"Fused Pos: ({self.fused_position[0]:.2f}, {self.fused_position[1]:.2f}, {self.fused_position[2]:.2f}), " \
                         f"Vel: ({self.fused_velocity[0]:.2f}, {self.fused_velocity[1]:.2f}, {self.fused_velocity[2]:.2f}), " \
                         f"PosCov: ({self.position_covariance[0]:.3f}, {self.position_covariance[1]:.3f}, {self.position_covariance[2]:.3f})"
        self.status_pub.publish(status_msg)

    def publish_fused_state(self, timestamp):
        """Publish the fused state estimate"""
        # Publish fused pose
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = timestamp.to_msg()
        pose_msg.header.frame_id = "map"

        pose_msg.pose.pose.position.x = float(self.fused_position[0])
        pose_msg.pose.pose.position.y = float(self.fused_position[1])
        pose_msg.pose.pose.position.z = float(self.fused_position[2])

        pose_msg.pose.pose.orientation.x = float(self.fused_orientation[0])
        pose_msg.pose.pose.orientation.y = float(self.fused_orientation[1])
        pose_msg.pose.pose.orientation.z = float(self.fused_orientation[2])
        pose_msg.pose.pose.orientation.w = float(self.fused_orientation[3])

        # Set covariance (diagonal elements only, for simplicity)
        for i in range(3):
            pose_msg.pose.covariance[i*6 + i] = float(self.position_covariance[i])
        for i in range(3):
            pose_msg.pose.covariance[(i+3)*6 + (i+3)] = float(self.orientation_covariance[i])

        self.fused_pose_pub.publish(pose_msg)

        # Publish fused twist
        twist_msg = TwistWithCovarianceStamped()
        twist_msg.header.stamp = timestamp.to_msg()
        twist_msg.header.frame_id = "base_link"

        twist_msg.twist.twist.linear.x = float(self.fused_velocity[0])
        twist_msg.twist.twist.linear.y = float(self.fused_velocity[1])
        twist_msg.twist.twist.linear.z = float(self.fused_velocity[2])

        if self.imu_data:
            twist_msg.twist.twist.angular.x = self.imu_data.angular_velocity.x
            twist_msg.twist.twist.angular.y = self.imu_data.angular_velocity.y
            twist_msg.twist.twist.angular.z = self.imu_data.angular_velocity.z

        # Set velocity covariance
        for i in range(3):
            twist_msg.twist.covariance[i*6 + i] = float(self.velocity_covariance[i])

        self.fused_twist_pub.publish(twist_msg)

def main(args=None):
    rclpy.init(args=args)
    fusion_demo = SensorFusionDemo()

    try:
        rclpy.spin(fusion_demo)
    except KeyboardInterrupt:
        fusion_demo.get_logger().info("Sensor Fusion Demo stopped by user")
    finally:
        fusion_demo.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Quick Recap

In this lesson, we've covered:

- **LIDAR Simulation**: Configuring realistic LIDAR sensors with proper noise models and characteristics
- **Camera Simulation**: Setting up depth cameras with realistic distortion and resolution parameters
- **IMU Modeling**: Creating IMU sensors with realistic drift, noise, and bias characteristics
- **Multi-Sensor Integration**: Combining multiple sensors for comprehensive perception
- **Sensor Validation**: Techniques for validating simulated sensor data against real-world expectations
- **Sensor Fusion**: Demonstrating how multiple sensors work together to improve robot perception

Sensor simulation is crucial for creating believable digital twins. The accuracy of your simulated sensors directly impacts how well your robot algorithms will perform when transferred to real hardware. Properly configured sensors with realistic noise models, drift characteristics, and limitations ensure that your robot's perception and navigation systems are robust enough to handle real-world conditions.

## Summary of Chapter 3

In Chapter 3: "The Digital Twin (Gazebo & Unity)", we've covered:

1. **Introduction to Gazebo Simulation**: Core concepts and basic robot simulation setup
2. **Physics and Collision Modeling**: Realistic physics properties and collision detection
3. **Unity-Based Visualization**: High-quality visualization and user interface creation
4. **Sensor Simulation**: Comprehensive sensor modeling and fusion techniques

This chapter provides the foundation for creating realistic digital twins that combine accurate physics simulation with high-quality visualization and realistic sensor models.