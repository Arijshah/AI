---
sidebar_label: 'Unity-Based Visualization'
title: 'Unity-Based Visualization'
---

# Unity-Based Visualization

## Overview

Unity provides a powerful real-time visualization platform that can serve as a complementary tool to Gazebo's physics simulation. While Gazebo excels at physics-based simulation, Unity offers high-quality graphics rendering, advanced visual effects, and immersive user interfaces. This lesson explores how to integrate Unity with ROS 2 for enhanced robot visualization, debugging, and human-robot interaction design.

Unity's capabilities include photorealistic rendering, virtual reality integration, and custom user interfaces that can provide additional insights into robot behavior beyond what traditional simulation environments offer.

## Learning Outcomes

By the end of this lesson, you will be able to:
- Understand the role of Unity in robotics visualization
- Set up Unity-ROS 2 communication bridges
- Create custom visualizations for robot states and sensor data
- Implement VR/AR interfaces for robot teleoperation
- Design intuitive interfaces for robot monitoring and control

## Hands-on Steps

1. **Unity-ROS Bridge Setup**: Configure communication between Unity and ROS 2
2. **Robot Visualization**: Create 3D models and animations for robot states
3. **Sensor Data Visualization**: Display sensor data in Unity environment
4. **User Interface Creation**: Build custom UI for robot monitoring
5. **VR Integration**: Implement basic VR teleoperation interface

### Prerequisites

- Understanding of ROS 2 concepts and message types
- Basic familiarity with Unity development (or willingness to learn)
- Knowledge of 3D modeling concepts

## Code Examples

Let's start by creating ROS 2 nodes that can interface with Unity for visualization purposes:

```python
# unity_bridge.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist, Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Image, JointState
from std_msgs.msg import String, Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
import math
import json

class UnityBridge(Node):
    """
    Bridge node for sending ROS 2 data to Unity for visualization
    """
    def __init__(self):
        super().__init__('unity_bridge')

        # Publishers for Unity visualization
        self.robot_pose_pub = self.create_publisher(Float32MultiArray, '/unity/robot_pose', 10)
        self.laser_data_pub = self.create_publisher(Float32MultiArray, '/unity/laser_data', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/unity/markers', 10)
        self.status_pub = self.create_publisher(String, '/unity/status', 10)

        # Subscribers for robot data
        self.odom_sub = self.create_subscription(Odometry, '/physics_robot/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/physics_robot/scan', self.scan_callback, 10)
        self.joint_sub = self.create_subscription(JointState, '/physics_robot/joint_states', self.joint_callback, 10)

        # Timer for publishing visualization data
        self.viz_timer = self.create_timer(0.033, self.publish_visualization_data)  # ~30 FPS

        # Robot state storage
        self.robot_pose = Pose()
        self.laser_ranges = []
        self.joint_positions = {}
        self.last_update_time = self.get_clock().now()

        # Visualization parameters
        self.robot_scale = [1.0, 1.0, 1.0]
        self.robot_color = [0.2, 0.6, 1.0, 1.0]  # RGBA

        self.get_logger().info("Unity Bridge initialized")

    def odom_callback(self, msg):
        """Process odometry data for visualization"""
        self.robot_pose = msg.pose.pose

    def scan_callback(self, msg):
        """Process laser scan data for visualization"""
        self.laser_ranges = msg.ranges

    def joint_callback(self, msg):
        """Process joint state data"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]

    def publish_visualization_data(self):
        """Publish robot data in Unity-compatible format"""
        # Publish robot pose (position and orientation as float array)
        pose_msg = Float32MultiArray()
        pose_msg.data = [
            self.robot_pose.position.x,
            self.robot_pose.position.y,
            self.robot_pose.position.z,
            self.robot_pose.orientation.x,
            self.robot_pose.orientation.y,
            self.robot_pose.orientation.z,
            self.robot_pose.orientation.w
        ]
        self.robot_pose_pub.publish(pose_msg)

        # Publish laser scan data
        if self.laser_ranges:
            laser_msg = Float32MultiArray()
            # Limit to 360 points for performance
            step = max(1, len(self.laser_ranges) // 360)
            laser_msg.data = self.laser_ranges[::step][:360]  # Take up to 360 points
            self.laser_data_pub.publish(laser_msg)

        # Publish markers for Unity visualization
        marker_array = MarkerArray()

        # Robot marker
        robot_marker = Marker()
        robot_marker.header.frame_id = "map"
        robot_marker.header.stamp = self.get_clock().now().to_msg()
        robot_marker.ns = "robot"
        robot_marker.id = 0
        robot_marker.type = Marker.MESH_RESOURCE
        robot_marker.mesh_resource = "package://unity_visualization/meshes/robot.dae"
        robot_marker.action = Marker.ADD
        robot_marker.pose = self.robot_pose
        robot_marker.scale.x = self.robot_scale[0]
        robot_marker.scale.y = self.robot_scale[1]
        robot_marker.scale.z = self.robot_scale[2]
        robot_marker.color.r = self.robot_color[0]
        robot_marker.color.g = self.robot_color[1]
        robot_marker.color.b = self.robot_color[2]
        robot_marker.color.a = self.robot_color[3]
        marker_array.markers.append(robot_marker)

        # Laser scan visualization as points
        if self.laser_ranges:
            scan_marker = Marker()
            scan_marker.header.frame_id = "base_link"  # Robot's frame
            scan_marker.header.stamp = self.get_clock().now().to_msg()
            scan_marker.ns = "laser_scan"
            scan_marker.id = 1
            scan_marker.type = Marker.POINTS
            scan_marker.action = Marker.ADD
            scan_marker.scale.x = 0.02  # Point size
            scan_marker.scale.y = 0.02
            scan_marker.color.r = 1.0
            scan_marker.color.g = 0.0
            scan_marker.color.b = 0.0
            scan_marker.color.a = 0.8

            # Convert laser ranges to points
            angle_min = -math.pi/2  # Assuming 90 degree FOV
            angle_increment = math.pi / len(self.laser_ranges) if len(self.laser_ranges) > 0 else 0.01

            for i, range_val in enumerate(self.laser_ranges[::10]):  # Sample every 10th point for performance
                if not math.isnan(range_val) and range_val > 0.1 and range_val < 10.0:
                    angle = angle_min + i * 10 * angle_increment
                    x = range_val * math.cos(angle)
                    y = range_val * math.sin(angle)

                    point = Point()
                    point.x = x
                    point.y = y
                    point.z = 0.1  # Slightly above ground
                    scan_marker.points.append(point)

                    # Add color based on distance
                    scan_marker.colors.append(self.get_color_for_distance(range_val))

            marker_array.markers.append(scan_marker)

        self.marker_pub.publish(marker_array)

        # Publish status
        status_msg = String()
        status_msg.data = f"Unity Bridge: Pos=({self.robot_pose.position.x:.2f}, {self.robot_pose.position.y:.2f}), " \
                         f"Scan Points={len(self.laser_ranges)}, Joints={len(self.joint_positions)}"
        self.status_pub.publish(status_msg)

    def get_color_for_distance(self, distance):
        """Get color based on distance value for visualization"""
        from std_msgs.msg import ColorRGBA
        color = ColorRGBA()

        # Color coding: red for close, green for far
        if distance < 1.0:
            color.r = 1.0
            color.g = 0.0
            color.b = 0.0
        elif distance < 3.0:
            color.r = 1.0
            color.g = 1.0
            color.b = 0.0
        else:
            color.r = 0.0
            color.g = 1.0
            color.b = 0.0

        color.a = 1.0
        return color

def main(args=None):
    rclpy.init(args=args)
    bridge = UnityBridge()

    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        bridge.get_logger().info("Unity Bridge stopped by user")
    finally:
        bridge.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Now let's create a Unity C# script that can receive and visualize the ROS 2 data:

```csharp
// UnityRobotVisualizer.cs
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Ros2ForUnity.Messages.Std_msgs;
using Ros2ForUnity.Messages.Geometry_msgs;
using Ros2ForUnity.Messages.Sensor_msgs;
using Ros2ForUnity.Messages.Nav_msgs;

public class UnityRobotVisualizer : MonoBehaviour
{
    [Header("ROS Connection")]
    public string rosAgentIp = "127.0.0.1";
    public int rosAgentPort = 8888;

    [Header("Robot Visualization")]
    public GameObject robotModel;
    public Material robotMaterial;
    public float robotScale = 1.0f;

    [Header("Laser Visualization")]
    public GameObject laserPointPrefab;
    public Material laserMaterial;
    public float maxLaserDistance = 10.0f;
    public int laserPointCount = 360;

    [Header("UI Elements")]
    public UnityEngine.UI.Text statusText;
    public UnityEngine.UI.Text positionText;

    private Ros2Node rosNode;
    private Ros2Publisher<Float32MultiArray> robotPosePublisher;
    private Ros2Subscriber<Float32MultiArray> robotPoseSubscriber;
    private Ros2Subscriber<Float32MultiArray> laserDataSubscriber;
    private Ros2Subscriber<Visualization_msgs.MarkerArray> markerSubscriber;

    private GameObject[] laserPoints;
    private Vector3 robotPosition;
    private Quaternion robotRotation;

    void Start()
    {
        // Initialize ROS 2 connection
        InitializeROSConnection();

        // Initialize laser visualization
        InitializeLaserVisualization();

        // Set initial values
        robotPosition = Vector3.zero;
        robotRotation = Quaternion.identity;
    }

    void InitializeROSConnection()
    {
        // Create ROS node
        rosNode = new Ros2Node("unity_visualizer");

        // Create subscribers
        robotPoseSubscriber = rosNode.CreateSubscription<Float32MultiArray>(
            "/unity/robot_pose",
            ReceiveRobotPose);

        laserDataSubscriber = rosNode.CreateSubscription<Float32MultiArray>(
            "/unity/laser_data",
            ReceiveLaserData);

        markerSubscriber = rosNode.CreateSubscription<Visualization_msgs.MarkerArray>(
            "/unity/markers",
            ReceiveMarkers);

        // Start ROS communication
        Ros2ForUnity.Ros2cs.Init();
        Ros2ForUnity.Ros2cs.CreateNode(rosNode);

        rosNode.Spinner.Start();
    }

    void InitializeLaserVisualization()
    {
        // Create laser point objects
        laserPoints = new GameObject[laserPointCount];
        for (int i = 0; i < laserPointCount; i++)
        {
            laserPoints[i] = Instantiate(laserPointPrefab);
            laserPoints[i].SetActive(false);
            laserPoints[i].GetComponent<Renderer>().material = laserMaterial;
        }
    }

    void ReceiveRobotPose(Float32MultiArray msg)
    {
        if (msg.Data.Count >= 7) // Position (3) + Orientation (4)
        {
            robotPosition = new Vector3(msg.Data[0], msg.Data[1], msg.Data[2]);
            robotRotation = new Quaternion(msg.Data[3], msg.Data[4], msg.Data[5], msg.Data[6]);

            // Update robot position in Unity
            if (robotModel != null)
            {
                robotModel.transform.position = robotPosition;
                robotModel.transform.rotation = robotRotation;
            }

            // Update UI
            if (positionText != null)
            {
                positionText.text = $"Position: ({robotPosition.x:F2}, {robotPosition.y:F2}, {robotPosition.z:F2})";
            }
        }
    }

    void ReceiveLaserData(Float32MultiArray msg)
    {
        if (msg.Data.Count > 0)
        {
            // Update laser visualization
            UpdateLaserVisualization(msg.Data);
        }
    }

    void ReceiveMarkers(Visualization_msgs.MarkerArray msg)
    {
        foreach (var marker in msg.Markers)
        {
            if (marker.Type == Visualization_msgs.Marker.POINTS)
            {
                // Handle point cloud visualization
                UpdatePointCloudVisualization(marker);
            }
            else if (marker.Type == Visualization_msgs.Marker.MESH_RESOURCE)
            {
                // Handle robot model update
                UpdateRobotModel(marker);
            }
        }
    }

    void UpdateLaserVisualization(List<float> ranges)
    {
        float angleStep = 2.0f * Mathf.PI / ranges.Count;

        for (int i = 0; i < Mathf.Min(ranges.Count, laserPointCount); i++)
        {
            float range = ranges[i];
            if (range > 0.1f && range < maxLaserDistance)
            {
                float angle = i * angleStep - Mathf.PI; // Center at -π to π

                Vector3 pointPos = new Vector3(
                    range * Mathf.Cos(angle),
                    0.1f, // Slightly above ground
                    range * Mathf.Sin(angle)
                );

                laserPoints[i].transform.position = robotModel.transform.TransformPoint(pointPos);
                laserPoints[i].SetActive(true);

                // Color based on distance
                float colorValue = Mathf.InverseLerp(0.1f, maxLaserDistance, range);
                laserPoints[i].GetComponent<Renderer>().material.color =
                    Color.Lerp(Color.red, Color.green, colorValue);
            }
            else
            {
                laserPoints[i].SetActive(false);
            }
        }

        // Hide remaining points
        for (int i = ranges.Count; i < laserPointCount; i++)
        {
            laserPoints[i].SetActive(false);
        }
    }

    void UpdatePointCloudVisualization(Visualization_msgs.Marker marker)
    {
        // Process point cloud data from ROS marker
        for (int i = 0; i < Mathf.Min(marker.Points.Count, laserPointCount); i++)
        {
            var rosPoint = marker.Points[i];
            Vector3 unityPoint = new Vector3(rosPoint.X, rosPoint.Z, rosPoint.Y); // Convert ROS to Unity coordinates

            if (i < laserPoints.Length)
            {
                laserPoints[i].transform.position = unityPoint;
                laserPoints[i].SetActive(true);

                if (i < marker.Colors.Count)
                {
                    var rosColor = marker.Colors[i];
                    laserPoints[i].GetComponent<Renderer>().material.color =
                        new Color(rosColor.R, rosColor.G, rosColor.B, rosColor.A);
                }
            }
        }
    }

    void UpdateRobotModel(Visualization_msgs.Marker marker)
    {
        // Update robot model based on marker data
        if (robotModel != null)
        {
            robotModel.transform.position = new Vector3(
                (float)marker.Pose.Position.X,
                (float)marker.Pose.Position.Z, // ROS Y -> Unity Z
                (float)marker.Pose.Position.Y  // ROS Z -> Unity Y
            );

            robotModel.transform.rotation = new Quaternion(
                (float)marker.Pose.Orientation.X,
                (float)marker.Pose.Orientation.Z,
                (float)marker.Pose.Orientation.Y,
                (float)marker.Pose.Orientation.W
            );
        }
    }

    void Update()
    {
        // Update status text
        if (statusText != null)
        {
            statusText.text = $"ROS Connection: {(rosNode != null ? "Connected" : "Disconnected")}\n" +
                             $"Laser Points: {(laserPoints != null ? laserPoints.Length : 0)}";
        }
    }

    void OnDestroy()
    {
        if (rosNode != null)
        {
            rosNode.Spinner.Stop();
            Ros2ForUnity.Ros2cs.DestroyNode(rosNode);
            Ros2ForUnity.Ros2cs.Shutdown();
        }
    }
}
```

Now let's create a ROS 2 node for Unity-based teleoperation interface:

```python
# unity_teleop.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import Joy
from std_msgs.msg import String, Bool
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl
import math

class UnityTeleop(Node):
    """
    Unity-based teleoperation interface for robot control
    """
    def __init__(self):
        super().__init__('unity_teleop')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/physics_robot/cmd_vel', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/unity/teleop_pose', 10)
        self.status_pub = self.create_publisher(String, '/unity/teleop_status', 10)

        # Subscribers
        self.joy_sub = self.create_subscription(Joy, '/unity/joy_input', self.joy_callback, 10)

        # Timers
        self.teleop_timer = self.create_timer(0.05, self.teleop_loop)

        # Teleoperation state
        self.linear_speed = 0.0
        self.angular_speed = 0.0
        self.last_joy_input = None
        self.teleop_active = False
        self.control_mode = "velocity"  # velocity, position, or trajectory

        # Velocity limits
        self.max_linear = 0.5
        self.max_angular = 1.0

        self.get_logger().info("Unity Teleoperation Interface initialized")

    def joy_callback(self, msg):
        """Process joystick input from Unity"""
        self.last_joy_input = msg
        self.teleop_active = True

        # Map joystick axes to robot commands
        # Axis 1: Left stick vertical (forward/backward)
        # Axis 0: Left stick horizontal (turn left/right)
        linear_input = msg.axes[1] if len(msg.axes) > 1 else 0.0
        angular_input = msg.axes[0] if len(msg.axes) > 0 else 0.0

        # Apply deadzone
        if abs(linear_input) < 0.2:
            linear_input = 0.0
        if abs(angular_input) < 0.2:
            angular_input = 0.0

        # Scale to max velocities
        self.linear_speed = linear_input * self.max_linear
        self.angular_speed = angular_input * self.max_angular

    def teleop_loop(self):
        """Main teleoperation loop"""
        cmd = Twist()

        if self.teleop_active and self.last_joy_input:
            # Apply current speeds
            cmd.linear.x = self.linear_speed
            cmd.angular.z = self.angular_speed

            # Check for special buttons
            if len(self.last_joy_input.buttons) > 0 and self.last_joy_input.buttons[0] == 1:  # A button
                # Emergency stop
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.get_logger().info("Emergency stop activated")

            # Check for mode change (e.g., right bumper button)
            if len(self.last_joy_input.buttons) > 5 and self.last_joy_input.buttons[5] == 1:  # Right bumper
                self.control_mode = "position" if self.control_mode == "velocity" else "velocity"
                self.get_logger().info(f"Control mode changed to: {self.control_mode}")

        # Publish command
        self.cmd_vel_pub.publish(cmd)

        # Publish status
        status_msg = String()
        status_msg.data = f"Mode: {self.control_mode}, Lin: {self.linear_speed:.2f}, Ang: {self.angular_speed:.2f}, " \
                         f"Active: {self.teleop_active}"
        self.status_pub.publish(status_msg)

        # Reset activity if no input for a while
        if self.last_joy_input is None:
            self.teleop_active = False
        else:
            # Update last input time
            pass

def main(args=None):
    rclpy.init(args=args)
    teleop = UnityTeleop()

    try:
        rclpy.spin(teleop)
    except KeyboardInterrupt:
        teleop.get_logger().info("Unity Teleoperation stopped by user")
    finally:
        # Send stop command on exit
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        teleop.cmd_vel_pub.publish(cmd)
        teleop.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Small Simulation

Let's create a Unity-ROS integration test that demonstrates advanced visualization capabilities:

```python
# unity_integration_test.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point, Vector3
from sensor_msgs.msg import LaserScan, Image
from std_msgs.msg import ColorRGBA, String, Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration
import math
import numpy as np
from datetime import datetime

class UnityIntegrationTest(Node):
    """
    Comprehensive test of Unity-ROS integration capabilities
    """
    def __init__(self):
        super().__init__('unity_integration_test')

        # Publishers for Unity visualization
        self.marker_pub = self.create_publisher(MarkerArray, '/unity_integration/markers', 10)
        self.path_pub = self.create_publisher(Marker, '/unity_integration/path', 10)
        self.heatmap_pub = self.create_publisher(Float32MultiArray, '/unity_integration/heatmap', 10)
        self.status_pub = self.create_publisher(String, '/unity_integration/status', 10)

        # Timer for dynamic visualization
        self.test_timer = self.create_timer(0.1, self.run_integration_test)

        # Test parameters
        self.test_phase = 0
        self.test_start_time = self.get_clock().now()
        self.robot_path = []
        self.heatmap_data = np.zeros((50, 50))  # 50x50 grid for heatmap
        self.animation_time = 0.0

        self.get_logger().info("Unity Integration Test initialized")

    def create_robot_marker(self, x, y, z, angle=0.0):
        """Create a marker for the robot"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "integration_test"
        marker.id = 0
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.pose.orientation.z = math.sin(angle / 2.0)
        marker.pose.orientation.w = math.cos(angle / 2.0)
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.color.r = 0.2
        marker.color.g = 0.6
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.mesh_resource = "package://unity_integration_test/meshes/robot.dae"
        return marker

    def create_path_marker(self, path_points):
        """Create a marker for the robot's path"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "integration_test"
        marker.id = 1
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05  # Line width
        marker.color.r = 1.0
        marker.color.a = 0.8

        for point in path_points:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = 0.1  # Slightly above ground
            marker.points.append(p)

            # Color gradient based on position in path
            color = ColorRGBA()
            progress = len(marker.points) / len(path_points) if path_points else 0
            color.r = progress
            color.g = 1.0 - progress
            color.b = 0.5
            color.a = 0.8
            marker.colors.append(color)

        return marker

    def create_heatmap_marker(self):
        """Create heatmap data for Unity"""
        heatmap_msg = Float32MultiArray()

        # Generate dynamic heatmap data
        current_time = (self.get_clock().now() - self.test_start_time).nanoseconds / 1e9
        for i in range(50):
            for j in range(50):
                # Create wave pattern
                x_norm = (i - 25) / 25.0
                y_norm = (j - 25) / 25.0
                distance = math.sqrt(x_norm*x_norm + y_norm*y_norm)

                # Add time-varying wave
                wave = math.sin(distance * 5 - current_time * 2)
                self.heatmap_data[i, j] = (wave + 1) / 2  # Normalize to 0-1

        # Flatten the 2D array for transmission
        heatmap_msg.data = self.heatmap_data.flatten().tolist()
        return heatmap_msg

    def create_sensor_fusion_marker(self):
        """Create visualization for sensor fusion results"""
        marker_array = MarkerArray()

        # Create multiple sensor data representations
        current_time = (self.get_clock().now() - self.test_start_time).nanoseconds / 1e9

        # Simulated LIDAR points (spiral pattern)
        lidar_marker = Marker()
        lidar_marker.header.frame_id = "map"
        lidar_marker.header.stamp = self.get_clock().now().to_msg()
        lidar_marker.ns = "sensor_fusion"
        lidar_marker.id = 10
        lidar_marker.type = Marker.POINTS
        lidar_marker.action = Marker.ADD
        lidar_marker.scale.x = 0.03
        lidar_marker.scale.y = 0.03
        lidar_marker.color.r = 1.0
        lidar_marker.color.a = 0.7

        for i in range(100):
            angle = i * 0.1 + current_time
            distance = 2.0 + math.sin(i * 0.05) * 0.5
            x = distance * math.cos(angle)
            y = distance * math.sin(angle)

            point = Point()
            point.x = x
            point.y = y
            point.z = 0.1
            lidar_marker.points.append(point)

            # Color based on distance
            color = ColorRGBA()
            color.r = min(1.0, distance / 3.0)
            color.g = 1.0 - min(1.0, distance / 3.0)
            color.b = 0.5
            color.a = 0.7
            lidar_marker.colors.append(color)

        marker_array.markers.append(lidar_marker)

        # Simulated camera FOV
        fov_marker = Marker()
        fov_marker.header.frame_id = "map"
        fov_marker.header.stamp = self.get_clock().now().to_msg()
        fov_marker.ns = "sensor_fusion"
        fov_marker.id = 11
        fov_marker.type = Marker.TRIANGLE_LIST
        fov_marker.action = Marker.ADD
        fov_marker.scale.x = 1.0
        fov_marker.scale.y = 1.0
        fov_marker.scale.z = 1.0
        fov_marker.color.b = 0.3
        fov_marker.color.a = 0.3

        # Create a simple camera FOV triangle
        robot_pos = Point()
        robot_pos.x = 0.0
        robot_pos.y = 0.0
        robot_pos.z = 0.1

        # FOV vertices
        left_corner = Point()
        left_corner.x = 2.0 * math.cos(-0.5)  # 60 degree FOV
        left_corner.y = 2.0 * math.sin(-0.5)
        left_corner.z = 0.1

        right_corner = Point()
        right_corner.x = 2.0 * math.cos(0.5)
        right_corner.y = 2.0 * math.sin(0.5)
        right_corner.z = 0.1

        # Add triangle (robot position to left corner to right corner)
        fov_marker.points.extend([robot_pos, left_corner, right_corner])

        marker_array.markers.append(fov_marker)

        return marker_array

    def run_integration_test(self):
        """Run comprehensive integration test"""
        marker_array = MarkerArray()
        current_time = (self.get_clock().now() - self.test_start_time).nanoseconds / 1e9

        # Phase 0: Basic robot visualization
        if self.test_phase == 0:
            # Moving robot in a circle
            radius = 2.0
            angle = current_time * 0.5  # Rotate at 0.5 rad/s
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            z = 0.1

            robot_marker = self.create_robot_marker(x, y, z, angle)
            marker_array.markers.append(robot_marker)

            # Add to path
            self.robot_path.append((x, y))
            if len(self.robot_path) > 100:  # Limit path length
                self.robot_path.pop(0)

            path_marker = self.create_path_marker(self.robot_path)
            marker_array.markers.append(path_marker)

            if current_time > 10.0:  # Move to next phase after 10 seconds
                self.test_phase = 1
                self.test_start_time = self.get_clock().now()

        # Phase 1: Sensor fusion visualization
        elif self.test_phase == 1:
            sensor_markers = self.create_sensor_fusion_marker()
            marker_array.markers.extend(sensor_markers.markers)

            if current_time > 10.0:  # Move to next phase
                self.test_phase = 2
                self.test_start_time = self.get_clock().now()

        # Phase 2: Heatmap visualization
        elif self.test_phase == 2:
            heatmap_msg = self.create_heatmap_marker()
            self.heatmap_pub.publish(heatmap_msg)

            # Also publish a simple status marker
            status_marker = Marker()
            status_marker.header.frame_id = "map"
            status_marker.header.stamp = self.get_clock().now().to_msg()
            status_marker.ns = "integration_test"
            status_marker.id = 20
            status_marker.type = Marker.TEXT_VIEW_FACING
            status_marker.action = Marker.ADD
            status_marker.pose.position.x = 0.0
            status_marker.pose.position.y = 0.0
            status_marker.pose.position.z = 1.0
            status_marker.scale.z = 0.3
            status_marker.color.r = 1.0
            status_marker.color.g = 1.0
            status_marker.color.b = 1.0
            status_marker.color.a = 1.0
            status_marker.text = f"Integration Test Phase 2\nTime: {current_time:.1f}s"
            marker_array.markers.append(status_marker)

            if current_time > 10.0:  # Cycle back to phase 0
                self.test_phase = 0
                self.test_start_time = self.get_clock().now()

        # Publish all markers
        self.marker_pub.publish(marker_array)

        # Publish path separately as well
        if len(self.robot_path) > 1 and self.test_phase == 0:
            path_marker = self.create_path_marker(self.robot_path)
            self.path_pub.publish(path_marker)

        # Publish status
        status_msg = String()
        status_msg.data = f"Phase: {self.test_phase}, Time: {current_time:.1f}s, Markers: {len(marker_array.markers)}"
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    test_node = UnityIntegrationTest()

    try:
        rclpy.spin(test_node)
    except KeyboardInterrupt:
        test_node.get_logger().info("Unity Integration Test stopped by user")
    finally:
        test_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Quick Recap

In this lesson, we've covered:

- **Unity-ROS Bridge**: Setting up communication between Unity and ROS 2 for visualization
- **Robot Visualization**: Creating 3D models and animations for robot states in Unity
- **Sensor Data Visualization**: Displaying LIDAR, camera, and other sensor data in Unity
- **Teleoperation Interface**: Building Unity-based interfaces for robot control
- **Advanced Visualization**: Creating heatmaps, path visualizations, and sensor fusion displays

Unity provides a powerful platform for creating high-quality visualizations that complement Gazebo's physics simulation. The combination allows for both accurate physics simulation and photorealistic rendering, which is invaluable for robot development, debugging, and human-robot interaction design.

In the next lesson, we'll explore sensor simulation including LIDAR, depth cameras, and IMUs, and how to integrate these with Unity for comprehensive robot perception simulation.