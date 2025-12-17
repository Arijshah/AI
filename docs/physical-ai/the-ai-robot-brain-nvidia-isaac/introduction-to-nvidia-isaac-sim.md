---
sidebar_label: 'Introduction to NVIDIA Isaac Sim'
title: 'Introduction to NVIDIA Isaac Sim'
---

# Introduction to NVIDIA Isaac Sim

## Overview

NVIDIA Isaac Sim is a powerful robotics simulation platform built on NVIDIA Omniverse, designed specifically for developing, testing, and training AI-powered robots. It combines high-fidelity physics simulation with advanced graphics rendering, synthetic data generation, and AI training capabilities. Isaac Sim provides a comprehensive environment for creating intelligent robots that can perceive, navigate, and interact with complex real-world scenarios.

Isaac Sim is particularly valuable for Physical AI development as it enables the creation of photorealistic simulation environments where AI algorithms can be trained on synthetic data that closely matches real-world conditions, significantly reducing the need for physical prototyping and real-world training data collection.

## Learning Outcomes

By the end of this lesson, you will be able to:
- Understand the architecture and capabilities of NVIDIA Isaac Sim
- Set up Isaac Sim for robotics development
- Create basic simulation environments with realistic lighting and physics
- Integrate Isaac Sim with ROS 2 for robot control
- Generate synthetic sensor data for AI training

## Hands-on Steps

1. **Isaac Sim Installation**: Set up Isaac Sim with Omniverse
2. **Basic Scene Creation**: Create a simple simulation environment
3. **Robot Integration**: Add a robot model to the simulation
4. **Sensor Configuration**: Set up synthetic sensors
5. **ROS 2 Bridge**: Connect to ROS 2 for control and data exchange

### Prerequisites

- Understanding of ROS 2 concepts and basic node development
- Knowledge of 3D modeling concepts
- Basic understanding of AI and machine learning principles

## Code Examples

Let's start by exploring how to create a basic Isaac Sim environment using Python and Omniverse Kit extensions:

```python
# isaac_sim_robot_control.py
import carb
import omni
import omni.ext
import omni.kit.ui
import omni.usd
from pxr import Usd, UsdGeom, Gf, Sdf
import numpy as np
import math

# This would be part of an actual Isaac Sim extension
# For demonstration purposes, we'll show the conceptual structure
class IsaacSimController:
    """
    Controller class for Isaac Sim operations
    This is a conceptual representation of how Isaac Sim extensions work
    """
    def __init__(self):
        self.stage = None
        self.robot_prim = None
        self.physics_world = None
        self.sensor_configs = {}

    def create_basic_scene(self):
        """
        Create a basic Isaac Sim scene with ground plane and simple objects
        """
        # Get the current stage
        self.stage = omni.usd.get_context().get_stage()

        # Create a ground plane
        ground_path = Sdf.Path("/World/groundPlane")
        ground_prim = self.stage.DefinePrim(ground_path, "Xform")
        plane_mesh = UsdGeom.Mesh.Define(self.stage, "/World/groundPlane/plane")

        # Configure the ground plane
        plane_mesh.CreatePointsAttr().Set([
            Gf.Vec3f(-10, -10, 0), Gf.Vec3f(10, -10, 0),
            Gf.Vec3f(10, 10, 0), Gf.Vec3f(-10, 10, 0)
        ])
        plane_mesh.CreateFaceVertexIndicesAttr().Set([0, 1, 2, 0, 2, 3])
        plane_mesh.CreateFaceVertexCountsAttr().Set([3, 3])

        # Add physics properties
        from omni.physx.scripts import particle_sample
        particle_sample.add_rigidbody(ground_prim, self.stage)

        carb.log_info("Basic scene created with ground plane")

    def add_robot_to_scene(self, robot_usd_path="/Isaac/Robots/Carter/carter_v1.usd"):
        """
        Add a robot to the scene (using Carter robot as example)
        """
        if self.stage:
            # Define robot prim
            robot_path = Sdf.Path("/World/Robot")
            self.robot_prim = self.stage.DefinePrim(robot_path, "Xform")

            # Add robot reference
            self.robot_prim.GetReferences().AddReference(robot_usd_path)

            # Set initial position
            xform_api = UsdGeom.XformCommonAPI(self.robot_prim)
            xform_api.SetTranslate((0.0, 0.0, 0.5))

            carb.log_info(f"Robot added to scene from {robot_usd_path}")

    def configure_sensors(self):
        """
        Configure synthetic sensors for the robot
        """
        # Configure RGB camera
        camera_config = {
            'type': 'rgb',
            'position': [0.2, 0.0, 0.8],  # x, y, z offset from robot base
            'rotation': [0.0, 0.0, 0.0],  # pitch, yaw, roll
            'resolution': [640, 480],
            'fov': 60.0,
            'sensor_tick': 0.033  # 30 FPS
        }

        # Configure depth camera
        depth_config = {
            'type': 'depth',
            'position': [0.2, 0.0, 0.8],
            'resolution': [640, 480],
            'min_depth': 0.1,
            'max_depth': 10.0
        }

        # Configure LIDAR
        lidar_config = {
            'type': 'lidar',
            'position': [0.25, 0.0, 0.9],
            'rotation': [0.0, 0.0, 0.0],
            'yaw_samples': 1080,  # Horizontal resolution
            'yaw_lower_bond': -2.35619,  # -135 degrees
            'yaw_upper_bond': 2.35619,   # 135 degrees
            'pitch_samples': 1,  # Vertical resolution (1 for 2D lidar)
            'range': 25.0,
            'rotation_frequency': 10  # Hz
        }

        self.sensor_configs = {
            'rgb_camera': camera_config,
            'depth_camera': depth_config,
            'lidar': lidar_config
        }

        carb.log_info("Sensors configured for robot")

    def setup_ros_bridge(self):
        """
        Setup ROS 2 bridge for communication
        """
        # This would configure the ROS bridge extension
        # In practice, this involves setting up ROS publishers/subscribers
        ros_config = {
            'enabled': True,
            'namespace': '/isaac_robot',
            'topics': {
                'cmd_vel': '/cmd_vel',
                'odom': '/odom',
                'scan': '/scan',
                'rgb_image': '/camera/rgb/image_raw',
                'depth_image': '/camera/depth/image_raw',
                'imu': '/imu'
            }
        }

        carb.log_info(f"ROS bridge configured with namespace: {ros_config['namespace']}")

        return ros_config

def main():
    """
    Main function demonstrating Isaac Sim setup
    Note: This is a conceptual example - actual Isaac Sim extensions
    would be integrated into the Omniverse extension system
    """
    controller = IsaacSimController()

    # Create basic scene
    controller.create_basic_scene()

    # Add robot
    controller.add_robot_to_scene()

    # Configure sensors
    controller.configure_sensors()

    # Setup ROS bridge
    ros_config = controller.setup_ros_bridge()

    print("Isaac Sim environment setup complete!")
    print(f"Configured sensors: {list(controller.sensor_configs.keys())}")
    print(f"ROS namespace: {ros_config['namespace']}")

if __name__ == "__main__":
    main()
```

Now let's create a more practical ROS 2 node that would interface with Isaac Sim:

```python
# isaac_sim_ros_bridge.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose, Point
from sensor_msgs.msg import LaserScan, Image, Imu, PointCloud2
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Header
from cv_bridge import CvBridge
import numpy as np
import math
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

class IsaacSimROSBridge(Node):
    """
    ROS 2 bridge node for interfacing with Isaac Sim
    """
    def __init__(self):
        super().__init__('isaac_sim_ros_bridge')

        # Publishers - These would publish to Isaac Sim
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/isaac_robot/cmd_vel', self.cmd_vel_callback, 10)

        # Publishers - These would publish sensor data from Isaac Sim
        self.odom_pub = self.create_publisher(Odometry, '/isaac_robot/odom', 10)
        self.scan_pub = self.create_publisher(LaserScan, '/isaac_robot/scan', 10)
        self.rgb_pub = self.create_publisher(Image, '/isaac_robot/camera/rgb/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, '/isaac_robot/camera/depth/image_raw', 10)
        self.imu_pub = self.create_publisher(Imu, '/isaac_robot/imu', 10)
        self.status_pub = self.create_publisher(String, '/isaac_robot/status', 10)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Timers
        self.sim_timer = self.create_timer(0.033, self.simulation_step)  # ~30 FPS
        self.sensor_timer = self.create_timer(0.05, self.publish_sensor_data)  # 20 Hz

        # Robot state
        self.robot_pose = Pose()
        self.robot_twist = Twist()
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.last_update_time = self.get_clock().now()

        # Sensor simulation parameters
        self.lidar_ranges = [float('inf')] * 1080  # 1080 samples
        self.image_width = 640
        self.image_height = 480

        # CV bridge for image conversion
        self.cv_bridge = CvBridge()

        # Simulated environment features
        self.simulated_objects = [
            {'type': 'box', 'position': (2.0, 1.0, 0.5), 'size': (1.0, 1.0, 1.0)},
            {'type': 'cylinder', 'position': (-1.5, -2.0, 0.5), 'radius': 0.5, 'height': 1.0},
            {'type': 'sphere', 'position': (0.0, 3.0, 0.5), 'radius': 0.3}
        ]

        self.get_logger().info("Isaac Sim ROS Bridge initialized")

    def cmd_vel_callback(self, msg):
        """Process velocity commands from ROS"""
        self.linear_velocity = msg.linear.x
        self.angular_velocity = msg.angular.z

    def simulation_step(self):
        """Main simulation step - update robot state based on commands"""
        current_time = self.get_clock().now()
        dt = (current_time - self.last_update_time).nanoseconds / 1e9
        self.last_update_time = current_time

        if dt > 0:
            # Update robot position based on velocities
            # Simple differential drive kinematics
            if abs(self.angular_velocity) > 0.001:  # Turning
                # Calculate turn radius and new position
                turn_radius = self.linear_velocity / self.angular_velocity if abs(self.angular_velocity) > 0.001 else 0
                angular_displacement = self.angular_velocity * dt
                linear_displacement = self.linear_velocity * dt

                # Update orientation
                current_yaw = math.atan2(
                    2 * (self.robot_pose.orientation.w * self.robot_pose.orientation.z +
                         self.robot_pose.orientation.x * self.robot_pose.orientation.y),
                    1 - 2 * (self.robot_pose.orientation.y**2 + self.robot_pose.orientation.z**2)
                )
                new_yaw = current_yaw + angular_displacement

                # Update position
                dx = linear_displacement * math.cos(new_yaw)
                dy = linear_displacement * math.sin(new_yaw)

                self.robot_pose.position.x += dx
                self.robot_pose.position.y += dy
                self.robot_pose.orientation.z = math.sin(new_yaw / 2)
                self.robot_pose.orientation.w = math.cos(new_yaw / 2)
            else:  # Moving straight
                # Calculate displacement
                dx = self.linear_velocity * dt * math.cos(
                    math.atan2(2 * self.robot_pose.orientation.z * self.robot_pose.orientation.w,
                              1 - 2 * self.robot_pose.orientation.z**2)
                )
                dy = self.linear_velocity * dt * math.sin(
                    math.atan2(2 * self.robot_pose.orientation.z * self.robot_pose.orientation.w,
                              1 - 2 * self.robot_pose.orientation.z**2)
                )

                self.robot_pose.position.x += dx
                self.robot_pose.position.y += dy

            # Update twist for odometry
            self.robot_twist.linear.x = self.linear_velocity
            self.robot_twist.angular.z = self.angular_velocity

    def generate_lidar_scan(self):
        """Generate simulated LIDAR scan based on environment"""
        # Create a simulated environment with objects
        ranges = []
        angle_min = -math.pi * 0.75  # -135 degrees
        angle_max = math.pi * 0.75   # 135 degrees
        angle_increment = (angle_max - angle_min) / len(self.lidar_ranges)

        for i in range(len(self.lidar_ranges)):
            angle = angle_min + i * angle_increment

            # Transform to world coordinates
            ray_direction = (
                math.cos(angle),
                math.sin(angle)
            )

            # Check for intersections with simulated objects
            min_distance = float('inf')

            for obj in self.simulated_objects:
                if obj['type'] == 'box':
                    # Simple box intersection (axis-aligned)
                    pos = obj['position']
                    size = obj['size']
                    distance = self.ray_box_intersection(
                        (self.robot_pose.position.x, self.robot_pose.position.y),
                        ray_direction,
                        pos[0], pos[1], size[0]/2, size[1]/2
                    )
                    if distance and distance < min_distance:
                        min_distance = distance
                elif obj['type'] == 'cylinder':
                    # Cylinder intersection
                    pos = obj['position']
                    distance = self.ray_cylinder_intersection(
                        (self.robot_pose.position.x, self.robot_pose.position.y),
                        ray_direction,
                        pos[0], pos[1], obj['radius']
                    )
                    if distance and distance < min_distance:
                        min_distance = distance
                elif obj['type'] == 'sphere':
                    # Sphere intersection
                    pos = obj['position']
                    distance = self.ray_sphere_intersection(
                        (self.robot_pose.position.x, self.robot_pose.position.y),
                        ray_direction,
                        pos[0], pos[1], obj['radius']
                    )
                    if distance and distance < min_distance:
                        min_distance = distance

            # Add some noise to make it more realistic
            noise = np.random.normal(0, 0.01)  # 1cm standard deviation
            final_range = min(min_distance, 25.0)  # Cap at max range
            ranges.append(max(0.1, final_range + noise))  # Min range 0.1m

        return ranges, angle_min, angle_increment

    def ray_box_intersection(self, ray_origin, ray_dir, box_x, box_y, half_width, half_height):
        """Calculate intersection of ray with axis-aligned box"""
        # Calculate intersection with box boundaries
        t1 = (box_x - half_width - ray_origin[0]) / ray_dir[0] if ray_dir[0] != 0 else float('inf')
        t2 = (box_x + half_width - ray_origin[0]) / ray_dir[0] if ray_dir[0] != 0 else float('inf')
        t3 = (box_y - half_height - ray_origin[1]) / ray_dir[1] if ray_dir[1] != 0 else float('inf')
        t4 = (box_y + half_height - ray_origin[1]) / ray_dir[1] if ray_dir[1] != 0 else float('inf')

        t_min = max(min(t1, t2), min(t3, t4))
        t_max = min(max(t1, t2), max(t3, t4))

        if t_max >= 0 and t_min <= t_max:
            return t_min if t_min >= 0 else t_max

        return None

    def ray_cylinder_intersection(self, ray_origin, ray_dir, cyl_x, cyl_y, radius):
        """Calculate intersection of ray with cylinder"""
        # Convert to cylinder-relative coordinates
        rel_x = ray_origin[0] - cyl_x
        rel_y = ray_origin[1] - cyl_y

        # Quadratic equation coefficients
        a = ray_dir[0]**2 + ray_dir[1]**2
        b = 2 * (rel_x * ray_dir[0] + rel_y * ray_dir[1])
        c = rel_x**2 + rel_y**2 - radius**2

        discriminant = b**2 - 4*a*c

        if discriminant < 0:
            return None

        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2*a)
        t2 = (-b + sqrt_disc) / (2*a)

        # Return the smallest positive intersection
        if t1 > 0:
            return t1
        elif t2 > 0:
            return t2
        else:
            return None

    def ray_sphere_intersection(self, ray_origin, ray_dir, sph_x, sph_y, radius):
        """Calculate intersection of ray with sphere"""
        # Convert to sphere-relative coordinates
        rel_x = ray_origin[0] - sph_x
        rel_y = ray_origin[1] - sph_y

        # Quadratic equation coefficients
        a = ray_dir[0]**2 + ray_dir[1]**2
        b = 2 * (rel_x * ray_dir[0] + rel_y * ray_dir[1])
        c = rel_x**2 + rel_y**2 - radius**2

        discriminant = b**2 - 4*a*c

        if discriminant < 0:
            return None

        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2*a)
        t2 = (-b + sqrt_disc) / (2*a)

        # Return the smallest positive intersection
        if t1 > 0:
            return t1
        elif t2 > 0:
            return t2
        else:
            return None

    def generate_camera_image(self):
        """Generate a simulated RGB image based on robot position"""
        # Create a simple synthetic image based on robot position and environment
        image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)

        # Create a simple gradient background
        for y in range(self.image_height):
            for x in range(self.image_width):
                # Gradient from blue (top) to green (bottom)
                image[y, x, 0] = int(255 * y / self.image_height)  # Blue
                image[y, x, 1] = int(255 * (1 - y / self.image_height))  # Green
                image[y, x, 2] = 50  # Red (constant)

        # Add simulated objects based on robot's view
        robot_x, robot_y = self.robot_pose.position.x, self.robot_pose.position.y

        for obj in self.simulated_objects:
            # Calculate object position relative to robot
            obj_x, obj_y = obj['position'][0], obj['position'][1]
            rel_x = obj_x - robot_x
            rel_y = obj_y - robot_y

            # Calculate apparent size based on distance
            distance = math.sqrt(rel_x**2 + rel_y**2)
            if distance < 10:  # Only draw if close enough
                # Project to image coordinates (simple perspective)
                # This is a simplified projection for demonstration
                img_x = int(self.image_width / 2 + rel_x * 50)  # Scale factor for visibility
                img_y = int(self.image_height / 2 - rel_y * 50)  # Invert Y

                if 0 <= img_x < self.image_width and 0 <= img_y < self.image_height:
                    if obj['type'] == 'box':
                        # Draw a rectangle
                        size = int(obj['size'][0] * 30 / max(distance, 1))  # Scale with distance
                        image[max(0, img_y-size):min(self.image_height, img_y+size),
                              max(0, img_x-size):min(self.image_width, img_x+size)] = [255, 0, 0]  # Red
                    elif obj['type'] == 'cylinder':
                        # Draw a circle
                        radius = int(obj['radius'] * 30 / max(distance, 1))
                        cv2.circle(image, (img_x, img_y), radius, (0, 255, 0), -1)  # Green
                    elif obj['type'] == 'sphere':
                        # Draw a circle
                        radius = int(obj['radius'] * 30 / max(distance, 1))
                        cv2.circle(image, (img_x, img_y), radius, (0, 0, 255), -1)  # Blue

        return image

    def publish_sensor_data(self):
        """Publish simulated sensor data"""
        current_time = self.get_clock().now()

        # Publish odometry
        odom_msg = Odometry()
        odom_msg.header.stamp = current_time.to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'
        odom_msg.pose.pose = self.robot_pose
        odom_msg.twist.twist = self.robot_twist

        # Add some covariance to make it realistic
        odom_msg.pose.covariance = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.1, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 1000000.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 1000000.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0, 1000000.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0, 0.0, 0.05]
        odom_msg.twist.covariance = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0,
                                    0.0, 0.1, 0.0, 0.0, 0.0, 0.0,
                                    0.0, 0.0, 1000000.0, 0.0, 0.0, 0.0,
                                    0.0, 0.0, 0.0, 1000000.0, 0.0, 0.0,
                                    0.0, 0.0, 0.0, 0.0, 1000000.0, 0.0,
                                    0.0, 0.0, 0.0, 0.0, 0.0, 0.05]

        self.odom_pub.publish(odom_msg)

        # Publish LIDAR scan
        ranges, angle_min, angle_increment = self.generate_lidar_scan()
        scan_msg = LaserScan()
        scan_msg.header.stamp = current_time.to_msg()
        scan_msg.header.frame_id = 'laser_frame'
        scan_msg.angle_min = angle_min
        scan_msg.angle_max = -angle_min  # 135 degrees
        scan_msg.angle_increment = angle_increment
        scan_msg.time_increment = 0.0
        scan_msg.scan_time = 0.1  # 10Hz
        scan_msg.range_min = 0.1
        scan_msg.range_max = 25.0
        scan_msg.ranges = ranges
        # Add intensities (simulated)
        scan_msg.intensities = [100.0] * len(ranges)

        self.scan_pub.publish(scan_msg)

        # Publish camera image
        image = self.generate_camera_image()
        image_msg = self.cv_bridge.cv2_to_imgmsg(image, encoding="bgr8")
        image_msg.header.stamp = current_time.to_msg()
        image_msg.header.frame_id = 'camera_rgb_frame'

        self.rgb_pub.publish(image_msg)

        # Publish depth image (simplified)
        depth_image = np.ones((self.image_height, self.image_width), dtype=np.float32) * 5.0  # Default depth
        # Add some variation based on the same objects
        for obj in self.simulated_objects:
            obj_x, obj_y = obj['position'][0], obj['position'][1]
            rel_x = obj_x - self.robot_pose.position.x
            rel_y = obj_y - self.robot_pose.position.y
            distance = math.sqrt(rel_x**2 + rel_y**2)

            # Project to image coordinates
            img_x = int(self.image_width / 2 + rel_x * 50)
            img_y = int(self.image_height / 2 - rel_y * 50)

            if 0 <= img_x < self.image_width and 0 <= img_y < self.image_height:
                size = max(1, int(30 / max(distance, 1)))
                depth_image[max(0, img_y-size):min(self.image_height, img_y+size),
                           max(0, img_x-size):min(self.image_width, img_x+size)] = distance

        depth_msg = self.cv_bridge.cv2_to_imgmsg(depth_image, encoding="32FC1")
        depth_msg.header.stamp = current_time.to_msg()
        depth_msg.header.frame_id = 'camera_depth_frame'

        self.depth_pub.publish(depth_msg)

        # Publish IMU data (simulated)
        imu_msg = Imu()
        imu_msg.header.stamp = current_time.to_msg()
        imu_msg.header.frame_id = 'imu_frame'

        # Set orientation (from robot pose)
        imu_msg.orientation = self.robot_pose.orientation

        # Add some noise to make realistic
        imu_msg.orientation_covariance = [0.01, 0.0, 0.0,
                                         0.0, 0.01, 0.0,
                                         0.0, 0.0, 0.01]

        # Angular velocity (from robot twist)
        imu_msg.angular_velocity.z = self.robot_twist.angular.z
        imu_msg.angular_velocity_covariance = [0.01, 0.0, 0.0,
                                              0.0, 0.01, 0.0,
                                              0.0, 0.0, 0.01]

        # Linear acceleration (simulate gravity and movement)
        imu_msg.linear_acceleration.x = self.robot_twist.linear.x * 2.0  # Simulate acceleration
        imu_msg.linear_acceleration.y = 0.0
        imu_msg.linear_acceleration.z = 9.81  # Gravity
        imu_msg.linear_acceleration_covariance = [0.01, 0.0, 0.0,
                                                 0.0, 0.01, 0.0,
                                                 0.0, 0.0, 0.01]

        self.imu_pub.publish(imu_msg)

        # Publish status
        status_msg = String()
        status_msg.data = f"Pos: ({self.robot_pose.position.x:.2f}, {self.robot_pose.position.y:.2f}), " \
                         f"Vel: ({self.linear_velocity:.2f}, {self.angular_velocity:.2f}), " \
                         f"Objects: {len(self.simulated_objects)}"
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    bridge = IsaacSimROSBridge()

    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        bridge.get_logger().info("Isaac Sim ROS Bridge stopped by user")
    finally:
        bridge.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    import cv2  # Import here to avoid issues if not available
    main()
```

## Small Simulation

Let's create a synthetic data generation example that demonstrates Isaac Sim's capability for AI training:

```python
# synthetic_data_generator.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, PointCloud2
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Header
from cv_bridge import CvBridge
import numpy as np
import math
import json
import os
from datetime import datetime

class SyntheticDataGenerator(Node):
    """
    Generate synthetic training data for AI models using Isaac Sim principles
    """
    def __init__(self):
        super().__init__('synthetic_data_generator')

        # Publishers
        self.rgb_pub = self.create_publisher(Image, '/synthetic_data/rgb', 10)
        self.depth_pub = self.create_publisher(Image, '/synthetic_data/depth', 10)
        self.seg_pub = self.create_publisher(Image, '/synthetic_data/segmentation', 10)
        self.scan_pub = self.create_publisher(LaserScan, '/synthetic_data/lidar', 10)
        self.status_pub = self.create_publisher(String, '/synthetic_data/status', 10)

        # Timers
        self.data_gen_timer = self.create_timer(0.5, self.generate_synthetic_data)

        # Data collection parameters
        self.cv_bridge = CvBridge()
        self.data_counter = 0
        self.collection_enabled = True
        self.collection_dir = "/tmp/isaac_synthetic_data"

        # Create collection directory
        os.makedirs(self.collection_dir, exist_ok=True)

        # Scene parameters for variation
        self.scene_configurations = [
            {'lighting': 'bright', 'weather': 'clear', 'objects': 5},
            {'lighting': 'dim', 'weather': 'overcast', 'objects': 8},
            {'lighting': 'bright', 'weather': 'sunny', 'objects': 3},
            {'lighting': 'variable', 'weather': 'partly_cloudy', 'objects': 6}
        ]
        self.current_scene_idx = 0

        # Object classes for segmentation
        self.object_classes = {
            1: 'robot',
            2: 'obstacle',
            3: 'furniture',
            4: 'wall',
            5: 'floor'
        }

        self.get_logger().info("Synthetic Data Generator initialized")

    def generate_random_scene(self):
        """Generate a random scene configuration"""
        scene = self.scene_configurations[self.current_scene_idx]
        self.current_scene_idx = (self.current_scene_idx + 1) % len(self.scene_configurations)

        # Generate random objects in the scene
        objects = []
        for i in range(scene['objects']):
            obj_type = np.random.choice(['box', 'cylinder', 'sphere'])
            obj_class = np.random.choice(list(self.object_classes.keys())[1:])  # Exclude robot (class 1)

            obj = {
                'type': obj_type,
                'class': obj_class,
                'position': (
                    np.random.uniform(-5, 5),
                    np.random.uniform(-5, 5),
                    np.random.uniform(0.1, 2.0)
                ),
                'size': (
                    np.random.uniform(0.2, 1.5),
                    np.random.uniform(0.2, 1.5),
                    np.random.uniform(0.2, 2.0)
                ) if obj_type == 'box' else (
                    np.random.uniform(0.2, 1.0),  # radius for cylinder/sphere
                    np.random.uniform(0.5, 2.0)   # height for cylinder
                ),
                'color': np.random.uniform(0.2, 1.0, 3).tolist()
            }
            objects.append(obj)

        return scene, objects

    def generate_synthetic_rgb(self, objects, lighting_condition):
        """Generate synthetic RGB image"""
        height, width = 480, 640
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Apply lighting condition
        if lighting_condition == 'bright':
            base_brightness = 0.8
        elif lighting_condition == 'dim':
            base_brightness = 0.3
        else:
            base_brightness = 0.6

        # Create a base environment
        for y in range(height):
            for x in range(width):
                # Create a gradient floor
                floor_intensity = 100 + int(50 * y / height)
                image[y, x] = [floor_intensity, floor_intensity, floor_intensity]

        # Add objects to the image
        for obj in objects:
            # Project 3D object to 2D image
            # Simplified projection for demonstration
            center_x = int(width / 2 + obj['position'][0] * 30)
            center_y = int(height / 2 - obj['position'][1] * 30)

            if 0 <= center_x < width and 0 <= center_y < height:
                # Draw the object based on its type
                if obj['type'] == 'box':
                    size = max(1, int(obj['size'][0] * 20))
                    color = [int(c * 255) for c in obj['color']]
                    image[max(0, center_y-size):min(height, center_y+size),
                          max(0, center_x-size):min(width, center_x+size)] = color
                elif obj['type'] in ['cylinder', 'sphere']:
                    radius = max(1, int(obj['size'][0] * 20))
                    color = [int(c * 255) for c in obj['color']]
                    for dy in range(-radius, radius):
                        for dx in range(-radius, radius):
                            if dx*dx + dy*dy <= radius*radius:
                                py, px = center_y + dy, center_x + dx
                                if 0 <= py < height and 0 <= px < width:
                                    image[py, px] = color

        # Apply lighting effects
        image = (image * base_brightness).astype(np.uint8)

        # Add some noise to make more realistic
        noise = np.random.normal(0, 5, image.shape).astype(np.int16)
        image = np.clip(image + noise, 0, 255).astype(np.uint8)

        return image

    def generate_synthetic_depth(self, objects):
        """Generate synthetic depth image"""
        height, width = 480, 640
        depth = np.ones((height, width), dtype=np.float32) * 10.0  # Default max distance

        # For each object, calculate its depth projection
        for obj in objects:
            center_x = int(width / 2 + obj['position'][0] * 30)
            center_y = int(height / 2 - obj['position'][1] * 30)
            distance = math.sqrt(obj['position'][0]**2 + obj['position'][1]**2 + obj['position'][2]**2)

            if 0 <= center_x < width and 0 <= center_y < height:
                if obj['type'] == 'box':
                    size = max(1, int(obj['size'][0] * 20))
                    depth[max(0, center_y-size):min(height, center_y+size),
                          max(0, center_x-size):min(width, center_x+size)] = distance
                elif obj['type'] in ['cylinder', 'sphere']:
                    radius = max(1, int(obj['size'][0] * 20))
                    for dy in range(-radius, radius):
                        for dx in range(-radius, radius):
                            if dx*dx + dy*dy <= radius*radius:
                                py, px = center_y + dy, center_x + dx
                                if 0 <= py < height and 0 <= px < width:
                                    depth[py, px] = distance

        # Add some noise
        noise = np.random.normal(0, 0.05, depth.shape).astype(np.float32)
        depth = np.maximum(0.1, depth + noise)  # Ensure minimum distance

        return depth

    def generate_synthetic_segmentation(self, objects):
        """Generate synthetic segmentation mask"""
        height, width = 480, 640
        segmentation = np.zeros((height, width), dtype=np.uint8)

        # Floor class
        segmentation[:, :] = 5  # Floor class

        # Add objects with their class IDs
        for obj in objects:
            center_x = int(width / 2 + obj['position'][0] * 30)
            center_y = int(height / 2 - obj['position'][1] * 30)

            if 0 <= center_x < width and 0 <= center_y < height:
                if obj['type'] == 'box':
                    size = max(1, int(obj['size'][0] * 20))
                    segmentation[max(0, center_y-size):min(height, center_y+size),
                                max(0, center_x-size):min(width, center_x+size)] = obj['class']
                elif obj['type'] in ['cylinder', 'sphere']:
                    radius = max(1, int(obj['size'][0] * 20))
                    for dy in range(-radius, radius):
                        for dx in range(-radius, radius):
                            if dx*dx + dy*dy <= radius*radius:
                                py, px = center_y + dy, center_x + dx
                                if 0 <= py < height and 0 <= px < width:
                                    segmentation[py, px] = obj['class']

        # Convert to 3-channel image for ROS compatibility
        seg_image = np.stack([segmentation, segmentation, segmentation], axis=2)
        return seg_image.astype(np.uint8)

    def generate_synthetic_lidar(self, objects):
        """Generate synthetic LIDAR scan"""
        num_points = 1080  # Common for 2D LIDAR
        angle_min = -math.pi * 0.75  # -135 degrees
        angle_max = math.pi * 0.75   # 135 degrees
        angle_increment = (angle_max - angle_min) / num_points

        ranges = []

        for i in range(num_points):
            angle = angle_min + i * angle_increment
            ray_direction = (math.cos(angle), math.sin(angle))

            # Find closest object in this direction
            min_distance = 25.0  # Max range

            for obj in objects:
                if obj['type'] == 'box':
                    distance = self.ray_box_intersection(
                        (0, 0), ray_direction,  # Robot at origin
                        obj['position'][0], obj['position'][1],
                        obj['size'][0]/2, obj['size'][1]/2
                    )
                elif obj['type'] in ['cylinder', 'sphere']:
                    distance = self.ray_cylinder_intersection(
                        (0, 0), ray_direction,
                        obj['position'][0], obj['position'][1],
                        obj['size'][0]  # radius
                    )

                if distance and distance < min_distance:
                    min_distance = distance

            # Add noise and ensure valid range
            noise = np.random.normal(0, 0.01)
            final_range = max(0.1, min(25.0, min_distance + noise))
            ranges.append(final_range)

        return ranges, angle_min, angle_increment

    def ray_box_intersection(self, ray_origin, ray_dir, box_x, box_y, half_width, half_height):
        """Calculate intersection of ray with axis-aligned box"""
        t1 = (box_x - half_width - ray_origin[0]) / ray_dir[0] if ray_dir[0] != 0 else float('inf')
        t2 = (box_x + half_width - ray_origin[0]) / ray_dir[0] if ray_dir[0] != 0 else float('inf')
        t3 = (box_y - half_height - ray_origin[1]) / ray_dir[1] if ray_dir[1] != 0 else float('inf')
        t4 = (box_y + half_height - ray_origin[1]) / ray_dir[1] if ray_dir[1] != 0 else float('inf')

        t_min = max(min(t1, t2), min(t3, t4))
        t_max = min(max(t1, t2), max(t3, t4))

        if t_max >= 0 and t_min <= t_max:
            return t_min if t_min >= 0 else t_max

        return None

    def ray_cylinder_intersection(self, ray_origin, ray_dir, cyl_x, cyl_y, radius):
        """Calculate intersection of ray with cylinder"""
        rel_x = ray_origin[0] - cyl_x
        rel_y = ray_origin[1] - cyl_y

        a = ray_dir[0]**2 + ray_dir[1]**2
        b = 2 * (rel_x * ray_dir[0] + rel_y * ray_dir[1])
        c = rel_x**2 + rel_y**2 - radius**2

        discriminant = b**2 - 4*a*c

        if discriminant < 0:
            return None

        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2*a)
        t2 = (-b + sqrt_disc) / (2*a)

        if t1 > 0:
            return t1
        elif t2 > 0:
            return t2
        else:
            return None

    def save_synthetic_data(self, rgb_image, depth_image, seg_image, lidar_ranges, metadata):
        """Save synthetic data to disk with metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # Save images
        import cv2
        cv2.imwrite(f"{self.collection_dir}/rgb_{timestamp}.png", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{self.collection_dir}/depth_{timestamp}.png", (depth_image * 256).astype(np.uint16))
        cv2.imwrite(f"{self.collection_dir}/seg_{timestamp}.png", seg_image)

        # Save LIDAR data
        np.save(f"{self.collection_dir}/lidar_{timestamp}.npy", np.array(lidar_ranges))

        # Save metadata
        metadata['timestamp'] = timestamp
        with open(f"{self.collection_dir}/metadata_{timestamp}.json", 'w') as f:
            json.dump(metadata, f, indent=2)

    def generate_synthetic_data(self):
        """Main function to generate synthetic data"""
        if not self.collection_enabled:
            return

        current_time = self.get_clock().now()

        # Generate random scene
        scene_config, objects = self.generate_random_scene()

        # Generate synthetic sensor data
        rgb_image = self.generate_synthetic_rgb(objects, scene_config['lighting'])
        depth_image = self.generate_synthetic_depth(objects)
        seg_image = self.generate_synthetic_segmentation(objects)
        lidar_ranges, angle_min, angle_increment = self.generate_synthetic_lidar(objects)

        # Create ROS messages
        rgb_msg = self.cv_bridge.cv2_to_imgmsg(rgb_image, encoding="rgb8")
        rgb_msg.header.stamp = current_time.to_msg()
        rgb_msg.header.frame_id = 'camera_rgb_frame'

        depth_msg = self.cv_bridge.cv2_to_imgmsg(depth_image, encoding="32FC1")
        depth_msg.header.stamp = current_time.to_msg()
        depth_msg.header.frame_id = 'camera_depth_frame'

        seg_msg = self.cv_bridge.cv2_to_imgmsg(seg_image, encoding="rgb8")
        seg_msg.header.stamp = current_time.to_msg()
        seg_msg.header.frame_id = 'camera_seg_frame'

        scan_msg = LaserScan()
        scan_msg.header.stamp = current_time.to_msg()
        scan_msg.header.frame_id = 'laser_frame'
        scan_msg.angle_min = angle_min
        scan_msg.angle_max = -angle_min
        scan_msg.angle_increment = angle_increment
        scan_msg.time_increment = 0.0
        scan_msg.scan_time = 0.1
        scan_msg.range_min = 0.1
        scan_msg.range_max = 25.0
        scan_msg.ranges = lidar_ranges

        # Publish synthetic data
        self.rgb_pub.publish(rgb_msg)
        self.depth_pub.publish(depth_msg)
        self.seg_pub.publish(seg_msg)
        self.scan_pub.publish(scan_msg)

        # Save to disk for AI training
        metadata = {
            'scene_config': scene_config,
            'object_count': len(objects),
            'objects': objects,
            'collection_id': self.data_counter
        }
        self.save_synthetic_data(rgb_image, depth_image, seg_image, lidar_ranges, metadata)

        # Publish status
        status_msg = String()
        status_msg.data = f"Synthetic Data #{self.data_counter}: Objects={len(objects)}, " \
                         f"Lighting={scene_config['lighting']}, Weather={scene_config['weather']}"
        self.status_pub.publish(status_msg)

        self.data_counter += 1

        self.get_logger().info(f"Generated synthetic data batch #{self.data_counter}")

def main(args=None):
    rclpy.init(args=args)
    generator = SyntheticDataGenerator()

    try:
        rclpy.spin(generator)
    except KeyboardInterrupt:
        generator.get_logger().info("Synthetic Data Generator stopped by user")
    finally:
        generator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Quick Recap

In this lesson, we've covered:

- **Isaac Sim Architecture**: Understanding the components and capabilities of NVIDIA Isaac Sim
- **Environment Setup**: Creating simulation environments with realistic physics and graphics
- **Robot Integration**: Adding robots to Isaac Sim with proper sensor configurations
- **ROS Bridge**: Connecting Isaac Sim to ROS 2 for control and data exchange
- **Synthetic Data Generation**: Creating training data for AI models with realistic variation

NVIDIA Isaac Sim provides a comprehensive platform for developing AI-powered robots by combining high-fidelity physics simulation with advanced graphics rendering. This enables the creation of photorealistic training data that can bridge the reality gap between simulation and real-world deployment.

In the next lesson, we'll explore synthetic data generation in more detail, focusing on how Isaac Sim can create diverse and realistic datasets for training perception and navigation models.