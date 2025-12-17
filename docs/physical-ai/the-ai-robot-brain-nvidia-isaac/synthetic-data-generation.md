---
sidebar_label: 'Synthetic Data Generation'
title: 'Synthetic Data Generation'
---

# Synthetic Data Generation

## Overview

Synthetic data generation is a revolutionary approach in AI development that addresses the critical challenge of data scarcity in robotics. NVIDIA Isaac Sim excels at creating diverse, labeled, and photorealistic synthetic datasets that can be used to train perception models, navigation systems, and other AI components. This lesson explores the principles, techniques, and best practices for generating high-quality synthetic data that effectively transfers to real-world applications.

The ability to generate unlimited, perfectly labeled training data with precise ground truth information makes synthetic data generation a cornerstone of modern AI-robotics development, significantly reducing the time and cost associated with real-world data collection.

## Learning Outcomes

By the end of this lesson, you will be able to:
- Understand the principles and benefits of synthetic data generation
- Configure Isaac Sim for diverse data generation scenarios
- Generate various types of synthetic sensor data (RGB, depth, segmentation, LIDAR)
- Apply domain randomization techniques to improve model generalization
- Validate synthetic-to-real transfer performance

## Hands-on Steps

1. **Environment Configuration**: Set up Isaac Sim for data generation
2. **Scene Randomization**: Create diverse scene variations
3. **Sensor Data Generation**: Generate multiple sensor modalities
4. **Domain Randomization**: Apply randomization techniques
5. **Data Validation**: Test synthetic-to-real transfer

### Prerequisites

- Understanding of Isaac Sim basics (from previous lesson)
- Knowledge of computer vision and machine learning concepts
- Experience with sensor data formats

## Code Examples

Let's start by creating a comprehensive synthetic data generation pipeline:

```python
# synthetic_data_pipeline.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, PointCloud2, CameraInfo
from geometry_msgs.msg import Pose, Point
from std_msgs.msg import String, Header, Bool
from cv_bridge import CvBridge
import numpy as np
import math
import json
import os
import cv2
from datetime import datetime
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class SceneConfiguration:
    """Data class for scene configuration parameters"""
    lighting_condition: str
    weather: str
    object_count: int
    texture_randomization: bool
    camera_position: Tuple[float, float, float]
    camera_orientation: Tuple[float, float, float]

@dataclass
class ObjectConfiguration:
    """Data class for individual object parameters"""
    object_id: int
    object_type: str  # 'box', 'cylinder', 'sphere', 'capsule'
    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]  # quaternion
    size: Tuple[float, float, float]  # x, y, z for box; radius, height for others
    color: Tuple[float, float, float]  # RGB normalized
    class_id: int  # for segmentation
    material_properties: Dict[str, float]  # roughness, metallic, etc.

class SyntheticDataGenerator(Node):
    """
    Advanced synthetic data generator with domain randomization
    """
    def __init__(self):
        super().__init__('synthetic_data_generator')

        # Publishers
        self.rgb_pub = self.create_publisher(Image, '/isaac_synthetic/rgb', 10)
        self.depth_pub = self.create_publisher(Image, '/isaac_synthetic/depth', 10)
        self.seg_pub = self.create_publisher(Image, '/isaac_synthetic/segmentation', 10)
        self.normal_pub = self.create_publisher(Image, '/isaac_synthetic/normal', 10)
        self.lidar_pub = self.create_publisher(LaserScan, '/isaac_synthetic/lidar', 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, '/isaac_synthetic/camera_info', 10)
        self.metadata_pub = self.create_publisher(String, '/isaac_synthetic/metadata', 10)

        # Control subscribers
        self.enable_sub = self.create_subscription(Bool, '/isaac_synthetic/enable', self.enable_callback, 10)
        self.config_sub = self.create_subscription(String, '/isaac_synthetic/config', self.config_callback, 10)

        # Timers
        self.data_gen_timer = self.create_timer(0.2, self.generate_data_cycle)  # 5Hz

        # Internal components
        self.cv_bridge = CvBridge()
        self.data_counter = 0
        self.collection_enabled = True
        self.collection_dir = "/tmp/isaac_synthetic_data"
        self.current_config = None

        # Create collection directory
        os.makedirs(self.collection_dir, exist_ok=True)

        # Define object classes for segmentation
        self.object_classes = {
            0: 'background',
            1: 'robot',
            2: 'obstacle',
            3: 'furniture',
            4: 'wall',
            5: 'floor',
            6: 'ceiling',
            7: 'decoration'
        }

        # Domain randomization parameters
        self.lighting_conditions = ['bright', 'dim', 'variable', 'backlight']
        self.weather_conditions = ['clear', 'overcast', 'foggy', 'rainy_simulation']
        self.texture_sets = ['indoor', 'outdoor', 'industrial', 'residential']

        # Camera parameters
        self.camera_params = {
            'width': 640,
            'height': 480,
            'fov': 60.0,  # degrees
            'near_clip': 0.1,
            'far_clip': 100.0
        }

        # Initialize first configuration
        self.current_config = self.generate_random_scene_config()

        self.get_logger().info("Advanced Synthetic Data Generator initialized")

    def enable_callback(self, msg):
        """Enable/disable data collection"""
        self.collection_enabled = msg.data
        self.get_logger().info(f"Data collection {'enabled' if self.collection_enabled else 'disabled'}")

    def config_callback(self, msg):
        """Receive configuration parameters"""
        try:
            config_dict = json.loads(msg.data)
            # Apply configuration changes
            if 'lighting' in config_dict:
                self.current_config.lighting_condition = config_dict['lighting']
            if 'object_count' in config_dict:
                self.current_config.object_count = config_dict['object_count']
            if 'weather' in config_dict:
                self.current_config.weather = config_dict['weather']

            self.get_logger().info(f"Configuration updated: {config_dict}")
        except json.JSONDecodeError:
            self.get_logger().error("Invalid configuration JSON received")

    def generate_random_scene_config(self) -> SceneConfiguration:
        """Generate a random scene configuration with domain randomization"""
        return SceneConfiguration(
            lighting_condition=random.choice(self.lighting_conditions),
            weather=random.choice(self.weather_conditions),
            object_count=random.randint(3, 15),
            texture_randomization=random.choice([True, False]),
            camera_position=(
                random.uniform(-2, 2),
                random.uniform(-2, 2),
                random.uniform(1, 3)
            ),
            camera_orientation=(
                random.uniform(-0.3, 0.3),  # pitch
                random.uniform(-0.3, 0.3),  # yaw
                random.uniform(-0.1, 0.1)   # roll
            )
        )

    def generate_random_object(self, object_id: int) -> ObjectConfiguration:
        """Generate a random object with domain randomization"""
        obj_type = random.choice(['box', 'cylinder', 'sphere', 'capsule'])
        class_id = random.choice(list(self.object_classes.keys())[2:])  # Exclude background and robot

        # Random position within scene bounds
        position = (
            random.uniform(-8, 8),
            random.uniform(-8, 8),
            random.uniform(0.1, 3.0)
        )

        # Random orientation (quaternion)
        roll = random.uniform(-0.5, 0.5)
        pitch = random.uniform(-0.5, 0.5)
        yaw = random.uniform(-math.pi, math.pi)
        orientation = self.euler_to_quaternion(roll, pitch, yaw)

        # Random size based on object type
        if obj_type == 'box':
            size = (
                random.uniform(0.2, 2.0),
                random.uniform(0.2, 2.0),
                random.uniform(0.2, 2.0)
            )
        elif obj_type in ['cylinder', 'capsule']:
            size = (
                random.uniform(0.1, 1.0),  # radius
                random.uniform(0.3, 2.0),  # height
                0.0  # unused for cylinder/capsule
            )
        else:  # sphere
            size = (
                random.uniform(0.1, 1.0),  # radius
                0.0,  # unused
                0.0   # unused
            )

        # Random color
        color = (
            random.uniform(0.1, 1.0),
            random.uniform(0.1, 1.0),
            random.uniform(0.1, 1.0)
        )

        # Material properties for realistic rendering
        material_properties = {
            'roughness': random.uniform(0.1, 0.9),
            'metallic': random.uniform(0.0, 0.5),
            'specular': random.uniform(0.1, 0.8)
        }

        return ObjectConfiguration(
            object_id=object_id,
            object_type=obj_type,
            position=position,
            orientation=orientation,
            size=size,
            color=color,
            class_id=class_id,
            material_properties=material_properties
        )

    def euler_to_quaternion(self, roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
        """Convert Euler angles to quaternion"""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return (x, y, z, w)

    def generate_scene_data(self) -> Tuple[List[ObjectConfiguration], SceneConfiguration]:
        """Generate complete scene with objects and configuration"""
        # Generate new scene configuration
        scene_config = self.generate_random_scene_config()

        # Generate objects
        objects = []
        for i in range(scene_config.object_count):
            obj = self.generate_random_object(i)
            objects.append(obj)

        return objects, scene_config

    def generate_synthetic_rgb(self, objects: List[ObjectConfiguration], scene_config: SceneConfiguration) -> np.ndarray:
        """Generate synthetic RGB image with domain randomization"""
        height = self.camera_params['height']
        width = self.camera_params['width']
        image = np.zeros((height, width, 3), dtype=np.float32)

        # Apply lighting condition
        lighting_factor = {
            'bright': 1.0,
            'dim': 0.3,
            'variable': random.uniform(0.4, 1.0),
            'backlight': 0.7
        }[scene_config.lighting_condition]

        # Create background based on weather
        if scene_config.weather == 'clear':
            # Sky gradient
            for y in range(height):
                sky_color = [
                    0.5 + 0.3 * y / height,  # Blue increases toward top
                    0.7 + 0.2 * y / height,  # Green increases toward top
                    1.0  # Sky blue
                ]
                image[y, :, :] = sky_color
        elif scene_config.weather == 'overcast':
            # Gray overcast
            image[:, :, :] = [0.7, 0.7, 0.8]
        elif scene_config.weather == 'foggy':
            # Foggy with low contrast
            base_color = [0.8, 0.8, 0.8]
            for y in range(height):
                fog_factor = 1.0 - 0.3 * y / height  # Less fog at top
                image[y, :, :] = [c * fog_factor for c in base_color]

        # Add ground plane
        ground_level = int(0.6 * height)  # Ground starts at 60% height
        for y in range(ground_level, height):
            # Ground color with texture variation
            ground_color = [
                0.4 + random.uniform(-0.1, 0.1),  # Brownish
                0.3 + random.uniform(-0.1, 0.1),
                0.2 + random.uniform(-0.1, 0.1)
            ]
            image[y, :, :] = ground_color

        # Add objects to scene
        for obj in objects:
            # Project 3D object to 2D image
            # Simplified projection for demonstration
            # In real Isaac Sim, this would use proper 3D rendering
            obj_x, obj_y, obj_z = obj.position
            screen_x = int(width / 2 + obj_x * 40)  # Scale factor for visibility
            screen_y = int(height / 2 - obj_y * 40)  # Invert Y

            if 0 <= screen_x < width and 0 <= screen_y < height:
                # Determine object size in pixels based on distance
                distance_factor = max(0.1, 5.0 / max(obj_z, 0.5))
                size_pixels = max(1, int(obj.size[0] * 30 * distance_factor))

                # Draw object based on type
                color = [int(c * 255 * lighting_factor) for c in obj.color]
                if obj.object_type == 'box':
                    # Draw rectangle
                    x1, x2 = max(0, screen_x - size_pixels), min(width, screen_x + size_pixels)
                    y1, y2 = max(0, screen_y - size_pixels), min(height, screen_y + size_pixels)
                    image[y1:y2, x1:x2] = [c / 255.0 for c in color]  # Normalize to 0-1
                elif obj.object_type in ['cylinder', 'sphere', 'capsule']:
                    # Draw circle
                    cv2.circle(image, (screen_x, screen_y), size_pixels,
                              [c / 255.0 for c in color], -1)

        # Apply weather effects
        if scene_config.weather == 'rainy_simulation':
            # Add rain streaks (simulated)
            for _ in range(50):
                x = random.randint(0, width-1)
                y = random.randint(0, height-10)
                length = random.randint(5, 15)
                cv2.line(image, (x, y), (x, min(y+length, height-1)), (0.7, 0.7, 1.0), 1)

        # Add noise to make more realistic
        noise = np.random.normal(0, 0.02, image.shape).astype(np.float32)
        image = np.clip(image + noise, 0, 1)

        return (image * 255).astype(np.uint8)

    def generate_synthetic_depth(self, objects: List[ObjectConfiguration]) -> np.ndarray:
        """Generate synthetic depth image"""
        height = self.camera_params['height']
        width = self.camera_params['width']
        depth = np.ones((height, width), dtype=np.float32) * self.camera_params['far_clip']

        # Calculate depth for each object
        for obj in objects:
            obj_x, obj_y, obj_z = obj.position
            screen_x = int(width / 2 + obj_x * 40)
            screen_y = int(height / 2 - obj_y * 40)

            if 0 <= screen_x < width and 0 <= screen_y < height:
                # Object distance from camera
                distance = math.sqrt(obj_x**2 + obj_y**2 + obj_z**2)

                # Determine object size in pixels
                distance_factor = max(0.1, 5.0 / max(obj_z, 0.5))
                size_pixels = max(1, int(obj.size[0] * 30 * distance_factor))

                # Fill object area with its distance
                if obj.object_type == 'box':
                    x1, x2 = max(0, screen_x - size_pixels), min(width, screen_x + size_pixels)
                    y1, y2 = max(0, screen_y - size_pixels), min(height, screen_y + size_pixels)
                    depth[y1:y2, x1:x2] = min(distance, self.camera_params['far_clip'])
                elif obj.object_type in ['cylinder', 'sphere', 'capsule']:
                    cv2.circle(depth, (screen_x, screen_y), size_pixels, distance, -1)

        # Add realistic depth noise
        noise = np.random.normal(0, 0.01, depth.shape).astype(np.float32)
        depth = np.maximum(self.camera_params['near_clip'], depth + noise)

        return depth

    def generate_synthetic_segmentation(self, objects: List[ObjectConfiguration], width: int, height: int) -> np.ndarray:
        """Generate synthetic segmentation mask"""
        segmentation = np.zeros((height, width), dtype=np.uint8)

        # Set background class
        segmentation[:, :] = 0  # background class

        # Add objects with their class IDs
        for obj in objects:
            obj_x, obj_y, obj_z = obj.position
            screen_x = int(width / 2 + obj_x * 40)
            screen_y = int(height / 2 - obj_y * 40)

            if 0 <= screen_x < width and 0 <= screen_y < height:
                distance_factor = max(0.1, 5.0 / max(obj_z, 0.5))
                size_pixels = max(1, int(obj.size[0] * 30 * distance_factor))

                if obj.object_type == 'box':
                    x1, x2 = max(0, screen_x - size_pixels), min(width, screen_x + size_pixels)
                    y1, y2 = max(0, screen_y - size_pixels), min(height, screen_y + size_pixels)
                    segmentation[y1:y2, x1:x2] = obj.class_id
                elif obj.object_type in ['cylinder', 'sphere', 'capsule']:
                    cv2.circle(segmentation, (screen_x, screen_y), size_pixels, obj.class_id, -1)

        # Convert to 3-channel for ROS compatibility
        seg_image = np.stack([segmentation, segmentation, segmentation], axis=2)
        return seg_image.astype(np.uint8)

    def generate_synthetic_normals(self, objects: List[ObjectConfiguration], width: int, height: int) -> np.ndarray:
        """Generate synthetic surface normal map"""
        normals = np.zeros((height, width, 3), dtype=np.float32)

        # For simplicity, assign face normals to objects
        for obj in objects:
            obj_x, obj_y, obj_z = obj.position
            screen_x = int(width / 2 + obj_x * 40)
            screen_y = int(height / 2 - obj_y * 40)

            if 0 <= screen_x < width and 0 <= screen_y < height:
                distance_factor = max(0.1, 5.0 / max(obj_z, 0.5))
                size_pixels = max(1, int(obj.size[0] * 30 * distance_factor))

                # Assign normal based on object type
                if obj.object_type == 'box':
                    normal = [0, 0, 1]  # Front face normal
                    x1, x2 = max(0, screen_x - size_pixels), min(width, screen_x + size_pixels)
                    y1, y2 = max(0, screen_y - size_pixels), min(height, screen_y + size_pixels)
                    normals[y1:y2, x1:x2] = normal
                elif obj.object_type in ['cylinder', 'sphere']:
                    # Radial normals for curved surfaces
                    for dy in range(-size_pixels, size_pixels):
                        for dx in range(-size_pixels, size_pixels):
                            if dx*dx + dy*dy <= size_pixels*size_pixels:
                                py, px = screen_y + dy, screen_x + dx
                                if 0 <= py < height and 0 <= px < width:
                                    # Normal points outward from center
                                    vec_x = px - screen_x
                                    vec_y = py - screen_y
                                    vec_z = 0
                                    length = max(0.1, math.sqrt(vec_x*vec_x + vec_y*vec_y + vec_z*vec_z))
                                    normals[py, px] = [vec_x/length, vec_y/length, vec_z/length]

        # Normalize to 0-255 range
        normals = ((normals + 1) * 127.5).astype(np.uint8)
        return normals

    def generate_synthetic_lidar(self, objects: List[ObjectConfiguration]) -> Tuple[List[float], float, float]:
        """Generate synthetic LIDAR scan"""
        num_points = 1080
        angle_min = -math.pi * 0.75  # -135 degrees
        angle_max = math.pi * 0.75   # 135 degrees
        angle_increment = (angle_max - angle_min) / num_points

        ranges = []

        for i in range(num_points):
            angle = angle_min + i * angle_increment
            ray_direction = (math.cos(angle), math.sin(angle))

            min_distance = self.camera_params['far_clip']

            # Check for intersections with all objects
            for obj in objects:
                if obj.object_type == 'box':
                    distance = self.ray_box_intersection(
                        (0, 0), ray_direction,  # Robot at origin
                        obj.position[0], obj.position[1],
                        obj.size[0]/2, obj.size[1]/2
                    )
                elif obj.object_type in ['cylinder', 'sphere']:
                    distance = self.ray_cylinder_intersection(
                        (0, 0), ray_direction,
                        obj.position[0], obj.position[1],
                        obj.size[0]  # radius
                    )
                else:
                    continue

                if distance and distance < min_distance:
                    min_distance = distance

            # Add realistic LIDAR noise
            noise = np.random.normal(0, 0.01)
            final_range = max(self.camera_params['near_clip'],
                             min(self.camera_params['far_clip'], min_distance + noise))
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

    def save_synthetic_data(self, rgb_image, depth_image, seg_image, normal_image, lidar_ranges, metadata):
        """Save synthetic data to disk with metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # Create data directory for this batch
        batch_dir = f"{self.collection_dir}/batch_{self.data_counter:06d}"
        os.makedirs(batch_dir, exist_ok=True)

        # Save images
        cv2.imwrite(f"{batch_dir}/rgb_{timestamp}.png", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{batch_dir}/depth_{timestamp}.png", (depth_image * 256).astype(np.uint16))
        cv2.imwrite(f"{batch_dir}/seg_{timestamp}.png", seg_image)
        cv2.imwrite(f"{batch_dir}/normal_{timestamp}.png", normal_image)

        # Save LIDAR data
        np.save(f"{batch_dir}/lidar_{timestamp}.npy", np.array(lidar_ranges))

        # Save metadata
        metadata['timestamp'] = timestamp
        metadata['batch_id'] = self.data_counter
        with open(f"{batch_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save individual object annotations
        obj_annotations = []
        for obj in metadata['objects']:
            obj_annotations.append({
                'id': obj.object_id,
                'type': obj.object_type,
                'class': self.object_classes.get(obj.class_id, 'unknown'),
                'position': obj.position,
                'size': obj.size,
                'color': obj.color
            })

        with open(f"{batch_dir}/object_annotations.json", 'w') as f:
            json.dump(obj_annotations, f, indent=2)

    def generate_data_cycle(self):
        """Main data generation cycle"""
        if not self.collection_enabled:
            return

        current_time = self.get_clock().now()

        # Generate new scene
        objects, scene_config = self.generate_scene_data()

        # Generate all sensor data
        rgb_image = self.generate_synthetic_rgb(objects, scene_config)
        depth_image = self.generate_synthetic_depth(objects)
        seg_image = self.generate_synthetic_segmentation(objects, self.camera_params['width'], self.camera_params['height'])
        normal_image = self.generate_synthetic_normals(objects, self.camera_params['width'], self.camera_params['height'])
        lidar_ranges, angle_min, angle_increment = self.generate_synthetic_lidar(objects)

        # Create and publish ROS messages
        rgb_msg = self.cv_bridge.cv2_to_imgmsg(rgb_image, encoding="rgb8")
        rgb_msg.header.stamp = current_time.to_msg()
        rgb_msg.header.frame_id = 'camera_rgb_frame'
        self.rgb_pub.publish(rgb_msg)

        depth_msg = self.cv_bridge.cv2_to_imgmsg(depth_image, encoding="32FC1")
        depth_msg.header.stamp = current_time.to_msg()
        depth_msg.header.frame_id = 'camera_depth_frame'
        self.depth_pub.publish(depth_msg)

        seg_msg = self.cv_bridge.cv2_to_imgmsg(seg_image, encoding="rgb8")
        seg_msg.header.stamp = current_time.to_msg()
        seg_msg.header.frame_id = 'camera_seg_frame'
        self.seg_pub.publish(seg_msg)

        normal_msg = self.cv_bridge.cv2_to_imgmsg(normal_image, encoding="rgb8")
        normal_msg.header.stamp = current_time.to_msg()
        normal_msg.header.frame_id = 'camera_normal_frame'
        self.normal_pub.publish(normal_msg)

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
        self.lidar_pub.publish(scan_msg)

        # Create camera info message
        camera_info_msg = CameraInfo()
        camera_info_msg.header.stamp = current_time.to_msg()
        camera_info_msg.header.frame_id = 'camera_rgb_frame'
        camera_info_msg.width = self.camera_params['width']
        camera_info_msg.height = self.camera_params['height']
        camera_info_msg.distortion_model = 'plumb_bob'
        # Simple pinhole camera model
        fx = self.camera_params['width'] / (2 * math.tan(math.radians(self.camera_params['fov']/2)))
        fy = fx
        cx = self.camera_params['width'] / 2
        cy = self.camera_params['height'] / 2
        camera_info_msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        self.camera_info_pub.publish(camera_info_msg)

        # Create metadata and publish
        metadata = {
            'scene_config': {
                'lighting_condition': scene_config.lighting_condition,
                'weather': scene_config.weather,
                'object_count': scene_config.object_count,
                'texture_randomization': scene_config.texture_randomization
            },
            'objects': [obj for obj in objects],
            'sensor_data_types': ['rgb', 'depth', 'segmentation', 'normals', 'lidar'],
            'domain_randomization_applied': True
        }

        metadata_msg = String()
        metadata_msg.data = json.dumps(metadata, default=lambda x: x.__dict__ if hasattr(x, '__dict__') else str(x))
        self.metadata_pub.publish(metadata_msg)

        # Save to disk
        self.save_synthetic_data(rgb_image, depth_image, seg_image, normal_image, lidar_ranges, metadata)

        # Publish status
        status_msg = String()
        status_msg.data = f"Batch {self.data_counter}: {len(objects)} objects, " \
                         f"Lighting={scene_config.lighting_condition}, Weather={scene_config.weather}"
        self.get_logger().info(status_msg.data)

        self.data_counter += 1

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

Now let's create a specialized data validation tool for synthetic-to-real transfer:

```python
# synthetic_data_validator.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String, Float32
import numpy as np
import cv2
from cv_bridge import CvBridge
import json
from sklearn.metrics import mean_squared_error
from scipy import stats
import matplotlib.pyplot as plt

class SyntheticDataValidator(Node):
    """
    Validate synthetic data quality and synthetic-to-real transfer potential
    """
    def __init__(self):
        super().__init__('synthetic_data_validator')

        # Publishers
        self.quality_score_pub = self.create_publisher(Float32, '/synthetic_validation/quality_score', 10)
        self.transfer_score_pub = self.create_publisher(Float32, '/synthetic_validation/transfer_score', 10)
        self.status_pub = self.create_publisher(String, '/synthetic_validation/status', 10)

        # Subscribers
        self.rgb_sub = self.create_subscription(Image, '/isaac_synthetic/rgb', self.rgb_callback, 10)
        self.real_rgb_sub = self.create_subscription(Image, '/real_camera/rgb', self.real_rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/isaac_synthetic/depth', self.depth_callback, 10)

        # Timers
        self.validation_timer = self.create_timer(1.0, self.run_validation)

        # Internal components
        self.cv_bridge = CvBridge()
        self.synthetic_rgb_buffer = []
        self.real_rgb_buffer = []
        self.synthetic_depth_buffer = []

        # Validation parameters
        self.buffer_size = 10
        self.validation_metrics = {
            'sharpness': 0.0,
            'color_distribution': 0.0,
            'texture_complexity': 0.0,
            'dynamic_range': 0.0
        }

        self.get_logger().info("Synthetic Data Validator initialized")

    def rgb_callback(self, msg):
        """Process synthetic RGB images"""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "rgb8")
            self.synthetic_rgb_buffer.append(cv_image.copy())

            if len(self.synthetic_rgb_buffer) > self.buffer_size:
                self.synthetic_rgb_buffer.pop(0)
        except Exception as e:
            self.get_logger().error(f"Error processing synthetic RGB: {e}")

    def real_rgb_callback(self, msg):
        """Process real RGB images for comparison"""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "rgb8")
            self.real_rgb_buffer.append(cv_image.copy())

            if len(self.real_rgb_buffer) > self.buffer_size:
                self.real_rgb_buffer.pop(0)
        except Exception as e:
            self.get_logger().error(f"Error processing real RGB: {e}")

    def depth_callback(self, msg):
        """Process synthetic depth images"""
        try:
            cv_depth = self.cv_bridge.imgmsg_to_cv2(msg, "32FC1")
            self.synthetic_depth_buffer.append(cv_depth.copy())

            if len(self.synthetic_depth_buffer) > self.buffer_size:
                self.synthetic_depth_buffer.pop(0)
        except Exception as e:
            self.get_logger().error(f"Error processing synthetic depth: {e}")

    def calculate_sharpness(self, image):
        """Calculate image sharpness using Laplacian variance"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var

    def calculate_color_distribution(self, image):
        """Calculate color distribution statistics"""
        if len(image.shape) == 3:
            # Calculate histogram for each channel
            hist_r = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
            hist_b = cv2.calcHist([image], [2], None, [256], [0, 256])

            # Normalize histograms
            hist_r = hist_r / hist_r.sum()
            hist_g = hist_g / hist_g.sum()
            hist_b = hist_b / hist_b.sum()

            return {
                'mean': [hist_r.mean(), hist_g.mean(), hist_b.mean()],
                'std': [hist_r.std(), hist_g.std(), hist_b.std()],
                'entropy': [
                    -np.sum(hist_r * np.log2(hist_r + 1e-10)),
                    -np.sum(hist_g * np.log2(hist_g + 1e-10)),
                    -np.sum(hist_b * np.log2(hist_b + 1e-10))
                ]
            }
        return {'mean': [0], 'std': [0], 'entropy': [0]}

    def calculate_texture_complexity(self, image):
        """Calculate texture complexity using Local Binary Patterns"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

        # Simple texture measure using gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        return gradient_magnitude.mean()

    def calculate_dynamic_range(self, image):
        """Calculate image dynamic range"""
        if len(image.shape) == 3:
            # Calculate for each channel
            ranges = []
            for i in range(3):
                channel = image[:, :, i]
                ranges.append(float(channel.max() - channel.min()))
            return sum(ranges) / len(ranges)
        else:
            return float(image.max() - image.min())

    def compare_synthetic_real(self):
        """Compare synthetic and real image characteristics"""
        if not self.synthetic_rgb_buffer or not self.real_rgb_buffer:
            return 0.0

        # Get latest images
        synth_img = self.synthetic_rgb_buffer[-1]
        real_img = self.real_rgb_buffer[-1]

        # Calculate metrics for both
        synth_metrics = {
            'sharpness': self.calculate_sharpness(synth_img),
            'color_dist': self.calculate_color_distribution(synth_img),
            'texture': self.calculate_texture_complexity(synth_img),
            'dynamic_range': self.calculate_dynamic_range(synth_img)
        }

        real_metrics = {
            'sharpness': self.calculate_sharpness(real_img),
            'color_dist': self.calculate_color_distribution(real_img),
            'texture': self.calculate_texture_complexity(real_img),
            'dynamic_range': self.calculate_dynamic_range(real_img)
        }

        # Calculate similarity scores (0-1, where 1 is most similar)
        sharpness_sim = 1 / (1 + abs(synth_metrics['sharpness'] - real_metrics['sharpness']))
        texture_sim = 1 / (1 + abs(synth_metrics['texture'] - real_metrics['texture']))

        # Color distribution similarity (using Bhattacharyya distance for histograms)
        color_sim = 0
        for i in range(3):  # RGB channels
            hist_synth = cv2.calcHist([synth_img], [i], None, [32], [0, 256])
            hist_real = cv2.calcHist([real_img], [i], None, [32], [0, 256])

            # Normalize
            hist_synth = hist_synth / hist_synth.sum()
            hist_real = hist_real / hist_real.sum()

            # Bhattacharyya distance
            bc = cv2.compareHist(hist_synth, hist_real, cv2.HISTCMP_BHATTACHARYYA)
            color_sim += max(0, 1 - bc)  # Convert to similarity

        color_sim = color_sim / 3  # Average across channels

        # Combine metrics
        combined_similarity = (sharpness_sim * 0.3 +
                              texture_sim * 0.3 +
                              color_sim * 0.4)

        return min(1.0, max(0.0, combined_similarity))

    def assess_synthetic_quality(self):
        """Assess overall quality of synthetic data"""
        if not self.synthetic_rgb_buffer:
            return 0.0

        latest_img = self.synthetic_rgb_buffer[-1]

        # Calculate various quality metrics
        sharpness = self.calculate_sharpness(latest_img)
        texture_complexity = self.calculate_texture_complexity(latest_img)
        color_stats = self.calculate_color_distribution(latest_img)

        # Normalize metrics to 0-1 scale
        sharpness_score = min(1.0, sharpness / 1000)  # Adjust normalization as needed
        texture_score = min(1.0, texture_complexity / 50)  # Adjust normalization as needed
        color_entropy_score = min(1.0, sum(color_stats['entropy']) / 15)  # Adjust normalization as needed

        # Combined quality score
        quality_score = (sharpness_score * 0.4 +
                        texture_score * 0.3 +
                        color_entropy_score * 0.3)

        return min(1.0, max(0.0, quality_score))

    def run_validation(self):
        """Run comprehensive validation"""
        if not self.synthetic_rgb_buffer:
            return

        # Assess synthetic data quality
        quality_score = self.assess_synthetic_quality()

        # Assess synthetic-to-real transfer potential
        transfer_score = self.compare_synthetic_real()

        # Publish scores
        quality_msg = Float32()
        quality_msg.data = quality_score
        self.quality_score_pub.publish(quality_msg)

        transfer_msg = Float32()
        transfer_msg.data = transfer_score
        self.transfer_score_pub.publish(transfer_msg)

        # Publish status
        status_msg = String()
        status_msg.data = f"Quality: {quality_score:.3f}, Transfer: {transfer_score:.3f}, " \
                         f"Buffer: {len(self.synthetic_rgb_buffer)}/{len(self.real_rgb_buffer)}"
        self.status_pub.publish(status_msg)

        # Log validation results
        self.get_logger().info(f"Validation - Quality: {quality_score:.3f}, Transfer: {transfer_score:.3f}")

        # Update metrics for reporting
        latest_img = self.synthetic_rgb_buffer[-1]
        self.validation_metrics['sharpness'] = self.calculate_sharpness(latest_img)
        self.validation_metrics['texture_complexity'] = self.calculate_texture_complexity(latest_img)

def main(args=None):
    rclpy.init(args=args)
    validator = SyntheticDataValidator()

    try:
        rclpy.spin(validator)
    except KeyboardInterrupt:
        validator.get_logger().info("Synthetic Data Validator stopped by user")
    finally:
        validator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Small Simulation

Let's create a domain randomization experiment that demonstrates how varying environmental parameters affects synthetic data quality:

```python
# domain_randomization_experiment.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String, Float32MultiArray
import numpy as np
import cv2
from cv_bridge import CvBridge
import json
from dataclasses import dataclass
from typing import List, Dict
import random

@dataclass
class DomainConfig:
    """Configuration for domain randomization experiment"""
    lighting_intensity: float
    fog_density: float
    texture_complexity: float
    object_count: int
    camera_noise: float
    weather_type: str

class DomainRandomizationExperiment(Node):
    """
    Experiment with different domain randomization parameters
    to optimize synthetic-to-real transfer
    """
    def __init__(self):
        super().__init__('domain_randomization_experiment')

        # Publishers
        self.config_pub = self.create_publisher(String, '/domain_randomization/config', 10)
        self.metrics_pub = self.create_publisher(Float32MultiArray, '/domain_randomization/metrics', 10)
        self.status_pub = self.create_publisher(String, '/domain_randomization/status', 10)

        # Timers
        self.experiment_timer = self.create_timer(2.0, self.run_experiment_cycle)

        # Internal components
        self.cv_bridge = CvBridge()
        self.experiment_counter = 0
        self.experiment_configs = []
        self.results_history = []

        # Define experiment parameters
        self.lighting_range = (0.3, 1.5)  # Factor applied to base lighting
        self.fog_range = (0.0, 0.3)       # Fog density
        self.texture_range = (0.1, 1.0)   # Texture complexity factor
        self.object_count_range = (3, 12) # Number of objects in scene
        self.camera_noise_range = (0.0, 0.05)  # Noise level
        self.weather_types = ['clear', 'foggy', 'rainy', 'snowy', 'overcast']

        # Initialize with random configurations
        self.generate_initial_configs(20)

        self.get_logger().info("Domain Randomization Experiment initialized")

    def generate_initial_configs(self, num_configs: int):
        """Generate initial random configurations for experiment"""
        for _ in range(num_configs):
            config = DomainConfig(
                lighting_intensity=random.uniform(*self.lighting_range),
                fog_density=random.uniform(*self.fog_range),
                texture_complexity=random.uniform(*self.texture_range),
                object_count=random.randint(*self.object_count_range),
                camera_noise=random.uniform(*self.camera_noise_range),
                weather_type=random.choice(self.weather_types)
            )
            self.experiment_configs.append(config)

    def generate_synthetic_scene(self, config: DomainConfig) -> Dict:
        """Generate synthetic scene based on configuration"""
        # This would interface with Isaac Sim in a real implementation
        # For this example, we'll simulate the effects of different parameters

        # Calculate how each parameter affects the resulting image quality
        # These are simplified models of how domain parameters affect data quality

        # Lighting affects visibility and contrast
        lighting_quality = min(1.0, config.lighting_intensity * 0.8)

        # Fog reduces visibility and contrast
        fog_quality = max(0.1, 1.0 - config.fog_density * 2)

        # Texture complexity affects feature richness
        texture_quality = config.texture_complexity

        # Object count affects training diversity
        object_diversity = min(1.0, config.object_count / 15)

        # Camera noise affects data precision
        noise_impact = max(0.5, 1.0 - config.camera_noise * 20)

        # Weather affects overall scene quality
        weather_impact = {
            'clear': 1.0,
            'foggy': 0.7,
            'rainy': 0.6,
            'snowy': 0.6,
            'overcast': 0.8
        }[config.weather_type]

        # Calculate combined quality score
        combined_quality = (lighting_quality * 0.2 +
                           fog_quality * 0.2 +
                           texture_quality * 0.2 +
                           object_diversity * 0.15 +
                           noise_impact * 0.15 +
                           weather_impact * 0.1)

        # Calculate transferability score (how well this config might transfer to real)
        # Based on how close parameters are to real-world conditions
        real_world_similarity = (
            (1 - abs(config.lighting_intensity - 1.0) * 0.3) * 0.25 +  # Close to 1.0 is more realistic
            (1 - config.fog_density * 0.5) * 0.25 +  # Less fog is more common
            min(1.0, config.texture_complexity * 0.8) * 0.2 +  # Rich textures are good
            min(1.0, config.object_count / 10) * 0.15 +  # Moderate object count
            max(0.5, 1 - config.camera_noise * 10) * 0.15 +  # Less noise is better
            0.05  # Base score
        )

        # Calculate diversity score (how different this config is from others)
        diversity_score = self.calculate_diversity_score(config)

        return {
            'config': config,
            'quality_score': combined_quality,
            'transferability_score': real_world_similarity,
            'diversity_score': diversity_score,
            'overall_score': (combined_quality * 0.5 + real_world_similarity * 0.3 + diversity_score * 0.2)
        }

    def calculate_diversity_score(self, config: DomainConfig) -> float:
        """Calculate how diverse this configuration is compared to others"""
        if not self.experiment_configs:
            return 1.0

        # Calculate distance to other configs
        distances = []
        for other_config in self.experiment_configs:
            if other_config != config:
                # Calculate euclidean distance in parameter space
                dist = (
                    (config.lighting_intensity - other_config.lighting_intensity) ** 2 +
                    (config.fog_density - other_config.fog_density) ** 2 +
                    (config.texture_complexity - other_config.texture_complexity) ** 2 +
                    (config.object_count - other_config.object_count) ** 2 +
                    (config.camera_noise - other_config.camera_noise) ** 2
                ) ** 0.5
                distances.append(dist)

        # Average distance to other configs (higher = more diverse)
        if distances:
            avg_distance = sum(distances) / len(distances)
            # Normalize to 0-1 scale
            return min(1.0, avg_distance / 2.0)  # 2.0 is arbitrary normalization factor
        else:
            return 1.0

    def optimize_config_distribution(self):
        """Optimize the distribution of configurations based on results"""
        if not self.results_history:
            return

        # Sort results by overall score
        sorted_results = sorted(self.results_history, key=lambda x: x['overall_score'], reverse=True)

        # Keep top performers and generate new variations
        top_configs = [r['config'] for r in sorted_results[:5]]  # Top 5 configs

        # Generate new configs based on successful patterns
        new_configs = []
        for top_config in top_configs:
            for _ in range(2):  # Generate 2 variations of each top config
                # Add small random perturbations to successful configs
                new_config = DomainConfig(
                    lighting_intensity=max(self.lighting_range[0],
                                         min(self.lighting_range[1],
                                             top_config.lighting_intensity + random.uniform(-0.1, 0.1))),
                    fog_density=max(self.fog_range[0],
                                  min(self.fog_range[1],
                                      top_config.fog_density + random.uniform(-0.05, 0.05))),
                    texture_complexity=max(self.texture_range[0],
                                         min(self.texture_range[1],
                                             top_config.texture_complexity + random.uniform(-0.1, 0.1))),
                    object_count=max(self.object_count_range[0],
                                   min(self.object_count_range[1],
                                       top_config.object_count + random.randint(-2, 2))),
                    camera_noise=max(self.camera_noise_range[0],
                                   min(self.camera_noise_range[1],
                                       top_config.camera_noise + random.uniform(-0.01, 0.01))),
                    weather_type=top_config.weather_type if random.random() > 0.3 else random.choice(self.weather_types)
                )
                new_configs.append(new_config)

        # Replace experiment configs with optimized set
        self.experiment_configs = new_configs

    def run_experiment_cycle(self):
        """Run one cycle of the domain randomization experiment"""
        if self.experiment_configs:
            # Get current configuration
            current_config = self.experiment_configs[self.experiment_counter % len(self.experiment_configs)]

            # Generate synthetic scene with current config
            result = self.generate_synthetic_scene(current_config)

            # Store result
            self.results_history.append(result)
            if len(self.results_history) > 50:  # Limit history size
                self.results_history.pop(0)

            # Publish configuration for Isaac Sim to use
            config_msg = String()
            config_dict = {
                'lighting_intensity': current_config.lighting_intensity,
                'fog_density': current_config.fog_density,
                'texture_complexity': current_config.texture_complexity,
                'object_count': current_config.object_count,
                'camera_noise': current_config.camera_noise,
                'weather_type': current_config.weather_type,
                'experiment_id': self.experiment_counter
            }
            config_msg.data = json.dumps(config_dict)
            self.config_pub.publish(config_msg)

            # Publish metrics
            metrics_msg = Float32MultiArray()
            metrics_msg.data = [
                result['quality_score'],
                result['transferability_score'],
                result['diversity_score'],
                result['overall_score']
            ]
            self.metrics_pub.publish(metrics_msg)

            # Publish status
            status_msg = String()
            status_msg.data = f"Exp {self.experiment_counter}: L={current_config.lighting_intensity:.2f}, " \
                             f"F={current_config.fog_density:.2f}, W={current_config.weather_type}, " \
                             f"Score={result['overall_score']:.3f}"
            self.status_pub.publish(status_msg)

            self.get_logger().info(f"Experiment {self.experiment_counter}: Config={current_config.weather_type}, " \
                                  f"Score={result['overall_score']:.3f}")

            # Every 10 experiments, optimize the configuration distribution
            if self.experiment_counter > 0 and self.experiment_counter % 10 == 0:
                self.optimize_config_distribution()
                self.get_logger().info(f"Optimized configuration distribution. History size: {len(self.results_history)}")

            self.experiment_counter += 1

def main(args=None):
    rclpy.init(args=args)
    experiment = DomainRandomizationExperiment()

    try:
        rclpy.spin(experiment)
    except KeyboardInterrupt:
        experiment.get_logger().info("Domain Randomization Experiment stopped by user")
    finally:
        experiment.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Quick Recap

In this lesson, we've covered:

- **Synthetic Data Principles**: Understanding the benefits and applications of synthetic data generation
- **Advanced Generation Pipeline**: Creating comprehensive synthetic data with multiple sensor modalities
- **Domain Randomization**: Techniques for improving model generalization through environmental variation
- **Quality Assessment**: Methods for validating synthetic data quality and transfer potential
- **Experimentation Framework**: Systematic approaches to optimize synthetic data generation

Synthetic data generation with NVIDIA Isaac Sim enables the creation of diverse, perfectly labeled training datasets that can significantly accelerate AI model development while reducing real-world data collection costs. The combination of photorealistic rendering, physics simulation, and domain randomization techniques makes synthetic data an essential tool for modern robotics AI development.

In the next lesson, we'll explore Isaac ROS for Visual Simultaneous Localization and Mapping (VSLAM), focusing on how Isaac Sim can be used to develop and test advanced perception systems.