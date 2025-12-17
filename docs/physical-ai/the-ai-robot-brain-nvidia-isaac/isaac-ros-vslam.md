---
sidebar_label: 'Isaac ROS for VSLAM'
title: 'Isaac ROS for VSLAM'
---

# Isaac ROS for VSLAM

## Overview

Visual Simultaneous Localization and Mapping (VSLAM) is a critical capability for autonomous robots, enabling them to understand and navigate in unknown environments. NVIDIA Isaac ROS provides specialized packages and tools that accelerate VSLAM development by leveraging GPU acceleration and optimized algorithms. This lesson explores how to implement and optimize VSLAM systems using Isaac ROS within the Isaac Sim environment.

Isaac ROS bridges the gap between high-performance GPU-accelerated computer vision algorithms and the ROS 2 ecosystem, making it possible to deploy sophisticated VSLAM systems that can run in real-time on robotic platforms.

## Learning Outcomes

By the end of this lesson, you will be able to:
- Understand the architecture of Isaac ROS VSLAM components
- Implement stereo visual odometry using Isaac ROS packages
- Configure and optimize VSLAM pipelines for different environments
- Integrate VSLAM with other ROS 2 navigation components
- Evaluate VSLAM performance and accuracy in simulation

## Hands-on Steps

1. **Isaac ROS Setup**: Install and configure Isaac ROS packages
2. **Stereo Camera Configuration**: Set up stereo cameras for visual odometry
3. **VSLAM Pipeline Implementation**: Create a complete VSLAM system
4. **Performance Optimization**: Tune parameters for optimal performance
5. **Integration Testing**: Connect VSLAM to navigation stack

### Prerequisites

- Understanding of ROS 2 concepts and message types
- Knowledge of computer vision fundamentals
- Experience with Isaac Sim (from previous lessons)
- Basic understanding of SLAM concepts

## Code Examples

Let's start by creating a node that interfaces with Isaac ROS stereo visual odometry:

```python
# isaac_ros_vslam_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, PointStamped
from std_msgs.msg import String, Header
from cv_bridge import CvBridge
import numpy as np
import math
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

class IsaacROSVisualOdometryNode(Node):
    """
    Node that simulates integration with Isaac ROS Visual Odometry
    In a real implementation, this would interface with Isaac ROS packages
    """
    def __init__(self):
        super().__init__('isaac_ros_vslam_node')

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, '/visual_odometry/odom', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/visual_odometry/pose', 10)
        self.status_pub = self.create_publisher(String, '/visual_odometry/status', 10)

        # Subscribers
        self.left_image_sub = self.create_subscription(
            Image, '/isaac_sim/camera/left/image_raw', self.left_image_callback, 10)
        self.right_image_sub = self.create_subscription(
            Image, '/isaac_sim/camera/right/image_raw', self.right_image_callback, 10)
        self.left_info_sub = self.create_subscription(
            CameraInfo, '/isaac_sim/camera/left/camera_info', self.left_info_callback, 10)
        self.right_info_sub = self.create_subscription(
            CameraInfo, '/isaac_sim/camera/right/camera_info', self.right_info_callback, 10)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Timers
        self.processing_timer = self.create_timer(0.1, self.process_visual_odometry)  # 10 Hz

        # Internal components
        self.cv_bridge = CvBridge()
        self.left_image = None
        self.right_image = None
        self.left_camera_info = None
        self.right_camera_info = None
        self.camera_baseline = 0.1  # meters (typical stereo baseline)

        # VSLAM state
        self.current_position = np.array([0.0, 0.0, 0.0])
        self.current_orientation = np.array([0.0, 0.0, 0.0, 1.0])  # x, y, z, w quaternion
        self.previous_features = None
        self.tracking_features = {}
        self.frame_count = 0

        # Performance metrics
        self.processing_times = []
        self.position_uncertainty = 0.1  # Initial uncertainty

        # Feature tracking parameters
        self.feature_detector_params = {
            'max_features': 1000,
            'quality_level': 0.01,
            'min_distance': 10,
            'block_size': 3
        }

        self.get_logger().info("Isaac ROS Visual Odometry Node initialized")

    def left_image_callback(self, msg):
        """Process left camera image"""
        try:
            self.left_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting left image: {e}")

    def right_image_callback(self, msg):
        """Process right camera image"""
        try:
            self.right_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting right image: {e}")

    def left_info_callback(self, msg):
        """Process left camera info"""
        self.left_camera_info = msg
        # Extract baseline from projection matrix if available
        if len(msg.p) >= 3:
            self.camera_baseline = abs(msg.p[3]) / msg.p[0]  # P(0,3) / P(0,0)

    def right_info_callback(self, msg):
        """Process right camera info"""
        self.right_camera_info = msg

    def detect_features(self, image):
        """Detect features in image using Shi-Tomasi corner detection"""
        if image is None:
            return np.array([])

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect good features to track
        features = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.feature_detector_params['max_features'],
            qualityLevel=self.feature_detector_params['quality_level'],
            minDistance=self.feature_detector_params['min_distance'],
            blockSize=self.feature_detector_params['block_size']
        )

        return features if features is not None else np.array([])

    def track_features(self, prev_img, curr_img, prev_features):
        """Track features between frames using Lucas-Kanade optical flow"""
        if prev_img is None or curr_img is None or len(prev_features) == 0:
            return np.array([]), np.array([]), np.array([])

        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        next_features, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_features, None,
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # Filter out bad matches
        good_old = prev_features[status == 1]
        good_new = next_features[status == 1]

        return good_old, good_new, status

    def estimate_essential_matrix(self, points1, points2):
        """Estimate essential matrix from corresponding points"""
        if len(points1) < 5:
            return None

        # Estimate essential matrix
        essential_matrix, mask = cv2.findEssentialMat(
            points2, points1,
            focal=self.left_camera_info.k[0] if self.left_camera_info else 1.0,
            pp=(self.left_camera_info.k[2], self.left_camera_info.k[5]) if self.left_camera_info else (0.0, 0.0),
            method=cv2.RANSAC,
            threshold=1.0,
            prob=0.999
        )

        return essential_matrix, mask

    def decompose_essential_matrix(self, essential_matrix):
        """Decompose essential matrix to get rotation and translation"""
        if essential_matrix is None:
            return None, None

        # Decompose essential matrix
        _, rotation, translation, _ = cv2.recoverPose(essential_matrix)

        return rotation, translation

    def triangulate_points(self, points1, points2):
        """Triangulate 3D points from stereo correspondences"""
        if self.left_camera_info is None or self.right_camera_info is None:
            return np.array([])

        # Create projection matrices
        # Left camera: identity rotation, zero translation
        proj_matrix1 = np.array([
            [self.left_camera_info.k[0], 0, self.left_camera_info.k[2], 0],
            [0, self.left_camera_info.k[4], self.left_camera_info.k[5], 0],
            [0, 0, 1, 0]
        ])

        # Right camera: identity rotation, baseline translation
        proj_matrix2 = np.array([
            [self.right_camera_info.k[0], 0, self.right_camera_info.k[2], -self.right_camera_info.k[0] * self.camera_baseline],
            [0, self.right_camera_info.k[4], self.right_camera_info.k[5], 0],
            [0, 0, 1, 0]
        ])

        # Triangulate points
        points4d = cv2.triangulatePoints(
            proj_matrix1, proj_matrix2,
            points1.T, points2.T
        )

        # Convert from homogeneous coordinates
        points3d = (points4d[:3] / points4d[3]).T

        return points3d

    def process_visual_odometry(self):
        """Main visual odometry processing function"""
        start_time = self.get_clock().now()

        # Check if we have both stereo images
        if self.left_image is None or self.right_image is None:
            return

        # Detect features in current left image
        current_features = self.detect_features(self.left_image)

        if self.previous_features is not None and len(self.previous_features) > 0 and len(current_features) > 0:
            # Track features between frames
            prev_matched, curr_matched, status = self.track_features(
                self.previous_left_image, self.left_image, self.previous_features
            )

            if len(prev_matched) > 5:  # Need minimum points for estimation
                # Estimate motion using essential matrix
                essential_matrix, mask = self.estimate_essential_matrix(prev_matched, curr_matched)

                if essential_matrix is not None:
                    rotation, translation = self.decompose_essential_matrix(essential_matrix)

                    if rotation is not None and translation is not None:
                        # Convert rotation matrix to quaternion
                        quat = self.rotation_matrix_to_quaternion(rotation)

                        # Update position and orientation
                        # Apply translation scaled by some factor (since we don't have true scale)
                        scale_factor = 0.1  # This would be determined by actual depth in real implementation
                        delta_pos = np.array([translation[0, 0] * scale_factor,
                                            translation[1, 0] * scale_factor,
                                            translation[2, 0] * scale_factor])

                        # Update position (in robot frame, transform to world frame)
                        # For simplicity, we'll just add the delta
                        self.current_position += delta_pos

                        # Update orientation
                        # This is a simplified update - in practice you'd compose rotations
                        self.current_orientation = self.multiply_quaternions(
                            self.current_orientation,
                            [quat[0], quat[1], quat[2], quat[3]]
                        )

                        # Update uncertainty based on number of tracked features
                        self.position_uncertainty = max(0.05, 10.0 / len(prev_matched))

        # Store current data for next iteration
        self.previous_features = current_features
        self.previous_left_image = self.left_image.copy()

        # Publish results
        self.publish_odometry()

        # Calculate processing time
        end_time = self.get_clock().now()
        processing_time = (end_time - start_time).nanoseconds / 1e6  # ms
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)

        # Publish status
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        status_msg = String()
        status_msg.data = f"Frame: {self.frame_count}, Features: {len(current_features) if current_features is not None else 0}, " \
                         f"Pos: ({self.current_position[0]:.2f}, {self.current_position[1]:.2f}, {self.current_position[2]:.2f}), " \
                         f"ProcTime: {avg_processing_time:.1f}ms, Uncertainty: {self.position_uncertainty:.3f}"
        self.status_pub.publish(status_msg)

        self.frame_count += 1

    def rotation_matrix_to_quaternion(self, rotation_matrix):
        """Convert rotation matrix to quaternion"""
        # Method from https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
        trace = np.trace(rotation_matrix)

        if trace > 0:
            s = math.sqrt(trace + 1.0) * 2  # s=4*qw
            qw = 0.25 * s
            qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
            qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
            qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
        else:
            if rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
                s = math.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]) * 2
                qw = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
                qx = 0.25 * s
                qy = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
                qz = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
            elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
                s = math.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]) * 2
                qw = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
                qx = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
                qy = 0.25 * s
                qz = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
            else:
                s = math.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) * 2
                qw = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
                qx = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
                qy = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
                qz = 0.25 * s

        return np.array([qx, qy, qz, qw])

    def multiply_quaternions(self, q1, q2):
        """Multiply two quaternions"""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2

        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

        # Normalize
        norm = math.sqrt(x*x + y*y + z*z + w*w)
        if norm > 0:
            return [x/norm, y/norm, z/norm, w/norm]
        else:
            return [0, 0, 0, 1]

    def publish_odometry(self):
        """Publish odometry information"""
        current_time = self.get_clock().now()

        # Create odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = current_time.to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'

        # Set position
        odom_msg.pose.pose.position.x = float(self.current_position[0])
        odom_msg.pose.pose.position.y = float(self.current_position[1])
        odom_msg.pose.pose.position.z = float(self.current_position[2])

        # Set orientation
        odom_msg.pose.pose.orientation.x = float(self.current_orientation[0])
        odom_msg.pose.pose.orientation.y = float(self.current_orientation[1])
        odom_msg.pose.pose.orientation.z = float(self.current_orientation[2])
        odom_msg.pose.pose.orientation.w = float(self.current_orientation[3])

        # Set covariance based on uncertainty
        uncertainty = self.position_uncertainty
        odom_msg.pose.covariance = [
            uncertainty, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, uncertainty, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, uncertainty, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, uncertainty
        ]

        # Publish odometry
        self.odom_pub.publish(odom_msg)

        # Create and publish pose stamped message
        pose_msg = PoseStamped()
        pose_msg.header.stamp = current_time.to_msg()
        pose_msg.header.frame_id = 'odom'
        pose_msg.pose = odom_msg.pose.pose
        self.pose_pub.publish(pose_msg)

        # Broadcast transform
        t = TransformStamped()
        t.header.stamp = current_time.to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = float(self.current_position[0])
        t.transform.translation.y = float(self.current_position[1])
        t.transform.translation.z = float(self.current_position[2])
        t.transform.rotation.x = float(self.current_orientation[0])
        t.transform.rotation.y = float(self.current_orientation[1])
        t.transform.rotation.z = float(self.current_orientation[2])
        t.transform.rotation.w = float(self.current_orientation[3])
        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    vo_node = IsaacROSVisualOdometryNode()

    try:
        rclpy.spin(vo_node)
    except KeyboardInterrupt:
        vo_node.get_logger().info("Isaac ROS Visual Odometry Node stopped by user")
    finally:
        vo_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    import cv2  # Import here to avoid issues if not available
    main()
```

Now let's create a more advanced VSLAM node that integrates with Isaac ROS packages and includes mapping capabilities:

```python
# isaac_ros_slam_mapper.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import String, Header
from cv_bridge import CvBridge
import numpy as np
import math
import open3d as o3d
from scipy.spatial import cKDTree
from tf2_ros import TransformListener, Buffer
import tf2_geometry_msgs
import threading
import time

class IsaacROSVSLAMMapper(Node):
    """
    Advanced VSLAM mapper node that integrates with Isaac ROS packages
    This node performs mapping in addition to localization
    """
    def __init__(self):
        super().__init__('isaac_ros_slam_mapper')

        # Publishers
        self.map_pub = self.create_publisher(OccupancyGrid, '/slam_map', 10)
        self.pointcloud_pub = self.create_publisher(PointCloud2, '/slam_pointcloud', 10)
        self.odom_pub = self.create_publisher(Odometry, '/slam/odom', 10)
        self.status_pub = self.create_publisher(String, '/slam/status', 10)

        # Subscribers
        self.stereo_sub = self.create_subscription(
            String, '/isaac_ros/stereo_processed', self.stereo_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/visual_odometry/odom', self.odom_callback, 10)

        # Timers
        self.mapping_timer = self.create_timer(0.5, self.update_map)  # 2 Hz
        self.publish_timer = self.create_timer(0.1, self.publish_results)  # 10 Hz

        # Internal components
        self.cv_bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # SLAM state
        self.current_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # x, y, z, qx, qy, qz, qw
        self.map_points = []  # List of 3D points in global frame
        self.local_map = {}  # Dictionary of points for efficient lookup
        self.keyframes = []  # Store key poses for loop closure
        self.graph_optimizer = None  # Placeholder for graph optimization

        # Mapping parameters
        self.map_resolution = 0.1  # meters per cell
        self.map_size = 50  # meters (50x50 map)
        self.map_center = np.array([0.0, 0.0])  # Center of map in world coordinates
        self.occupancy_grid = np.zeros((int(self.map_size / self.map_resolution),
                                       int(self.map_size / self.map_resolution)), dtype=np.int8)

        # Performance tracking
        self.frame_count = 0
        self.processing_times = []
        self.map_size_history = []

        # Feature management
        self.feature_lifetime = 10  # How many frames to keep features
        self.max_map_points = 10000  # Limit for memory management

        self.get_logger().info("Isaac ROS VSLAM Mapper initialized")

    def stereo_callback(self, msg):
        """Process stereo vision results from Isaac ROS"""
        # In a real implementation, this would receive processed stereo data
        # For this example, we'll simulate receiving 3D points from stereo processing
        try:
            # Parse the message (in real implementation, this would be actual stereo data)
            # For simulation, we'll generate points based on the environment
            points_3d = self.generate_simulated_points()

            # Transform points to global frame using current pose
            global_points = self.transform_points_to_global(points_3d)

            # Add points to map
            self.add_points_to_map(global_points)

            # Update keyframe if significant movement occurred
            if self.should_add_keyframe():
                self.add_keyframe()

        except Exception as e:
            self.get_logger().error(f"Error processing stereo data: {e}")

    def odom_callback(self, msg):
        """Update current pose from visual odometry"""
        self.current_pose[0] = msg.pose.pose.position.x
        self.current_pose[1] = msg.pose.pose.position.y
        self.current_pose[2] = msg.pose.pose.position.z
        self.current_pose[3] = msg.pose.pose.orientation.x
        self.current_pose[4] = msg.pose.pose.orientation.y
        self.current_pose[5] = msg.pose.pose.orientation.z
        self.current_pose[6] = msg.pose.pose.orientation.w

    def generate_simulated_points(self):
        """Generate simulated 3D points from stereo processing"""
        # This simulates what would come from Isaac ROS stereo packages
        # In reality, this would be actual points from stereo reconstruction
        points = []

        # Generate points based on current position to simulate environment
        for i in range(50):  # Generate 50 points per frame
            # Simulate points around the robot's current position
            angle = np.random.uniform(0, 2 * math.pi)
            distance = np.random.uniform(1, 10)  # 1-10 meters away

            x = distance * math.cos(angle)
            y = distance * math.sin(angle)
            z = np.random.uniform(0, 2)  # Ground to 2m height

            points.append([x, y, z])

        return np.array(points)

    def transform_points_to_global(self, points_local):
        """Transform points from camera frame to global map frame"""
        # Extract current pose
        pos = self.current_pose[:3]
        quat = self.current_pose[3:]

        # Convert quaternion to rotation matrix
        rotation = self.quaternion_to_rotation_matrix(quat)

        # Transform points
        points_global = []
        for point in points_local:
            # Rotate and translate
            rotated_point = rotation @ point
            global_point = rotated_point + pos
            points_global.append(global_point)

        return np.array(points_global)

    def quaternion_to_rotation_matrix(self, quat):
        """Convert quaternion to rotation matrix"""
        x, y, z, w = quat

        r11 = 1 - 2*(y*y + z*z)
        r12 = 2*(x*y - w*z)
        r13 = 2*(x*z + w*y)

        r21 = 2*(x*y + w*z)
        r22 = 1 - 2*(x*x + z*z)
        r23 = 2*(y*z - w*x)

        r31 = 2*(x*z - w*y)
        r32 = 2*(y*z + w*x)
        r33 = 1 - 2*(x*x + y*y)

        return np.array([
            [r11, r12, r13],
            [r21, r22, r23],
            [r31, r32, r33]
        ])

    def add_points_to_map(self, points):
        """Add 3D points to the map with proper filtering"""
        for point in points:
            x, y, z = point

            # Filter points that are too far from robot
            robot_x, robot_y = self.current_pose[0], self.current_pose[1]
            dist_to_robot = math.sqrt((x - robot_x)**2 + (y - robot_y)**2)

            if dist_to_robot > 15:  # Too far away
                continue

            # Add to local map with position as key
            pos_key = (round(x, 2), round(y, 2), round(z, 2))
            self.local_map[pos_key] = {
                'position': point,
                'observations': 1,
                'first_observed': self.frame_count,
                'last_observed': self.frame_count
            }

            # Add to global point list
            self.map_points.append(point)

        # Limit map size for performance
        if len(self.map_points) > self.max_map_points:
            # Keep only recent points
            self.map_points = self.map_points[-self.max_map_points:]

    def should_add_keyframe(self):
        """Determine if we should add a new keyframe"""
        if not self.keyframes:
            return True

        # Add keyframe if enough movement occurred
        last_keyframe_pose = self.keyframes[-1]
        current_pos = self.current_pose[:3]

        distance = np.linalg.norm(current_pos - last_keyframe_pose[:3])

        return distance > 1.0  # Add keyframe every meter

    def add_keyframe(self):
        """Add current pose as a keyframe"""
        self.keyframes.append(self.current_pose.copy())

    def update_map(self):
        """Update the occupancy grid map"""
        start_time = self.get_clock().now()

        # Clear the occupancy grid
        self.occupancy_grid.fill(-1)  # Unknown

        # Convert map points to occupancy grid
        map_shape = self.occupancy_grid.shape
        center_x, center_y = self.map_center

        for point in self.map_points[-1000:]:  # Only recent points for performance
            x, y, z = point

            # Convert world coordinates to grid indices
            grid_x = int((x - (center_x - self.map_size/2)) / self.map_resolution)
            grid_y = int((y - (center_y - self.map_size/2)) / self.map_resolution)

            # Check bounds
            if 0 <= grid_x < map_shape[1] and 0 <= grid_y < map_shape[0]:
                # Mark as occupied (100) if it's a ground point or obstacle
                # For simplicity, mark all points as occupied
                self.occupancy_grid[grid_y, grid_x] = 100

        # Calculate processing time
        end_time = self.get_clock().now()
        processing_time = (end_time - start_time).nanoseconds / 1e6  # ms
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)

        self.frame_count += 1

    def publish_results(self):
        """Publish SLAM results"""
        current_time = self.get_clock().now()

        # Publish occupancy grid map
        map_msg = OccupancyGrid()
        map_msg.header.stamp = current_time.to_msg()
        map_msg.header.frame_id = 'map'
        map_msg.info.resolution = float(self.map_resolution)
        map_msg.info.width = self.occupancy_grid.shape[1]
        map_msg.info.height = self.occupancy_grid.shape[0]
        map_msg.info.origin.position.x = float(self.map_center[0] - self.map_size/2)
        map_msg.info.origin.position.y = float(self.map_center[1] - self.map_size/2)
        map_msg.data = self.occupancy_grid.flatten().tolist()
        self.map_pub.publish(map_msg)

        # Publish odometry (same as input but with updated frame_id)
        odom_msg = Odometry()
        odom_msg.header.stamp = current_time.to_msg()
        odom_msg.header.frame_id = 'map'
        odom_msg.child_frame_id = 'base_link'
        odom_msg.pose.pose.position.x = float(self.current_pose[0])
        odom_msg.pose.pose.position.y = float(self.current_pose[1])
        odom_msg.pose.pose.position.z = float(self.current_pose[2])
        odom_msg.pose.pose.orientation.x = float(self.current_pose[3])
        odom_msg.pose.pose.orientation.y = float(self.current_pose[4])
        odom_msg.pose.pose.orientation.z = float(self.current_pose[5])
        odom_msg.pose.pose.orientation.w = float(self.current_pose[6])
        self.odom_pub.publish(odom_msg)

        # Publish status
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        status_msg = String()
        status_msg.data = f"Frame: {self.frame_count}, MapPoints: {len(self.map_points)}, " \
                         f"Keyframes: {len(self.keyframes)}, MapSize: {len(self.map_points)}, " \
                         f"ProcTime: {avg_processing_time:.1f}ms"
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    slam_node = IsaacROSVSLAMMapper()

    try:
        rclpy.spin(slam_node)
    except KeyboardInterrupt:
        slam_node.get_logger().info("Isaac ROS VSLAM Mapper stopped by user")
    finally:
        slam_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Small Simulation

Let's create a performance evaluation node for the VSLAM system:

```python
# vslam_performance_evaluator.py
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, PointStamped
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String, Float32
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import math
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime

class VSLAMEvaluator(Node):
    """
    Evaluate VSLAM performance by comparing against ground truth
    """
    def __init__(self):
        super().__init__('vslam_evaluator')

        # Publishers
        self.rmse_pub = self.create_publisher(Float32, '/vslam_evaluation/rmse', 10)
        self.ate_pub = self.create_publisher(Float32, '/vslam_evaluation/ate', 10)
        self.rpe_pub = self.create_publisher(Float32, '/vslam_evaluation/rpe', 10)
        self.status_pub = self.create_publisher(String, '/vslam_evaluation/status', 10)
        self.error_markers_pub = self.create_publisher(MarkerArray, '/vslam_evaluation/error_markers', 10)

        # Subscribers
        self.vslam_odom_sub = self.create_subscription(
            Odometry, '/slam/odom', self.vslam_odom_callback, 10)
        self.ground_truth_sub = self.create_subscription(
            Odometry, '/ground_truth/odom', self.ground_truth_callback, 10)

        # Timers
        self.evaluation_timer = self.create_timer(1.0, self.evaluate_performance)

        # Data storage
        self.vslam_trajectory = []
        self.ground_truth_trajectory = []
        self.evaluation_results = {
            'rmse': float('inf'),
            'ate': float('inf'),
            'rpe': float('inf'),
            'trajectory_length': 0,
            'processing_times': []
        }

        # Evaluation parameters
        self.max_trajectory_length = 1000
        self.evaluation_window = 50  # For RPE calculation

        self.get_logger().info("VSLAM Performance Evaluator initialized")

    def vslam_odom_callback(self, msg):
        """Store VSLAM estimated poses"""
        pose = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ])
        self.vslam_trajectory.append({
            'timestamp': msg.header.stamp,
            'pose': pose,
            'position': pose[:3]
        })

        # Limit trajectory size
        if len(self.vslam_trajectory) > self.max_trajectory_length:
            self.vslam_trajectory.pop(0)

    def ground_truth_callback(self, msg):
        """Store ground truth poses"""
        pose = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ])
        self.ground_truth_trajectory.append({
            'timestamp': msg.header.stamp,
            'pose': pose,
            'position': pose[:3]
        })

        # Limit trajectory size
        if len(self.ground_truth_trajectory) > self.max_trajectory_length:
            self.ground_truth_trajectory.pop(0)

    def calculate_rmse(self):
        """Calculate Root Mean Square Error between trajectories"""
        if len(self.vslam_trajectory) == 0 or len(self.ground_truth_trajectory) == 0:
            return float('inf')

        # Align trajectories by time (simplified approach)
        min_len = min(len(self.vslam_trajectory), len(self.ground_truth_trajectory))

        if min_len < 2:
            return float('inf')

        # Calculate position errors
        errors = []
        for i in range(min_len):
            vslam_pos = self.vslam_trajectory[i]['position']
            gt_pos = self.ground_truth_trajectory[i]['position']

            error = np.linalg.norm(vslam_pos - gt_pos)
            errors.append(error)

        if errors:
            rmse = np.sqrt(np.mean(np.array(errors) ** 2))
            return float(rmse)
        else:
            return float('inf')

    def calculate_ate(self):
        """Calculate Absolute Trajectory Error"""
        if len(self.vslam_trajectory) < 2 or len(self.ground_truth_trajectory) < 2:
            return float('inf')

        # Find optimal alignment between trajectories
        min_len = min(len(self.vslam_trajectory), len(self.ground_truth_trajectory))

        if min_len < 2:
            return float('inf')

        vslam_positions = np.array([t['position'] for t in self.vslam_trajectory[:min_len]])
        gt_positions = np.array([t['position'] for t in self.ground_truth_trajectory[:min_len]])

        # Calculate ATE (after alignment, simplified as direct comparison)
        ate = np.mean(np.linalg.norm(vslam_positions - gt_positions, axis=1))
        return float(ate)

    def calculate_rpe(self):
        """Calculate Relative Pose Error"""
        if len(self.vslam_trajectory) < 3 or len(self.ground_truth_trajectory) < 3:
            return float('inf')

        min_len = min(len(self.vslam_trajectory), len(self.ground_truth_trajectory))

        if min_len < 3:
            return float('inf')

        rpe_values = []

        # Calculate RPE for consecutive pairs
        for i in range(min_len - 1):
            # VSLAM relative motion
            vslam_pos1 = self.vslam_trajectory[i]['position']
            vslam_pos2 = self.vslam_trajectory[i + 1]['position']
            vslam_rel = vslam_pos2 - vslam_pos1

            # Ground truth relative motion
            gt_pos1 = self.ground_truth_trajectory[i]['position']
            gt_pos2 = self.ground_truth_trajectory[i + 1]['position']
            gt_rel = gt_pos2 - gt_pos1

            # Calculate relative error
            rel_error = np.linalg.norm(vslam_rel - gt_rel)
            rpe_values.append(rel_error)

        if rpe_values:
            rpe = np.mean(rpe_values)
            return float(rpe)
        else:
            return float('inf')

    def create_error_markers(self):
        """Create visualization markers for errors"""
        marker_array = MarkerArray()

        # Position error markers
        for i in range(min(len(self.vslam_trajectory), len(self.ground_truth_trajectory))):
            if i % 10 == 0:  # Only show every 10th error for performance
                vslam_pos = self.vslam_trajectory[i]['position']
                gt_pos = self.ground_truth_trajectory[i]['position']

                # Line marker showing error
                error_marker = Marker()
                error_marker.header.frame_id = "map"
                error_marker.header.stamp = self.get_clock().now().to_msg()
                error_marker.ns = "vslam_errors"
                error_marker.id = i
                error_marker.type = Marker.LINE_STRIP
                error_marker.action = Marker.ADD
                error_marker.scale.x = 0.02
                error_marker.color.r = 1.0
                error_marker.color.a = 0.8

                # Add start (GT) and end (VSLAM) points
                start_point = Point()
                start_point.x = float(gt_pos[0])
                start_point.y = float(gt_pos[1])
                start_point.z = float(gt_pos[2])
                error_marker.points.append(start_point)

                end_point = Point()
                end_point.x = float(vslam_pos[0])
                end_point.y = float(vslam_pos[1])
                end_point.z = float(vslam_pos[2])
                error_marker.points.append(end_point)

                marker_array.markers.append(error_marker)

        return marker_array

    def evaluate_performance(self):
        """Run comprehensive performance evaluation"""
        if len(self.vslam_trajectory) == 0 or len(self.ground_truth_trajectory) == 0:
            return

        # Calculate metrics
        rmse = self.calculate_rmse()
        ate = self.calculate_ate()
        rpe = self.calculate_rpe()

        # Store results
        self.evaluation_results['rmse'] = rmse
        self.evaluation_results['ate'] = ate
        self.evaluation_results['rpe'] = rpe
        self.evaluation_results['trajectory_length'] = min(len(self.vslam_trajectory),
                                                          len(self.ground_truth_trajectory))

        # Publish metrics
        rmse_msg = Float32()
        rmse_msg.data = rmse
        self.rmse_pub.publish(rmse_msg)

        ate_msg = Float32()
        ate_msg.data = ate
        self.ate_pub.publish(ate_msg)

        rpe_msg = Float32()
        rpe_msg.data = rpe
        self.rpe_pub.publish(rpe_msg)

        # Publish error visualization
        error_markers = self.create_error_markers()
        self.error_markers_pub.publish(error_markers)

        # Publish status
        status_msg = String()
        status_msg.data = f"RMSE: {rmse:.3f}m, ATE: {ate:.3f}m, RPE: {rpe:.3f}m, " \
                         f"TrajLen: {self.evaluation_results['trajectory_length']}"
        self.status_pub.publish(status_msg)

        # Log evaluation results
        self.get_logger().info(f"VSLAM Evaluation - RMSE: {rmse:.3f}m, ATE: {ate:.3f}m, RPE: {rpe:.3f}m")

def main(args=None):
    rclpy.init(args=args)
    evaluator = VSLAMEvaluator()

    try:
        rclpy.spin(evaluator)
    except KeyboardInterrupt:
        evaluator.get_logger().info("VSLAM Performance Evaluator stopped by user")

        # Print final evaluation summary
        results = evaluator.evaluation_results
        print("\n" + "="*50)
        print("VSLAM Performance Evaluation Summary")
        print("="*50)
        print(f"Root Mean Square Error (RMSE): {results['rmse']:.3f} m")
        print(f"Absolute Trajectory Error (ATE): {results['ate']:.3f} m")
        print(f"Relative Pose Error (RPE): {results['rpe']:.3f} m")
        print(f"Trajectory Length: {results['trajectory_length']} poses")
        print("="*50)

    finally:
        evaluator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Quick Recap

In this lesson, we've covered:

- **Isaac ROS Architecture**: Understanding the components of Isaac ROS VSLAM packages
- **Visual Odometry**: Implementing stereo visual odometry with feature tracking and motion estimation
- **Mapping Integration**: Combining localization with map building capabilities
- **Performance Evaluation**: Techniques for assessing VSLAM accuracy and robustness
- **GPU Acceleration**: Leveraging Isaac ROS's optimized algorithms for real-time performance

Isaac ROS provides a powerful framework for implementing sophisticated VSLAM systems that can run efficiently on robotic platforms. The integration of GPU-accelerated computer vision algorithms with the ROS 2 ecosystem enables the deployment of advanced perception systems that were previously computationally prohibitive on mobile robots.

In the next lesson, we'll explore Nav2-based path planning for humanoid robots, focusing on how to adapt traditional navigation approaches for robots with complex kinematics and dynamics.

## Summary of Chapter 4 So Far

In Chapter 4: "The AI-Robot Brain (NVIDIA Isaac)", we've covered:

1. **Introduction to NVIDIA Isaac Sim**: Core concepts and capabilities of the Isaac Sim platform
2. **Synthetic Data Generation**: Creating diverse, labeled datasets for AI training with domain randomization
3. **Isaac ROS for VSLAM**: Implementing visual SLAM systems with Isaac ROS packages

These lessons provide the foundation for understanding how NVIDIA's AI tools can accelerate robotics development, from simulation to perception and mapping.