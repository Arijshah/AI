---
sidebar_label: 'URDF and Humanoid Robot Modeling'
title: 'URDF and Humanoid Robot Modeling'
---

# URDF and Humanoid Robot Modeling

## Overview

URDF (Unified Robot Description Format) is the standard way to describe robot models in ROS 2. It defines the physical and visual properties of a robot, including its links (rigid bodies), joints (connections between links), and other properties like mass, inertia, and visual appearance. For humanoid robots, URDF becomes particularly important as it captures the complex kinematic structure needed for human-like movement.

This lesson covers creating detailed URDF models for humanoid robots and integrating them with ROS 2 for simulation and control.

## Learning Outcomes

By the end of this lesson, you will be able to:
- Create URDF files for simple and complex robots
- Define links, joints, and materials in URDF
- Model humanoid robot kinematics with proper joint limits and dynamics
- Visualize robot models in RViz
- Integrate URDF with robot state publishers for joint visualization

## Hands-on Steps

1. **Basic URDF Creation**: Create a simple robot model with links and joints
2. **Humanoid Structure**: Model a basic humanoid robot with limbs
3. **Visual and Collision Properties**: Add visual and collision elements
4. **Kinematic Chain Setup**: Create proper kinematic chains for limbs
5. **Integration with ROS 2**: Use robot state publisher to visualize the model

### Prerequisites

- Understanding of ROS 2 nodes and topics
- Basic knowledge of 3D coordinate systems and transformations
- Understanding of robot kinematics concepts

## Code Examples

Let's start by creating a basic URDF model for a simple robot:

```xml
<!-- basic_robot.urdf -->
<?xml version="1.0"?>
<robot name="basic_robot">
  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Sensor Mount -->
  <link name="sensor_mount">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Joint connecting base to sensor mount -->
  <joint name="sensor_joint" type="fixed">
    <parent link="base_link"/>
    <child link="sensor_mount"/>
    <origin xyz="0.3 0 0.3" rpy="0 0 0"/>
  </joint>
</robot>
```

Now let's create a more complex humanoid robot model:

```xml
<!-- humanoid_robot.urdf -->
<?xml version="1.0"?>
<robot name="humanoid_robot">
  <!-- Materials -->
  <material name="black">
    <color rgba="0.0 0.0 0.0 1.0"/>
  </material>
  <material name="blue">
    <color rgba="0.0 0.0 0.8 1.0"/>
  </material>
  <material name="green">
    <color rgba="0.0 0.8 0.0 1.0"/>
  </material>
  <material name="grey">
    <color rgba="0.2 0.2 0.2 1.0"/>
  </material>
  <material name="orange">
    <color rgba="1.0 0.423529411765 0.0392156862745 1.0"/>
  </material>
  <material name="brown">
    <color rgba="0.870588235294 0.811764705882 0.764705882353 1.0"/>
  </material>
  <material name="red">
    <color rgba="0.8 0.0 0.0 1.0"/>
  </material>
  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>

  <!-- Base Footprint -->
  <link name="base_footprint">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.001 0.001 0.001"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.001 0.001 0.001"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.0001"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <!-- Base Link (Pelvis) -->
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.2 0.2"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.2 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <inertia ixx="0.116666666667" ixy="0.0" ixz="0.0" iyy="0.166666666667" iyz="0.0" izz="0.166666666667"/>
    </inertial>
  </link>

  <joint name="base_joint" type="fixed">
    <origin xyz="0 0 0.15" rpy="0 0 0"/>
    <parent link="base_footprint"/>
    <child link="base_link"/>
  </joint>

  <!-- Torso -->
  <link name="torso">
    <visual>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <geometry>
        <box size="0.25 0.15 0.6"/>
      </geometry>
      <material name="grey"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <geometry>
        <box size="0.25 0.15 0.6"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="8.0"/>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <inertia ixx="0.313333333333" ixy="0.0" ixz="0.0" iyy="0.341666666667" iyz="0.0" izz="0.0866666666667"/>
    </inertial>
  </link>

  <joint name="torso_joint" type="fixed">
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
    <parent link="base_link"/>
    <child link="torso"/>
  </joint>

  <!-- Head -->
  <link name="head">
    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.12"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.12"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <inertia ixx="0.0192" ixy="0.0" ixz="0.0" iyy="0.0192" iyz="0.0" izz="0.0192"/>
    </inertial>
  </link>

  <joint name="neck_joint" type="revolute">
    <origin xyz="0 0 0.6" rpy="0 0 0"/>
    <parent link="torso"/>
    <child link="head"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="100" velocity="1"/>
  </joint>

  <!-- Left Arm -->
  <link name="left_shoulder">
    <visual>
      <origin xyz="0 0.05 0" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.1" radius="0.05"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0.05 0" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.1" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_shoulder_joint" type="revolute">
    <origin xyz="0.1 0.1 0.3" rpy="0 0 0"/>
    <parent link="torso"/>
    <child link="left_shoulder"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="2"/>
  </joint>

  <link name="left_upper_arm">
    <visual>
      <origin xyz="0 0 -0.15" rpy="1.57 0 0"/>
      <geometry>
        <cylinder length="0.3" radius="0.04"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="1.57 0 0"/>
      <geometry>
        <cylinder length="0.3" radius="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.005625" ixy="0.0" ixz="0.0" iyy="0.005625" iyz="0.0" izz="0.00024"/>
    </inertial>
  </link>

  <joint name="left_elbow_joint" type="revolute">
    <origin xyz="0 0.1 -0.1" rpy="0 0 0"/>
    <parent link="left_shoulder"/>
    <child link="left_upper_arm"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="1.0" effort="50" velocity="2"/>
  </joint>

  <link name="left_forearm">
    <visual>
      <origin xyz="0 0 -0.12" rpy="1.57 0 0"/>
      <geometry>
        <cylinder length="0.24" radius="0.03"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.12" rpy="1.57 0 0"/>
      <geometry>
        <cylinder length="0.24" radius="0.03"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <inertia ixx="0.00384" ixy="0.0" ixz="0.0" iyy="0.00384" iyz="0.0" izz="0.000144"/>
    </inertial>
  </link>

  <joint name="left_wrist_joint" type="revolute">
    <origin xyz="0 0 -0.24" rpy="0 0 0"/>
    <parent link="left_upper_arm"/>
    <child link="left_forearm"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="20" velocity="2"/>
  </joint>

  <!-- Right Arm (mirror of left) -->
  <link name="right_shoulder">
    <visual>
      <origin xyz="0 -0.05 0" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.1" radius="0.05"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 -0.05 0" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.1" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="right_shoulder_joint" type="revolute">
    <origin xyz="0.1 -0.1 0.3" rpy="0 0 0"/>
    <parent link="torso"/>
    <child link="right_shoulder"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="2"/>
  </joint>

  <link name="right_upper_arm">
    <visual>
      <origin xyz="0 0 -0.15" rpy="1.57 0 0"/>
      <geometry>
        <cylinder length="0.3" radius="0.04"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="1.57 0 0"/>
      <geometry>
        <cylinder length="0.3" radius="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.005625" ixy="0.0" ixz="0.0" iyy="0.005625" iyz="0.0" izz="0.00024"/>
    </inertial>
  </link>

  <joint name="right_elbow_joint" type="revolute">
    <origin xyz="0 -0.1 -0.1" rpy="0 0 0"/>
    <parent link="right_shoulder"/>
    <child link="right_upper_arm"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="1.0" effort="50" velocity="2"/>
  </joint>

  <link name="right_forearm">
    <visual>
      <origin xyz="0 0 -0.12" rpy="1.57 0 0"/>
      <geometry>
        <cylinder length="0.24" radius="0.03"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.12" rpy="1.57 0 0"/>
      <geometry>
        <cylinder length="0.24" radius="0.03"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <inertia ixx="0.00384" ixy="0.0" ixz="0.0" iyy="0.00384" iyz="0.0" izz="0.000144"/>
    </inertial>
  </link>

  <joint name="right_wrist_joint" type="revolute">
    <origin xyz="0 0 -0.24" rpy="0 0 0"/>
    <parent link="right_upper_arm"/>
    <child link="right_forearm"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="20" velocity="2"/>
  </joint>

  <!-- Left Leg -->
  <link name="left_hip">
    <visual>
      <origin xyz="0 0.05 0" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.1" radius="0.06"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <origin xyz="0 0.05 0" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.1" radius="0.06"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.0015" ixy="0.0" ixz="0.0" iyy="0.0015" iyz="0.0" izz="0.0018"/>
    </inertial>
  </link>

  <joint name="left_hip_joint" type="revolute">
    <origin xyz="-0.05 0.08 0" rpy="0 0 0"/>
    <parent link="base_link"/>
    <child link="left_hip"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.785" upper="0.785" effort="100" velocity="1"/>
  </joint>

  <link name="left_thigh">
    <visual>
      <origin xyz="0 0 -0.2" rpy="1.57 0 0"/>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="1.57 0 0"/>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.5"/>
      <inertia ixx="0.0333333333333" ixy="0.0" ixz="0.0" iyy="0.0333333333333" iyz="0.0" izz="0.003125"/>
    </inertial>
  </link>

  <joint name="left_knee_joint" type="revolute">
    <origin xyz="0 0.1 -0.1" rpy="0 0 0"/>
    <parent link="left_hip"/>
    <child link="left_thigh"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.0" effort="100" velocity="1"/>
  </joint>

  <link name="left_shin">
    <visual>
      <origin xyz="0 0 -0.2" rpy="1.57 0 0"/>
      <geometry>
        <cylinder length="0.4" radius="0.04"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="1.57 0 0"/>
      <geometry>
        <cylinder length="0.4" radius="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.8"/>
      <inertia ixx="0.024" ixy="0.0" ixz="0.0" iyy="0.024" iyz="0.0" izz="0.00144"/>
    </inertial>
  </link>

  <joint name="left_ankle_joint" type="revolute">
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <parent link="left_thigh"/>
    <child link="left_shin"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.785" upper="0.785" effort="50" velocity="1"/>
  </joint>

  <link name="left_foot">
    <visual>
      <origin xyz="0 0 -0.025" rpy="0 0 0"/>
      <geometry>
        <box size="0.15 0.08 0.05"/>
      </geometry>
      <material name="black"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.025" rpy="0 0 0"/>
      <geometry>
        <box size="0.15 0.08 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.000525" ixy="0.0" ixz="0.0" iyy="0.0010625" iyz="0.0" izz="0.00125"/>
    </inertial>
  </link>

  <joint name="left_foot_joint" type="fixed">
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    <parent link="left_shin"/>
    <child link="left_foot"/>
  </joint>

  <!-- Right Leg (mirror of left) -->
  <link name="right_hip">
    <visual>
      <origin xyz="0 -0.05 0" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.1" radius="0.06"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <origin xyz="0 -0.05 0" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.1" radius="0.06"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.0015" ixy="0.0" ixz="0.0" iyy="0.0015" iyz="0.0" izz="0.0018"/>
    </inertial>
  </link>

  <joint name="right_hip_joint" type="revolute">
    <origin xyz="-0.05 -0.08 0" rpy="0 0 0"/>
    <parent link="base_link"/>
    <child link="right_hip"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.785" upper="0.785" effort="100" velocity="1"/>
  </joint>

  <link name="right_thigh">
    <visual>
      <origin xyz="0 0 -0.2" rpy="1.57 0 0"/>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="1.57 0 0"/>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.5"/>
      <inertia ixx="0.0333333333333" ixy="0.0" ixz="0.0" iyy="0.0333333333333" iyz="0.0" izz="0.003125"/>
    </inertial>
  </link>

  <joint name="right_knee_joint" type="revolute">
    <origin xyz="0 -0.1 -0.1" rpy="0 0 0"/>
    <parent link="right_hip"/>
    <child link="right_thigh"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.0" effort="100" velocity="1"/>
  </joint>

  <link name="right_shin">
    <visual>
      <origin xyz="0 0 -0.2" rpy="1.57 0 0"/>
      <geometry>
        <cylinder length="0.4" radius="0.04"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="1.57 0 0"/>
      <geometry>
        <cylinder length="0.4" radius="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.8"/>
      <inertia ixx="0.024" ixy="0.0" ixz="0.0" iyy="0.024" iyz="0.0" izz="0.00144"/>
    </inertial>
  </link>

  <joint name="right_ankle_joint" type="revolute">
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <parent link="right_thigh"/>
    <child link="right_shin"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.785" upper="0.785" effort="50" velocity="1"/>
  </joint>

  <link name="right_foot">
    <visual>
      <origin xyz="0 0 -0.025" rpy="0 0 0"/>
      <geometry>
        <box size="0.15 0.08 0.05"/>
      </geometry>
      <material name="black"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.025" rpy="0 0 0"/>
      <geometry>
        <box size="0.15 0.08 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.000525" ixy="0.0" ixz="0.0" iyy="0.0010625" iyz="0.0" izz="0.00125"/>
    </inertial>
  </link>

  <joint name="right_foot_joint" type="fixed">
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    <parent link="right_shin"/>
    <child link="right_foot"/>
  </joint>
</robot>
```

Now let's create a ROS 2 node to publish joint states for visualization:

```python
# joint_state_publisher.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import math
import random

class JointStatePublisher(Node):
    """
    Node to publish joint states for URDF visualization
    """
    def __init__(self):
        super().__init__('joint_state_publisher')

        # Create publisher for joint states
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)

        # Timer for publishing joint states
        self.timer = self.create_timer(0.1, self.publish_joint_states)

        # Joint names for our humanoid robot
        self.joint_names = [
            'neck_joint',
            'left_shoulder_joint', 'left_elbow_joint', 'left_wrist_joint',
            'right_shoulder_joint', 'right_elbow_joint', 'right_wrist_joint',
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint'
        ]

        # Initialize joint positions
        self.joint_positions = [0.0] * len(self.joint_names)

        # For animation
        self.time = 0.0

        self.get_logger().info("Joint State Publisher initialized")

    def publish_joint_states(self):
        # Create joint state message
        msg = JointState()
        msg.name = self.joint_names

        # Update time for animation
        self.time += 0.01

        # Animate some joints for demonstration
        self.joint_positions[0] = 0.2 * math.sin(self.time)  # neck
        self.joint_positions[1] = 0.5 * math.sin(self.time * 0.5)  # left shoulder
        self.joint_positions[2] = 0.8 * math.sin(self.time * 0.7)  # left elbow
        self.joint_positions[4] = 0.5 * math.sin(self.time * 0.5 + math.pi)  # right shoulder (opposite)
        self.joint_positions[5] = 0.8 * math.sin(self.time * 0.7 + math.pi)  # right elbow (opposite)

        # Leg movements
        self.joint_positions[7] = 0.2 * math.sin(self.time * 0.3)  # left hip
        self.joint_positions[8] = 1.0 + 0.3 * math.sin(self.time * 0.3 + math.pi/2)  # left knee
        self.joint_positions[10] = 0.2 * math.sin(self.time * 0.3 + math.pi)  # right hip (opposite)
        self.joint_positions[11] = 1.0 + 0.3 * math.sin(self.time * 0.3 + 3*math.pi/2)  # right knee (opposite)

        msg.position = self.joint_positions
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"

        self.joint_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    joint_publisher = JointStatePublisher()

    try:
        rclpy.spin(joint_publisher)
    except KeyboardInterrupt:
        joint_publisher.get_logger().info("Joint State Publisher stopped by user")
    finally:
        joint_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Small Simulation

Let's create a simple inverse kinematics example that demonstrates how to control the humanoid robot arms:

```python
# humanoid_ik_controller.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point
from std_msgs.msg import Header
import math
import numpy as np

class HumanoidIKController(Node):
    """
    Simple inverse kinematics controller for humanoid arms
    """
    def __init__(self):
        super().__init__('humanoid_ik_controller')

        # Publishers
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.target_pub = self.create_publisher(Point, 'target_position', 10)

        # Timers
        self.control_timer = self.create_timer(0.05, self.control_loop)

        # Joint names
        self.joint_names = [
            'left_shoulder_joint', 'left_elbow_joint', 'left_wrist_joint',
            'right_shoulder_joint', 'right_elbow_joint', 'right_wrist_joint'
        ]

        # Initial joint positions (in radians)
        self.joint_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Arm lengths (approximate for our URDF)
        self.upper_arm_length = 0.3
        self.forearm_length = 0.24

        # Target positions (x, y, z in robot coordinate frame)
        self.left_target = np.array([0.3, 0.2, 0.3])  # Left arm target
        self.right_target = np.array([0.3, -0.2, 0.3])  # Right arm target

        # Animation time
        self.time = 0.0

        self.get_logger().info("Humanoid IK Controller initialized")

    def calculate_ik_2dof(self, target_x, target_y, l1, l2):
        """
        Calculate inverse kinematics for 2-DOF arm
        target_x, target_y: desired end effector position
        l1, l2: lengths of upper arm and forearm
        Returns: (shoulder_angle, elbow_angle)
        """
        # Distance from shoulder to target
        dist = math.sqrt(target_x**2 + target_y**2)

        # Check if target is reachable
        if dist > l1 + l2:
            # Target is too far, extend arm fully
            shoulder_angle = math.atan2(target_y, target_x)
            elbow_angle = 0.0
        elif dist < abs(l1 - l2):
            # Target is too close, fold arm
            shoulder_angle = math.atan2(target_y, target_x)
            elbow_angle = math.pi
        else:
            # Calculate elbow angle using law of cosines
            cos_elbow = (l1**2 + l2**2 - dist**2) / (2 * l1 * l2)
            cos_elbow = max(-1, min(1, cos_elbow))  # Clamp to [-1, 1]
            elbow_angle = math.pi - math.acos(cos_elbow)

            # Calculate shoulder angle
            k1 = l1 + l2 * math.cos(elbow_angle)
            k2 = l2 * math.sin(elbow_angle)
            shoulder_angle = math.atan2(target_y, target_x) - math.atan2(k2, k1)

        return shoulder_angle, elbow_angle

    def control_loop(self):
        # Animate targets in circular motion
        self.time += 0.02

        # Move left target in a circle
        self.left_target[0] = 0.3 + 0.1 * math.cos(self.time)
        self.left_target[1] = 0.2 + 0.1 * math.sin(self.time * 1.3)

        # Move right target in a different pattern
        self.right_target[0] = 0.3 + 0.1 * math.cos(self.time * 1.2)
        self.right_target[1] = -0.2 + 0.1 * math.sin(self.time)

        # Calculate IK for left arm (X-Z plane for simplicity)
        left_shoulder, left_elbow = self.calculate_ik_2dof(
            self.left_target[0], self.left_target[2],
            self.upper_arm_length, self.forearm_length
        )

        # Calculate IK for right arm
        right_shoulder, right_elbow = self.calculate_ik_2dof(
            self.right_target[0], self.right_target[2],
            self.upper_arm_length, self.forearm_length
        )

        # Update joint positions
        self.joint_positions[0] = left_shoulder  # left shoulder
        self.joint_positions[1] = left_elbow     # left elbow
        self.joint_positions[2] = 0.0           # left wrist (not used in 2-DOF)
        self.joint_positions[3] = right_shoulder  # right shoulder
        self.joint_positions[4] = right_elbow     # right elbow
        self.joint_positions[5] = 0.0            # right wrist (not used in 2-DOF)

        # Create and publish joint state message
        msg = JointState()
        msg.name = self.joint_names
        msg.position = self.joint_positions
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"

        self.joint_pub.publish(msg)

        # Publish target positions for visualization
        left_target_msg = Point()
        left_target_msg.x = float(self.left_target[0])
        left_target_msg.y = float(self.left_target[1])
        left_target_msg.z = float(self.left_target[2])
        self.target_pub.publish(left_target_msg)

def main(args=None):
    rclpy.init(args=args)
    ik_controller = HumanoidIKController()

    try:
        rclpy.spin(ik_controller)
    except KeyboardInterrupt:
        ik_controller.get_logger().info("Humanoid IK Controller stopped by user")
    finally:
        ik_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Quick Recap

In this lesson, we've covered:

- **URDF Basics**: Creating links, joints, visual elements, and inertial properties
- **Humanoid Modeling**: Building a complex humanoid robot with proper kinematic chains
- **Joint Types**: Understanding different joint types (revolute, fixed) and their parameters
- **Physical Properties**: Adding mass, inertia, and collision properties for realistic simulation
- **Integration**: Connecting URDF models with ROS 2 for visualization and control
- **Inverse Kinematics**: Simple examples of controlling robot limbs through mathematical models

URDF is essential for humanoid robotics as it provides a standardized way to describe the complex kinematic structure of human-like robots. The proper definition of links, joints, and their physical properties is crucial for accurate simulation and control.

In the next lesson, we'll explore practical ROS 2 workflows and best practices for building robust robotic applications.