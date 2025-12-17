---
sidebar_label: 'Physics and Collision Modeling'
title: 'Physics and Collision Modeling'
---

# Physics and Collision Modeling

## Overview

Physics simulation is the cornerstone of creating realistic digital twins for robotic systems. In Gazebo, accurate physics modeling ensures that robots behave similarly in simulation and reality. This lesson explores the fundamental concepts of physics simulation, including mass properties, friction, collision detection, and realistic material interactions. Understanding these concepts is crucial for creating simulations that faithfully represent real-world robot behaviors.

Proper physics modeling allows for accurate testing of robot dynamics, interaction with objects, and validation of control algorithms before deployment on physical hardware.

## Learning Outcomes

By the end of this lesson, you will be able to:
- Configure mass properties and inertial tensors for robot links
- Set up realistic friction and contact properties
- Model collision geometries and materials accurately
- Implement custom physics properties for special objects
- Validate physics models against real-world behavior

## Hands-on Steps

1. **Mass Properties Setup**: Define mass and inertial properties for robot components
2. **Friction Modeling**: Configure friction coefficients for different surfaces
3. **Collision Detection**: Set up collision properties and contact models
4. **Material Properties**: Define realistic material behaviors
5. **Physics Validation**: Compare simulation vs. real-world behavior

### Prerequisites

- Understanding of URDF and SDF formats (from Chapter 2)
- Basic knowledge of physics concepts (mass, friction, collision)
- Experience with Gazebo simulation setup

## Code Examples

Let's start by creating a detailed robot model with proper physics properties:

```xml
<!-- physics_robot.urdf -->
<?xml version="1.0"?>
<robot name="physics_robot">
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

  <!-- Base Link with realistic physics properties -->
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <box size="0.4 0.3 0.2"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <box size="0.4 0.3 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <!-- Realistic mass for a robot base (~10kg) -->
      <mass value="10.0"/>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <!-- Calculated inertia for a box: Ixx=1/12*m*(h²+w²), etc. -->
      <inertia ixx="0.108333" ixy="0.0" ixz="0.0"
               iyy="0.133333" iyz="0.0"
               izz="0.241667"/>
    </inertial>
  </link>

  <!-- Left Wheel with realistic friction -->
  <link name="left_wheel">
    <visual>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <!-- Inertia for a cylinder about its center axis -->
      <inertia ixx="0.0006" ixy="0.0" ixz="0.0"
               iyy="0.0006" iyz="0.0"
               izz="0.0025"/>
    </inertial>
  </link>

  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="0 0.175 0.05" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <!-- Right Wheel -->
  <link name="right_wheel">
    <visual>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.0006" ixy="0.0" ixz="0.0"
               iyy="0.0006" iyz="0.0"
               izz="0.0025"/>
    </inertial>
  </link>

  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="0 -0.175 0.05" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <!-- Manipulator Arm Base -->
  <link name="arm_base">
    <visual>
      <geometry>
        <cylinder length="0.1" radius="0.05"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.1" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.00052" ixy="0.0" ixz="0.0"
               iyy="0.00052" iyz="0.0"
               izz="0.00025"/>
    </inertial>
  </link>

  <joint name="arm_base_joint" type="revolute">
    <parent link="base_link"/>
    <child link="arm_base"/>
    <origin xyz="0.15 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="10.0" velocity="2.0"/>
  </joint>

  <!-- Upper Arm -->
  <link name="upper_arm">
    <visual>
      <origin xyz="0 0 0.15" rpy="1.5708 0 0"/>
      <geometry>
        <cylinder length="0.3" radius="0.03"/>
      </geometry>
      <material name="green"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.15" rpy="1.5708 0 0"/>
      <geometry>
        <cylinder length="0.3" radius="0.03"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <inertia ixx="0.00608" ixy="0.0" ixz="0.0"
               iyy="0.00608" iyz="0.0"
               izz="0.00036"/>
    </inertial>
  </link>

  <joint name="shoulder_joint" type="revolute">
    <parent link="arm_base"/>
    <child link="upper_arm"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="15.0" velocity="1.5"/>
  </joint>

  <!-- Gazebo-specific physics properties -->
  <gazebo reference="base_link">
    <material>Gazebo/White</material>
    <!-- Buoyancy if needed -->
    <gravity>1</gravity>
    <mu1>0.3</mu1>  <!-- Friction coefficient -->
    <mu2>0.3</mu2>  <!-- Secondary friction coefficient -->
    <kp>1000000.0</kp>  <!-- Contact stiffness -->
    <kd>100.0</kd>     <!-- Contact damping -->
    <max_vel>10.0</max_vel>
    <min_depth>0.001</min_depth>
  </gazebo>

  <gazebo reference="left_wheel">
    <material>Gazebo/Blue</material>
    <!-- Higher friction for wheels to grip the ground -->
    <mu1>1.0</mu1>
    <mu2>1.0</mu2>
    <kp>1000000.0</kp>
    <kd>100.0</kd>
    <!-- Wheel-specific properties -->
    <fdir1>1 0 0</fdir1>  <!-- Direction of wheel friction -->
  </gazebo>

  <gazebo reference="right_wheel">
    <material>Gazebo/Blue</material>
    <mu1>1.0</mu1>
    <mu2>1.0</mu2>
    <kp>1000000.0</kp>
    <kd>100.0</kd>
    <fdir1>1 0 0</fdir1>
  </gazebo>

  <gazebo reference="arm_base">
    <material>Gazebo/Red</material>
    <mu1>0.8</mu1>
    <mu2>0.8</mu2>
    <kp>1000000.0</kp>
    <kd>100.0</kd>
  </gazebo>

  <gazebo reference="upper_arm">
    <material>Gazebo/Green</material>
    <mu1>0.6</mu1>
    <mu2>0.6</mu2>
    <kp>1000000.0</kp>
    <kd>100.0</kd>
  </gazebo>

  <!-- Differential drive plugin -->
  <gazebo>
    <plugin name="differential_drive" filename="libgazebo_ros_diff_drive.so">
      <ros>
        <namespace>/physics_robot</namespace>
        <remapping>cmd_vel:=cmd_vel</remapping>
        <remapping>odom:=odom</remapping>
      </ros>
      <left_joint>left_wheel_joint</left_joint>
      <right_joint>right_wheel_joint</right_joint>
      <wheel_separation>0.35</wheel_separation>
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

  <!-- Joint state publisher -->
  <gazebo>
    <plugin name="joint_state_publisher" filename="libgazebo_ros_joint_state_publisher.so">
      <ros>
        <namespace>/physics_robot</namespace>
        <remapping>joint_states:=joint_states</remapping>
      </ros>
      <update_rate>30</update_rate>
      <joint_name>left_wheel_joint</joint_name>
      <joint_name>right_wheel_joint</joint_name>
      <joint_name>arm_base_joint</joint_name>
      <joint_name>shoulder_joint</joint_name>
    </plugin>
  </gazebo>
</robot>
```

Now let's create a complex world with different physics properties for various surfaces:

```xml
<!-- physics_world.world -->
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="physics_world">
    <!-- Include the default sun -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Add ground plane with specific physics properties -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Modify ground plane physics properties -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <surface>
            <friction>
              <ode>
                <mu>0.8</mu>  <!-- High friction for good grip -->
                <mu2>0.8</mu2>
                <fdir1>0 0 1</fdir1>
                <slip1>0.0</slip1>
                <slip2>0.0</slip2>
              </ode>
            </friction>
            <bounce>
              <restitution_coefficient>0.1</restitution_coefficient>  <!-- Low bounce -->
              <threshold>100000</threshold>
            </bounce>
            <contact>
              <ode>
                <soft_cfm>0</soft_cfm>
                <soft_erp>0.2</soft_erp>
                <kp>1e+13</kp>
                <kd>1</kd>
                <max_vel>100.0</max_vel>
                <min_depth>0.001</min_depth>
              </ode>
            </contact>
          </surface>
        </collision>
        <visual name="visual">
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
            <specular>0.01 0.01 0.01 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <!-- Add a low-friction surface (ice) -->
    <model name="ice_surface">
      <pose>3 0 0.01 0 0 0</pose>
      <link name="ice_link">
        <collision name="ice_collision">
          <geometry>
            <box>
              <size>2 2 0.02</size>
            </box>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>0.1</mu>  <!-- Very low friction -->
                <mu2>0.1</mu2>
              </ode>
            </friction>
          </surface>
        </collision>
        <visual name="ice_visual">
          <geometry>
            <box>
              <size>2 2 0.02</size>
            </box>
          </geometry>
          <material>
            <ambient>0.9 0.9 1.0 0.8</ambient>
            <diffuse>0.9 0.9 1.0 0.8</diffuse>
            <specular>0.5 0.5 0.8 0.8</specular>
          </material>
        </visual>
        <static>true</static>
      </link>
    </model>

    <!-- Add a high-friction surface (rubber mat) -->
    <model name="rubber_mat">
      <pose>-3 0 0.01 0 0 0</pose>
      <link name="rubber_link">
        <collision name="rubber_collision">
          <geometry>
            <box>
              <size>2 2 0.02</size>
            </box>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>1.5</mu>  <!-- High friction -->
                <mu2>1.5</mu2>
              </ode>
            </friction>
          </surface>
        </collision>
        <visual name="rubber_visual">
          <geometry>
            <box>
              <size>2 2 0.02</size>
            </box>
          </geometry>
          <material>
            <ambient>0.2 0.2 0.2 1</ambient>
            <diffuse>0.2 0.2 0.2 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
        <static>true</static>
      </link>
    </model>

    <!-- Add various objects with different physics properties -->

    <!-- Light box (easy to push) -->
    <model name="light_box">
      <pose>1 1 0.2 0 0 0</pose>
      <link name="box_link">
        <collision name="box_collision">
          <geometry>
            <box>
              <size>0.2 0.2 0.2</size>
            </box>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>0.5</mu>
                <mu2>0.5</mu2>
              </ode>
            </friction>
          </surface>
        </collision>
        <visual name="box_visual">
          <geometry>
            <box>
              <size>0.2 0.2 0.2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.2 1</ambient>
            <diffuse>0.8 0.8 0.2 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>0.5</mass>  <!-- Light mass -->
          <inertia>
            <ixx>0.003333</ixx>
            <ixy>0.0</ixy>
            <ixz>0.0</ixz>
            <iyy>0.003333</iyy>
            <iyz>0.0</iyz>
            <izz>0.003333</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <!-- Heavy cylinder (hard to move) -->
    <model name="heavy_cylinder">
      <pose>-1 1 0.3 0 0 0</pose>
      <link name="cylinder_link">
        <collision name="cylinder_collision">
          <geometry>
            <cylinder>
              <radius>0.15</radius>
              <length>0.3</length>
            </cylinder>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>0.8</mu>
                <mu2>0.8</mu2>
              </ode>
            </friction>
          </surface>
        </collision>
        <visual name="cylinder_visual">
          <geometry>
            <cylinder>
              <radius>0.15</radius>
              <length>0.3</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0.8 0.2 0.2 1</ambient>
            <diffuse>0.8 0.2 0.2 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>5.0</mass>  <!-- Heavy mass -->
          <inertia>
            <ixx>0.140625</ixx>
            <ixy>0.0</ixy>
            <ixz>0.0</ixz>
            <iyy>0.140625</iyy>
            <iyz>0.0</iyz>
            <izz>0.1125</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <!-- Bouncy ball -->
    <model name="bouncy_ball">
      <pose>0 -1 1 0 0 0</pose>
      <link name="ball_link">
        <collision name="ball_collision">
          <geometry>
            <sphere>
              <radius>0.1</radius>
            </sphere>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>0.3</mu>
                <mu2>0.3</mu2>
              </ode>
            </friction>
            <bounce>
              <restitution_coefficient>0.8</restitution_coefficient>  <!-- High bounce -->
              <threshold>100000</threshold>
            </bounce>
          </surface>
        </collision>
        <visual name="ball_visual">
          <geometry>
            <sphere>
              <radius>0.1</radius>
            </sphere>
          </geometry>
          <material>
            <ambient>0.2 0.8 0.2 1</ambient>
            <diffuse>0.2 0.8 0.2 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>0.3</mass>
          <inertia>
            <ixx>0.0006</ixx>
            <ixy>0.0</ixy>
            <ixz>0.0</ixz>
            <iyy>0.0006</iyy>
            <iyz>0.0</iyz>
            <izz>0.0006</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <!-- Physics engine parameters -->
    <physics name="default_physics" default="true" type="ode">
      <max_step_size>0.001</max_step_size>  <!-- Small step size for accuracy -->
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
      <ode>
        <solver>
          <type>quick</type>
          <iters>10</iters>
          <sor>1.3</sor>
        </solver>
        <constraints>
          <cfm>0.0</cfm>
          <erp>0.2</erp>
          <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
    </physics>
  </world>
</sdf>
```

Now let's create a ROS 2 node to test physics interactions:

```python
# physics_interactor.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Wrench, Point
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ContactsState
from gazebo_msgs.srv import ApplyBodyWrench, GetModelState
import math
import time

class PhysicsInteractor(Node):
    """
    Node to interact with physics simulation and test collision detection
    """
    def __init__(self):
        super().__init__('physics_interactor')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/physics_robot/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/physics_status', 10)

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, '/physics_robot/joint_states', self.joint_state_callback, 10)

        self.contact_sub = self.create_subscription(
            ContactsState, '/gazebo/contact', self.contact_callback, 10)

        # Services
        self.apply_wrench_cli = self.create_client(ApplyBodyWrench, '/gazebo/apply_body_wrench')
        self.get_model_state_cli = self.create_client(GetModelState, '/gazebo/get_model_state')

        # Timers
        self.interaction_timer = self.create_timer(0.1, self.interaction_loop)

        # Robot state
        self.joint_positions = {}
        self.joint_velocities = {}
        self.contact_states = []
        self.robot_position = Point()
        self.robot_orientation = Point()

        # Test parameters
        self.test_phase = 0
        self.test_start_time = self.get_clock().now()
        self.test_sequence = [
            self.test_basic_movement,
            self.test_surface_interaction,
            self.test_object_manipulation,
            self.test_collision_response
        ]

        self.get_logger().info("Physics Interactor initialized")

    def joint_state_callback(self, msg):
        """Process joint state data"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.joint_velocities[name] = msg.velocity[i]

    def contact_callback(self, msg):
        """Process contact sensor data"""
        self.contact_states = msg.states

    def apply_force_to_body(self, body_name, force_x, force_y, force_z, duration=1.0):
        """Apply a force to a specific body in the simulation"""
        if not self.apply_wrench_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Service /gazebo/apply_body_wrench not available')
            return False

        req = ApplyBodyWrench.Request()
        req.body_name = body_name
        req.wrench.force.x = force_x
        req.wrench.force.y = force_y
        req.wrench.force.z = force_z
        req.duration.sec = int(duration)
        req.duration.nanosec = int((duration - int(duration)) * 1e9)

        future = self.apply_wrench_cli.call_async(req)
        return future

    def get_robot_state(self):
        """Get the current state of the robot model"""
        if not self.get_model_state_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Service /gazebo/get_model_state not available')
            return None

        req = GetModelState.Request()
        req.model_name = 'physics_robot'
        req.relative_entity_name = 'world'

        future = self.get_model_state_cli.call_async(req)
        return future

    def test_basic_movement(self):
        """Test basic robot movement on normal surface"""
        cmd = Twist()
        cmd.linear.x = 0.3  # Move forward
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)

        self.get_logger().info("Test 1: Moving robot forward on normal surface")
        return True

    def test_surface_interaction(self):
        """Test robot movement on different friction surfaces"""
        cmd = Twist()
        cmd.linear.x = 0.3
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)

        self.get_logger().info("Test 2: Moving robot across different surfaces (normal, ice, rubber)")
        return True

    def test_object_manipulation(self):
        """Test interaction with objects of different masses"""
        cmd = Twist()
        cmd.linear.x = 0.2
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)

        # Apply a small force to the light box to see if it moves
        self.apply_force_to_body('light_box::box_link', 5.0, 0.0, 0.0, 0.5)

        self.get_logger().info("Test 3: Interacting with light and heavy objects")
        return True

    def test_collision_response(self):
        """Test collision detection and response"""
        cmd = Twist()
        cmd.linear.x = 0.1  # Slow movement to observe collisions
        cmd.angular.z = 0.2  # Gentle rotation
        self.cmd_vel_pub.publish(cmd)

        self.get_logger().info("Test 4: Observing collision responses")
        return True

    def interaction_loop(self):
        """Main interaction loop that runs different physics tests"""
        current_time = self.get_clock().now()
        elapsed = (current_time - self.test_start_time).nanoseconds / 1e9

        # Run tests in sequence
        if elapsed > 5.0 and self.test_phase < len(self.test_sequence):
            self.test_sequence[self.test_phase]()
            self.test_phase += 1
            self.test_start_time = current_time

        # Reset test sequence after all tests are done
        if self.test_phase >= len(self.test_sequence):
            self.test_phase = 0
            self.test_start_time = current_time

        # Publish status
        status_msg = String()
        status_msg.data = f"Test Phase: {self.test_phase}, Elapsed: {elapsed:.1f}s, " \
                         f"Contacts: {len(self.contact_states)}, Joints: {len(self.joint_positions)}"
        self.status_pub.publish(status_msg)

        # Log contact information
        if self.contact_states:
            for contact in self.contact_states:
                if len(contact.collision1_name) > 0 and len(contact.collision2_name) > 0:
                    self.get_logger().info(f"Contact: {contact.collision1_name} vs {contact.collision2_name}")

def main(args=None):
    rclpy.init(args=args)
    interactor = PhysicsInteractor()

    try:
        rclpy.spin(interactor)
    except KeyboardInterrupt:
        interactor.get_logger().info("Physics Interactor stopped by user")
    finally:
        interactor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Small Simulation

Let's create a physics validation tool that compares simulation behavior with expected real-world physics:

```python
# physics_validator.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float32, String
import numpy as np
import math
import time

class PhysicsValidator(Node):
    """
    Validates physics simulation against expected real-world behavior
    """
    def __init__(self):
        super().__init__('physics_validator')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/physics_robot/cmd_vel', 10)
        self.validation_pub = self.create_publisher(String, '/physics_validation', 10)

        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, '/physics_robot/odom', self.odom_callback, 10)
        self.joint_sub = self.create_subscription(JointState, '/physics_robot/joint_states', self.joint_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/physics_robot/imu', self.imu_callback, 10)

        # Timers
        self.validation_timer = self.create_timer(0.1, self.validation_loop)

        # Robot state tracking
        self.position_history = []
        self.velocity_history = []
        self.acceleration_history = []
        self.joint_positions = {}
        self.joint_velocities = {}
        self.linear_velocity = Vector3()
        self.angular_velocity = Vector3()
        self.orientation = Vector3()

        # Validation parameters
        self.start_time = self.get_clock().now()
        self.last_position = None
        self.last_velocity = None
        self.validation_results = {
            'friction_coefficient': None,
            'momentum_conservation': None,
            'energy_loss': None,
            'collision_response': None
        }

        # Test parameters
        self.test_stage = 0
        self.test_start_time = self.get_clock().now()
        self.test_commands = [
            {'linear': 0.3, 'angular': 0.0, 'duration': 3.0},  # Move straight
            {'linear': 0.0, 'angular': 0.5, 'duration': 2.0},  # Rotate
            {'linear': 0.2, 'angular': 0.0, 'duration': 3.0},  # Move again
            {'linear': 0.0, 'angular': 0.0, 'duration': 2.0}   # Stop
        ]
        self.current_command = 0

        self.get_logger().info("Physics Validator initialized")

    def odom_callback(self, msg):
        """Process odometry data for validation"""
        current_pos = msg.pose.pose.position
        current_vel = msg.twist.twist.linear

        # Store position and velocity for analysis
        self.position_history.append((current_pos.x, current_pos.y, current_pos.z))
        self.velocity_history.append((current_vel.x, current_vel.y, current_vel.z))

        # Calculate acceleration if we have previous velocity
        if self.last_velocity is not None:
            dt = 0.1  # Assuming 10Hz update rate
            ax = (current_vel.x - self.last_velocity.x) / dt
            ay = (current_vel.y - self.last_velocity.y) / dt
            az = (current_vel.z - self.last_velocity.z) / dt
            self.acceleration_history.append((ax, ay, az))

        self.last_velocity = current_vel

    def joint_callback(self, msg):
        """Process joint state data"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.joint_velocities[name] = msg.velocity[i]

    def imu_callback(self, msg):
        """Process IMU data for validation"""
        self.linear_velocity = msg.linear_acceleration
        self.angular_velocity = msg.angular_velocity
        self.orientation = msg.orientation

    def calculate_physics_metrics(self):
        """Calculate various physics validation metrics"""
        if len(self.position_history) < 10:
            return  # Need enough data points

        # Calculate average velocity over recent history
        recent_positions = self.position_history[-10:]
        dx = recent_positions[-1][0] - recent_positions[0][0]
        dy = recent_positions[-1][1] - recent_positions[0][1]
        dt = len(recent_positions) * 0.1  # 10 samples at 10Hz
        avg_velocity = math.sqrt(dx*dx + dy*dy) / dt if dt > 0 else 0

        # Calculate velocity from wheel joints (for comparison)
        if 'left_wheel_joint' in self.joint_velocities and 'right_wheel_joint' in self.joint_velocities:
            left_vel = self.joint_velocities['left_wheel_joint']
            right_vel = self.joint_velocities['right_wheel_joint']
            wheel_radius = 0.1  # From URDF
            expected_vel = (left_vel + right_vel) * wheel_radius / 2.0

        # Calculate energy loss (should be minimal in ideal conditions)
        if len(self.velocity_history) >= 2:
            current_speed = math.sqrt(
                self.velocity_history[-1][0]**2 +
                self.velocity_history[-1][1]**2
            )
            initial_speed = math.sqrt(
                self.velocity_history[0][0]**2 +
                self.velocity_history[0][1]**2
            )
            energy_loss = abs(initial_speed - current_speed) / initial_speed if initial_speed > 0 else 0

        return {
            'avg_velocity': avg_velocity,
            'expected_velocity': expected_vel if 'expected_vel' in locals() else 0,
            'energy_loss': energy_loss,
            'position_history_length': len(self.position_history)
        }

    def validation_loop(self):
        """Main validation loop"""
        # Execute test commands in sequence
        current_time = self.get_clock().now()
        elapsed = (current_time - self.test_start_time).nanoseconds / 1e9

        if self.current_command < len(self.test_commands):
            command = self.test_commands[self.current_command]

            if elapsed < command['duration']:
                # Execute current command
                cmd = Twist()
                cmd.linear.x = command['linear']
                cmd.angular.z = command['angular']
                self.cmd_vel_pub.publish(cmd)
            else:
                # Move to next command
                self.current_command += 1
                self.test_start_time = current_time

        # Calculate physics metrics
        metrics = self.calculate_physics_metrics()

        if metrics:
            # Validate physics behavior
            validation_passed = True
            validation_details = []

            # Check if velocity is reasonable (not too high or too low)
            if metrics['avg_velocity'] > 2.0:  # Too fast for our robot
                validation_passed = False
                validation_details.append(f"Velocity too high: {metrics['avg_velocity']:.2f}")
            elif metrics['avg_velocity'] < 0.01 and self.current_command < len(self.test_commands) - 1:
                validation_passed = False
                validation_details.append(f"Velocity too low: {metrics['avg_velocity']:.2f}")

            # Check energy conservation (should be mostly conserved in frictionless environment)
            if metrics['energy_loss'] > 0.3:  # More than 30% energy loss is suspicious
                validation_details.append(f"High energy loss: {metrics['energy_loss']:.2f}")

            # Publish validation results
            validation_msg = String()
            validation_msg.data = f"Valid: {validation_passed}, Avg Vel: {metrics['avg_velocity']:.2f}, " \
                                 f"Energy Loss: {metrics['energy_loss']:.2f}, " \
                                 f"Details: {'; '.join(validation_details) if validation_details else 'OK'}"
            self.validation_pub.publish(validation_msg)

            if not validation_passed:
                self.get_logger().warn(f"Physics validation issue: {'; '.join(validation_details)}")

    def get_validation_report(self):
        """Generate a comprehensive physics validation report"""
        report = "Physics Validation Report\n"
        report += "=" * 30 + "\n"

        metrics = self.calculate_physics_metrics()
        if metrics:
            report += f"Average Velocity: {metrics['avg_velocity']:.3f} m/s\n"
            report += f"Expected Velocity: {metrics['expected_velocity']:.3f} m/s\n"
            report += f"Energy Loss: {metrics['energy_loss']:.3f}\n"
            report += f"Position Samples: {metrics['position_history_length']}\n"

        report += f"Total Runtime: {(self.get_clock().now() - self.start_time).nanoseconds / 1e9:.1f} s\n"
        report += f"Current Test Stage: {self.current_command}/{len(self.test_commands)}\n"

        return report

def main(args=None):
    rclpy.init(args=args)
    validator = PhysicsValidator()

    try:
        rclpy.spin(validator)
    except KeyboardInterrupt:
        validator.get_logger().info("Physics Validator stopped by user")
        print("\n" + validator.get_validation_report())
    finally:
        validator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Quick Recap

In this lesson, we've covered:

- **Mass Properties**: Setting realistic mass and inertial tensors for robot components
- **Friction Modeling**: Configuring friction coefficients for different surfaces and materials
- **Collision Detection**: Setting up proper collision geometries and contact properties
- **Physics Validation**: Comparing simulation behavior with expected real-world physics
- **Material Properties**: Defining realistic material behaviors for different surfaces

Proper physics modeling is essential for creating believable digital twins. The physics properties we've explored - mass, friction, collision detection, and material interactions - determine how accurately our simulated robots will behave compared to their real-world counterparts.

In the next lesson, we'll explore Unity-based visualization and how it can complement Gazebo's physics simulation for enhanced robot development and debugging.