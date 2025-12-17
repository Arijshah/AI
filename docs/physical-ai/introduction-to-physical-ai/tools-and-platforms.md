---
sidebar_label: 'Tools and Platforms'
title: 'Tools and Platforms'
---

# Tools and Platforms

## Overview

Developing Physical AI systems requires specialized tools and platforms that bridge the gap between digital algorithms and physical reality. These tools range from simulation environments that allow safe testing, to robotics frameworks that handle low-level control, to hardware platforms that provide tangible interfaces for experimentation.

This lesson introduces the essential tools and platforms for Physical AI development, providing hands-on experience with popular frameworks and practical examples you can run yourself.

## Learning Outcomes

By the end of this lesson, you will be able to:
- Identify key simulation and development platforms for Physical AI
- Set up basic Physical AI development environments
- Implement simple Physical AI algorithms using popular frameworks
- Compare different tools based on their strengths and applications

## Hands-on Steps

1. **Set up Gazebo Simulation Environment**: Install and configure Gazebo for robot simulation
2. **Explore PyBullet Physics Engine**: Learn to use PyBullet for physics simulation
3. **Implement a Simple Robot Controller**: Create a controller using ROS/ROS2 concepts
4. **Test with Real Hardware Concepts**: Understand how to transition from simulation to real robots

### Prerequisites

- Basic Python programming knowledge
- Understanding of robotics concepts (from previous lessons)
- Access to a computer capable of running physics simulations

## Code Examples

Let's start by exploring PyBullet, a powerful physics engine that's excellent for Physical AI experimentation:

```python
import pybullet as p
import pybullet_data
import time
import numpy as np

def setup_pybullet_environment():
    """
    Set up a basic PyBullet environment for Physical AI experiments
    """
    # Connect to physics server
    physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version

    # Set gravity
    p.setGravity(0, 0, -9.81)

    # Load plane
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    planeId = p.loadURDF("plane.urdf")

    # Load a simple robot (KUKA LBR iiwa)
    startPos = [0, 0, 1]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    robotId = p.loadURDF("kuka_iiwa/model.urdf", startPos, startOrientation)

    # Get joint information
    numJoints = p.getNumJoints(robotId)
    print(f"Robot has {numJoints} joints")

    for i in range(numJoints):
        jointInfo = p.getJointInfo(robotId, i)
        print(f"Joint {i}: {jointInfo[1].decode('utf-8')}, Type: {jointInfo[2]}")

    return physicsClient, robotId

def simple_robot_control(robotId):
    """
    Simple control loop to move the robot arm
    """
    # Enable torque control for the relevant joints
    controlJoints = [1, 2, 3, 4, 5, 6, 7]  # Joint indices for the robot arm

    # Set initial joint positions
    initialPositions = [0, 0, 0, 0, 0, 0, 0]

    # Move to initial position
    for i, jointIndex in enumerate(controlJoints):
        p.setJointMotorControl2(
            bodyIndex=robotId,
            jointIndex=jointIndex,
            controlMode=p.POSITION_CONTROL,
            targetPosition=initialPositions[i],
            force=500
        )

    # Run simulation
    for i in range(1000):
        # Simple oscillating motion
        t = i * 0.01  # Time in seconds

        # Create oscillating target positions
        targetPositions = [
            0.5 * np.sin(t),      # Joint 1
            0.3 * np.sin(t*0.7),  # Joint 2
            0.4 * np.sin(t*1.3),  # Joint 3
            0.2 * np.sin(t*0.5),  # Joint 4
            0.6 * np.sin(t*1.1),  # Joint 5
            0.3 * np.sin(t*0.9),  # Joint 6
            0.5 * np.sin(t*1.5)   # Joint 7
        ]

        # Apply control
        p.setJointMotorControlArray(
            bodyIndex=robotId,
            jointIndices=controlJoints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=targetPositions,
            forces=[500] * len(controlJoints)
        )

        p.stepSimulation()
        time.sleep(0.01)

    return robotId

def create_simple_obstacle_avoidance(robotId):
    """
    Implement a simple obstacle avoidance behavior
    """
    # Add an obstacle
    obstacleStartPos = [0.5, 0, 0.5]
    obstacleId = p.loadURDF("sphere.urdf", obstacleStartPos, p.getQuaternionFromEuler([0, 0, 0]),
                           globalScaling=0.2)

    # Define a simple path
    path_points = [
        [0, 0, 1],
        [0.5, 0.3, 1.2],
        [1.0, 0, 1.5],
        [1.5, -0.3, 1.2],
        [2.0, 0, 1]
    ]

    # Simple path following with obstacle avoidance
    for step in range(len(path_points) - 1):
        start_point = np.array(path_points[step])
        end_point = np.array(path_points[step + 1])

        # Break path into smaller steps
        for t in np.linspace(0, 1, 50):
            # Interpolate between points
            target_pos = (1 - t) * start_point + t * end_point

            # Check for potential collision with obstacle
            obstacle_pos, _ = p.getBasePositionAndOrientation(obstacleId)
            dist_to_obstacle = np.linalg.norm(np.array(obstacle_pos) - target_pos)

            # If too close to obstacle, modify path
            if dist_to_obstacle < 0.4:
                # Move slightly away from obstacle
                obstacle_vec = np.array(obstacle_pos) - target_pos
                avoidance_vec = obstacle_vec / np.linalg.norm(obstacle_vec) * 0.3
                target_pos += avoidance_vec

            # Move end effector to target position (simplified)
            # In a real implementation, you'd use inverse kinematics here
            p.stepSimulation()
            time.sleep(0.01)

    print("Path following with obstacle avoidance completed")

# Note: To run these examples, you would need to install PyBullet:
# pip install pybullet
#
# Then execute:
# physicsClient, robotId = setup_pybullet_environment()
# simple_robot_control(robotId)
# create_simple_obstacle_avoidance(robotId)
# p.disconnect()
```

Now let's look at another popular platform, the Robot Operating System (ROS), with a simplified example:

```python
# ROS concepts example (this is pseudocode since ROS requires specific setup)
class SimpleROSNode:
    """
    Simplified representation of a ROS node for Physical AI
    Note: This is conceptual - actual ROS requires specific installation
    """
    def __init__(self, node_name):
        self.node_name = node_name
        self.subscribers = {}
        self.publishers = {}
        print(f"Initialized {node_name} node")

    def create_subscriber(self, topic_name, msg_type, callback):
        """Create a subscriber to receive messages"""
        self.subscribers[topic_name] = {
            'type': msg_type,
            'callback': callback
        }
        print(f"Created subscriber to {topic_name}")

    def create_publisher(self, topic_name, msg_type):
        """Create a publisher to send messages"""
        self.publishers[topic_name] = {
            'type': msg_type,
            'queue': []
        }
        print(f"Created publisher to {topic_name}")

    def publish(self, topic_name, message):
        """Publish a message to a topic"""
        if topic_name in self.publishers:
            self.publishers[topic_name]['queue'].append(message)
            print(f"Published to {topic_name}: {message}")

    def spin_once(self):
        """Process one cycle of messages"""
        # Process messages in queue
        for topic_name, pub_data in self.publishers.items():
            while pub_data['queue']:
                msg = pub_data['queue'].pop(0)
                # In real ROS, this would be sent to subscribers
                pass

def ros_concepts_example():
    """
    Demonstrate ROS concepts with pseudocode
    """
    # Create a robot controller node
    controller = SimpleROSNode("robot_controller")

    # Create subscribers for sensor data
    def laser_callback(data):
        print(f"Laser scan received: {len(data)} points")
        # Process laser data for obstacle detection

    def imu_callback(data):
        print(f"IMU data: orientation={data['orientation']}, angular_velocity={data['angular_vel']}")
        # Process IMU data for balance control

    controller.create_subscriber("/scan", "LaserScan", laser_callback)
    controller.create_subscriber("/imu/data", "Imu", imu_callback)

    # Create publishers for control commands
    controller.create_publisher("/cmd_vel", "Twist")
    controller.create_publisher("/joint_commands", "JointState")

    # Simulate a simple control loop
    for i in range(10):
        # Publish velocity command
        cmd_vel = {
            'linear': {'x': 0.5, 'y': 0.0, 'z': 0.0},
            'angular': {'x': 0.0, 'y': 0.0, 'z': 0.1}
        }
        controller.publish("/cmd_vel", cmd_vel)

        # Publish joint commands
        joint_cmd = {
            'positions': [0.1, 0.2, 0.3, 0.4],
            'velocities': [0.0, 0.0, 0.0, 0.0]
        }
        controller.publish("/joint_commands", joint_cmd)

        # Process messages
        controller.spin_once()

        print(f"Control cycle {i+1} completed")

    return controller

# Run ROS concepts example
# ros_concepts_example()
```

## Small Simulation

Let's create a simple simulation environment using Python to demonstrate Physical AI concepts without external dependencies:

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class SimplePhysicalAISim:
    """
    Simple Physical AI simulation environment without external dependencies
    """
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.robots = []
        self.obstacles = []
        self.goals = []

        # Add some random obstacles
        for _ in range(5):
            x = np.random.uniform(1, width-1)
            y = np.random.uniform(1, height-1)
            self.obstacles.append((x, y, 0.5))  # (x, y, radius)

    def add_robot(self, x, y, name="Robot"):
        """Add a robot to the environment"""
        robot = {
            'name': name,
            'x': x,
            'y': y,
            'theta': 0,  # orientation
            'path': [(x, y)]
        }
        self.robots.append(robot)
        return robot

    def add_goal(self, x, y):
        """Add a goal location"""
        self.goals.append((x, y))

    def sense_environment(self, robot_idx):
        """Simple sensor model for a robot"""
        robot = self.robots[robot_idx]

        # Detect obstacles within sensor range
        sensor_range = 2.0
        obstacles_in_range = []

        for obs_x, obs_y, obs_radius in self.obstacles:
            dist = np.sqrt((obs_x - robot['x'])**2 + (obs_y - robot['y'])**2)
            if dist < sensor_range:
                obstacles_in_range.append({
                    'x': obs_x,
                    'y': obs_y,
                    'distance': dist,
                    'angle': np.arctan2(obs_y - robot['y'], obs_x - robot['x']) - robot['theta']
                })

        # Detect goals
        goals_in_range = []
        for goal_x, goal_y in self.goals:
            dist = np.sqrt((goal_x - robot['x'])**2 + (goal_y - robot['y'])**2)
            if dist < sensor_range * 2:  # Goal detection range
                goals_in_range.append({
                    'x': goal_x,
                    'y': goal_y,
                    'distance': dist,
                    'angle': np.arctan2(goal_y - robot['y'], goal_x - robot['x']) - robot['theta']
                })

        return {
            'obstacles': obstacles_in_range,
            'goals': goals_in_range
        }

    def simple_navigation(self, robot_idx):
        """Simple navigation algorithm"""
        robot = self.robots[robot_idx]
        sensors = self.sense_environment(robot_idx)

        # Find closest goal
        if sensors['goals']:
            closest_goal = min(sensors['goals'], key=lambda g: g['distance'])

            # Simple obstacle avoidance
            avoidance_vector = np.array([0.0, 0.0])

            for obs in sensors['obstacles']:
                if obs['distance'] < 1.0:  # Too close to obstacle
                    # Create repulsive force away from obstacle
                    angle_to_robot = obs['angle'] + robot['theta']
                    avoidance_force = (1.0 / obs['distance']) * 0.5
                    avoidance_vector[0] -= avoidance_force * np.cos(angle_to_robot)
                    avoidance_vector[1] -= avoidance_force * np.sin(angle_to_robot)

            # Desired direction toward goal
            goal_direction = np.array([
                np.cos(closest_goal['angle'] + robot['theta']),
                np.sin(closest_goal['angle'] + robot['theta'])
            ])

            # Combine goal seeking with obstacle avoidance
            final_direction = goal_direction + avoidance_vector
            final_direction = final_direction / np.linalg.norm(final_direction) if np.linalg.norm(final_direction) > 0 else goal_direction

            # Move robot
            step_size = 0.1
            robot['x'] += step_size * final_direction[0]
            robot['y'] += step_size * final_direction[1]
            robot['theta'] = np.arctan2(final_direction[1], final_direction[0])

            # Store path
            robot['path'].append((robot['x'], robot['y']))

    def visualize(self):
        """Visualize the environment"""
        fig, ax = plt.subplots(figsize=(10, 10))

        # Draw environment boundaries
        ax.add_patch(plt.Rectangle((0, 0), self.width, self.height, fill=False, linewidth=2))

        # Draw obstacles
        for obs_x, obs_y, obs_radius in self.obstacles:
            circle = plt.Circle((obs_x, obs_y), obs_radius, color='red', alpha=0.5)
            ax.add_patch(circle)

        # Draw goals
        for goal_x, goal_y in self.goals:
            ax.plot(goal_x, goal_y, 'go', markersize=10, label='Goal' if goal_x == self.goals[0][0] else "")

        # Draw robots and their paths
        for i, robot in enumerate(self.robots):
            # Draw path
            path_x, path_y = zip(*robot['path'])
            ax.plot(path_x, path_y, label=f"{robot['name']} Path", linewidth=2)

            # Draw robot position
            ax.plot(robot['x'], robot['y'], 'bo', markersize=8, label=f"{robot['name']}" if i == 0 else "")

            # Draw orientation
            dx = 0.3 * np.cos(robot['theta'])
            dy = 0.3 * np.sin(robot['theta'])
            ax.arrow(robot['x'], robot['y'], dx, dy, head_width=0.1, head_length=0.1,
                    fc='blue', ec='blue')

        ax.set_xlim(-0.5, self.width + 0.5)
        ax.set_ylim(-0.5, self.height + 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('Simple Physical AI Simulation')
        ax.legend()

        plt.show()

    def run_simulation(self, steps=100):
        """Run the simulation for a number of steps"""
        for step in range(steps):
            for i in range(len(self.robots)):
                self.simple_navigation(i)

            # Stop if robot reaches goal
            for robot in self.robots:
                for goal_x, goal_y in self.goals:
                    dist_to_goal = np.sqrt((robot['x'] - goal_x)**2 + (robot['y'] - goal_y)**2)
                    if dist_to_goal < 0.3:  # Reached goal
                        print(f"{robot['name']} reached goal at step {step}")
                        return

def run_physical_ai_simulation():
    """Run the complete Physical AI simulation"""
    # Create simulation environment
    sim = SimplePhysicalAISim(width=10, height=10)

    # Add a robot
    robot = sim.add_robot(1, 1, "ExplorerBot")

    # Add a goal
    sim.add_goal(8, 8)

    # Run simulation
    print("Starting Physical AI simulation...")
    sim.run_simulation(steps=200)

    # Visualize results
    sim.visualize()

    print("Simulation completed!")
    return sim

# Run the simulation
# sim_result = run_physical_ai_simulation()
print("Physical AI simulation environment created!")
print("To run the simulation, uncomment the last lines and ensure matplotlib is installed.")
```

## Quick Recap

In this lesson, we've covered:

- **Simulation Tools**: PyBullet for physics simulation and robot control
- **Development Frameworks**: ROS concepts for building Physical AI systems
- **Hands-on Implementation**: Created a simple simulation environment from scratch
- **Platform Comparison**: Understanding different tools based on their applications

The Physical AI ecosystem offers various tools and platforms, each with specific strengths:

- **PyBullet/Gazebo**: Excellent for physics simulation and testing
- **ROS/ROS2**: Comprehensive robotics framework with extensive libraries
- **Custom Environments**: For specific research needs or learning purposes

Choosing the right platform depends on your specific needs: simulation fidelity, hardware compatibility, community support, and development complexity. For beginners, starting with simulation tools like PyBullet allows for safe experimentation before moving to real hardware.

In the next chapters, we'll explore more advanced Physical AI concepts, including learning algorithms, perception systems, and human-robot interaction.