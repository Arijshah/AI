---
sidebar_position: 4
---

# Capstone Project: Autonomous Humanoid Robot in Simulation

## Overview
This capstone lesson integrates all the concepts from the Vision-Language-Action (VLA) chapter into a complete autonomous humanoid robot system. We'll create a robot that can understand natural language commands, plan complex tasks using LLMs, perceive its environment, and execute sophisticated behaviors in simulation. This project demonstrates the full VLA pipeline in action.

## Learning Outcomes
By the end of this lesson, you will:
- Integrate VLA components into a complete autonomous system
- Implement a multi-modal perception system combining vision, language, and action
- Create a humanoid robot controller that responds to voice commands
- Design a cognitive architecture for autonomous decision-making
- Validate the system in simulation with complex scenarios
- Evaluate the performance of the integrated VLA system

## Hands-on Steps

### Step 1: Set up the Capstone Project Structure
First, let's create the main package for our capstone project:

```bash
# Create the capstone project package
cd ~/ros2_ws/src
ros2 pkg create --dependencies rclpy std_msgs sensor_msgs geometry_msgs vision_msgs nav_msgs moveit_msgs --node-name autonomous_humanoid humanoid_capstone

cd humanoid_capstone
mkdir -p humanoid_capstone/{perception,planning,control,navigation,utils}
```

### Step 2: Create the Main Autonomous Humanoid Node
Create the central node that orchestrates all VLA components:

```python
# humanoid_capstone/humanoid_capstone/autonomous_humanoid_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from humanoid_capstone.perception.vision_system import VisionSystem
from humanoid_capstone.planning.cognitive_planner import CognitivePlanner
from humanoid_capstone.control.motion_controller import MotionController
from humanoid_capstone.navigation.path_planner import PathPlanner
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

class AutonomousHumanoidNode(Node):
    def __init__(self):
        super().__init__('autonomous_humanoid')

        # Initialize all subsystems
        self.vision_system = VisionSystem(self)
        self.cognitive_planner = CognitivePlanner(self)
        self.motion_controller = MotionController(self)
        self.path_planner = PathPlanner(self)

        # Publishers
        self.status_pub = self.create_publisher(String, 'system_status', 10)
        self.behavior_pub = self.create_publisher(String, 'behavior_command', 10)
        self.navigation_goal_pub = self.create_publisher(PoseStamped, 'navigation_goal', 10)

        # Subscribers
        self.voice_command_sub = self.create_subscription(
            String,
            'voice_command',
            self.voice_command_callback,
            10
        )

        self.vision_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.vision_callback,
            10
        )

        self.laser_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.laser_callback,
            10
        )

        # Internal state
        self.current_task = None
        self.robot_state = {
            'position': {'x': 0.0, 'y': 0.0, 'theta': 0.0},
            'battery': 1.0,
            'detected_objects': [],
            'navigation_status': 'idle',
            'current_behavior': 'idle'
        }

        # Executor for async operations
        self.executor_pool = ThreadPoolExecutor(max_workers=8)
        self.asyncio_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.asyncio_loop)

        # Timer for system status updates
        self.status_timer = self.create_timer(1.0, self.publish_system_status)

        self.get_logger().info('Autonomous Humanoid Node initialized')

    def voice_command_callback(self, msg):
        """Handle voice commands from the voice-to-action system"""
        command_text = msg.data
        self.get_logger().info(f'Received voice command: {command_text}')

        # Process command asynchronously
        future = asyncio.run_coroutine_threadsafe(
            self.process_voice_command(command_text),
            self.asyncio_loop
        )

    async def process_voice_command(self, command_text: str):
        """Process a voice command through the full VLA pipeline"""
        try:
            self.get_logger().info(f'Processing voice command: {command_text}')

            # Update system status
            self.update_system_status('processing_command', command_text)

            # Use cognitive planner to generate a plan
            plan = await self.cognitive_planner.plan_task(command_text, self.robot_state)

            if plan:
                self.get_logger().info(f'Generated plan with {len(plan)} steps')
                self.current_task = plan

                # Execute the plan
                success = await self.execute_plan(plan)

                if success:
                    self.get_logger().info('Task completed successfully')
                    self.update_system_status('task_completed', command_text)
                else:
                    self.get_logger().error('Task execution failed')
                    self.update_system_status('task_failed', command_text)
            else:
                self.get_logger().error('Failed to generate plan')
                self.update_system_status('planning_failed', command_text)

        except Exception as e:
            self.get_logger().error(f'Error processing voice command: {str(e)}')
            self.update_system_status('error', str(e))

    async def execute_plan(self, plan: list) -> bool:
        """Execute a plan step by step"""
        self.update_system_status('executing_plan', f'Plan with {len(plan)} steps')

        for i, step in enumerate(plan):
            self.get_logger().info(f'Executing step {i+1}/{len(plan)}: {step["action"]}')

            try:
                success = await self.execute_plan_step(step)

                if success:
                    self.get_logger().info(f'Step {i+1} completed successfully')
                else:
                    self.get_logger().error(f'Step {i+1} failed')
                    return False

            except Exception as e:
                self.get_logger().error(f'Step {i+1} execution error: {str(e)}')
                return False

        return True

    async def execute_plan_step(self, step: dict) -> bool:
        """Execute a single step in the plan"""
        action_type = step.get('action', '')
        parameters = step.get('parameters', {})

        if action_type == 'navigate_to':
            return await self.execute_navigation_step(parameters)
        elif action_type == 'detect_object':
            return await self.execute_detection_step(parameters)
        elif action_type == 'manipulate_object':
            return await self.execute_manipulation_step(parameters)
        elif action_type == 'speak':
            return await self.execute_speech_step(parameters)
        elif action_type == 'wait':
            return await self.execute_wait_step(parameters)
        else:
            self.get_logger().error(f'Unknown action type: {action_type}')
            return False

    async def execute_navigation_step(self, params: dict) -> bool:
        """Execute navigation step"""
        target_x = params.get('x', 0.0)
        target_y = params.get('y', 0.0)

        # Plan path using navigation system
        path = await self.path_planner.plan_path(
            self.robot_state['position']['x'],
            self.robot_state['position']['y'],
            target_x, target_y
        )

        if path:
            # Execute navigation
            success = await self.motion_controller.navigate_to_pose(target_x, target_y)
            if success:
                # Update robot state
                self.robot_state['position']['x'] = target_x
                self.robot_state['position']['y'] = target_y
                return True

        return False

    async def execute_detection_step(self, params: dict) -> bool:
        """Execute object detection step"""
        target_object = params.get('target_object', 'any')

        # Use vision system to detect objects
        detections = await self.vision_system.detect_objects(target_object)

        if detections:
            self.robot_state['detected_objects'] = detections
            self.get_logger().info(f'Detected {len(detections)} objects')
            return True

        return False

    async def execute_manipulation_step(self, params: dict) -> bool:
        """Execute manipulation step"""
        object_name = params.get('object', '')
        action = params.get('manipulation_action', 'pick_up')

        # For now, just log the action
        # In a real implementation, this would control the robot's arms
        self.get_logger().info(f'Manipulation: {action} {object_name}')
        return True

    async def execute_speech_step(self, params: dict) -> bool:
        """Execute speech step"""
        text = params.get('text', '')
        self.get_logger().info(f'Speaking: {text}')
        # In a real implementation, this would use TTS
        return True

    async def execute_wait_step(self, params: dict) -> bool:
        """Execute wait step"""
        duration = params.get('duration', 1.0)
        import time
        time.sleep(duration)  # In async context, use asyncio.sleep
        return True

    def vision_callback(self, msg):
        """Handle incoming vision data"""
        # Process image through vision system
        future = asyncio.run_coroutine_threadsafe(
            self.vision_system.process_image(msg),
            self.asyncio_loop
        )

    def laser_callback(self, msg):
        """Handle incoming laser scan data"""
        # Update navigation safety based on laser data
        self.path_planner.update_obstacles(msg)

    def update_system_status(self, status: str, details: str = ""):
        """Update and publish system status"""
        status_msg = String()
        status_msg.data = json.dumps({
            'status': status,
            'details': details,
            'timestamp': self.get_clock().now().nanoseconds,
            'robot_state': self.robot_state
        })
        self.status_pub.publish(status_msg)

        # Update internal state
        self.robot_state['current_behavior'] = status

    def publish_system_status(self):
        """Publish periodic system status updates"""
        self.update_system_status('operational')

    def destroy_node(self):
        """Clean up resources"""
        self.executor_pool.shutdown(wait=True)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AutonomousHumanoidNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Autonomous Humanoid Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Step 3: Create the Vision System Module
Now let's create the vision system that will handle perception tasks:

```python
# humanoid_capstone/humanoid_capstone/perception/vision_system.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import cv2
import numpy as np
import asyncio
import logging

class VisionSystem:
    def __init__(self, node):
        self.node = node
        self.logger = node.get_logger()
        self.cv_bridge = CvBridge()

        # Initialize vision processing components
        self.object_detector = self._init_object_detector()
        self.pose_estimator = self._init_pose_estimator()

        # Store recent detections
        self.recent_detections = []

    def _init_object_detector(self):
        """Initialize object detection model"""
        # For simulation, we'll use a simple color-based detector
        # In real implementation, this would be YOLO, Detectron2, etc.
        return ColorBasedDetector()

    def _init_pose_estimator(self):
        """Initialize pose estimation"""
        # For simulation, we'll use simple geometric pose estimation
        return GeometricPoseEstimator()

    async def process_image(self, image_msg: Image):
        """Process incoming image message"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

            # Perform object detection
            detections = self.object_detector.detect(cv_image)

            # Estimate poses
            for detection in detections:
                pose = self.pose_estimator.estimate_pose(cv_image, detection)
                detection['pose'] = pose

            # Update recent detections
            self.recent_detections = detections

            return detections

        except Exception as e:
            self.logger.error(f'Error processing image: {str(e)}')
            return []

    async def detect_objects(self, target_object: str = "any"):
        """Detect specific objects in the current view"""
        # In simulation, return predefined objects
        if target_object.lower() == "any":
            return self.recent_detections
        else:
            # Filter detections by target object
            filtered = [obj for obj in self.recent_detections
                       if target_object.lower() in obj.get('class', '').lower()]
            return filtered

    def get_object_location(self, object_name: str):
        """Get the location of a specific object"""
        for detection in self.recent_detections:
            if detection.get('class', '').lower() == object_name.lower():
                return detection.get('pose', {}).get('position', {})
        return None

    def has_clear_path_to_object(self, object_name: str):
        """Check if there's a clear path to the object"""
        # This would check navigation map for obstacles
        # For simulation, assume path is clear if object is detected
        obj_location = self.get_object_location(object_name)
        return obj_location is not None


class ColorBasedDetector:
    """Simple color-based object detector for simulation"""
    def __init__(self):
        # Define color ranges for different objects
        self.color_ranges = {
            'red_cup': ([0, 50, 50], [10, 255, 255]),
            'blue_box': ([100, 50, 50], [130, 255, 255]),
            'green_ball': ([40, 50, 50], [80, 255, 255]),
            'yellow_book': ([20, 50, 50], [30, 255, 255])
        }

    def detect(self, image):
        """Detect objects based on color ranges"""
        detections = []

        for obj_name, (lower, upper) in self.color_ranges.items():
            # Create mask for color range
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > 100:  # Filter small contours
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)

                    detection = {
                        'class': obj_name,
                        'confidence': 0.8,  # Simulation confidence
                        'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                        'center': {'x': x + w/2, 'y': y + h/2}
                    }
                    detections.append(detection)

        return detections


class GeometricPoseEstimator:
    """Simple geometric pose estimator for simulation"""
    def __init__(self):
        self.focal_length = 500  # Simulation focal length
        self.known_object_sizes = {
            'red_cup': 0.1,  # 10cm diameter
            'blue_box': 0.15,  # 15cm side
            'green_ball': 0.08,  # 8cm diameter
            'yellow_book': 0.2  # 20cm length
        }

    def estimate_pose(self, image, detection):
        """Estimate 3D pose from 2D detection"""
        obj_class = detection.get('class', '')
        bbox = detection.get('bbox', {})
        center = detection.get('center', {})

        if obj_class in self.known_object_sizes:
            # Estimate distance based on object size in image
            pixel_size = max(bbox.get('width', 1), bbox.get('height', 1))
            known_size = self.known_object_sizes[obj_class]

            # Simple distance estimation (in simulation units)
            distance = (known_size * self.focal_length) / pixel_size

            # Convert image coordinates to world coordinates
            # This is a simplified simulation model
            world_x = (center['x'] - image.shape[1]/2) * distance / self.focal_length
            world_y = (center['y'] - image.shape[0]/2) * distance / self.focal_length

            pose = {
                'position': {'x': world_x, 'y': world_y, 'z': distance},
                'orientation': {'w': 1.0, 'x': 0.0, 'y': 0.0, 'z': 0.0}
            }

            return pose

        return {'position': {'x': 0, 'y': 0, 'z': 1}, 'orientation': {'w': 1.0, 'x': 0.0, 'y': 0.0, 'z': 0.0}}
```

### Step 4: Create the Cognitive Planner Module
Now let's create the cognitive planner that will use LLMs for high-level reasoning:

```python
# humanoid_capstone/humanoid_capstone/planning/cognitive_planner.py
import openai
import json
from typing import Dict, List, Any, Optional
import asyncio
import logging

class CognitivePlanner:
    def __init__(self, node):
        self.node = node
        self.logger = node.get_logger()

        # Initialize LLM client
        self.client = openai.OpenAI()  # Configure with your API key

        self.system_prompt = """
        You are a cognitive planning assistant for an autonomous humanoid robot. Your task is to decompose high-level human commands into detailed, executable plans that the robot can follow.

        The robot has the following capabilities:
        - Navigation: Move to specific locations (x, y coordinates)
        - Object Detection: Identify and locate objects in the environment
        - Manipulation: Pick up and place objects (simulation only)
        - Speech: Communicate with humans
        - Perception: Understand the environment through vision and sensors

        Available actions:
        1. navigate_to: Move robot to a specific location
           Parameters: x (float), y (float)
        2. detect_object: Look for a specific object
           Parameters: target_object (string)
        3. manipulate_object: Pick up or place an object
           Parameters: object (string), manipulation_action (string: "pick_up", "place")
        4. speak: Say something to the human
           Parameters: text (string)
        5. wait: Wait for a certain duration
           Parameters: duration (float)

        When creating plans:
        1. Break down complex tasks into simple, sequential steps
        2. Consider the robot's current state and environment
        3. Include necessary perception steps before manipulation
        4. Add verification steps after critical actions
        5. Consider safety and feasibility

        Respond in JSON format with the following structure:
        {
            "plan": [
                {
                    "id": 1,
                    "action": "action_name",
                    "description": "What the robot should do",
                    "parameters": {"param1": "value1"},
                    "required_conditions": ["condition1"],
                    "expected_outcomes": ["outcome1"]
                }
            ]
        }
        """

    async def plan_task(self, goal: str, robot_state: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Generate a plan for a given goal using LLM"""
        try:
            # Create detailed prompt with context
            prompt = f"""
            Robot State: {json.dumps(robot_state, indent=2)}

            Human Goal: {goal}

            Please create a detailed plan for the robot to achieve this goal. Consider the current state and environment.
            """

            response = await self.client.chat.completions.create(
                model="gpt-4",  # Use appropriate model
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2048,
                response_format={"type": "json_object"}
            )

            plan_json = response.choices[0].message.content
            plan_data = json.loads(plan_json)

            return plan_data.get('plan', [])

        except Exception as e:
            self.logger.error(f"Error generating plan: {str(e)}")
            return None

    async def refine_plan(self, plan: List[Dict[str, Any]], feedback: str) -> Optional[List[Dict[str, Any]]]:
        """Refine an existing plan based on feedback"""
        try:
            prompt = f"""
            Current Plan: {json.dumps(plan, indent=2)}

            Feedback: {feedback}

            Please refine the plan to address the feedback while maintaining the original goal.
            """

            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2048,
                response_format={"type": "json_object"}
            )

            plan_json = response.choices[0].message.content
            refined_plan = json.loads(plan_json)

            return refined_plan.get('plan', [])

        except Exception as e:
            self.logger.error(f"Error refining plan: {str(e)}")
            return None

    async def validate_plan(self, plan: List[Dict[str, Any]], environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate if a plan is feasible in the current environment"""
        try:
            prompt = f"""
            Plan: {json.dumps(plan, indent=2)}

            Environment State: {json.dumps(environment_state, indent=2)}

            Please evaluate if this plan is feasible and safe. Return a JSON object with:
            {{
                "feasible": true/false,
                "issues": ["issue1", "issue2"],
                "confidence": 0.0-1.0,
                "suggestions": ["suggestion1", "suggestion2"]
            }}
            """

            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1024,
                response_format={"type": "json_object"}
            )

            validation = json.loads(response.choices[0].message.content)
            return validation

        except Exception as e:
            self.logger.error(f"Error validating plan: {str(e)}")
            return {"feasible": False, "issues": [str(e)], "confidence": 0.0, "suggestions": []}

    def analyze_plan_complexity(self, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the complexity and risk of a plan"""
        complexity = {
            "total_steps": len(plan),
            "action_types": set(),
            "estimated_duration": 0.0,
            "risk_level": "low",
            "dependencies": 0
        }

        for step in plan:
            action_type = step.get("action", "unknown")
            complexity["action_types"].add(action_type)

            # Estimate time for each action type
            if action_type == "navigate_to":
                complexity["estimated_duration"] += 10.0  # 10 seconds per navigation
            elif action_type == "detect_object":
                complexity["estimated_duration"] += 5.0
            elif action_type == "manipulate_object":
                complexity["estimated_duration"] += 8.0
            elif action_type == "speak":
                complexity["estimated_duration"] += 2.0
            elif action_type == "wait":
                complexity["estimated_duration"] += step.get("parameters", {}).get("duration", 1.0)

        complexity["action_types"] = list(complexity["action_types"])

        # Determine risk based on plan length and action types
        if len(plan) > 10:
            complexity["risk_level"] = "high"
        elif len(plan) > 5:
            complexity["risk_level"] = "medium"

        return complexity
```

### Step 5: Create the Motion Controller
Now let's create the motion controller for the humanoid robot:

```python
# humanoid_capstone/humanoid_capstone/control/motion_controller.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
from humanoid_capstone.utils.humanoid_kinematics import HumanoidKinematics
import asyncio
import math

class MotionController:
    def __init__(self, node):
        self.node = node
        self.logger = node.get_logger()

        # Initialize publishers for different motion types
        self.cmd_vel_pub = node.create_publisher(Twist, 'cmd_vel', 10)
        self.body_pose_pub = node.create_publisher(PoseStamped, 'body_pose', 10)

        # Initialize humanoid kinematics
        self.kinematics = HumanoidKinematics()

        # Robot state
        self.current_pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        self.is_moving = False

    async def navigate_to_pose(self, target_x: float, target_y: float) -> bool:
        """Navigate to a target pose using simple proportional control"""
        try:
            self.logger.info(f'Navigating to ({target_x}, {target_y})')

            # Simple navigation using proportional control
            distance_threshold = 0.1  # 10cm threshold
            angular_threshold = 0.1   # 0.1 rad threshold

            while True:
                # Calculate distance and angle to target
                dx = target_x - self.current_pose['x']
                dy = target_y - self.current_pose['y']
                distance = math.sqrt(dx*dx + dy*dy)

                target_angle = math.atan2(dy, dx)
                angle_diff = target_angle - self.current_pose['theta']

                # Normalize angle difference
                while angle_diff > math.pi:
                    angle_diff -= 2 * math.pi
                while angle_diff < -math.pi:
                    angle_diff += 2 * math.pi

                # Check if we're close enough
                if distance < distance_threshold and abs(angle_diff) < angular_threshold:
                    # Stop the robot
                    self._stop_robot()
                    self.logger.info('Navigation completed')
                    return True

                # Create velocity command
                cmd = Twist()

                # Proportional control for linear velocity
                if distance > distance_threshold:
                    cmd.linear.x = min(0.5, distance * 0.5)  # Max 0.5 m/s
                else:
                    cmd.linear.x = 0.0

                # Proportional control for angular velocity
                if abs(angle_diff) > angular_threshold:
                    cmd.angular.z = max(-0.5, min(0.5, angle_diff * 1.0))  # Max 0.5 rad/s
                else:
                    cmd.angular.z = 0.0

                # Publish command
                self.cmd_vel_pub.publish(cmd)

                # Update current pose (in simulation, assume perfect odometry)
                self.current_pose['x'] += cmd.linear.x * 0.1 * math.cos(self.current_pose['theta'])
                self.current_pose['y'] += cmd.linear.x * 0.1 * math.sin(self.current_pose['theta'])
                self.current_pose['theta'] += cmd.angular.z * 0.1

                # Small delay
                await asyncio.sleep(0.1)

        except Exception as e:
            self.logger.error(f'Navigation error: {str(e)}')
            self._stop_robot()
            return False

    def _stop_robot(self):
        """Stop all robot motion"""
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)
        self.is_moving = False

    async def execute_gesture(self, gesture_name: str) -> bool:
        """Execute a predefined gesture"""
        try:
            self.logger.info(f'Executing gesture: {gesture_name}')

            # Define gesture patterns
            gestures = {
                'wave': self._execute_wave_gesture,
                'point': self._execute_point_gesture,
                'greet': self._execute_greet_gesture,
                'think': self._execute_think_gesture
            }

            if gesture_name in gestures:
                return await gestures[gesture_name]()
            else:
                self.logger.error(f'Unknown gesture: {gesture_name}')
                return False

        except Exception as e:
            self.logger.error(f'Gesture execution error: {str(e)}')
            return False

    async def _execute_wave_gesture(self) -> bool:
        """Execute waving gesture"""
        # In simulation, just log the action
        self.logger.info('Executing wave gesture')
        await asyncio.sleep(2.0)  # Simulate gesture duration
        return True

    async def _execute_point_gesture(self) -> bool:
        """Execute pointing gesture"""
        self.logger.info('Executing point gesture')
        await asyncio.sleep(1.5)
        return True

    async def _execute_greet_gesture(self) -> bool:
        """Execute greeting gesture"""
        self.logger.info('Executing greet gesture')
        await asyncio.sleep(2.0)
        return True

    async def _execute_think_gesture(self) -> bool:
        """Execute thinking gesture (tilt head, etc.)"""
        self.logger.info('Executing think gesture')
        await asyncio.sleep(1.0)
        return True

    async def balance_control(self) -> bool:
        """Maintain balance for humanoid robot"""
        # Implement balance control algorithms
        # This would involve PID control of joint positions
        # to maintain center of mass over support polygon
        self.logger.info('Balance control active')
        return True

    def update_robot_pose(self, pose_msg: PoseStamped):
        """Update robot's current pose from localization system"""
        self.current_pose['x'] = pose_msg.pose.position.x
        self.current_pose['y'] = pose_msg.pose.position.y
        # Convert quaternion to euler for theta
        quat = pose_msg.pose.orientation
        self.current_pose['theta'] = math.atan2(
            2 * (quat.w * quat.z + quat.x * quat.y),
            1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        )
```

### Step 6: Create the Navigation System
Now let's create the path planning and navigation system:

```python
# humanoid_capstone/humanoid_capstone/navigation/path_planner.py
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
import numpy as np
import heapq
from typing import List, Tuple, Optional
import asyncio

class PathPlanner:
    def __init__(self, node):
        self.node = node
        self.logger = node.get_logger()

        # Initialize navigation map
        self.navigation_map = None
        self.map_resolution = 0.05  # 5cm resolution
        self.map_origin = {'x': 0.0, 'y': 0.0}

        # Store obstacles from laser scans
        self.obstacles = []

    async def plan_path(self, start_x: float, start_y: float, goal_x: float, goal_y: float) -> Optional[List[Tuple[float, float]]]:
        """Plan a path from start to goal using A* algorithm"""
        try:
            self.logger.info(f'Planning path from ({start_x}, {start_y}) to ({goal_x}, {goal_y})')

            # For simulation, create a simple grid-based map
            # In real implementation, this would use the actual navigation map
            grid_size = 100  # 100x100 grid
            grid = self._create_simulation_grid(grid_size)

            # Convert world coordinates to grid coordinates
            start_grid = self._world_to_grid(start_x, start_y, grid_size)
            goal_grid = self._world_to_grid(goal_x, goal_y, grid_size)

            # Check if start and goal are valid
            if not self._is_valid_cell(start_grid[0], start_grid[1], grid) or \
               not self._is_valid_cell(goal_grid[0], goal_grid[1], grid):
                self.logger.error('Start or goal position is invalid')
                return None

            # Run A* pathfinding
            path = self._a_star(grid, start_grid, goal_grid)

            if path:
                # Convert grid path back to world coordinates
                world_path = []
                for grid_x, grid_y in path:
                    world_x, world_y = self._grid_to_world(grid_x, grid_y, grid_size)
                    world_path.append((world_x, world_y))

                self.logger.info(f'Found path with {len(world_path)} waypoints')
                return world_path

        except Exception as e:
            self.logger.error(f'Path planning error: {str(e)}')

        return None

    def _create_simulation_grid(self, size: int) -> np.ndarray:
        """Create a simulation grid with some obstacles"""
        grid = np.zeros((size, size), dtype=np.uint8)

        # Add some static obstacles (for simulation)
        # These represent walls or furniture in the environment
        obstacles = [
            (20, 20, 5, 5),    # x, y, width, height
            (60, 30, 10, 3),
            (40, 70, 3, 10),
            (80, 80, 5, 5)
        ]

        for x, y, w, h in obstacles:
            if x + w < size and y + h < size:
                grid[y:y+h, x:x+w] = 1  # 1 = occupied

        return grid

    def _a_star(self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """A* pathfinding algorithm"""
        rows, cols = grid.shape

        # Directions: up, down, left, right, and diagonals
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

        # Cost for diagonal movement
        diagonal_cost = 1.414

        # Initialize open and closed sets
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            for i, (dx, dy) in enumerate(directions):
                neighbor = (current[0] + dx, current[1] + dy)

                # Check bounds
                if not (0 <= neighbor[0] < cols and 0 <= neighbor[1] < rows):
                    continue

                # Check if cell is walkable
                if grid[neighbor[1], neighbor[0]] == 1:  # Occupied
                    continue

                # Calculate movement cost
                move_cost = diagonal_cost if i >= 4 else 1.0  # Diagonal vs cardinal
                tentative_g_score = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Heuristic function for A* (Euclidean distance)"""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def _is_valid_cell(self, x: int, y: int, grid: np.ndarray) -> bool:
        """Check if a cell is valid (within bounds and not occupied)"""
        rows, cols = grid.shape
        return (0 <= x < cols and 0 <= y < rows and grid[y, x] == 0)

    def _world_to_grid(self, x: float, y: float, grid_size: int) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates"""
        grid_x = int((x - self.map_origin['x']) / self.map_resolution)
        grid_y = int((y - self.map_origin['y']) / self.map_resolution)
        return (max(0, min(grid_size-1, grid_x)), max(0, min(grid_size-1, grid_y)))

    def _grid_to_world(self, grid_x: int, grid_y: int, grid_size: int) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates"""
        world_x = grid_x * self.map_resolution + self.map_origin['x']
        world_y = grid_y * self.map_resolution + self.map_origin['y']
        return (world_x, world_y)

    def update_obstacles(self, laser_scan: LaserScan):
        """Update obstacle map based on laser scan data"""
        try:
            # Process laser scan to detect obstacles
            angle_min = laser_scan.angle_min
            angle_increment = laser_scan.angle_increment

            new_obstacles = []
            for i, range_val in enumerate(laser_scan.ranges):
                if not (laser_scan.range_min <= range_val <= laser_scan.range_max):
                    continue  # Invalid range

                angle = angle_min + i * angle_increment
                x = range_val * math.cos(angle)
                y = range_val * math.sin(angle)

                # Convert to grid coordinates and mark as occupied
                new_obstacles.append((x, y))

            self.obstacles = new_obstacles
            self.logger.debug(f'Updated obstacles from laser scan: {len(new_obstacles)} detected')

        except Exception as e:
            self.logger.error(f'Error processing laser scan: {str(e)}')

    def is_path_clear(self, path: List[Tuple[float, float]]) -> bool:
        """Check if a path is clear of obstacles"""
        # For simulation, assume path is clear if no dynamic obstacles detected
        # In real implementation, this would check the path against the costmap
        return len(self.obstacles) == 0
```

### Step 7: Create the Main Launch File
Now let's create a launch file to bring up the entire system:

```python
# humanoid_capstone/launch/autonomous_humanoid.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Declare launch arguments
    ld = LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'
        ),

        DeclareLaunchArgument(
            'robot_model',
            default_value='humanoid',
            description='Robot model to use'
        )
    ])

    # Autonomous Humanoid Node
    autonomous_humanoid_node = Node(
        package='humanoid_capstone',
        executable='autonomous_humanoid_node',
        name='autonomous_humanoid',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'robot_model': LaunchConfiguration('robot_model')}
        ],
        output='screen'
    )

    # Voice Recognition Node (from previous lesson)
    voice_recognition_node = Node(
        package='voice_to_action',
        executable='voice_recognition_node',
        name='voice_recognition',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        output='screen'
    )

    # Vision Processing Node
    vision_node = Node(
        package='humanoid_capstone',
        executable='vision_processor_node',
        name='vision_processor',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        output='screen'
    )

    # Navigation Node
    navigation_node = Node(
        package='nav2_bringup',
        executable='nav2_launch',
        name='navigation_system',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        output='screen'
    )

    # Add all nodes to launch description
    ld.add_action(autonomous_humanoid_node)
    ld.add_action(voice_recognition_node)
    ld.add_action(vision_node)
    ld.add_action(navigation_node)

    return ld
```

### Step 8: Create a Simulation Test Script
Finally, let's create a comprehensive test script that demonstrates the complete system:

```python
# humanoid_capstone/test/capstone_demo.py
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import time
import asyncio

class CapstoneDemo(Node):
    def __init__(self):
        super().__init__('capstone_demo')

        # Publisher for voice commands
        self.voice_cmd_pub = self.create_publisher(String, 'voice_command', 10)

        # Publisher for system status requests
        self.status_req_pub = self.create_publisher(String, 'status_request', 10)

        self.demo_commands = [
            "Navigate to the kitchen and find the red cup",
            "Pick up the blue box and place it on the table",
            "Go to the living room and wait there",
            "Find the green ball and bring it to me"
        ]

        self.get_logger().info('Capstone Demo Node initialized')

    def run_demo_scenario(self):
        """Run a complete demo scenario"""
        self.get_logger().info('Starting Capstone Demo Scenario')

        for i, command in enumerate(self.demo_commands):
            self.get_logger().info(f'Executing demo command {i+1}: {command}')

            # Publish voice command
            cmd_msg = String()
            cmd_msg.data = command
            self.voice_cmd_pub.publish(cmd_msg)

            # Wait for execution to complete
            time.sleep(15)  # Wait 15 seconds per command (simulation time)

            self.get_logger().info(f'Completed command {i+1}')

        self.get_logger().info('Capstone Demo completed successfully!')

    def demonstrate_vla_integration(self):
        """Demonstrate the full VLA pipeline"""
        self.get_logger().info('Demonstrating VLA Integration...')

        # 1. Vision: The robot perceives its environment
        self.get_logger().info('1. Vision System Active - Detecting objects and environment')

        # 2. Language: Processing natural language command
        command = "Please go to the kitchen, find the red cup, and bring it to the table"
        self.get_logger().info(f'2. Language Understanding - Processing: "{command}"')

        # Publish the command
        cmd_msg = String()
        cmd_msg.data = command
        self.voice_cmd_pub.publish(cmd_msg)

        # 3. Action: Executing the plan
        self.get_logger().info('3. Action Execution - Planning and executing task...')

        # Wait for execution
        time.sleep(20)

        self.get_logger().info('VLA Integration demonstration complete!')

def main(args=None):
    rclpy.init(args=args)
    demo = CapstoneDemo()

    # Run the demo
    demo.demonstrate_vla_integration()
    demo.run_demo_scenario()

    # Keep node alive briefly to see results
    time.sleep(5)

    demo.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Code Examples

Here's a complete example of how to integrate the VLA system with a humanoid robot in simulation:

```python
# Example: Complete VLA system integration
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from humanoid_capstone.autonomous_humanoid_node import AutonomousHumanoidNode

class VLASystemDemo(Node):
    def __init__(self):
        super().__init__('vla_system_demo')

        # Initialize the complete VLA system
        self.vla_system = AutonomousHumanoidNode()

        # Publisher for high-level commands
        self.command_pub = self.create_publisher(String, 'high_level_goal', 10)

        self.get_logger().info('VLA System Demo initialized')

    def demonstrate_complex_task(self):
        """Demonstrate a complex multi-step task"""
        complex_goal = "I need you to go to the kitchen, find the red cup on the counter, pick it up, then go to the living room and place it on the coffee table. After that, come back and tell me the task is complete."

        self.get_logger().info(f'Executing complex task: {complex_goal}')

        cmd_msg = String()
        cmd_msg.data = complex_goal
        self.command_pub.publish(cmd_msg)

    def evaluate_system_performance(self):
        """Evaluate the performance of the VLA system"""
        # This would include metrics like:
        # - Task completion rate
        # - Planning time
        # - Execution accuracy
        # - Natural language understanding accuracy
        # - System response time
        pass

def run_vla_evaluation():
    """Run comprehensive evaluation of the VLA system"""
    rclpy.init()
    demo = VLASystemDemo()

    try:
        demo.demonstrate_complex_task()
        rclpy.spin_once(demo, timeout_sec=30.0)  # Let it run for 30 seconds
    except KeyboardInterrupt:
        pass
    finally:
        demo.destroy_node()
        rclpy.shutdown()
```

## Small Simulation

Let's create a simple simulation to demonstrate the complete VLA system:

```python
# humanoid_capstone/simulations/vla_simulation.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import asyncio
import time
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class RobotState:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    battery: float = 1.0
    holding_object: str = None

class VLASimulation:
    def __init__(self):
        self.robot = RobotState()
        self.objects = {
            'red_cup': {'x': 5.0, 'y': 3.0, 'picked_up': False},
            'blue_box': {'x': -2.0, 'y': 4.0, 'picked_up': False},
            'green_ball': {'x': 1.0, 'y': -1.0, 'picked_up': False}
        }
        self.path_history = [(0, 0)]
        self.current_task = None
        self.task_log = []

        # Setup visualization
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.robot_plot = self.ax.plot([], [], 'ro', markersize=15, label='Robot')[0]
        self.path_plot = self.ax.plot([], [], 'b-', alpha=0.7, label='Path')[0]
        self.objects_plots = {}

        # Set up the environment
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_xlabel('X Position (m)')
        self.ax.set_ylabel('Y Position (m)')
        self.ax.set_title('VLA System Simulation: Vision-Language-Action Integration')
        self.ax.grid(True)
        self.ax.legend()

        # Plot objects
        for obj_name, obj_data in self.objects.items():
            color = 'red' if 'red' in obj_name else 'blue' if 'blue' in obj_name else 'green'
            plot = self.ax.plot([obj_data['x']], [obj_data['y']], f'{color}o', markersize=10, label=obj_name)[0]
            self.objects_plots[obj_name] = plot

    def process_command(self, command: str):
        """Process a natural language command"""
        self.task_log.append(f"Processing: {command}")
        print(f"VLA System: Processing command - {command}")

        # Simple command parsing for simulation
        if 'kitchen' in command.lower() and 'red cup' in command.lower():
            self.current_task = {'action': 'fetch_object', 'target': 'red_cup', 'destination': (6, 3)}
        elif 'living room' in command.lower() and 'place' in command.lower():
            self.current_task = {'action': 'place_object', 'destination': (8, 1)}
        elif 'find' in command.lower():
            target = 'red_cup' if 'red cup' in command.lower() else 'blue_box' if 'blue box' in command.lower() else 'green_ball'
            self.current_task = {'action': 'navigate_to', 'target': target}

    def execute_task(self):
        """Execute the current task"""
        if not self.current_task:
            return

        action = self.current_task['action']

        if action == 'navigate_to':
            target_obj = self.current_task['target']
            target_x = self.objects[target_obj]['x']
            target_y = self.objects[target_obj]['y']

            # Move towards target
            dx = target_x - self.robot.x
            dy = target_y - self.robot.y
            distance = np.sqrt(dx*dx + dy*dy)

            if distance > 0.5:  # Not close enough
                self.robot.x += 0.1 * dx / distance
                self.robot.y += 0.1 * dy / distance
                self.path_history.append((self.robot.x, self.robot.y))
            else:
                self.task_log.append(f"Reached {target_obj}")
                print(f"VLA System: Reached {target_obj}")
                self.current_task = None  # Task completed

        elif action == 'fetch_object':
            target_obj = self.current_task['target']
            if self.is_close_to_object(target_obj):
                self.objects[target_obj]['picked_up'] = True
                self.robot.holding_object = target_obj
                self.task_log.append(f"Fetched {target_obj}")
                print(f"VLA System: Fetched {target_obj}")
                # Move to next part of task - go to destination
                dest = self.current_task['destination']
                self.current_task = {'action': 'navigate_to', 'target': dest}

    def is_close_to_object(self, obj_name: str) -> bool:
        """Check if robot is close to an object"""
        obj_x = self.objects[obj_name]['x']
        obj_y = self.objects[obj_name]['y']
        distance = np.sqrt((obj_x - self.robot.x)**2 + (obj_y - self.robot.y)**2)
        return distance < 1.0

    def animate(self, frame):
        """Animation function"""
        # Process commands periodically
        if frame == 10:
            self.process_command("Go find the red cup in the kitchen")
        elif frame == 50:
            self.process_command("Now place it on the table in the living room")

        # Execute current task
        self.execute_task()

        # Update robot position
        self.robot_plot.set_data([self.robot.x], [self.robot.y])
        x_vals, y_vals = zip(*self.path_history)
        self.path_plot.set_data(x_vals, y_vals)

        # Update object positions (mark picked up objects)
        for obj_name, plot in self.objects_plots.items():
            if self.objects[obj_name]['picked_up']:
                # Move picked up object with robot
                plot.set_data([self.robot.x + 0.3], [self.robot.y + 0.3])
            else:
                plot.set_data([self.objects[obj_name]['x']], [self.objects[obj_name]['y']])

        # Add task log as text on plot
        self.ax.text(0.02, 0.98, '\n'.join(self.task_log[-3:]),
                    transform=self.ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Add robot orientation indicator
        arrow_length = 0.5
        arrow_dx = arrow_length * np.cos(self.robot.theta)
        arrow_dy = arrow_length * np.sin(self.robot.theta)
        self.ax.arrow(self.robot.x, self.robot.y, arrow_dx, arrow_dy,
                     head_width=0.1, head_length=0.1, fc='red', ec='red')

        return [self.robot_plot, self.path_plot] + list(self.objects_plots.values())

    def run_simulation(self):
        """Run the VLA system simulation"""
        print("Starting VLA System Simulation...")
        print("The robot will process natural language commands and execute tasks")
        print("Green = Objects, Red = Robot, Blue = Path")

        ani = FuncAnimation(self.fig, self.animate, frames=200, interval=100, blit=False)
        plt.show()

if __name__ == "__main__":
    sim = VLASimulation()
    sim.run_simulation()
```

## Quick Recap
In this capstone lesson, we've created a complete autonomous humanoid robot system that integrates all the Vision-Language-Action (VLA) components:

1. **Vision System**: Object detection and pose estimation using computer vision
2. **Language Understanding**: LLM-based cognitive planning and task decomposition
3. **Action Execution**: Motion control, navigation, and manipulation capabilities
4. **Integration**: A central node that orchestrates all components in a unified system

The system can understand natural language commands, plan complex multi-step tasks using LLMs, perceive its environment through vision, and execute sophisticated behaviors in simulation. This demonstrates the full potential of VLA integration for autonomous humanoid robots.

The capstone project showcases how all the individual components from this chapter work together to create an intelligent, autonomous robotic system capable of complex interactions with its environment based on natural language instructions.