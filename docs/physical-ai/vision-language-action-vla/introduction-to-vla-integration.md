---
sidebar_label: 'Introduction to Vision-Language-Action Integration'
title: 'Introduction to Vision-Language-Action Integration'
---

# Introduction to Vision-Language-Action Integration

## Overview

Vision-Language-Action (VLA) represents the cutting edge of AI robotics, where visual perception, natural language understanding, and physical action are seamlessly integrated. This paradigm enables robots to understand and execute complex commands expressed in natural language, perceive their environment visually, and perform appropriate physical actions. VLA systems are crucial for creating intuitive human-robot interaction and enabling robots to operate effectively in human-centered environments.

The VLA framework combines large language models (LLMs) for understanding commands, computer vision for environmental perception, and robotics control for executing actions. This integration allows robots to interpret high-level, ambiguous human instructions and translate them into specific, executable robot behaviors.

## Learning Outcomes

By the end of this lesson, you will be able to:
- Understand the architecture and components of VLA systems
- Explain the integration points between vision, language, and action systems
- Design cognitive planning pipelines that map language to robot actions
- Implement basic VLA system components using modern AI tools
- Evaluate the effectiveness of VLA systems for robot control

## Hands-on Steps

1. **VLA Architecture Setup**: Design the system architecture connecting vision, language, and action components
2. **Perception Pipeline**: Create visual perception for environmental understanding
3. **Language Processing**: Implement natural language understanding for commands
4. **Action Mapping**: Connect language understanding to robot action execution
5. **Integration Testing**: Test the complete VLA pipeline with robot commands

### Prerequisites

- Understanding of ROS 2 concepts and robotics frameworks
- Knowledge of large language models and their applications
- Basic understanding of computer vision concepts
- Experience with Python and AI/ML frameworks

## Code Examples

Let's start by creating the core architecture for a VLA system:

```python
# vla_architecture.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist, Pose, Point
from std_msgs.msg import String
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np
import json
import asyncio
import openai
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class VLACommand:
    """Data class for VLA commands"""
    text: str
    visual_context: Optional[np.ndarray] = None
    robot_state: Dict = None
    action_sequence: List[str] = None

@dataclass
class VLAActionResult:
    """Data class for VLA action results"""
    success: bool
    executed_actions: List[str]
    reasoning_trace: List[str]
    confidence: float

class VLAArchitecture(Node):
    """
    Vision-Language-Action architecture node
    Integrates perception, language understanding, and action execution
    """
    def __init__(self):
        super().__init__('vla_architecture')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/robot/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/vla/status', 10)
        self.result_pub = self.create_publisher(String, '/vla/result', 10)

        # Subscribers
        self.voice_cmd_sub = self.create_subscription(String, '/voice/command', self.voice_command_callback, 10)
        self.camera_sub = self.create_subscription(Image, '/camera/rgb/image_raw', self.camera_callback, 10)
        self.robot_state_sub = self.create_subscription(String, '/robot/state', self.robot_state_callback, 10)

        # Internal components
        self.cv_bridge = CvBridge()
        self.current_image = None
        self.current_robot_state = {}
        self.command_history = []

        # VLA components
        self.perception_module = VisionPerceptionModule(self)
        self.language_module = LanguageUnderstandingModule(self)
        self.action_module = ActionExecutionModule(self)

        # VLA processing timer
        self.vla_timer = self.create_timer(0.1, self.process_vla_cycle)

        self.get_logger().info("VLA Architecture initialized")

    def voice_command_callback(self, msg):
        """Process voice commands"""
        self.get_logger().info(f"Received voice command: {msg.data}")

        # Create VLA command with current visual context
        vla_cmd = VLACommand(
            text=msg.data,
            visual_context=self.current_image,
            robot_state=self.current_robot_state.copy()
        )

        # Add to processing queue
        self.command_history.append(vla_cmd)

    def camera_callback(self, msg):
        """Process camera images for visual context"""
        try:
            self.current_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")

    def robot_state_callback(self, msg):
        """Update robot state"""
        try:
            self.current_robot_state = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error("Error parsing robot state")

    def process_vla_cycle(self):
        """Main VLA processing cycle"""
        if not self.command_history:
            return

        # Process the oldest command
        cmd = self.command_history[0]

        # Step 1: Visual perception
        visual_info = self.perception_module.analyze_scene(cmd.visual_context)

        # Step 2: Language understanding with visual context
        action_sequence = self.language_module.parse_command_with_context(
            cmd.text,
            visual_info,
            cmd.robot_state
        )

        # Step 3: Action execution
        result = self.action_module.execute_action_sequence(action_sequence)

        # Publish results
        result_msg = String()
        result_msg.data = json.dumps({
            'command': cmd.text,
            'actions': action_sequence,
            'success': result.success,
            'confidence': result.confidence,
            'reasoning': result.reasoning_trace
        })
        self.result_pub.publish(result_msg)

        status_msg = String()
        status_msg.data = f"Processed: '{cmd.text}', Actions: {len(action_sequence)}, Success: {result.success}"
        self.status_pub.publish(status_msg)

        self.get_logger().info(f"VLA Result: {status_msg.data}")

        # Remove processed command
        self.command_history.pop(0)

class VisionPerceptionModule:
    """
    Vision module for scene understanding
    """
    def __init__(self, parent_node):
        self.node = parent_node
        # In a real implementation, this would connect to a vision model (e.g., CLIP, DETR, etc.)

    def analyze_scene(self, image: np.ndarray) -> Dict:
        """
        Analyze the current scene to extract relevant information
        In a real implementation, this would use computer vision models
        """
        if image is None:
            return {'objects': [], 'spatial_relations': [], 'environment': 'unknown'}

        # Simulated scene analysis
        scene_info = {
            'objects': [
                {'name': 'table', 'position': [1.0, 0.0, 0.0], 'size': 'medium'},
                {'name': 'chair', 'position': [1.5, -0.5, 0.0], 'size': 'medium'},
                {'name': 'cup', 'position': [1.2, 0.2, 0.8], 'size': 'small'}
            ],
            'spatial_relations': [
                'cup is on table',
                'chair is near table'
            ],
            'environment': 'indoor',
            'navigation_clear': True  # Whether path is clear
        }

        self.node.get_logger().info(f"Scene analysis: {len(scene_info['objects'])} objects detected")
        return scene_info

class LanguageUnderstandingModule:
    """
    Language understanding module for command parsing
    """
    def __init__(self, parent_node):
        self.node = parent_node
        # In a real implementation, this would connect to an LLM (e.g., GPT, Claude, etc.)
        self.action_mapping = {
            'move to': ['navigate_to_location'],
            'go to': ['navigate_to_location'],
            'pick up': ['approach_object', 'grasp_object'],
            'grasp': ['approach_object', 'grasp_object'],
            'take': ['approach_object', 'grasp_object'],
            'place': ['navigate_to_location', 'release_object'],
            'put': ['navigate_to_location', 'release_object'],
            'find': ['scan_environment', 'identify_object'],
            'look for': ['scan_environment', 'identify_object'],
            'bring': ['approach_object', 'grasp_object', 'navigate_to_location', 'release_object']
        }

    def parse_command_with_context(self, command: str, visual_info: Dict, robot_state: Dict) -> List[str]:
        """
        Parse natural language command with visual and robot state context
        """
        command_lower = command.lower()
        actions = []

        # Map command to actions based on keywords
        for keyword, mapped_actions in self.action_mapping.items():
            if keyword in command_lower:
                actions.extend(mapped_actions)

        # Add context-aware refinements
        if 'navigate_to_location' in actions:
            # Extract location from command
            target_location = self.extract_location_from_command(command, visual_info)
            if target_location:
                actions.append(f'navigate_to_{target_location}')

        if 'approach_object' in actions:
            # Extract object from command
            target_object = self.extract_object_from_command(command, visual_info)
            if target_object:
                actions.append(f'approach_{target_object}')

        # Remove duplicates while preserving order
        unique_actions = []
        for action in actions:
            if action not in unique_actions:
                unique_actions.append(action)

        self.node.get_logger().info(f"Parsed command '{command}' to actions: {unique_actions}")
        return unique_actions

    def extract_location_from_command(self, command: str, visual_info: Dict) -> Optional[str]:
        """Extract target location from command using visual context"""
        command_lower = command.lower()

        # Look for location indicators in visual info
        for obj in visual_info.get('objects', []):
            if obj['name'] in command_lower:
                return obj['name']

        # Look for spatial terms
        if 'table' in command_lower:
            return 'table'
        elif 'kitchen' in command_lower or 'counter' in command_lower:
            return 'kitchen'
        elif 'bedroom' in command_lower:
            return 'bedroom'

        return 'default_location'

    def extract_object_from_command(self, command: str, visual_info: Dict) -> Optional[str]:
        """Extract target object from command using visual context"""
        command_lower = command.lower()

        # Look for objects in visual info
        for obj in visual_info.get('objects', []):
            if obj['name'] in command_lower:
                return obj['name']

        # Look for object descriptors
        if 'cup' in command_lower or 'mug' in command_lower:
            return 'cup'
        elif 'book' in command_lower:
            return 'book'
        elif 'ball' in command_lower:
            return 'ball'

        return 'unknown_object'

class ActionExecutionModule:
    """
    Action execution module for robot control
    """
    def __init__(self, parent_node):
        self.node = parent_node
        self.action_functions = {
            'navigate_to_location': self.execute_navigate_to_location,
            'approach_object': self.execute_approach_object,
            'grasp_object': self.execute_grasp_object,
            'release_object': self.execute_release_object,
            'scan_environment': self.execute_scan_environment,
            'identify_object': self.execute_identify_object
        }

    def execute_action_sequence(self, action_sequence: List[str]) -> VLAActionResult:
        """
        Execute a sequence of actions
        """
        executed_actions = []
        reasoning_trace = []
        success = True
        confidence = 0.9  # Initial high confidence

        for action in action_sequence:
            self.node.get_logger().info(f"Executing action: {action}")

            # Extract specific action and parameters
            if '_' in action:
                base_action = action.split('_', 1)[0]
                param = action.split('_', 1)[1]
            else:
                base_action = action
                param = None

            # Execute action if supported
            if base_action in self.action_functions:
                try:
                    action_result = self.action_functions[base_action](param)
                    executed_actions.append(action)
                    reasoning_trace.append(f"Executed {action}: {action_result}")

                    # Update confidence based on action success
                    if not action_result.get('success', True):
                        confidence *= 0.8  # Reduce confidence on failure
                        success = False
                except Exception as e:
                    self.node.get_logger().error(f"Error executing action {action}: {e}")
                    reasoning_trace.append(f"Failed to execute {action}: {str(e)}")
                    success = False
                    confidence *= 0.5  # Significantly reduce confidence on error
            else:
                self.node.get_logger().warn(f"Unknown action: {action}")
                reasoning_trace.append(f"Unknown action: {action}")
                success = False

        return VLAActionResult(
            success=success,
            executed_actions=executed_actions,
            reasoning_trace=reasoning_trace,
            confidence=confidence
        )

    def execute_navigate_to_location(self, location: str) -> Dict:
        """Execute navigation to a specific location"""
        self.node.get_logger().info(f"Navigating to {location}")

        # In a real implementation, this would send navigation commands
        # For simulation, just return success
        return {'success': True, 'location': location, 'time_taken': 0.5}

    def execute_approach_object(self, obj_name: str) -> Dict:
        """Execute approach to a specific object"""
        self.node.get_logger().info(f"Approaching {obj_name}")
        return {'success': True, 'object': obj_name, 'distance': 0.1}

    def execute_grasp_object(self, obj_name: str = None) -> Dict:
        """Execute grasping action"""
        self.node.get_logger().info(f"Grasping {'object' if obj_name is None else obj_name}")
        return {'success': True, 'object': obj_name, 'grasped': True}

    def execute_release_object(self, obj_name: str = None) -> Dict:
        """Execute releasing action"""
        self.node.get_logger().info(f"Releasing {'object' if obj_name is None else obj_name}")
        return {'success': True, 'object': obj_name, 'released': True}

    def execute_scan_environment(self, param: str = None) -> Dict:
        """Execute environment scanning"""
        self.node.get_logger().info("Scanning environment")
        return {'success': True, 'objects_found': 5, 'features': ['table', 'chair', 'cup']}

    def execute_identify_object(self, obj_name: str = None) -> Dict:
        """Execute object identification"""
        self.node.get_logger().info(f"Identifying {obj_name}")
        return {'success': True, 'object': obj_name, 'confidence': 0.95}

def main(args=None):
    rclpy.init(args=args)
    vla_node = VLAArchitecture()

    try:
        rclpy.spin(vla_node)
    except KeyboardInterrupt:
        vla_node.get_logger().info("VLA Architecture stopped by user")
    finally:
        vla_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Now let's create a more advanced cognitive planning module that maps LLMs to ROS 2 actions:

```python
# cognitive_planning_module.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Point
from sensor_msgs.msg import Image
from nav_msgs.msg import Path
from builtin_interfaces.msg import Time
import json
import asyncio
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import requests
import re

@dataclass
class CognitivePlan:
    """Data class for cognitive plans"""
    plan_id: str
    original_command: str
    plan_steps: List[Dict]
    execution_context: Dict
    created_at: float

class CognitivePlanningNode(Node):
    """
    Cognitive planning node that maps LLMs to ROS 2 actions
    """
    def __init__(self):
        super().__init__('cognitive_planning_node')

        # Publishers
        self.plan_pub = self.create_publisher(String, '/cognitive_plan', 10)
        self.action_pub = self.create_publisher(String, '/robot/action', 10)
        self.status_pub = self.create_publisher(String, '/cognitive_planning/status', 10)

        # Subscribers
        self.command_sub = self.create_subscription(String, '/natural_language/command', self.command_callback, 10)
        self.vision_sub = self.create_subscription(Image, '/camera/rgb/image_raw', self.vision_callback, 10)
        self.world_state_sub = self.create_subscription(String, '/world_state', self.world_state_callback, 10)

        # Timers
        self.planning_timer = self.create_timer(0.5, self.planning_cycle)

        # Internal state
        self.pending_commands = []
        self.world_state = {}
        self.vision_context = None
        self.plan_history = []
        self.llm_client = SimpleLLMClient()  # Simulated LLM client

        self.get_logger().info("Cognitive Planning Node initialized")

    def command_callback(self, msg):
        """Process natural language commands"""
        self.get_logger().info(f"Received command: {msg.data}")
        self.pending_commands.append({
            'command': msg.data,
            'timestamp': time.time(),
            'context': self.world_state.copy()
        })

    def vision_callback(self, msg):
        """Process vision context"""
        # In a real implementation, this would process the image
        # For this example, we'll just note that vision data is available
        self.vision_context = {'available': True, 'timestamp': time.time()}

    def world_state_callback(self, msg):
        """Update world state"""
        try:
            self.world_state = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error("Error parsing world state")

    def planning_cycle(self):
        """Main planning cycle"""
        if not self.pending_commands:
            return

        # Process the oldest command
        cmd_data = self.pending_commands[0]

        # Create cognitive plan
        plan = self.create_cognitive_plan(cmd_data['command'], cmd_data['context'])

        if plan:
            # Publish plan
            plan_msg = String()
            plan_msg.data = json.dumps({
                'plan_id': plan.plan_id,
                'original_command': plan.original_command,
                'steps': plan.plan_steps,
                'context': plan.execution_context
            })
            self.plan_pub.publish(plan_msg)

            # Execute plan
            self.execute_plan(plan)

            # Update status
            status_msg = String()
            status_msg.data = f"Executed plan for: '{cmd_data['command']}', Steps: {len(plan.plan_steps)}"
            self.status_pub.publish(status_msg)

            self.get_logger().info(status_msg.data)

        # Remove processed command
        self.pending_commands.pop(0)

    def create_cognitive_plan(self, command: str, context: Dict) -> Optional[CognitivePlan]:
        """Create a cognitive plan from natural language command"""
        try:
            # Use LLM to generate plan
            plan_prompt = self.create_planning_prompt(command, context)
            plan_response = self.llm_client.generate(plan_prompt)

            # Parse LLM response into executable plan
            plan_steps = self.parse_plan_response(plan_response, command, context)

            if plan_steps:
                plan = CognitivePlan(
                    plan_id=f"plan_{int(time.time())}",
                    original_command=command,
                    plan_steps=plan_steps,
                    execution_context=context,
                    created_at=time.time()
                )

                # Add to history
                self.plan_history.append(plan)

                return plan

        except Exception as e:
            self.get_logger().error(f"Error creating cognitive plan: {e}")

        return None

    def create_planning_prompt(self, command: str, context: Dict) -> str:
        """Create prompt for LLM-based planning"""
        prompt = f"""
        You are a cognitive planning assistant for a robot. Given the user command and current world state,
        break down the command into executable steps for a robot.

        Command: {command}

        World State: {json.dumps(context, indent=2)}

        Provide a sequence of robot actions as a JSON list. Each action should be:
        - Navigate to a location
        - Detect an object
        - Grasp an object
        - Place an object
        - Manipulate an object
        - Wait for something
        - Communicate with user

        Example output format:
        [
          {{"action": "navigate", "target": "kitchen_table", "description": "Move to kitchen table"}},
          {{"action": "detect", "object": "red_mug", "description": "Look for red mug"}},
          {{"action": "grasp", "object": "red_mug", "description": "Pick up red mug"}},
          {{"action": "navigate", "target": "counter", "description": "Move to counter"}},
          {{"action": "place", "object": "red_mug", "target": "counter", "description": "Place mug on counter"}}
        ]

        Respond with ONLY the JSON list, nothing else:
        """

        return prompt

    def parse_plan_response(self, response: str, original_command: str, context: Dict) -> List[Dict]:
        """Parse LLM response into structured plan steps"""
        # Try to extract JSON from response
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            try:
                plan_steps = json.loads(json_match.group(0))
                return plan_steps
            except json.JSONDecodeError:
                pass

        # If no JSON found, try to create simple plan based on keywords
        command_lower = original_command.lower()

        simple_plan = []
        if 'bring' in command_lower or 'get' in command_lower:
            # Find object to bring
            obj_match = re.search(r'bring (.+?) to|get (.+?) to|bring (.+)', command_lower)
            obj = obj_match.group(1) if obj_match else 'object'

            simple_plan = [
                {"action": "detect", "object": obj, "description": f"Detect {obj}"},
                {"action": "navigate", "target": obj, "description": f"Navigate to {obj}"},
                {"action": "grasp", "object": obj, "description": f"Grasp {obj}"},
                {"action": "navigate", "target": "delivery_location", "description": "Navigate to delivery location"},
                {"action": "place", "object": obj, "target": "delivery_location", "description": f"Place {obj}"}
            ]
        elif 'move' in command_lower or 'go' in command_lower:
            # Extract destination
            dest_match = re.search(r'to (.+)|at (.+)', command_lower)
            dest = dest_match.group(1) if dest_match else 'destination'

            simple_plan = [
                {"action": "navigate", "target": dest, "description": f"Navigate to {dest}"}
            ]
        elif 'pick up' in command_lower or 'take' in command_lower:
            # Extract object
            obj_match = re.search(r'pick up (.+)|take (.+)', command_lower)
            obj = obj_match.group(1) if obj_match else 'object'

            simple_plan = [
                {"action": "detect", "object": obj, "description": f"Detect {obj}"},
                {"action": "navigate", "target": obj, "description": f"Navigate to {obj}"},
                {"action": "grasp", "object": obj, "description": f"Grasp {obj}"}
            ]

        return simple_plan

    def execute_plan(self, plan: CognitivePlan):
        """Execute the cognitive plan"""
        for step_idx, step in enumerate(plan.plan_steps):
            self.get_logger().info(f"Executing step {step_idx + 1}/{len(plan.plan_steps)}: {step}")

            # Map action to ROS 2 message
            action_msg = String()
            action_msg.data = json.dumps({
                'plan_id': plan.plan_id,
                'step_number': step_idx + 1,
                'action': step,
                'original_command': plan.original_command
            })

            self.action_pub.publish(action_msg)

            # Simple delay to allow action execution
            time.sleep(0.5)  # In real implementation, wait for action completion

class SimpleLLMClient:
    """
    Simulated LLM client for demonstration purposes
    In a real implementation, this would connect to an actual LLM service
    """
    def __init__(self):
        self.model = "gpt-4"  # Simulated model
        self.response_cache = {}

    def generate(self, prompt: str) -> str:
        """Generate response from LLM"""
        # In a real implementation, this would call an LLM API
        # For this example, we'll return simulated responses based on common commands

        if "bring" in prompt.lower() or "get" in prompt.lower():
            return """[
  {"action": "detect", "object": "coffee mug", "description": "Detect coffee mug on table"},
  {"action": "navigate", "target": "table", "description": "Navigate to table"},
  {"action": "grasp", "object": "coffee mug", "description": "Grasp coffee mug"},
  {"action": "navigate", "target": "counter", "description": "Navigate to counter"},
  {"action": "place", "object": "coffee mug", "target": "counter", "description": "Place coffee mug on counter"}
]"""
        elif "move" in prompt.lower() or "go" in prompt.lower():
            return """[
  {"action": "navigate", "target": "kitchen", "description": "Navigate to kitchen"},
  {"action": "wait", "duration": 1.0, "description": "Wait for confirmation"}
]"""
        else:
            # Default response
            return """[
  {"action": "detect", "object": "unknown", "description": "Scan environment for relevant objects"},
  {"action": "communicate", "message": "I need more specific instructions", "description": "Ask for clarification"}
]"""

def main(args=None):
    rclpy.init(args=args)
    planner = CognitivePlanningNode()

    try:
        rclpy.spin(planner)
    except KeyboardInterrupt:
        planner.get_logger().info("Cognitive Planning Node stopped by user")
    finally:
        planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Small Simulation

Let's create a simulation environment that demonstrates the VLA integration:

```python
# vla_integration_simulator.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist, Pose, Point
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np
import cv2
import json
import time
import random

class VLAIntegrationSimulator(Node):
    """
    Simulation environment for VLA integration
    Demonstrates the complete Vision-Language-Action pipeline
    """
    def __init__(self):
        super().__init__('vla_integration_simulator')

        # Publishers
        self.camera_pub = self.create_publisher(Image, '/camera/rgb/image_raw', 10)
        self.world_state_pub = self.create_publisher(String, '/world_state', 10)
        self.voice_cmd_pub = self.create_publisher(String, '/voice/command', 10)
        self.status_pub = self.create_publisher(String, '/vla_simulation/status', 10)

        # Subscribers
        self.vla_result_sub = self.create_subscription(String, '/vla/result', self.vla_result_callback, 10)
        self.robot_action_sub = self.create_subscription(String, '/robot/action', self.robot_action_callback, 10)

        # Timers
        self.camera_timer = self.create_timer(0.1, self.publish_camera_feed)  # 10 Hz
        self.world_state_timer = self.create_timer(1.0, self.publish_world_state)  # 1 Hz
        self.command_timer = self.create_timer(5.0, self.publish_random_command)  # Every 5 seconds
        self.simulation_timer = self.create_timer(0.05, self.simulation_step)  # 20 Hz

        # Internal components
        self.cv_bridge = CvBridge()
        self.sim_time = 0.0
        self.robot_position = np.array([0.0, 0.0])
        self.objects = [
            {'name': 'table', 'position': np.array([2.0, 1.0]), 'type': 'furniture'},
            {'name': 'chair', 'position': np.array([2.5, 0.5]), 'type': 'furniture'},
            {'name': 'cup', 'position': np.array([2.2, 1.2]), 'type': 'object'},
            {'name': 'book', 'position': np.array([1.8, 0.8]), 'type': 'object'},
        ]
        self.commands = [
            "Bring me the cup from the table",
            "Go to the kitchen",
            "Pick up the red book",
            "Move to the chair",
            "Find the cup and bring it to me"
        ]

        self.get_logger().info("VLA Integration Simulator initialized")

    def publish_camera_feed(self):
        """Generate and publish simulated camera images"""
        # Create a simple simulated environment image
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        # Draw floor grid
        for y in range(0, 480, 50):
            cv2.line(image, (0, y), (640, y), (50, 50, 50), 1)
        for x in range(0, 640, 50):
            cv2.line(image, (x, 0), (x, 480), (50, 50, 50), 1)

        # Draw robot position (as a circle)
        robot_img_x = int(320 + self.robot_position[0] * 50)  # Scale position for image
        robot_img_y = int(240 - self.robot_position[1] * 50)  # Invert Y for image coordinates
        cv2.circle(image, (robot_img_x, robot_img_y), 15, (0, 255, 0), -1)

        # Draw objects
        for obj in self.objects:
            obj_img_x = int(320 + obj['position'][0] * 50)
            obj_img_y = int(240 - obj['position'][1] * 50)

            if obj['type'] == 'furniture':
                cv2.rectangle(image,
                            (obj_img_x - 20, obj_img_y - 20),
                            (obj_img_x + 20, obj_img_y + 20),
                            (100, 100, 200), -1)
            else:  # object
                cv2.circle(image, (obj_img_x, obj_img_y), 10, (200, 100, 100), -1)

            # Label the object
            cv2.putText(image, obj['name'], (obj_img_x - 15, obj_img_y - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Add timestamp text
        cv2.putText(image, f"Time: {self.sim_time:.1f}s", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Publish image
        img_msg = self.cv_bridge.cv2_to_imgmsg(image, encoding="bgr8")
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = 'camera_rgb_optical_frame'
        self.camera_pub.publish(img_msg)

    def publish_world_state(self):
        """Publish current world state"""
        world_state = {
            'timestamp': self.sim_time,
            'robot_position': self.robot_position.tolist(),
            'robot_orientation': 0.0,  # Facing angle
            'objects': [
                {
                    'name': obj['name'],
                    'position': obj['position'].tolist(),
                    'type': obj['type'],
                    'visible': True  # All objects are visible in simulation
                }
                for obj in self.objects
            ],
            'navigation_map': 'simulated_map',
            'battery_level': 0.85
        }

        state_msg = String()
        state_msg.data = json.dumps(world_state)
        self.world_state_pub.publish(state_msg)

    def publish_random_command(self):
        """Publish a random voice command"""
        if self.commands:
            command = random.choice(self.commands)
            cmd_msg = String()
            cmd_msg.data = command
            self.voice_cmd_pub.publish(cmd_msg)

            self.get_logger().info(f"Published command: '{command}'")

    def vla_result_callback(self, msg):
        """Process VLA results"""
        try:
            result = json.loads(msg.data)
            self.get_logger().info(f"VLA Result: Command '{result['command']}' -> Actions: {result['actions']}")
        except json.JSONDecodeError:
            self.get_logger().error("Error parsing VLA result")

    def robot_action_callback(self, msg):
        """Process robot actions"""
        try:
            action_data = json.loads(msg.data)
            action = action_data['action']

            self.get_logger().info(f"Robot executing: {action['action']} - {action['description']}")

            # Update robot state based on action
            if action['action'] == 'navigate' and 'target' in action:
                # Move robot towards target
                target_name = action['target']

                # Find target position
                target_pos = None
                for obj in self.objects:
                    if obj['name'] == target_name or target_name in obj['name']:
                        target_pos = obj['position']
                        break

                if target_pos is not None:
                    # Simple navigation: move towards target
                    direction = target_pos - self.robot_position
                    distance = np.linalg.norm(direction)
                    if distance > 0.1:  # If not already at target
                        # Move 0.1 units towards target
                        move_vector = direction / distance * min(0.1, distance)
                        self.robot_position += move_vector

        except json.JSONDecodeError:
            self.get_logger().error("Error parsing robot action")

    def simulation_step(self):
        """Main simulation step"""
        self.sim_time += 0.05  # 20 Hz

        # Publish status
        status_msg = String()
        status_msg.data = f"Time: {self.sim_time:.2f}s, Robot: ({self.robot_position[0]:.2f}, {self.robot_position[1]:.2f}), " \
                         f"Objects: {len(self.objects)}, Commands: {len(self.commands)}"
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    simulator = VLAIntegrationSimulator()

    try:
        rclpy.spin(simulator)
    except KeyboardInterrupt:
        simulator.get_logger().info("VLA Integration Simulator stopped by user")
    finally:
        simulator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Quick Recap

In this lesson, we've covered:

- **VLA Architecture**: Understanding the components that connect vision, language, and action systems
- **Perception Pipeline**: Creating visual perception for environmental understanding
- **Language Processing**: Implementing natural language understanding for commands
- **Action Mapping**: Connecting language understanding to robot action execution
- **Integration Testing**: Complete VLA pipeline with simulation environment

The Vision-Language-Action (VLA) framework represents a significant advancement in robotics, enabling more intuitive human-robot interaction. By combining visual perception, natural language understanding, and action execution, VLA systems allow robots to interpret high-level, ambiguous human instructions and translate them into specific, executable behaviors.

In the next lesson, we'll explore voice-to-action systems using Whisper for speech recognition and how to map voice commands to specific robot actions.