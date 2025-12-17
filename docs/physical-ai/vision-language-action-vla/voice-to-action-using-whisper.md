---
sidebar_position: 2
---

# Voice-to-Action using Whisper

## Overview
This lesson explores how to implement voice-to-action systems using OpenAI's Whisper speech recognition model. We'll create a pipeline that captures voice commands, transcribes them using Whisper, and converts them into ROS 2 actions for humanoid robots. This approach enables natural human-robot interaction through spoken language.

## Learning Outcomes
By the end of this lesson, you will:
- Understand the architecture of voice-to-action systems for robotics
- Implement speech recognition using OpenAI's Whisper model
- Map voice commands to ROS 2 actions and services
- Create a robust voice command parser for humanoid robots
- Integrate voice control with existing ROS 2 navigation and manipulation systems

## Hands-on Steps

### Step 1: Set up the Voice Recognition Environment
First, let's create the necessary packages and dependencies for our voice-to-action system.

```bash
# Create the voice recognition package
cd ~/ros2_ws/src
ros2 pkg create --dependencies rclpy std_msgs sensor_msgs geometry_msgs action_msgs --node-name voice_recognition_node voice_to_action

cd voice_to_action
mkdir -p voice_to_action/{perception,utils,nlp}
```

Install the required Python dependencies:

```bash
pip install openai-whisper torch torchaudio ros2_numpy
```

### Step 2: Create the Voice Recognition Node
Create the main voice recognition node that will capture audio, process it with Whisper, and publish recognized commands:

```python
# voice_to_action/voice_to_action/voice_recognition_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import AudioData
import whisper
import torch
import numpy as np
import threading
import queue
import pyaudio
import wave
import io
from vosk import Model, KaldiRecognizer

class VoiceRecognitionNode(Node):
    def __init__(self):
        super().__init__('voice_recognition_node')

        # Initialize Whisper model
        self.get_logger().info('Loading Whisper model...')
        self.model = whisper.load_model("base")  # Use "small" or "medium" for better accuracy

        # Publishers
        self.command_pub = self.create_publisher(String, 'voice_command', 10)
        self.velocity_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Subscribers
        self.audio_sub = self.create_subscription(
            AudioData,
            'audio_input',
            self.audio_callback,
            10
        )

        # Parameters
        self.declare_parameter('sample_rate', 16000)
        self.sample_rate = self.get_parameter('sample_rate').value

        self.declare_parameter('chunk_size', 1024)
        self.chunk_size = self.get_parameter('chunk_size').value

        # Audio recording setup
        self.audio_queue = queue.Queue()
        self.recording = False
        self.audio_thread = None

        # Start audio recording thread
        self.start_audio_recording()

        # Timer for processing audio chunks
        self.timer = self.create_timer(1.0, self.process_audio)

        self.get_logger().info('Voice Recognition Node initialized')

    def start_audio_recording(self):
        """Start audio recording thread"""
        self.recording = True
        self.audio_thread = threading.Thread(target=self.record_audio)
        self.audio_thread.daemon = True
        self.audio_thread.start()

    def record_audio(self):
        """Record audio from microphone"""
        p = pyaudio.PyAudio()

        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

        self.get_logger().info('Recording audio...')

        while self.recording:
            data = stream.read(self.chunk_size)
            self.audio_queue.put(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

    def audio_callback(self, msg):
        """Callback for audio data from ROS topic"""
        # Convert audio message to numpy array
        audio_data = np.frombuffer(msg.data, dtype=np.int16).astype(np.float32) / 32768.0

        # Add to processing queue
        self.audio_queue.put(audio_data.tobytes())

    def process_audio(self):
        """Process accumulated audio data with Whisper"""
        if self.audio_queue.empty():
            return

        # Collect audio data from queue
        audio_chunks = []
        while not self.audio_queue.empty():
            chunk = self.audio_queue.get()
            if isinstance(chunk, bytes):
                # Convert bytes to float32 numpy array
                chunk_array = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                audio_chunks.append(chunk_array)
            else:
                audio_chunks.append(chunk)

        if len(audio_chunks) == 0:
            return

        # Concatenate all audio chunks
        full_audio = np.concatenate(audio_chunks)

        # Process with Whisper
        try:
            # Convert to tensor and process
            audio_tensor = torch.from_numpy(full_audio).float()

            # Transcribe using Whisper
            result = self.model.transcribe(audio_tensor.numpy(), fp16=False)
            transcription = result['text'].strip()

            if transcription:
                self.get_logger().info(f'Recognized: {transcription}')

                # Publish the recognized command
                cmd_msg = String()
                cmd_msg.data = transcription
                self.command_pub.publish(cmd_msg)

                # Parse and execute the command
                self.parse_voice_command(transcription)

        except Exception as e:
            self.get_logger().error(f'Error processing audio: {str(e)}')

    def parse_voice_command(self, command_text):
        """Parse voice command and convert to ROS 2 actions"""
        command_text = command_text.lower().strip()

        # Define command mappings
        if 'move forward' in command_text or 'go forward' in command_text:
            self.execute_navigation_command('forward')
        elif 'move backward' in command_text or 'go backward' in command_text:
            self.execute_navigation_command('backward')
        elif 'turn left' in command_text:
            self.execute_navigation_command('left')
        elif 'turn right' in command_text:
            self.execute_navigation_command('right')
        elif 'stop' in command_text:
            self.execute_stop_command()
        elif 'wave' in command_text:
            self.execute_arm_command('wave')
        elif 'raise arm' in command_text:
            self.execute_arm_command('raise')
        elif 'lower arm' in command_text:
            self.execute_arm_command('lower')
        elif 'dance' in command_text:
            self.execute_dance_command()
        else:
            self.get_logger().info(f'Unknown command: {command_text}')

    def execute_navigation_command(self, direction):
        """Execute navigation commands"""
        twist_msg = Twist()

        if direction == 'forward':
            twist_msg.linear.x = 0.5  # Forward speed
        elif direction == 'backward':
            twist_msg.linear.x = -0.5  # Backward speed
        elif direction == 'left':
            twist_msg.angular.z = 0.5  # Left turn
        elif direction == 'right':
            twist_msg.angular.z = -0.5  # Right turn

        self.velocity_pub.publish(twist_msg)
        self.get_logger().info(f'Executing navigation command: {direction}')

    def execute_stop_command(self):
        """Stop all movement"""
        twist_msg = Twist()
        self.velocity_pub.publish(twist_msg)
        self.get_logger().info('Stopping robot')

    def execute_arm_command(self, action):
        """Execute arm manipulation commands"""
        # This would typically call a service or publish to joint controllers
        self.get_logger().info(f'Executing arm command: {action}')

    def execute_dance_command(self):
        """Execute dance sequence"""
        self.get_logger().info('Executing dance sequence')
        # Complex sequence of movements
        # This would involve multiple joint controllers and timing

    def destroy_node(self):
        """Clean up resources"""
        self.recording = False
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VoiceRecognitionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Voice Recognition Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Step 3: Create the Voice Command Parser
Now let's create a sophisticated parser that can handle complex voice commands and map them to ROS 2 actions:

```python
# voice_to_action/voice_to_action/nlp/command_parser.py
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class CommandType(Enum):
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    INTERACTION = "interaction"
    SYSTEM = "system"

@dataclass
class ParsedCommand:
    command_type: CommandType
    action: str
    parameters: Dict[str, any]
    confidence: float
    raw_text: str

class VoiceCommandParser:
    def __init__(self):
        # Navigation commands
        self.nav_patterns = {
            r'(?:move|go|drive|navigate|walk)\s+(?:forward|ahead)': ('move', {'direction': 'forward'}),
            r'(?:move|go|drive|navigate|walk)\s+(?:backward|back|reverse)': ('move', {'direction': 'backward'}),
            r'(?:turn|rotate)\s+(?:left|anti-clockwise)': ('turn', {'direction': 'left'}),
            r'(?:turn|rotate)\s+(?:right|clockwise)': ('turn', {'direction': 'right'}),
            r'(?:move|step)\s+(left|right)': ('strafe', {'direction': '{0}'}),
            r'stop': ('stop', {}),
            r'go to (?:position|location|waypoint)\s+(.+?)\s*$': ('navigate_to', {'target': '{0}'}),
            r'follow (me|person|leader)': ('follow', {'target': '{0}'}),
        }

        # Manipulation commands
        self.manip_patterns = {
            r'(?:pick up|grasp|grab|take)\s+(.+?)\s*$': ('pick_up', {'object': '{0}'}),
            r'(?:put down|place|release|drop)\s+(.+?)\s*$': ('place', {'object': '{0}'}),
            r'(?:open|close)\s+(door|jar|box)': ('manipulate', {'object': '{0}', 'action': '{0}'}),
            r'(?:wave|raise|lift)\s+(?:your)?\s*(?:left|right|both)?\s*(?:arm|hand)': ('gesture', {'gesture': 'wave'}),
            r'(?:point|show|indicate)\s+(?:to)?\s*(.+?)\s*$': ('point_to', {'target': '{0}'}),
        }

        # Interaction commands
        self.interaction_patterns = {
            r'(?:say|speak|tell)\s+(.+?)\s*$': ('speak', {'text': '{0}'}),
            r'(?:hello|hi|greet)\s+(?:everyone|people|humans?)': ('greet', {'target': 'everyone'}),
            r'(?:introduce|present)\s+yourself': ('introduce', {}),
            r'(?:dance|perform|show)\s+dance': ('dance', {}),
        }

        # System commands
        self.system_patterns = {
            r'(?:shutdown|power off|turn off)\s+system': ('shutdown', {}),
            r'(?:reboot|restart)\s+system': ('reboot', {}),
            r'(?:sleep|standby|rest)': ('sleep', {}),
            r'(?:wake up|activate|start)': ('wakeup', {}),
        }

        self.all_patterns = {
            CommandType.NAVIGATION: self.nav_patterns,
            CommandType.MANIPULATION: self.manipulation_patterns,
            CommandType.INTERACTION: self.interaction_patterns,
            CommandType.SYSTEM: self.system_patterns,
        }

    def parse_command(self, text: str) -> Optional[ParsedCommand]:
        """Parse voice command and return structured command"""
        text = text.strip().lower()

        # Try each command type
        for cmd_type, patterns in self.all_patterns.items():
            for pattern, (action, params_template) in patterns.items():
                match = re.search(pattern, text)
                if match:
                    # Extract groups and fill template
                    groups = match.groups()
                    params = params_template.copy()

                    # Replace placeholders in parameters
                    for key, value in params.items():
                        if isinstance(value, str) and '{0}' in value:
                            if groups:
                                params[key] = value.format(*groups)
                        elif isinstance(value, str) and '{' in value and '}' in value:
                            # Handle numbered placeholders like {1}, {2}, etc.
                            for i, group in enumerate(groups):
                                placeholder = f'{{{i}}}'
                                if placeholder in value:
                                    params[key] = value.replace(placeholder, group)

                    # Calculate confidence based on match length
                    confidence = min(len(match.group()) / len(text), 1.0)

                    return ParsedCommand(
                        command_type=cmd_type,
                        action=action,
                        parameters=params,
                        confidence=confidence,
                        raw_text=text
                    )

        return None

    def get_suggested_commands(self) -> List[str]:
        """Return list of supported voice commands for user guidance"""
        suggestions = []

        for cmd_type, patterns in self.all_patterns.items():
            for pattern in patterns.keys():
                # Convert regex patterns to user-friendly examples
                example = self._regex_to_example(pattern)
                suggestions.append(f"{cmd_type.value}: {example}")

        return suggestions

    def _regex_to_example(self, pattern: str) -> str:
        """Convert regex pattern to example command"""
        # Simplify regex for user presentation
        example = pattern.replace(r'(?:', '').replace(r')', '')
        example = example.replace(r'\s+', ' ').replace(r'\s*', ' ')
        example = example.replace('|', ' or ')
        example = example.replace(r'.+?', '[object]')
        example = example.replace(r'.+?$', '[object]')
        return example.strip()
```

### Step 4: Create the Voice Action Executor
Now let's create a node that executes the parsed commands by calling appropriate ROS 2 services and publishing to topics:

```python
# voice_to_action/voice_to_action/action_execution_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist, PoseStamped
from action_msgs.msg import GoalStatus
from voice_to_action.nlp.command_parser import VoiceCommandParser, ParsedCommand, CommandType
import math

class VoiceActionExecutor(Node):
    def __init__(self):
        super().__init__('voice_action_executor')

        # Subscribe to parsed voice commands
        self.command_sub = self.create_subscription(
            String,
            'parsed_voice_command',
            self.command_callback,
            10
        )

        # Publishers for different action types
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.speech_pub = self.create_publisher(String, 'tts_request', 10)

        # Service clients for complex actions
        self.nav_client = self.create_client(NavigateToPose, '/navigate_to_pose')
        self.arm_client = self.create_client(MoveArm, '/move_arm')

        # Initialize command parser
        self.parser = VoiceCommandParser()

        # Store robot state
        self.current_pose = None
        self.is_moving = False

        self.get_logger().info('Voice Action Executor initialized')

    def command_callback(self, msg):
        """Process incoming voice command"""
        command_text = msg.data

        # Parse the command
        parsed_cmd = self.parser.parse_command(command_text)

        if parsed_cmd and parsed_cmd.confidence > 0.5:  # Confidence threshold
            self.get_logger().info(f'Executing command: {parsed_cmd.action} with confidence {parsed_cmd.confidence:.2f}')
            self.execute_command(parsed_cmd)
        else:
            # Unknown command - respond appropriately
            self.respond_unknown_command(command_text)

    def execute_command(self, parsed_cmd: ParsedCommand):
        """Execute the parsed command based on type"""
        if parsed_cmd.command_type == CommandType.NAVIGATION:
            self.execute_navigation_command(parsed_cmd)
        elif parsed_cmd.command_type == CommandType.MANIPULATION:
            self.execute_manipulation_command(parsed_cmd)
        elif parsed_cmd.command_type == CommandType.INTERACTION:
            self.execute_interaction_command(parsed_cmd)
        elif parsed_cmd.command_type == CommandType.SYSTEM:
            self.execute_system_command(parsed_cmd)

    def execute_navigation_command(self, cmd: ParsedCommand):
        """Execute navigation commands"""
        if cmd.action == 'move':
            self.move_robot(cmd.parameters['direction'])
        elif cmd.action == 'turn':
            self.turn_robot(cmd.parameters['direction'])
        elif cmd.action == 'strafe':
            self.strafe_robot(cmd.parameters['direction'])
        elif cmd.action == 'stop':
            self.stop_robot()
        elif cmd.action == 'navigate_to':
            self.navigate_to_location(cmd.parameters['target'])
        elif cmd.action == 'follow':
            self.follow_target(cmd.parameters['target'])

    def execute_manipulation_command(self, cmd: ParsedCommand):
        """Execute manipulation commands"""
        if cmd.action == 'pick_up':
            self.pick_up_object(cmd.parameters['object'])
        elif cmd.action == 'place':
            self.place_object(cmd.parameters['object'])
        elif cmd.action == 'gesture':
            self.perform_gesture(cmd.parameters['gesture'])

    def execute_interaction_command(self, cmd: ParsedCommand):
        """Execute interaction commands"""
        if cmd.action == 'speak':
            self.speak_text(cmd.parameters['text'])
        elif cmd.action == 'greet':
            self.greet_people(cmd.parameters.get('target', 'everyone'))

    def execute_system_command(self, cmd: ParsedCommand):
        """Execute system commands"""
        if cmd.action == 'shutdown':
            self.shutdown_system()
        elif cmd.action == 'sleep':
            self.enter_sleep_mode()

    def move_robot(self, direction: str):
        """Move robot in specified direction"""
        twist = Twist()

        if direction == 'forward':
            twist.linear.x = 0.5
        elif direction == 'backward':
            twist.linear.x = -0.5
        elif direction == 'left':
            twist.linear.y = 0.5
        elif direction == 'right':
            twist.linear.y = -0.5

        self.cmd_vel_pub.publish(twist)
        self.is_moving = True

    def turn_robot(self, direction: str):
        """Turn robot in specified direction"""
        twist = Twist()

        if direction == 'left':
            twist.angular.z = 0.5
        elif direction == 'right':
            twist.angular.z = -0.5

        self.cmd_vel_pub.publish(twist)
        self.is_moving = True

    def strafe_robot(self, direction: str):
        """Strafe robot sideways"""
        twist = Twist()

        if direction == 'left':
            twist.linear.y = 0.3
        elif direction == 'right':
            twist.linear.y = -0.3

        self.cmd_vel_pub.publish(twist)

    def stop_robot(self):
        """Stop all robot movement"""
        twist = Twist()
        self.cmd_vel_pub.publish(twist)
        self.is_moving = False

    def navigate_to_location(self, location: str):
        """Navigate to a specific location using navigation stack"""
        # This would typically use Nav2 or similar navigation system
        self.get_logger().info(f'Navigating to: {location}')

        # Example: Send navigation goal
        goal_msg = NavigateToPose.Goal()
        # Set goal pose based on location name
        # This would require a location database or semantic map
        self.nav_client.send_goal(goal_msg)

    def speak_text(self, text: str):
        """Publish text-to-speech request"""
        tts_msg = String()
        tts_msg.data = text
        self.speech_pub.publish(tts_msg)

    def respond_unknown_command(self, command_text: str):
        """Respond when command is not understood"""
        response = f"I'm sorry, I didn't understand the command: {command_text}. Could you please repeat that?"
        self.speak_text(response)

        # Provide command suggestions
        suggestions = self.parser.get_suggested_commands()
        if suggestions:
            help_text = "You can say commands like: " + ", ".join(suggestions[:3])  # Limit to 3 suggestions
            self.speak_text(help_text)

    def pick_up_object(self, object_name: str):
        """Pick up specified object"""
        self.get_logger().info(f'Attempting to pick up: {object_name}')
        # Call manipulation service here

    def place_object(self, object_name: str):
        """Place object"""
        self.get_logger.info(f'Attempting to place: {object_name}')
        # Call manipulation service here

    def perform_gesture(self, gesture_name: str):
        """Perform specified gesture"""
        self.get_logger().info(f'Performing gesture: {gesture_name}')
        # Call gesture execution service here

    def greet_people(self, target: str):
        """Greet people"""
        greeting = f"Hello {target}! Nice to meet you."
        self.speak_text(greeting)

    def shutdown_system(self):
        """Shutdown the system"""
        self.get_logger().info('Shutting down system...')
        # Perform graceful shutdown procedures

    def enter_sleep_mode(self):
        """Enter sleep/standby mode"""
        self.stop_robot()
        self.get_logger().info('Entering sleep mode')


def main(args=None):
    rclpy.init(args=args)
    node = VoiceActionExecutor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Voice Action Executor')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Step 5: Create the Main Voice Control Launch File
Finally, let's create a launch file that brings together all the voice control components:

```python
# voice_to_action/launch/voice_control.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'sample_rate',
            default_value='16000',
            description='Audio sample rate for voice recognition'
        ),

        # Voice recognition node
        Node(
            package='voice_to_action',
            executable='voice_recognition_node',
            name='voice_recognition',
            parameters=[
                {'sample_rate': LaunchConfiguration('sample_rate')}
            ],
            output='screen'
        ),

        # Voice action executor node
        Node(
            package='voice_to_action',
            executable='voice_action_executor',
            name='voice_action_executor',
            output='screen'
        ),

        # Text-to-speech node (if available)
        Node(
            package='tts_package',  # Replace with actual TTS package
            executable='tts_node',
            name='text_to_speech',
            output='screen'
        )
    ])
```

### Step 6: Test the Voice Control System
Let's create a simple test script to verify our voice control system works:

```python
# voice_to_action/test/test_voice_control.py
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import time

class VoiceControlTester(Node):
    def __init__(self):
        super().__init__('voice_control_tester')

        # Publisher for simulated voice commands
        self.voice_pub = self.create_publisher(String, 'voice_command', 10)

        # Subscriber to verify actions
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10
        )

        self.last_velocity = Twist()
        self.test_passed = True

        self.get_logger().info('Voice Control Tester initialized')

    def cmd_vel_callback(self, msg):
        """Store received velocity commands"""
        self.last_velocity = msg

    def run_tests(self):
        """Run comprehensive tests of voice control system"""
        self.get_logger().info('Starting voice control tests...')

        # Test 1: Move forward command
        self.get_logger().info('Test 1: Sending "move forward" command')
        cmd_msg = String()
        cmd_msg.data = 'move forward'
        self.voice_pub.publish(cmd_msg)
        time.sleep(2.0)

        if self.last_velocity.linear.x > 0:
            self.get_logger().info('✓ Move forward test passed')
        else:
            self.get_logger().warn('✗ Move forward test failed')
            self.test_passed = False

        # Test 2: Turn left command
        self.get_logger.info('Test 2: Sending "turn left" command')
        cmd_msg.data = 'turn left'
        self.voice_pub.publish(cmd_msg)
        time.sleep(2.0)

        if self.last_velocity.angular.z > 0:
            self.get_logger.info('✓ Turn left test passed')
        else:
            self.get_logger().warn('✗ Turn left test failed')
            self.test_passed = False

        # Test 3: Stop command
        self.get_logger.info('Test 3: Sending "stop" command')
        cmd_msg.data = 'stop'
        self.voice_pub.publish(cmd_msg)
        time.sleep(1.0)

        if abs(self.last_velocity.linear.x) < 0.01 and abs(self.last_velocity.angular.z) < 0.01:
            self.get_logger.info('✓ Stop test passed')
        else:
            self.get_logger().warn('✗ Stop test failed')
            self.test_passed = False

        # Final result
        if self.test_passed:
            self.get_logger().info('🎉 All voice control tests passed!')
        else:
            self.get_logger().error('❌ Some tests failed')

    def destroy_node(self):
        """Clean up resources"""
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    tester = VoiceControlTester()

    # Schedule tests after a brief delay to allow system to initialize
    timer = tester.create_timer(3.0, lambda: tester.run_tests())

    try:
        rclpy.spin(tester)
    except KeyboardInterrupt:
        tester.get_logger().info('Test interrupted')
    finally:
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Code Examples

Here's a complete example of how to use the voice-to-action system in practice:

```python
# Example: Integrating voice control with humanoid robot navigation
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from voice_to_action.nlp.command_parser import VoiceCommandParser

class HumanoidVoiceController(Node):
    def __init__(self):
        super().__init__('humanoid_voice_controller')

        # Initialize voice command parser
        self.parser = VoiceCommandParser()

        # Publishers for humanoid-specific actions
        self.body_cmd_pub = self.create_publisher(String, 'humanoid_body_commands', 10)
        self.arm_cmd_pub = self.create_publisher(String, 'humanoid_arm_commands', 10)

        # Example voice command processing
        self.demo_voice_commands()

    def demo_voice_commands(self):
        """Demonstrate voice command processing for humanoid robot"""
        commands = [
            "move forward slowly",
            "turn left 90 degrees",
            "raise both arms",
            "wave hello",
            "balance on one foot",
            "sit down",
            "stand up"
        ]

        for cmd_text in commands:
            parsed_cmd = self.parser.parse_command(cmd_text)
            if parsed_cmd:
                self.get_logger().info(f'Parsed: {parsed_cmd.action} - {parsed_cmd.parameters}')
                self.execute_humanoid_command(parsed_cmd)

    def execute_humanoid_command(self, cmd):
        """Execute command for humanoid robot"""
        if cmd.command_type == 'navigation':
            # Handle navigation for bipedal locomotion
            body_cmd = String()
            body_cmd.data = f"locomotion_{cmd.action}"
            self.body_cmd_pub.publish(body_cmd)
        elif cmd.action == 'gesture':
            # Handle arm gestures
            arm_cmd = String()
            arm_cmd.data = f"gesture_{cmd.parameters['gesture']}"
            self.arm_cmd_pub.publish(arm_cmd)

def main():
    rclpy.init()
    controller = HumanoidVoiceController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()
```

## Small Simulation

Let's create a simple simulation to test our voice-to-action system:

```python
# voice_to_action/simulations/voice_simulation.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import time

class VoiceToActionSimulation:
    def __init__(self):
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0  # Heading angle in radians
        self.is_moving = False
        self.command_history = []

        # Setup plot
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.robot_plot = self.ax.plot([], [], 'ro', markersize=15, label='Robot')[0]
        self.path_plot = self.ax.plot([], [], 'b-', alpha=0.7, label='Path')[0]

        # Set up the environment
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_xlabel('X Position (m)')
        self.ax.set_ylabel('Y Position (m)')
        self.ax.set_title('Voice-to-Action Robot Simulation')
        self.ax.grid(True)
        self.ax.legend()

        # Robot path storage
        self.path_x = [0]
        self.path_y = [0]

    def process_voice_command(self, command_text):
        """Process voice command and update robot state"""
        command_text = command_text.lower()

        if 'forward' in command_text:
            self.move_forward()
            self.command_history.append(('forward', time.time()))
        elif 'backward' in command_text:
            self.move_backward()
            self.command_history.append(('backward', time.time()))
        elif 'left' in command_text and 'turn' in command_text:
            self.turn_left()
            self.command_history.append(('turn_left', time.time()))
        elif 'right' in command_text and 'turn' in command_text:
            self.turn_right()
            self.command_history.append(('turn_right', time.time()))
        elif 'stop' in command_text:
            self.stop_robot()
            self.command_history.append(('stop', time.time()))

    def move_forward(self):
        """Move robot forward"""
        self.robot_x += 0.1 * np.cos(self.robot_theta)
        self.robot_y += 0.1 * np.sin(self.robot_theta)
        self.update_path()

    def move_backward(self):
        """Move robot backward"""
        self.robot_x -= 0.1 * np.cos(self.robot_theta)
        self.robot_y -= 0.1 * np.sin(self.robot_theta)
        self.update_path()

    def turn_left(self):
        """Turn robot left"""
        self.robot_theta += np.pi / 8  # 22.5 degrees

    def turn_right(self):
        """Turn robot right"""
        self.robot_theta -= np.pi / 8  # 22.5 degrees

    def stop_robot(self):
        """Stop robot movement"""
        pass  # Just stop the movement

    def update_path(self):
        """Update the path history"""
        self.path_x.append(self.robot_x)
        self.path_y.append(self.robot_y)

    def animate(self, frame):
        """Animation function for the plot"""
        # Simulate some voice commands periodically
        if frame % 50 == 0:  # Every 50 frames, send a command
            commands = ['move forward', 'turn left', 'move forward', 'turn right']
            cmd = commands[(frame // 50) % len(commands)]
            self.process_voice_command(cmd)
            print(f"Voice command: {cmd}")

        # Update robot position on plot
        self.robot_plot.set_data([self.robot_x], [self.robot_y])
        self.path_plot.set_data(self.path_x, self.path_y)

        # Add robot orientation indicator
        arrow_length = 0.5
        arrow_dx = arrow_length * np.cos(self.robot_theta)
        arrow_dy = arrow_length * np.sin(self.robot_theta)
        self.ax.arrow(self.robot_x, self.robot_y, arrow_dx, arrow_dy,
                     head_width=0.1, head_length=0.1, fc='red', ec='red')

        return self.robot_plot, self.path_plot

    def run_simulation(self):
        """Run the voice-to-action simulation"""
        ani = FuncAnimation(self.fig, self.animate, frames=500, interval=100, blit=True)
        plt.show()

if __name__ == "__main__":
    sim = VoiceToActionSimulation()
    sim.run_simulation()
```

## Quick Recap
In this lesson, we've implemented a comprehensive voice-to-action system using OpenAI's Whisper model for speech recognition. We created:

1. A voice recognition node that captures audio and processes it with Whisper
2. A sophisticated command parser that maps natural language to ROS 2 actions
3. An action executor that translates voice commands into robot behaviors
4. Integration with humanoid robot control systems
5. A simulation environment to test the voice control system

The system handles various command types including navigation, manipulation, interaction, and system commands. It's designed to be robust and extensible, allowing for easy addition of new voice commands and robotic capabilities.

The next lesson will cover cognitive planning with LLMs mapped to ROS 2 actions, building on this voice-to-action foundation.