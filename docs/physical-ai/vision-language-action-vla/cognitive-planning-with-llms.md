---
sidebar_position: 3
---

# Cognitive Planning with LLMs Mapped to ROS 2 Actions

## Overview
This lesson explores how to implement cognitive planning using Large Language Models (LLMs) and map their high-level reasoning to concrete ROS 2 actions. We'll create a cognitive planning system that can interpret complex user requests, break them down into executable steps, and coordinate multiple ROS 2 nodes to accomplish sophisticated tasks. This approach enables robots to perform complex, multi-step operations based on natural language instructions.

## Learning Outcomes
By the end of this lesson, you will:
- Understand cognitive planning architectures for robotics
- Implement LLM-based task decomposition and planning
- Map high-level plans to ROS 2 action sequences
- Create a cognitive planning node that coordinates multiple ROS 2 services
- Integrate LLM reasoning with robot execution systems
- Handle plan failures and replanning scenarios

## Hands-on Steps

### Step 1: Set up the Cognitive Planning Environment
First, let's create the necessary packages and dependencies for our cognitive planning system.

```bash
# Create the cognitive planning package
cd ~/ros2_ws/src
ros2 pkg create --dependencies rclpy std_msgs sensor_msgs geometry_msgs action_msgs ros2_actions --node-name cognitive_planner cognitive_planning

cd cognitive_planning
mkdir -p cognitive_planning/{planning,llm,utils,execution}
```

Install the required Python dependencies:

```bash
pip install openai anthropic transformers torch ros2_numpy
```

### Step 2: Create the Cognitive Planning Node
Create the main cognitive planning node that will receive high-level goals, use an LLM to decompose them into subtasks, and execute them through ROS 2:

```python
# cognitive_planning/cognitive_planning/cognitive_planner_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from cognitive_planning.llm.planner import LLMPlanner
from cognitive_planning.execution.executor import ActionExecutor
from cognitive_planning.planning.task_decomposer import TaskDecomposer
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

class CognitivePlannerNode(Node):
    def __init__(self):
        super().__init__('cognitive_planner')

        # Initialize cognitive planning components
        self.llm_planner = LLMPlanner()
        self.task_decomposer = TaskDecomposer()
        self.action_executor = ActionExecutor(self)

        # Publishers
        self.plan_status_pub = self.create_publisher(String, 'plan_status', 10)
        self.execution_feedback_pub = self.create_publisher(String, 'execution_feedback', 10)

        # Subscribers
        self.goal_sub = self.create_subscription(
            String,
            'high_level_goal',
            self.goal_callback,
            10
        )

        self.plan_execution_sub = self.create_subscription(
            Bool,
            'execute_plan',
            self.execute_plan_callback,
            10
        )

        # Internal state
        self.current_plan = None
        self.plan_lock = asyncio.Lock()
        self.executor_pool = ThreadPoolExecutor(max_workers=4)

        # Parameters
        self.declare_parameter('llm_model', 'gpt-4')
        self.declare_parameter('max_replan_attempts', 3)
        self.declare_parameter('plan_timeout', 30.0)

        self.get_logger().info('Cognitive Planner Node initialized')

    def goal_callback(self, msg):
        """Handle high-level goals from user or other nodes"""
        goal_text = msg.data
        self.get_logger().info(f'Received high-level goal: {goal_text}')

        # Plan the goal in a separate thread to avoid blocking
        future = asyncio.run_coroutine_threadsafe(
            self.plan_goal_async(goal_text),
            asyncio.get_event_loop()
        )

    async def plan_goal_async(self, goal_text: str):
        """Asynchronously plan a goal using LLM"""
        try:
            self.get_logger().info(f'Planning goal: {goal_text}')

            # Decompose the goal into subtasks using LLM
            plan = await self.task_decomposer.decompose_goal(goal_text)

            if plan:
                self.current_plan = plan
                self.get_logger().info(f'Generated plan with {len(plan)} steps')

                # Publish plan for review
                plan_msg = String()
                plan_msg.data = json.dumps({
                    'goal': goal_text,
                    'plan': plan,
                    'status': 'planned'
                })
                self.plan_status_pub.publish(plan_msg)

                # Auto-execute if configured to do so
                if self.get_parameter('auto_execute').value:
                    await self.execute_current_plan()
            else:
                self.get_logger().error('Failed to generate plan for goal')

        except Exception as e:
            self.get_logger().error(f'Error planning goal: {str(e)}')

    def execute_plan_callback(self, msg):
        """Execute the current plan if available"""
        if msg.data and self.current_plan:
            future = asyncio.run_coroutine_threadsafe(
                self.execute_current_plan(),
                asyncio.get_event_loop()
            )

    async def execute_current_plan(self):
        """Execute the current plan step by step"""
        if not self.current_plan:
            self.get_logger().warn('No plan to execute')
            return

        self.get_logger().info('Starting plan execution')

        execution_msg = String()
        execution_msg.data = json.dumps({
            'status': 'executing',
            'plan': self.current_plan
        })
        self.execution_feedback_pub.publish(execution_msg)

        # Execute each step in the plan
        for i, step in enumerate(self.current_plan):
            self.get_logger().info(f'Executing step {i+1}/{len(self.current_plan)}: {step["action"]}')

            try:
                # Execute the action
                success = await self.action_executor.execute_action(step)

                if success:
                    self.get_logger().info(f'Step {i+1} completed successfully')
                else:
                    self.get_logger().error(f'Step {i+1} failed')
                    # Try to recover or replan
                    await self.handle_step_failure(i, step)
                    break

            except Exception as e:
                self.get_logger().error(f'Step {i+1} execution error: {str(e)}')
                await self.handle_step_failure(i, step)
                break

        # Mark plan completion
        completion_msg = String()
        completion_msg.data = json.dumps({
            'status': 'completed',
            'plan': self.current_plan
        })
        self.execution_feedback_pub.publish(completion_msg)

    async def handle_step_failure(self, step_index: int, step: dict):
        """Handle step failure and attempt recovery"""
        self.get_logger().warn(f'Handling failure for step {step_index}: {step["action"]}')

        # Publish failure status
        failure_msg = String()
        failure_msg.data = json.dumps({
            'status': 'failed',
            'failed_step': step_index,
            'step': step
        })
        self.execution_feedback_pub.publish(failure_msg)

        # Attempt recovery based on failure type
        recovery_type = step.get('recovery', 'none')

        if recovery_type == 'retry':
            # Retry the failed step
            success = await self.action_executor.execute_action(step)
            if success:
                self.get_logger().info('Recovery successful')
                return

        elif recovery_type == 'skip':
            # Skip to next step
            self.get_logger().info('Skipping failed step')
            # Continue execution from next step
            remaining_plan = self.current_plan[step_index + 1:]
            for j, next_step in enumerate(remaining_plan):
                success = await self.action_executor.execute_action(next_step)
                if not success:
                    self.get_logger().error(f'Step {step_index + 1 + j} also failed')
                    break

        elif recovery_type == 'replan':
            # Replan from the current state
            await self.replan_from_failure(step_index)

        else:
            # No recovery strategy - plan fails
            self.get_logger().error('Plan execution failed with no recovery')

    async def replan_from_failure(self, failed_step_index: int):
        """Replan the remaining tasks after a failure"""
        self.get_logger().info(f'Replanning from step {failed_step_index}')

        # Get remaining tasks
        remaining_tasks = self.current_plan[failed_step_index + 1:]

        if not remaining_tasks:
            self.get_logger().info('No remaining tasks to replan')
            return

        # Ask LLM to replan based on current state
        try:
            replanned_tasks = await self.task_decomposer.replan_remaining_tasks(
                self.current_plan[:failed_step_index],  # Completed tasks
                remaining_tasks  # Remaining tasks
            )

            if replanned_tasks:
                # Update the plan with replanned tasks
                self.current_plan = self.current_plan[:failed_step_index] + replanned_tasks
                self.get_logger().info(f'Replanned with {len(replanned_tasks)} new tasks')

                # Execute the replanned tasks
                for i, step in enumerate(replanned_tasks):
                    success = await self.action_executor.execute_action(step)
                    if not success:
                        self.get_logger().error(f'Replanned step {i} failed')
                        break
            else:
                self.get_logger().error('Failed to replan remaining tasks')

        except Exception as e:
            self.get_logger().error(f'Error during replanning: {str(e)}')

    def destroy_node(self):
        """Clean up resources"""
        self.executor_pool.shutdown(wait=True)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CognitivePlannerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Cognitive Planner Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Step 3: Create the LLM Planner Module
Now let's create the LLM-based planner that will handle the cognitive reasoning:

```python
# cognitive_planning/cognitive_planning/llm/planner.py
import openai
import json
from typing import Dict, List, Any, Optional
import asyncio
import logging

class LLMPlanner:
    def __init__(self, model="gpt-4", api_key=None):
        """
        Initialize the LLM Planner
        """
        self.model = model
        self.client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()
        self.logger = logging.getLogger(__name__)

    async def plan_task(self, goal: str, context: Dict[str, Any] = None) -> Optional[List[Dict[str, Any]]]:
        """
        Generate a plan for a given goal using LLM
        """
        prompt = self._create_planning_prompt(goal, context)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2048,
                response_format={"type": "json_object"}
            )

            plan_json = response.choices[0].message.content
            plan = json.loads(plan_json)

            return plan.get('plan', [])

        except Exception as e:
            self.logger.error(f"Error generating plan: {str(e)}")
            return None

    def _create_planning_prompt(self, goal: str, context: Dict[str, Any]) -> str:
        """
        Create a structured prompt for task planning
        """
        context_str = json.dumps(context, indent=2) if context else "{}"

        prompt = f"""
        You are a cognitive planning assistant for a humanoid robot. Your task is to decompose high-level goals into executable steps.

        Goal: {goal}

        Context: {context_str}

        Please provide a detailed plan in JSON format with the following structure:
        {{
            "plan": [
                {{
                    "id": 1,
                    "action": "action_name",
                    "description": "Human-readable description of the action",
                    "parameters": {{"param1": "value1", "param2": "value2"}},
                    "dependencies": ["action_id"],  # Actions that must be completed before this one
                    "recovery": "retry|skip|replan|none",  # How to handle failure
                    "timeout": 30.0  # Timeout in seconds
                }}
            ]
        }}

        Available actions:
        - navigate_to: Move robot to a specific location
        - detect_object: Use vision system to detect objects
        - pick_up: Pick up an object
        - place: Place an object at a location
        - speak: Speak a text message
        - wait: Wait for a condition or time period
        - call_service: Call a specific ROS 2 service

        Ensure the plan is feasible, safe, and considers the robot's capabilities.
        """

        return prompt

    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for the LLM
        """
        return """
        You are a cognitive planning assistant for a humanoid robot. Your role is to decompose complex goals into simple, executable steps that can be performed by a robot system.

        Guidelines:
        1. Break down complex tasks into simple, atomic actions
        2. Consider the dependencies between actions
        3. Account for the robot's physical limitations and capabilities
        4. Include appropriate recovery strategies for each action
        5. Provide clear, unambiguous instructions
        6. Ensure the plan is safe and feasible

        Always respond in valid JSON format with the specified structure.
        """

    async def refine_plan(self, plan: List[Dict[str, Any]], feedback: str) -> Optional[List[Dict[str, Any]]]:
        """
        Refine an existing plan based on feedback
        """
        prompt = f"""
        You are refining a robot execution plan based on feedback.

        Original Plan: {json.dumps(plan, indent=2)}

        Feedback: {feedback}

        Please provide an improved plan in the same JSON format, addressing the feedback while maintaining the original goal.
        """

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
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

    async def evaluate_plan_feasibility(self, plan: List[Dict[str, Any]], environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate if a plan is feasible given the current environment state
        """
        prompt = f"""
        Evaluate the feasibility of this robot plan given the environment state.

        Plan: {json.dumps(plan, indent=2)}

        Environment State: {json.dumps(environment_state, indent=2)}

        Please respond with a JSON object containing:
        {{
            "feasible": true/false,
            "issues": ["issue1", "issue2", ...],
            "confidence": 0.0-1.0,
            "suggestions": ["suggestion1", "suggestion2", ...]
        }}
        """

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1024,
                response_format={"type": "json_object"}
            )

            evaluation = json.loads(response.choices[0].message.content)
            return evaluation

        except Exception as e:
            self.logger.error(f"Error evaluating plan feasibility: {str(e)}")
            return {"feasible": False, "issues": [str(e)], "confidence": 0.0, "suggestions": []}
```

### Step 4: Create the Task Decomposer
Now let's create the task decomposer that will handle the cognitive planning process:

```python
# cognitive_planning/cognitive_planning/planning/task_decomposer.py
import asyncio
from typing import Dict, List, Any, Optional
from cognitive_planning.llm.planner import LLMPlanner
import json
import logging

class TaskDecomposer:
    def __init__(self):
        self.llm_planner = LLMPlanner()
        self.logger = logging.getLogger(__name__)

    async def decompose_goal(self, goal: str) -> Optional[List[Dict[str, Any]]]:
        """
        Decompose a high-level goal into executable steps
        """
        # Get current environment state (this would come from perception system)
        environment_state = await self._get_environment_state()

        # Generate plan using LLM
        plan = await self.llm_planner.plan_task(goal, environment_state)

        if plan:
            # Validate and refine the plan
            is_feasible = await self._validate_plan(plan, environment_state)
            if is_feasible:
                return plan
            else:
                # Try to refine the plan
                refined_plan = await self._refine_plan_with_feedback(plan, environment_state)
                if refined_plan:
                    return refined_plan

        return None

    async def replan_remaining_tasks(self, completed_tasks: List[Dict[str, Any]],
                                   remaining_tasks: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """
        Replan remaining tasks after a failure, considering completed tasks
        """
        environment_state = await self._get_environment_state()

        # Create context about what has been completed
        context = {
            "completed_tasks": completed_tasks,
            "remaining_goals": remaining_tasks,
            "current_state": environment_state
        }

        # Generate new plan for remaining tasks
        prompt = f"""
        You are replanning robot tasks after a failure. Some tasks have been completed successfully,
        but there was a failure in the middle of execution.

        Completed Tasks: {json.dumps(completed_tasks, indent=2)}
        Remaining Goals: {json.dumps(remaining_tasks, indent=2)}
        Current Environment State: {json.dumps(environment_state, indent=2)}

        Please generate a new plan that continues from the current state to achieve the remaining goals.
        """

        # This would involve calling the LLM with the specific replanning prompt
        # For now, we'll use a simplified approach
        new_plan = []

        for task in remaining_tasks:
            # Adjust parameters based on current state
            new_task = task.copy()
            # Add any necessary adjustments based on completed tasks
            new_plan.append(new_task)

        return new_plan

    async def _get_environment_state(self) -> Dict[str, Any]:
        """
        Get the current environment state from perception and other systems
        """
        # This would typically query various ROS 2 topics and services
        # to get the current state of the world
        return {
            "robot_position": {"x": 0.0, "y": 0.0, "theta": 0.0},
            "robot_battery": 0.85,
            "detected_objects": [],
            "navigation_map": "available",
            "manipulation_capabilities": ["pick_up", "place", "move_arm"],
            "current_task": "idle"
        }

    async def _validate_plan(self, plan: List[Dict[str, Any]], environment_state: Dict[str, Any]) -> bool:
        """
        Validate if the plan is feasible in the current environment
        """
        evaluation = await self.llm_planner.evaluate_plan_feasibility(plan, environment_state)
        return evaluation.get("feasible", False)

    async def _refine_plan_with_feedback(self, plan: List[Dict[str, Any]],
                                       environment_state: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """
        Refine a plan based on feasibility feedback
        """
        evaluation = await self.llm_planner.evaluate_plan_feasibility(plan, environment_state)

        if evaluation.get("issues"):
            feedback = f"The plan has the following issues: {', '.join(evaluation['issues'])}. Please address these issues."
            refined_plan = await self.llm_planner.refine_plan(plan, feedback)
            return refined_plan

        return plan

    def get_plan_complexity(self, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the complexity of a plan
        """
        complexity_metrics = {
            "total_steps": len(plan),
            "action_types": set(),
            "estimated_duration": 0.0,
            "risk_level": "low",  # low, medium, high
            "dependency_depth": 0
        }

        for step in plan:
            action_type = step.get("action", "unknown")
            complexity_metrics["action_types"].add(action_type)

            # Estimate duration based on action type
            if action_type in ["navigate_to", "move"]:
                complexity_metrics["estimated_duration"] += 10.0
            elif action_type in ["pick_up", "place"]:
                complexity_metrics["estimated_duration"] += 5.0
            elif action_type in ["detect_object"]:
                complexity_metrics["estimated_duration"] += 3.0
            else:
                complexity_metrics["estimated_duration"] += 2.0

        complexity_metrics["action_types"] = list(complexity_metrics["action_types"])

        # Determine risk level based on complexity
        if len(plan) > 10:
            complexity_metrics["risk_level"] = "high"
        elif len(plan) > 5:
            complexity_metrics["risk_level"] = "medium"

        return complexity_metrics

    async def optimize_plan(self, plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize a plan for efficiency
        """
        # This would involve various optimization techniques
        # such as parallelizing independent actions, reducing redundant steps, etc.

        optimized_plan = []

        # Group independent actions that can be executed in parallel
        parallelizable_actions = []
        sequential_actions = []

        for step in plan:
            if step.get("dependencies", []):
                sequential_actions.append(step)
            else:
                parallelizable_actions.append(step)

        # For now, just return the original plan
        # In a real implementation, this would contain optimization logic
        return plan
```

### Step 5: Create the Action Executor
Now let's create the action executor that will map the planned actions to ROS 2 services and actions:

```python
# cognitive_planning/cognitive_planning/execution/executor.py
import rclpy
from rclpy.action import ActionClient
from rclpy.client import Client
from rclpy.qos import QoSProfile
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from typing import Dict, Any, Optional
import asyncio
import time

class ActionExecutor:
    def __init__(self, node):
        self.node = node
        self.logger = node.get_logger()

        # Initialize action and service clients
        self._init_clients()

    def _init_clients(self):
        """Initialize all necessary ROS 2 clients"""
        # Navigation action client
        self.nav_client = ActionClient(
            self.node,
            'nav2_msgs/action/NavigateToPose',
            'navigate_to_pose'
        )

        # Manipulation service clients
        self.pick_service = self.node.create_client(
            'manipulation_msgs/srv/PickUp',
            'pick_up_object'
        )

        self.place_service = self.node.create_client(
            'manipulation_msgs/srv/Place',
            'place_object'
        )

        # Speech service client
        self.speech_service = self.node.create_client(
            'tts_msgs/srv/Speak',
            'speak_text'
        )

        # Perception service client
        self.detection_service = self.node.create_client(
            'vision_msgs/srv/DetectObjects',
            'detect_objects'
        )

    async def execute_action(self, action: Dict[str, Any]) -> bool:
        """
        Execute a single action from the plan
        """
        action_type = action.get("action", "")
        parameters = action.get("parameters", {})
        timeout = action.get("timeout", 30.0)

        self.logger.info(f"Executing action: {action_type} with params: {parameters}")

        try:
            if action_type == "navigate_to":
                return await self._execute_navigation_action(parameters, timeout)
            elif action_type == "detect_object":
                return await self._execute_detection_action(parameters, timeout)
            elif action_type == "pick_up":
                return await self._execute_pickup_action(parameters, timeout)
            elif action_type == "place":
                return await self._execute_place_action(parameters, timeout)
            elif action_type == "speak":
                return await self._execute_speak_action(parameters, timeout)
            elif action_type == "wait":
                return await self._execute_wait_action(parameters, timeout)
            elif action_type == "call_service":
                return await self._execute_service_call(parameters, timeout)
            else:
                self.logger.error(f"Unknown action type: {action_type}")
                return False

        except Exception as e:
            self.logger.error(f"Error executing action {action_type}: {str(e)}")
            return False

    async def _execute_navigation_action(self, params: Dict[str, Any], timeout: float) -> bool:
        """Execute navigation action"""
        target_pose = PoseStamped()
        target_pose.header.frame_id = params.get("frame_id", "map")
        target_pose.header.stamp = self.node.get_clock().now().to_msg()

        target_pose.pose.position.x = params.get("x", 0.0)
        target_pose.pose.position.y = params.get("y", 0.0)
        target_pose.pose.position.z = params.get("z", 0.0)

        # Set orientation (assuming quaternion parameters)
        target_pose.pose.orientation.w = params.get("orientation_w", 1.0)
        target_pose.pose.orientation.x = params.get("orientation_x", 0.0)
        target_pose.pose.orientation.y = params.get("orientation_y", 0.0)
        target_pose.pose.orientation.z = params.get("orientation_z", 0.0)

        # Wait for action server
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.logger.error("Navigation action server not available")
            return False

        # Send goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = target_pose

        future = self.nav_client.send_goal_async(goal_msg)

        # Wait for result with timeout
        try:
            goal_handle = await asyncio.wait_for(future, timeout=timeout)
            if not goal_handle.accepted:
                self.logger.error("Navigation goal was rejected")
                return False

            result_future = goal_handle.get_result_async()
            result = await asyncio.wait_for(result_future, timeout=timeout)

            return result.result.status == GoalStatus.STATUS_SUCCEEDED

        except asyncio.TimeoutError:
            self.logger.error("Navigation action timed out")
            return False

    async def _execute_detection_action(self, params: Dict[str, Any], timeout: float) -> bool:
        """Execute object detection action"""
        if not self.detection_service.wait_for_service(timeout_sec=5.0):
            self.logger.error("Detection service not available")
            return False

        request = DetectObjects.Request()
        request.target_object = params.get("target", "")
        request.search_area = params.get("area", "current_view")

        future = self.detection_service.call_async(request)

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result.success
        except asyncio.TimeoutError:
            self.logger.error("Detection action timed out")
            return False

    async def _execute_pickup_action(self, params: Dict[str, Any], timeout: float) -> bool:
        """Execute pickup action"""
        if not self.pick_service.wait_for_service(timeout_sec=5.0):
            self.logger.error("Pickup service not available")
            return False

        request = PickUp.Request()
        request.object_name = params.get("object", "")
        request.grasp_pose = self._dict_to_pose(params.get("grasp_pose", {}))

        future = self.pick_service.call_async(request)

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result.success
        except asyncio.TimeoutError:
            self.logger.error("Pickup action timed out")
            return False

    async def _execute_place_action(self, params: Dict[str, Any], timeout: float) -> bool:
        """Execute place action"""
        if not self.place_service.wait_for_service(timeout_sec=5.0):
            self.logger.error("Place service not available")
            return False

        request = Place.Request()
        request.object_name = params.get("object", "")
        request.place_pose = self._dict_to_pose(params.get("place_pose", {}))

        future = self.place_service.call_async(request)

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result.success
        except asyncio.TimeoutError:
            self.logger.error("Place action timed out")
            return False

    async def _execute_speak_action(self, params: Dict[str, Any], timeout: float) -> bool:
        """Execute speech action"""
        if not self.speech_service.wait_for_service(timeout_sec=5.0):
            self.logger.error("Speech service not available")
            return False

        request = Speak.Request()
        request.text = params.get("text", "")

        future = self.speech_service.call_async(request)

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result.success
        except asyncio.TimeoutError:
            self.logger.error("Speech action timed out")
            return False

    async def _execute_wait_action(self, params: Dict[str, Any], timeout: float) -> bool:
        """Execute wait action"""
        wait_duration = params.get("duration", 1.0)

        # Use a timer to wait
        start_time = time.time()
        while time.time() - start_time < wait_duration:
            await asyncio.sleep(0.1)  # Non-blocking sleep

        return True

    async def _execute_service_call(self, params: Dict[str, Any], timeout: float) -> bool:
        """Execute a generic service call"""
        service_name = params.get("service_name", "")
        service_type = params.get("service_type", "")
        service_params = params.get("parameters", {})

        # This would dynamically create and call the appropriate service
        # For now, we'll log the call
        self.logger.info(f"Calling service {service_name} with params: {service_params}")

        # In a real implementation, this would dynamically create the service client
        # and call it with the provided parameters
        return True

    def _dict_to_pose(self, pose_dict: Dict[str, Any]):
        """Convert dictionary to Pose message"""
        from geometry_msgs.msg import Pose
        pose = Pose()

        pose.position.x = pose_dict.get("x", 0.0)
        pose.position.y = pose_dict.get("y", 0.0)
        pose.position.z = pose_dict.get("z", 0.0)

        pose.orientation.w = pose_dict.get("w", 1.0)
        pose.orientation.x = pose_dict.get("x", 0.0)
        pose.orientation.y = pose_dict.get("y", 0.0)
        pose.orientation.z = pose_dict.get("z", 0.0)

        return pose

    def cancel_current_action(self):
        """Cancel the currently executing action"""
        # This would cancel the current action if possible
        self.logger.info("Canceling current action")
        # Implementation would depend on the specific action type