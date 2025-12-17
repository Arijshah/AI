---
sidebar_label: 'Nav2 Path Planning for Humanoid Robots'
title: 'Nav2 Path Planning for Humanoid Robots'
---

# Nav2-based Path Planning for Humanoid Robots

## Overview

Navigation 2 (Nav2) is the latest navigation stack for ROS 2, providing state-of-the-art path planning and execution capabilities. When applied to humanoid robots, Nav2 requires special considerations due to their complex kinematics, balance requirements, and unique locomotion patterns. This lesson explores how to adapt Nav2 for humanoid robot navigation, including specialized planners, controllers, and safety considerations that account for the robot's bipedal nature.

Humanoid robots present unique challenges for navigation systems due to their narrow support base, balance constraints, and complex movement patterns. Understanding how to configure and customize Nav2 for these requirements is crucial for developing effective autonomous humanoid robot navigation systems.

## Learning Outcomes

By the end of this lesson, you will be able to:
- Configure Nav2 for humanoid robot kinematics and dynamics
- Implement specialized path planners for bipedal locomotion
- Set up safety and balance-aware navigation behaviors
- Integrate whole-body planning with Nav2's framework
- Adapt Nav2 for complex humanoid robot locomotion patterns

## Hands-on Steps

1. **Nav2 Configuration**: Set up Nav2 for humanoid robot parameters
2. **Specialized Planning**: Create planners that account for bipedal constraints
3. **Balance-Aware Control**: Implement balance-aware navigation behaviors
4. **Safety Integration**: Add humanoid-specific safety considerations
5. **Performance Testing**: Validate navigation on humanoid robot models

### Prerequisites

- Understanding of Nav2 architecture and components
- Knowledge of humanoid robot kinematics and dynamics
- Experience with Isaac Sim and ROS 2 navigation

## Code Examples

Let's start by creating a configuration file for Nav2 adapted for humanoid robots:

```yaml
# humanoid_nav2_params.yaml
amcl:
  ros__parameters:
    use_sim_time: True
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_footprint"
    beam_skip_distance: 0.5
    beam_skip_error_threshold: 0.9
    beam_skip_threshold: 0.3
    do_beamskip: false
    global_frame_id: "map"
    lambda_short: 0.1
    laser_likelihood_max_dist: 2.0
    laser_max_range: 10.0
    laser_min_range: -1.0
    laser_model_type: "likelihood_field"
    max_beams: 60
    max_particles: 2000
    min_particles: 500
    odom_frame_id: "odom"
    pf_err: 0.05
    pf_z: 0.99
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    resample_interval: 1
    robot_model_type: "nav2_amcl::DifferentialMotionModel"
    save_pose_rate: 0.5
    sigma_hit: 0.2
    tf_broadcast: true
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.25
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05

amcl_map_client:
  ros__parameters:
    use_sim_time: True

amcl_rclcpp_node:
  ros__parameters:
    use_sim_time: True

bt_navigator:
  ros__parameters:
    use_sim_time: True
    enable_groot_monitoring: True
    groot_zmq_publisher_port: 1666
    groot_zmq_server_port: 1667
    default_bt_xml_filename: "humanoid_nav2_default_bt.xml"
    plugin_lib_names:
    - nav2_compute_path_to_pose_action_bt_node
    - nav2_compute_path_through_poses_action_bt_node
    - nav2_follow_path_action_bt_node
    - nav2_spin_action_bt_node
    - nav2_wait_action_bt_node
    - nav2_assisted_teleop_action_bt_node
    - nav2_back_up_action_bt_node
    - nav2_drive_on_heading_bt_node
    - nav2_clear_costmap_service_bt_node
    - nav2_is_stuck_condition_bt_node
    - nav2_goal_reached_condition_bt_node
    - nav2_goal_updated_condition_bt_node
    - nav2_initial_pose_received_condition_bt_node
    - nav2_reinitialize_global_localization_service_bt_node
    - nav2_rate_controller_bt_node
    - nav2_distance_controller_bt_node
    - nav2_speed_controller_bt_node
    - nav2_truncate_path_action_bt_node
    - nav2_truncate_path_local_action_bt_node
    - nav2_goal_updater_node_bt_node
    - nav2_recovery_node_bt_node
    - nav2_pipeline_sequence_bt_node
    - nav2_round_robin_node_bt_node
    - nav2_transform_available_condition_bt_node
    - nav2_time_expired_condition_bt_node
    - nav2_path_expiring_timer_condition
    - nav2_distance_traveled_condition_bt_node
    - nav2_single_trigger_bt_node
    - nav2_is_battery_low_condition_bt_node
    - nav2_navigate_through_poses_action_bt_node
    - nav2_navigate_to_pose_action_bt_node
    - nav2_remove_passed_goals_action_bt_node
    - nav2_planner_selector_bt_node
    - nav2_controller_selector_bt_node
    - nav2_goal_checker_selector_bt_node
    - nav2_controller_cancel_bt_node
    - nav2_path_longer_on_approach_bt_node
    - nav2_wait_cancel_bt_node

bt_navigator_rclcpp_node:
  ros__parameters:
    use_sim_time: True

controller_server:
  ros__parameters:
    use_sim_time: True
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    failure_tolerance: 0.3
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    # Humanoid-specific controller
    FollowPath:
      plugin: "nav2_mppi_controller::MPPICtrl"
      time_steps: 32
      control_freq: 20.0
      horizon: 3.2
      control_horizon: 3.2
      velocity_scaling_factor: 0.8  # Conservative for humanoid balance
      frequency: 20.0

      # Humanoid-specific parameters
      max_linear_speed: 0.3  # Slower for stability
      min_linear_speed: 0.05
      max_angular_speed: 0.5  # Limited for balance
      min_angular_speed: 0.05

      # Balance constraints
      balance_margin: 0.1  # Safety margin for support polygon
      step_size_limit: 0.2  # Maximum step size
      step_duration: 0.8    # Time per step
      zmp_margin: 0.05      # Zero Moment Point safety margin

    progress_checker:
      plugin: "nav2_controller::SimpleProgressChecker"
      required_movement_radius: 0.5
      movement_time_allowance: 10.0

    goal_checker:
      plugin: "nav2_controller::SimpleGoalChecker"
      xy_goal_tolerance: 0.2  # Larger for humanoid stepping accuracy
      yaw_goal_tolerance: 0.2
      stateful: True

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: "odom"
      robot_base_frame: "base_footprint"
      use_sim_time: True
      rolling_window: true
      width: 6
      height: 6
      resolution: 0.05
      robot_radius: 0.3  # Larger radius for humanoid safety
      plugins: ["voxel_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: True
        publish_voxel_map: True
        origin_z: 0.0
        z_resolution: 0.2
        z_voxels: 10
        max_obstacle_height: 2.0
        mark_threshold: 0
        observation_sources: "scan"
        scan:
          topic: "/humanoid_robot/scan"
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
  local_costmap_client:
    ros__parameters:
      use_sim_time: True
  local_costmap_rclcpp_node:
    ros__parameters:
      use_sim_time: True

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 1.0
      global_frame: "map"
      robot_base_frame: "base_footprint"
      use_sim_time: True
      robot_radius: 0.3
      resolution: 0.05
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        observation_sources: "scan"
        scan:
          topic: "/humanoid_robot/scan"
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.6
  global_costmap_client:
    ros__parameters:
      use_sim_time: True
  global_costmap_rclcpp_node:
    ros__parameters:
      use_sim_time: True

planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    use_sim_time: True
    planner_plugins: ["GridBased"]

    # Humanoid-specific planner
    GridBased:
      plugin: "humanoid_nav2_plugins::HumanoidGridBasedPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: true
      max_iterations: 10000
      max_on_approach_iterations: 1000

      # Humanoid-specific parameters
      min_distance_from_wall: 0.4  # Extra safety for humanoid width
      step_size: 0.1              # Conservative step size
      step_height: 0.15           # Maximum step height
      foot_separation: 0.3        # Distance between feet
      support_polygon_buffer: 0.1 # Buffer for support polygon

      # Balance-aware planning
      balance_constraint_weight: 10.0
      zmp_constraint_weight: 5.0
      com_height: 0.8             # Center of mass height for humanoid

smoother_server:
  ros__parameters:
    use_sim_time: True
    smoother_plugins: ["simple_smoother"]
    simple_smoother:
      plugin: "nav2_smoother::SimpleSmoother"
      tolerance: 1.0e-10
      max_its: 1000
      do_refinement: True

behavior_server:
  ros__parameters:
    costmap_topic: global_costmap/costmap_raw
    footprint_topic: global_costmap/published_footprint
    cycle_frequency: 10.0
    behavior_plugins: ["spin", "backup", "wait", "assisted_teleop"]
    spin:
      plugin: "nav2_behaviors::Spin"
      server_timeout: 20
      sim_frequency: 20
      angle_thresh: 0.785
      angle_offset: 1.57
      scale_vel: 0.5
    backup:
      plugin: "nav2_behaviors::BackUp"
      server_timeout: 20
      sim_frequency: 20
      distance: 0.15  # Shorter backup distance for humanoid safety
      forward_sampling_distance: 0.05
      max_vel: 0.025
    wait:
      plugin: "nav2_behaviors::Wait"
      server_timeout: 20
      sim_frequency: 20
      duration: 1.0
    assisted_teleop:
      plugin: "nav2_behaviors::AssistedTeleop"
      server_timeout: 20
      sim_frequency: 20
      smooth_move: true
      use_duration: false
      duration: 2.0

waypoint_follower:
  ros__parameters:
    loop_rate: 20
    stop_on_failure: false
    waypoint_task_executor_plugin: "wait_at_waypoint"
    wait_at_waypoint:
      plugin: "nav2_waypoint_follower::WaitAtWaypoint"
      enabled: true
      waypoint_pause_duration: 0
```

Now let's create a custom path planner that's aware of humanoid balance constraints:

```python
# humanoid_path_planner.py
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from builtin_interfaces.msg import Duration
from nav2_msgs.action import ComputePathToPose
from nav2_msgs.srv import GetCostmap
from nav2_util.lifecycle_node import LifecycleNode
from nav2_costmap_2d.costmap_2d_ros import Costmap2DROS
from nav2_costmap_2d.costmap_2d import Costmap2D
from geometry_msgs.msg import PoseWithCovarianceStamped
import numpy as np
import math
from scipy.spatial.distance import cdist
import heapq

class HumanoidGridBasedPlanner(LifecycleNode):
    """
    Custom path planner for humanoid robots with balance constraints
    """
    def __init__(self, name):
        super().__init__(name)
        self.name = name
        self.costmap_ros = None
        self.costmap = None
        self.global_frame = 'map'
        self.robot_base_frame = 'base_footprint'
        self.planner_frequency = 20.0

        # Humanoid-specific parameters
        self.min_distance_from_wall = 0.4
        self.step_size = 0.1
        self.step_height = 0.15
        self.foot_separation = 0.3
        self.support_polygon_buffer = 0.1
        self.balance_constraint_weight = 10.0
        self.zmp_constraint_weight = 5.0
        self.com_height = 0.8

        # Path smoothing parameters
        self.smoothing_weight = 0.5
        self.curvature_weight = 0.3

    def configure(self, node, plugin_name, plugin_type):
        """Configure the planner"""
        self.logger.info(f"Configuring {plugin_name}")

        # Get costmap
        self.costmap_ros = node.get_parameter('global_costmap').value
        self.global_frame = self.costmap_ros.get_parameter('global_frame').value

        # Get humanoid-specific parameters
        self.min_distance_from_wall = node.get_parameter('min_distance_from_wall').value
        self.step_size = node.get_parameter('step_size').value
        self.com_height = node.get_parameter('com_height').value
        self.balance_constraint_weight = node.get_parameter('balance_constraint_weight').value

        self.logger.info(f"{plugin_name} configured successfully")

    def activate(self):
        """Activate the planner"""
        self.logger.info(f"{self.name} activated")
        return True

    def deactivate(self):
        """Deactivate the planner"""
        self.logger.info(f"{self.name} deactivated")
        return True

    def cleanup(self):
        """Clean up the planner"""
        self.logger.info(f"{self.name} cleaned up")
        return True

    def create_plan(self, start, goal, tolerance):
        """
        Create a path from start to goal considering humanoid constraints
        """
        self.get_logger().info(f"Creating plan from ({start.pose.position.x}, {start.pose.position.y}) "
                              f"to ({goal.pose.position.x}, {goal.pose.position.y})")

        # Get current costmap
        costmap = self.costmap_ros.get_costmap()

        # Convert start and goal to costmap coordinates
        start_x = int((start.pose.position.x - costmap.getOriginX()) / costmap.getResolution())
        start_y = int((start.pose.position.y - costmap.getOriginY()) / costmap.getResolution())

        goal_x = int((goal.pose.position.x - costmap.getOriginX()) / costmap.getResolution())
        goal_y = int((goal.pose.position.y - costmap.getOriginY()) / costmap.getResolution())

        # Check if start and goal are valid
        if not self.is_valid_cell(costmap, start_x, start_y) or not self.is_valid_cell(costmap, goal_x, goal_y):
            self.get_logger().warn("Start or goal position is not valid")
            return self.create_empty_path()

        # Plan path using modified A* with humanoid constraints
        path_cells = self.humanoid_astar(costmap, (start_x, start_y), (goal_x, goal_y))

        if not path_cells:
            self.get_logger().warn("No valid path found")
            return self.create_empty_path()

        # Convert path cells to world coordinates
        world_path = self.cells_to_world_path(path_cells, costmap)

        # Apply humanoid-specific path smoothing
        smoothed_path = self.humanoid_path_smoothing(world_path)

        # Create and return path message
        path_msg = Path()
        path_msg.header.frame_id = self.global_frame
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for point in smoothed_path:
            pose = PoseStamped()
            pose.header.frame_id = self.global_frame
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = 0.0
            # Set orientation to point towards next point
            if len(smoothed_path) > 1 and np.array_equal(point, smoothed_path[0]):
                # First point - look towards second point
                dx = smoothed_path[1][0] - point[0]
                dy = smoothed_path[1][1] - point[1]
                yaw = math.atan2(dy, dx)
                pose.pose.orientation.z = math.sin(yaw / 2)
                pose.pose.orientation.w = math.cos(yaw / 2)
            elif np.array_equal(point, smoothed_path[-1]):
                # Last point - keep orientation from previous
                pass
            else:
                # Intermediate points - look towards next
                idx = smoothed_path.index(list(point) if isinstance(point, np.ndarray) else point)
                if idx < len(smoothed_path) - 1:
                    dx = smoothed_path[idx + 1][0] - point[0]
                    dy = smoothed_path[idx + 1][1] - point[1]
                    yaw = math.atan2(dy, dx)
                    pose.pose.orientation.z = math.sin(yaw / 2)
                    pose.pose.orientation.w = math.cos(yaw / 2)

            path_msg.poses.append(pose)

        self.get_logger().info(f"Path created with {len(path_msg.poses)} waypoints")
        return path_msg

    def is_valid_cell(self, costmap, x, y):
        """Check if a cell is valid for humanoid navigation"""
        # Check bounds
        if x < 0 or x >= costmap.getSizeInCellsX() or y < 0 or y >= costmap.getSizeInCellsY():
            return False

        # Get cost
        cost = costmap.getCost(x, y)

        # Check if cell is in lethal obstacle or unknown (but allow unknown if configured)
        lethal_cost = costmap.getCostmap().LETHAL_OBSTACLE
        unknown_cost = costmap.getCostmap().NO_INFORMATION

        # Humanoid-specific: need to maintain safety distance from obstacles
        if cost >= lethal_cost * 0.9:  # 90% of lethal cost is too dangerous
            return False

        return True

    def humanoid_astar(self, costmap, start, goal):
        """A* algorithm modified for humanoid robot constraints"""
        # Heuristic function
        def heuristic(a, b):
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

        # Priority queue: (f_score, g_score, position)
        open_set = [(0, 0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}

        # Keep track of visited cells to avoid cycles
        closed_set = set()

        # 8-directional movement (for smoother paths)
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

        while open_set:
            current = heapq.heappop(open_set)[2]

            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            closed_set.add(current)

            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)

                # Check bounds
                if (neighbor[0] < 0 or neighbor[0] >= costmap.getSizeInCellsX() or
                    neighbor[1] < 0 or neighbor[1] >= costmap.getSizeInCellsY()):
                    continue

                # Skip if already visited
                if neighbor in closed_set:
                    continue

                # Check if valid for humanoid
                if not self.is_valid_cell(costmap, neighbor[0], neighbor[1]):
                    continue

                # Calculate tentative g_score
                # For diagonal moves, use longer distance
                move_cost = math.sqrt(2) if dx != 0 and dy != 0 else 1

                # Add humanoid-specific cost based on balance constraints
                balance_cost = self.calculate_balance_cost(costmap, neighbor, goal)
                tentative_g_score = g_score[current] + move_cost + balance_cost

                # If this path to neighbor is better
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)

                    heapq.heappush(open_set, (f_score[neighbor], g_score[neighbor], neighbor))

        # No path found
        return []

    def calculate_balance_cost(self, costmap, cell, goal):
        """Calculate additional cost based on humanoid balance constraints"""
        # This is a simplified balance cost calculation
        # In reality, this would involve complex ZMP (Zero Moment Point) calculations

        # Get real-world coordinates
        resolution = costmap.getResolution()
        world_x = costmap.getOriginX() + cell[0] * resolution
        world_y = costmap.getOriginY() + cell[1] * resolution

        # Calculate distance to goal (for goal-direction bias)
        goal_x = costmap.getOriginX() + goal[0] * resolution
        goal_y = costmap.getOriginY() + goal[1] * resolution
        dist_to_goal = math.sqrt((world_x - goal_x)**2 + (world_y - goal_y)**2)

        # Check local terrain for roughness (simplified)
        roughness_cost = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor = (cell[0] + dx, cell[1] + dy)
                if (0 <= neighbor[0] < costmap.getSizeInCellsX() and
                    0 <= neighbor[1] < costmap.getSizeInCellsY()):
                    neighbor_cost = costmap.getCost(neighbor[0], neighbor[1])
                    # Higher cost for areas with obstacles nearby
                    if neighbor_cost > 50:  # Some threshold
                        roughness_cost += 0.1

        # Balance cost is higher for areas with obstacles nearby
        # and slightly biased towards goal direction
        balance_cost = roughness_cost * self.balance_constraint_weight
        goal_bias = max(0, 1 - dist_to_goal / 10.0) * 0.1  # Slightly prefer goal direction

        return balance_cost + goal_bias

    def cells_to_world_path(self, path_cells, costmap):
        """Convert path from cell coordinates to world coordinates"""
        resolution = costmap.getResolution()
        origin_x = costmap.getOriginX()
        origin_y = costmap.getOriginY()

        world_path = []
        for cell_x, cell_y in path_cells:
            world_x = origin_x + (cell_x + 0.5) * resolution
            world_y = origin_y + (cell_y + 0.5) * resolution
            world_path.append([world_x, world_y])

        return np.array(world_path)

    def humanoid_path_smoothing(self, path):
        """Apply smoothing that respects humanoid movement constraints"""
        if len(path) < 3:
            return path

        # Apply smoothing with constraints
        smoothed_path = [path[0]]  # Always keep start point

        i = 0
        while i < len(path) - 1:
            # Look ahead to find a point that can be directly reached with smooth motion
            j = i + 1
            while j < len(path) - 1:
                # Check if path from path[i] to path[j] is smooth enough for humanoid
                if self.is_smooth_enough(path[i], path[j], path[min(j+1, len(path)-1)]):
                    j += 1
                else:
                    break

            # Add the furthest smooth point we found
            smoothed_path.append(path[j-1])
            i = j - 1

        # Always add the goal point
        if not np.array_equal(smoothed_path[-1], path[-1]):
            smoothed_path.append(path[-1])

        return np.array(smoothed_path)

    def is_smooth_enough(self, p1, p2, p3):
        """Check if the path segment through these points is smooth enough for humanoid"""
        # Calculate curvature between three points
        # For simplicity, use a basic curvature calculation
        if np.array_equal(p1, p2) or np.array_equal(p2, p3):
            return True

        # Calculate angles to check for sharp turns
        v1 = p2 - p1
        v2 = p3 - p2

        # Calculate angle between vectors
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1, 1))

        # Humanoids can't make very sharp turns, so limit the angle
        max_angle = math.radians(45)  # 45 degrees maximum

        return angle < max_angle

    def create_empty_path(self):
        """Create an empty path message"""
        path_msg = Path()
        path_msg.header.frame_id = self.global_frame
        path_msg.header.stamp = self.get_clock().now().to_msg()
        return path_msg

# Node for testing the planner
class HumanoidPathPlannerTestNode(Node):
    """
    Test node for the humanoid path planner
    """
    def __init__(self):
        super().__init__('humanoid_path_planner_test')

        # Publishers
        self.path_pub = self.create_publisher(Path, '/humanoid_plan', 10)
        self.status_pub = self.create_publisher(String, '/humanoid_planner_status', 10)

        # Timers
        self.test_timer = self.create_timer(5.0, self.test_planning)

        # Test parameters
        self.start_pose = PoseStamped()
        self.start_pose.pose.position.x = 0.0
        self.start_pose.pose.position.y = 0.0

        self.goal_pose = PoseStamped()
        self.goal_pose.pose.position.x = 5.0
        self.goal_pose.pose.position.y = 5.0

        self.get_logger().info("Humanoid Path Planner Test Node initialized")

    def test_planning(self):
        """Test the path planning functionality"""
        self.get_logger().info(f"Testing path planning from ({self.start_pose.pose.position.x}, {self.start_pose.pose.position.y}) "
                              f"to ({self.goal_pose.pose.position.x}, {self.goal_pose.pose.position.y})")

        # In a real implementation, we would call the planner service here
        # For this example, we'll simulate the planner
        self.simulate_planning()

    def simulate_planning(self):
        """Simulate planning to show expected behavior"""
        # Create a sample path (this would be generated by the actual planner)
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()

        # Create a curved path to demonstrate humanoid-aware planning
        for i in range(20):
            t = i / 19.0  # Parameter from 0 to 1
            x = t * 5.0  # From 0 to 5
            y = t * 5.0 + 0.5 * math.sin(t * 3)  # Slight curve
            theta = math.atan2(5.0 + 0.5 * 3 * math.cos(t * 3), 5.0)  # Orientation

            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.z = math.sin(theta / 2)
            pose.pose.orientation.w = math.cos(theta / 2)

            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

        status_msg = String()
        status_msg.data = f"Test path published with {len(path_msg.poses)} waypoints"
        self.status_pub.publish(status_msg)

        self.get_logger().info(f"Test path published with {len(path_msg.poses)} waypoints")

def main(args=None):
    rclpy.init(args=args)

    # For testing, create the test node
    test_node = HumanoidPathPlannerTestNode()

    try:
        rclpy.spin(test_node)
    except KeyboardInterrupt:
        test_node.get_logger().info("Humanoid Path Planner Test Node stopped by user")
    finally:
        test_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Now let's create a humanoid-specific controller that works with Nav2:

```python
# humanoid_controller.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import String, Float64
from tf2_ros import TransformListener, Buffer
from tf2_geometry_msgs import do_transform_pose
import numpy as np
import math
from scipy.spatial.distance import cdist
from geometry_msgs.msg import Vector3

class HumanoidController(Node):
    """
    Specialized controller for humanoid robot navigation
    Implements balance-aware control and bipedal locomotion patterns
    """
    def __init__(self):
        super().__init__('humanoid_controller')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/humanoid_robot/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/humanoid_controller/status', 10)
        self.balance_pub = self.create_publisher(Float64, '/humanoid_controller/balance_metric', 10)

        # Subscribers
        self.path_sub = self.create_subscription(Path, '/humanoid_plan', self.path_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/humanoid_robot/odom', self.odom_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/humanoid_robot/imu', self.imu_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/humanoid_robot/scan', self.scan_callback, 10)

        # Timers
        self.control_timer = self.create_timer(0.05, self.control_loop)  # 20 Hz

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Robot state
        self.current_pose = None
        self.current_twist = None
        self.current_imu = None
        self.path = []
        self.current_path_index = 0
        self.path_following_active = False

        # Controller parameters
        self.linear_vel_limit = 0.3  # Conservative for balance
        self.angular_vel_limit = 0.5
        self.lookahead_distance = 0.5
        self.arrival_threshold = 0.2
        self.orientation_threshold = 0.2

        # Balance control parameters
        self.com_height = 0.8  # Center of mass height
        self.balance_margin = 0.1  # Safety margin for support polygon
        self.zmp_tolerance = 0.05  # Zero Moment Point tolerance

        # Locomotion pattern parameters
        self.step_height = 0.05
        self.step_duration = 0.8
        self.swing_phase_ratio = 0.3  # 30% of step for swing phase

        # PID controllers
        self.linear_pid = {
            'kp': 1.0,
            'ki': 0.1,
            'kd': 0.05,
            'error_sum': 0.0,
            'last_error': 0.0
        }

        self.angular_pid = {
            'kp': 2.0,
            'ki': 0.2,
            'kd': 0.1,
            'error_sum': 0.0,
            'last_error': 0.0
        }

        self.get_logger().info("Humanoid Controller initialized")

    def path_callback(self, msg):
        """Receive path and prepare for following"""
        self.path = msg.poses
        self.current_path_index = 0
        self.path_following_active = True

        self.get_logger().info(f"Received path with {len(self.path)} waypoints")

    def odom_callback(self, msg):
        """Update current pose from odometry"""
        self.current_pose = msg.pose.pose
        self.current_twist = msg.twist.twist

    def imu_callback(self, msg):
        """Update IMU data for balance control"""
        self.current_imu = msg

    def scan_callback(self, msg):
        """Process laser scan for obstacle avoidance"""
        # Check for obstacles in path
        if self.path_following_active and len(self.path) > 0:
            self.check_path_clearance(msg)

    def check_path_clearance(self, scan_msg):
        """Check if path is clear of obstacles"""
        # Simple implementation - check if there are obstacles too close in front
        front_ranges = scan_msg.ranges[330:] + scan_msg.ranges[:30]  # ±30 degrees
        front_ranges = [r for r in front_ranges if not math.isnan(r) and 0.1 < r < 3.0]

        if front_ranges:
            min_range = min(front_ranges)
            if min_range < 0.5:  # Obstacle too close
                self.path_following_active = False
                self.get_logger().warn(f"Path blocked! Minimum range: {min_range:.2f}m")

    def compute_control_command(self):
        """Compute control command based on current state and path"""
        if not self.path_following_active or not self.path or not self.current_pose:
            return Twist()

        # Find next target point along path
        target_point = self.get_next_target_point()
        if target_point is None:
            # Check if we're near the end
            last_point = self.path[-1].pose.position
            current_pos = self.current_pose.position
            dist_to_goal = math.sqrt(
                (current_pos.x - last_point.x)**2 +
                (current_pos.y - last_point.y)**2
            )

            if dist_to_goal < self.arrival_threshold:
                # Reached goal
                self.path_following_active = False
                self.get_logger().info("Reached goal position")
                return Twist()  # Stop
            else:
                # Can't find target, stop
                self.get_logger().warn("Could not find target point on path")
                return Twist()

        # Calculate desired velocity towards target
        dx = target_point.x - self.current_pose.position.x
        dy = target_point.y - self.current_pose.position.y
        distance_to_target = math.sqrt(dx**2 + dy**2)

        # Calculate desired orientation
        desired_yaw = math.atan2(dy, dx)

        # Get current orientation
        current_yaw = math.atan2(
            2 * (self.current_pose.orientation.w * self.current_pose.orientation.z +
                 self.current_pose.orientation.x * self.current_pose.orientation.y),
            1 - 2 * (self.current_pose.orientation.y**2 + self.current_pose.orientation.z**2)
        )

        # Calculate orientation error
        orientation_error = math.atan2(math.sin(desired_yaw - current_yaw),
                                      math.cos(desired_yaw - current_yaw))

        # Apply PID control for linear velocity
        linear_error = distance_to_target
        self.linear_pid['error_sum'] += linear_error * 0.05  # dt = 0.05s
        linear_derivative = (linear_error - self.linear_pid['last_error']) / 0.05
        linear_output = (self.linear_pid['kp'] * linear_error +
                        self.linear_pid['ki'] * self.linear_pid['error_sum'] +
                        self.linear_pid['kd'] * linear_derivative)

        self.linear_pid['last_error'] = linear_error

        # Limit linear velocity
        linear_vel = max(-self.linear_vel_limit, min(self.linear_vel_limit, linear_output))

        # Apply PID control for angular velocity
        angular_error = orientation_error
        self.angular_pid['error_sum'] += angular_error * 0.05  # dt = 0.05s
        angular_derivative = (angular_error - self.angular_pid['last_error']) / 0.05
        angular_output = (self.angular_pid['kp'] * angular_error +
                         self.angular_pid['ki'] * self.angular_pid['error_sum'] +
                         self.angular_pid['kd'] * angular_derivative)

        self.angular_pid['last_error'] = angular_error

        # Limit angular velocity
        angular_vel = max(-self.angular_vel_limit, min(self.angular_vel_limit, angular_output))

        # Balance-aware adjustments
        if self.current_imu:
            # Use IMU data to adjust for balance
            roll = math.atan2(
                2 * (self.current_imu.orientation.w * self.current_imu.orientation.x -
                     self.current_imu.orientation.y * self.current_imu.orientation.z),
                1 - 2 * (self.current_imu.orientation.x**2 + self.current_imu.orientation.y**2)
            )
            pitch = math.asin(
                2 * (self.current_imu.orientation.w * self.current_imu.orientation.y +
                     self.current_imu.orientation.x * self.current_imu.orientation.z)
            )

            # If robot is tilting too much, reduce speed
            tilt_magnitude = math.sqrt(roll**2 + pitch**2)
            if tilt_magnitude > 0.1:  # 0.1 rad = ~5.7 degrees
                safety_factor = max(0.1, 1.0 - tilt_magnitude)  # Reduce speed based on tilt
                linear_vel *= safety_factor
                angular_vel *= safety_factor

        # Create and return command
        cmd = Twist()
        cmd.linear.x = linear_vel
        cmd.angular.z = angular_vel

        return cmd

    def get_next_target_point(self):
        """Find the next point on the path to navigate towards"""
        if not self.current_pose or not self.path:
            return None

        # Find the point on the path that is closest to the current position
        current_pos = np.array([self.current_pose.position.x, self.current_pose.position.y])

        # Look for the closest point on the path
        min_distance = float('inf')
        closest_idx = self.current_path_index

        for i in range(self.current_path_index, len(self.path)):
            path_pos = np.array([self.path[i].pose.position.x, self.path[i].pose.position.y])
            distance = np.linalg.norm(current_pos - path_pos)

            if distance < min_distance:
                min_distance = distance
                closest_idx = i

        # Now find a lookahead point
        lookahead_point = None
        for i in range(closest_idx, len(self.path)):
            path_pos = np.array([self.path[i].pose.position.x, self.path[i].pose.position.y])
            distance = np.linalg.norm(current_pos - path_pos)

            if distance >= self.lookahead_distance:
                lookahead_point = self.path[i].pose.position
                self.current_path_index = i
                break

        # If no point is far enough, use the last point
        if lookahead_point is None and len(self.path) > 0:
            lookahead_point = self.path[-1].pose.position

        return lookahead_point

    def calculate_balance_metric(self):
        """Calculate a metric representing how balanced the robot is"""
        if not self.current_imu:
            return 0.5  # Return neutral value if no IMU data

        # Calculate tilt angles from IMU
        roll = math.atan2(
            2 * (self.current_imu.orientation.w * self.current_imu.orientation.x -
                 self.current_imu.orientation.y * self.current_imu.orientation.z),
            1 - 2 * (self.current_imu.orientation.x**2 + self.current_imu.orientation.y**2)
        )
        pitch = math.asin(
            2 * (self.current_imu.orientation.w * self.current_imu.orientation.y +
                 self.current_imu.orientation.x * self.current_imu.orientation.z)
        )

        # Calculate tilt magnitude
        tilt_magnitude = math.sqrt(roll**2 + pitch**2)

        # Convert to balance metric (0.0 = unbalanced, 1.0 = perfectly balanced)
        max_acceptable_tilt = 0.2  # 0.2 rad = ~11.5 degrees
        balance_metric = max(0.0, min(1.0, 1.0 - (tilt_magnitude / max_acceptable_tilt)))

        return balance_metric

    def control_loop(self):
        """Main control loop"""
        cmd = self.compute_control_command()

        # Publish command
        self.cmd_vel_pub.publish(cmd)

        # Calculate and publish balance metric
        balance_metric = self.calculate_balance_metric()
        balance_msg = Float64()
        balance_msg.data = balance_metric
        self.balance_pub.publish(balance_metric)

        # Publish status
        status_msg = String()
        status_msg.data = f"PathIdx: {self.current_path_index}, Cmd: ({cmd.linear.x:.2f}, {cmd.angular.z:.2f}), " \
                         f"Balance: {balance_metric:.2f}, Active: {self.path_following_active}"
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    controller = HumanoidController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info("Humanoid Controller stopped by user")
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Small Simulation

Let's create a simulation environment that demonstrates the humanoid navigation system:

```python
# humanoid_navigation_simulator.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose, Point
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import LaserScan, Imu
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import numpy as np
import math

class HumanoidNavigationSimulator(Node):
    """
    Simulation environment for humanoid navigation
    Demonstrates the interaction between Nav2 and humanoid-specific controllers
    """
    def __init__(self):
        super().__init__('humanoid_navigation_simulator')

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, '/humanoid_robot/odom', 10)
        self.scan_pub = self.create_publisher(LaserScan, '/humanoid_robot/scan', 10)
        self.imu_pub = self.create_publisher(Imu, '/humanoid_robot/imu', 10)
        self.status_pub = self.create_publisher(String, '/navigation_sim_status', 10)

        # Subscribers
        self.cmd_vel_sub = self.create_subscription(Twist, '/humanoid_robot/cmd_vel', self.cmd_vel_callback, 10)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Timers
        self.sim_timer = self.create_timer(0.05, self.simulation_step)  # 20 Hz
        self.sensor_timer = self.create_timer(0.1, self.publish_sensors)  # 10 Hz

        # Robot state
        self.position = np.array([0.0, 0.0, 0.0])
        self.orientation = np.array([0.0, 0.0, 0.0, 1.0])  # x, y, z, w
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.cmd_twist = Twist()

        # Simulation parameters
        self.sim_time = 0.0
        self.robot_radius = 0.3
        self.com_height = 0.8  # Center of mass height

        # Environment setup
        self.obstacles = [
            {'position': np.array([2.0, 1.0]), 'radius': 0.5},
            {'position': np.array([3.0, -1.0]), 'radius': 0.4},
            {'position': np.array([0.0, 3.0]), 'radius': 0.6},
        ]

        # Balance simulation parameters
        self.balance_state = 0.0  # -1.0 to 1.0, where 0.0 is perfectly balanced
        self.balance_damping = 0.95
        self.balance_noise = 0.01

        self.get_logger().info("Humanoid Navigation Simulator initialized")

    def cmd_vel_callback(self, msg):
        """Process velocity commands"""
        self.cmd_twist = msg
        # Apply limits to make it more realistic for humanoid
        self.linear_velocity = max(-0.3, min(0.3, msg.linear.x))
        self.angular_velocity = max(-0.5, min(0.5, msg.angular.z))

    def update_robot_dynamics(self, dt):
        """Update robot position and orientation based on commands"""
        # Get current yaw from orientation quaternion
        current_yaw = math.atan2(
            2 * (self.orientation[3] * self.orientation[2] + self.orientation[0] * self.orientation[1]),
            1 - 2 * (self.orientation[1]**2 + self.orientation[2]**2)
        )

        # Update orientation based on angular velocity
        new_yaw = current_yaw + self.angular_velocity * dt
        self.orientation[2] = math.sin(new_yaw / 2)
        self.orientation[3] = math.cos(new_yaw / 2)

        # Update position based on linear velocity and current orientation
        dx = self.linear_velocity * math.cos(new_yaw) * dt
        dy = self.linear_velocity * math.sin(new_yaw) * dt

        self.position[0] += dx
        self.position[1] += dy

        # Simple balance model - the faster we move, the more we might lose balance
        balance_change = self.linear_velocity * 0.1 + self.angular_velocity * 0.2
        self.balance_state += balance_change * dt
        self.balance_state *= self.balance_damping  # Apply damping
        self.balance_state += np.random.normal(0, self.balance_noise)  # Add noise
        self.balance_state = max(-1.0, min(1.0, self.balance_state))  # Clamp

    def generate_laser_scan(self):
        """Generate simulated laser scan data"""
        num_points = 360
        angle_min = -math.pi
        angle_max = math.pi
        angle_increment = (angle_max - angle_min) / num_points

        ranges = []

        for i in range(num_points):
            angle = angle_min + i * angle_increment

            # Calculate ray direction
            ray_dir = np.array([math.cos(angle), math.sin(angle)])

            # Find closest obstacle in this direction
            min_range = 10.0  # Max range

            for obstacle in self.obstacles:
                # Vector from robot to obstacle
                to_obstacle = obstacle['position'] - self.position[:2]

                # Calculate distance from ray to obstacle center
                # Using ray-circle intersection
                obstacle_distance = np.linalg.norm(to_obstacle)
                obstacle_angle = math.atan2(to_obstacle[1], to_obstacle[0])

                # Check if ray is pointing toward obstacle
                angle_diff = abs(math.atan2(
                    math.sin(angle - obstacle_angle),
                    math.cos(angle - obstacle_angle)
                ))

                if angle_diff < 0.2:  # Within 0.2 rad of obstacle direction
                    # Calculate closest approach
                    closest_approach = abs(np.cross(to_obstacle, ray_dir))

                    if closest_approach < obstacle['radius']:
                        # Ray intersects obstacle circle
                        ray_to_center_proj = np.dot(to_obstacle, ray_dir)

                        if ray_to_center_proj > 0:  # Obstacle is in front
                            range_to_obstacle = ray_to_center_proj - math.sqrt(
                                obstacle['radius']**2 - closest_approach**2
                            )

                            if 0 < range_to_obstacle < min_range:
                                min_range = range_to_obstacle

            # Add some noise to make it more realistic
            noise = np.random.normal(0, 0.02)
            final_range = max(0.1, min_range + noise)
            ranges.append(final_range)

        scan_msg = LaserScan()
        scan_msg.header.stamp = self.get_clock().now().to_msg()
        scan_msg.header.frame_id = 'laser_frame'
        scan_msg.angle_min = angle_min
        scan_msg.angle_max = angle_max
        scan_msg.angle_increment = angle_increment
        scan_msg.time_increment = 0.0
        scan_msg.scan_time = 0.1
        scan_msg.range_min = 0.1
        scan_msg.range_max = 10.0
        scan_msg.ranges = ranges

        return scan_msg

    def generate_imu_data(self):
        """Generate simulated IMU data"""
        imu_msg = Imu()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = 'imu_frame'

        # Simulate orientation with some drift based on balance state
        drift_x = self.balance_state * 0.1
        drift_y = self.balance_state * 0.05

        # Calculate orientation with drift
        current_yaw = math.atan2(
            2 * (self.orientation[3] * self.orientation[2] + self.orientation[0] * self.orientation[1]),
            1 - 2 * (self.orientation[1]**2 + self.orientation[2]**2)
        )

        # Apply small drift based on balance state
        corrected_yaw = current_yaw + drift_x

        # Convert to quaternion
        imu_msg.orientation.x = 0.0 + np.random.normal(0, 0.001)  # Small noise
        imu_msg.orientation.y = drift_y + np.random.normal(0, 0.001)
        imu_msg.orientation.z = math.sin(corrected_yaw / 2) + np.random.normal(0, 0.001)
        imu_msg.orientation.w = math.cos(corrected_yaw / 2) + np.random.normal(0, 0.001)

        # Angular velocity (simulated)
        imu_msg.angular_velocity.x = np.random.normal(0, 0.01)
        imu_msg.angular_velocity.y = np.random.normal(0, 0.01)
        imu_msg.angular_velocity.z = self.angular_velocity + np.random.normal(0, 0.01)

        # Linear acceleration (simulated gravity + movement)
        imu_msg.linear_acceleration.x = self.linear_velocity * 0.1 + np.random.normal(0, 0.05)
        imu_msg.linear_acceleration.y = np.random.normal(0, 0.05)
        imu_msg.linear_acceleration.z = 9.81 + np.random.normal(0, 0.05)  # Gravity

        return imu_msg

    def simulation_step(self):
        """Main simulation step"""
        dt = 0.05  # 20 Hz
        self.sim_time += dt

        # Update robot dynamics
        self.update_robot_dynamics(dt)

        # Create and publish odometry
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_footprint'

        odom_msg.pose.pose.position.x = float(self.position[0])
        odom_msg.pose.pose.position.y = float(self.position[1])
        odom_msg.pose.pose.position.z = float(self.position[2])
        odom_msg.pose.pose.orientation.x = float(self.orientation[0])
        odom_msg.pose.pose.orientation.y = float(self.orientation[1])
        odom_msg.pose.pose.orientation.z = float(self.orientation[2])
        odom_msg.pose.pose.orientation.w = float(self.orientation[3])

        # Set twist
        odom_msg.twist.twist.linear.x = self.linear_velocity
        odom_msg.twist.twist.angular.z = self.angular_velocity

        self.odom_pub.publish(odom_msg)

        # Broadcast transform
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_footprint'
        t.transform.translation.x = float(self.position[0])
        t.transform.translation.y = float(self.position[1])
        t.transform.translation.z = float(self.position[2])
        t.transform.rotation.x = float(self.orientation[0])
        t.transform.rotation.y = float(self.orientation[1])
        t.transform.rotation.z = float(self.orientation[2])
        t.transform.rotation.w = float(self.orientation[3])
        self.tf_broadcaster.sendTransform(t)

    def publish_sensors(self):
        """Publish sensor data"""
        scan_msg = self.generate_laser_scan()
        self.scan_pub.publish(scan_msg)

        imu_msg = self.generate_imu_data()
        self.imu_pub.publish(imu_msg)

        # Publish status
        status_msg = String()
        status_msg.data = f"Pos: ({self.position[0]:.2f}, {self.position[1]:.2f}), " \
                         f"Vel: ({self.linear_velocity:.2f}, {self.angular_velocity:.2f}), " \
                         f"Balance: {self.balance_state:.2f}, Cmd: ({self.cmd_twist.linear.x:.2f}, {self.cmd_twist.angular.z:.2f})"
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    simulator = HumanoidNavigationSimulator()

    try:
        rclpy.spin(simulator)
    except KeyboardInterrupt:
        simulator.get_logger().info("Humanoid Navigation Simulator stopped by user")
    finally:
        simulator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Quick Recap

In this lesson, we've covered:

- **Nav2 Configuration**: Adapting Nav2 parameters for humanoid robot kinematics and safety requirements
- **Custom Path Planning**: Implementing balance-aware path planning with ZMP (Zero Moment Point) considerations
- **Humanoid Controllers**: Creating controllers that respect bipedal locomotion constraints
- **Balance Integration**: Incorporating balance and stability metrics into navigation
- **Simulation Environment**: Demonstrating the complete navigation system in simulation

The Nav2 framework can be successfully adapted for humanoid robots by considering their unique constraints: balance requirements, limited support polygon, and complex locomotion patterns. The key is to modify both the planning and control layers to account for the robot's bipedal nature and stability requirements.

## Summary of Chapter 4

In Chapter 4: "The AI-Robot Brain (NVIDIA Isaac)", we've covered:

1. **Introduction to NVIDIA Isaac Sim**: Core concepts and capabilities of the Isaac Sim platform
2. **Synthetic Data Generation**: Creating diverse, labeled datasets for AI training with domain randomization
3. **Isaac ROS for VSLAM**: Implementing visual SLAM systems with Isaac ROS packages
4. **Nav2 Path Planning for Humanoid Robots**: Adapting navigation for complex bipedal robots

This chapter has provided a comprehensive overview of how AI techniques, particularly those enabled by NVIDIA's Isaac platform, can enhance robotic perception, mapping, and navigation capabilities.