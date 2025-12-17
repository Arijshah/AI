// sidebars.js
/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render navigation automatically
 */

module.exports = {
  docs: [
    'intro',
    {
      type: 'category',
      label: 'Physical AI',
      items: [
        {
          type: 'category',
          label: 'Introduction to Physical AI',
          items: [
            'physical-ai/introduction-to-physical-ai/what-is-physical-ai',
            'physical-ai/introduction-to-physical-ai/humanoid-robotics-overview',
            'physical-ai/introduction-to-physical-ai/tools-and-platforms'
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'The Robotic Nervous System (ROS 2)',
      items: [
        'physical-ai/the-robotic-nervous-system-ros2/introduction-to-ros2',
        'physical-ai/the-robotic-nervous-system-ros2/ros2-nodes-and-parameters',
        'physical-ai/the-robotic-nervous-system-ros2/urdf-and-humanoid-robot-modeling',
        'physical-ai/the-robotic-nervous-system-ros2/practical-ros2-workflows'
      ],
    },
    {
      type: 'category',
      label: 'The Digital Twin (Gazebo & Unity)',
      items: [
        'physical-ai/the-digital-twin-gazebo-unity/introduction-to-gazebo-simulation',
        'physical-ai/the-digital-twin-gazebo-unity/physics-and-collision-modeling',
        'physical-ai/the-digital-twin-gazebo-unity/unity-based-visualization',
        'physical-ai/the-digital-twin-gazebo-unity/sensor-simulation'
      ],
    },
    {
      type: 'category',
      label: 'The AI-Robot Brain (NVIDIA Isaac)',
      items: [
        'physical-ai/the-ai-robot-brain-nvidia-isaac/introduction-to-nvidia-isaac-sim',
        'physical-ai/the-ai-robot-brain-nvidia-isaac/synthetic-data-generation',
        'physical-ai/the-ai-robot-brain-nvidia-isaac/isaac-ros-vslam',
        'physical-ai/the-ai-robot-brain-nvidia-isaac/nav2-path-planning-humanoid'
      ],
    },
    {
      type: 'category',
      label: 'Vision-Language-Action (VLA)',
      items: [
        'physical-ai/vision-language-action-vla/introduction-to-vla-integration',
        'physical-ai/vision-language-action-vla/voice-to-action-using-whisper',
        'physical-ai/vision-language-action-vla/cognitive-planning-with-llms',
        'physical-ai/vision-language-action-vla/capstone-project-autonomous-humanoid-robot'
      ],
    },
  ],
};