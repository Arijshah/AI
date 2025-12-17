# Physical AI Book Development Guidelines

Auto-generated from all feature plans. Last updated: 2025-12-17

## Active Technologies

- Docusaurus v3 (documentation framework)
- Markdown (content format)
- Node.js (runtime environment)
- npm/yarn (package management)
- Python 3.8+ (code examples)
- ROS 2 (robotics middleware)
- Gazebo (simulation environment)
- Unity (visualization platform)
- NVIDIA Isaac Sim (AI robotics platform)
- OpenAI Whisper (speech recognition)
- Large Language Models (cognitive planning)

## Project Structure

```text
docs/
├── physical-ai/
│   ├── introduction-to-physical-ai/           # Chapter 1 (existing)
│   │   ├── what-is-physical-ai.md
│   │   ├── humanoid-robotics-overview.md
│   │   └── tools-and-platforms.md
│   ├── the-robotic-nervous-system-ros2/       # Chapter 2 (new)
│   │   ├── introduction-to-ros2.md
│   │   ├── ros2-nodes-and-parameters.md
│   │   ├── urdf-and-humanoid-robot-modeling.md
│   │   └── practical-ros2-workflows.md
│   ├── the-digital-twin-gazebo-unity/         # Chapter 3 (new)
│   │   ├── introduction-to-gazebo-simulation.md
│   │   ├── physics-and-collision-modeling.md
│   │   ├── unity-based-visualization.md
│   │   └── sensor-simulation.md
│   ├── the-ai-robot-brain-nvidia-isaac/       # Chapter 4 (new)
│   │   ├── introduction-to-nvidia-isaac-sim.md
│   │   ├── synthetic-data-generation.md
│   │   ├── isaac-ros-vslam.md
│   │   └── nav2-path-planning-humanoid.md
│   └── vision-language-action-vla/            # Chapter 5 (new)
│       ├── introduction-to-vla-integration.md
│       ├── voice-to-action-using-whisper.md
│       ├── cognitive-planning-with-llms.md
│       └── capstone-project-autonomous-humanoid-robot.md
├── intro.md
└── ...
specs/
├── 1-physical-ai/
│   └── spec.md
└── 2-extend-physical-ai-book/
    ├── plan.md
    ├── research.md
    ├── data-model.md
    ├── quickstart.md
    ├── contracts/
    └── tasks.md
sidebars.js
package.json
docusaurus.config.js
```

## Commands

### Docusaurus Commands
```bash
npm start                    # Start development server
npm run build               # Build static site
npm run serve               # Serve built site locally
npm run deploy              # Deploy to GitHub Pages
```

### Content Creation Commands
```bash
# Create new lesson
touch docs/physical-ai/chapter-name/lesson-name.md

# Update sidebar
# Edit sidebars.js to add new content
```

### Development Workflow
```bash
git checkout -b feature-branch
# Make changes
npm start  # Test locally
git add .
git commit -m "Add descriptive commit message"
git push origin feature-branch
```

## Code Style

### Markdown Style
- Use proper frontmatter with required fields
- Follow lesson format: Overview → Learning Outcomes → Hands-on Steps → Code Examples → Small Simulation → Quick Recap
- Use fenced code blocks with language specification
- Include learning outcomes as bullet points
- Use consistent heading hierarchy (##, ###)

### Python Code Style (for examples)
- Use clear variable names
- Include comments explaining complex concepts
- Add error handling where appropriate
- Follow PEP 8 guidelines
- Include docstrings for functions/classes

### Docusaurus-Specific Style
- Use sidebar_position for proper navigation order
- Include proper relative links
- Use Docusaurus-specific components when needed
- Follow Docusaurus styling conventions

## Recent Changes

### 2-extend-physical-ai-book: Extend Physical AI Book
- Added 4 new chapters (2-5) covering ROS 2, simulation, AI perception, and language-action integration
- Created 16-20 lessons with hands-on examples and code
- Implemented beginner-to-intermediate friendly content
- Added proper navigation and sidebar configuration

### 1-physical-ai: Initial Physical AI Book
- Created basic Docusaurus structure
- Added Chapter 1: Introduction to Physical AI
- Implemented 3 lessons: What is Physical AI?, Humanoid Robotics Overview, Tools and Platforms
- Set up proper documentation framework

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->