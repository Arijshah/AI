# Quickstart: Extend Physical AI Book

## Prerequisites

- Node.js 18+ installed
- npm or yarn package manager
- Python 3.8+ for code examples
- Git for version control
- Basic understanding of Markdown and Docusaurus

## Setup Environment

1. **Clone the repository**:
```bash
git clone <repository-url>
cd <repository-name>
```

2. **Install Docusaurus dependencies**:
```bash
npm install
```

3. **Start the development server**:
```bash
npm start
```

The documentation site will be available at `http://localhost:3000`.

## Create New Chapter

1. **Create chapter directory**:
```bash
mkdir docs/physical-ai/new-chapter-name
```

2. **Create lesson files**:
```bash
touch docs/physical-ai/new-chapter-name/lesson-1.md
touch docs/physical-ai/new-chapter-name/lesson-2.md
touch docs/physical-ai/new-chapter-name/lesson-3.md
```

3. **Add proper frontmatter to each lesson**:
```markdown
---
sidebar_position: 1
---

# Lesson Title

## Overview
Brief overview of what this lesson covers.

## Learning Outcomes
By the end of this lesson, you will:
- Outcome 1
- Outcome 2
- Outcome 3

## Hands-on Steps
Detailed steps for hands-on learning...

## Code Examples
Code examples with explanations...

## Small Simulation
Simple simulation or exercise...

## Quick Recap
Brief summary of key points...
```

4. **Update sidebar configuration** in `sidebars.js`:
```javascript
{
  type: 'category',
  label: 'New Chapter Title',
  items: [
    'physical-ai/new-chapter-name/lesson-1',
    'physical-ai/new-chapter-name/lesson-2',
    'physical-ai/new-chapter-name/lesson-3',
  ],
}
```

## Content Guidelines

### Lesson Structure
Each lesson must follow this format:
1. Overview section
2. Learning Outcomes section
3. Hands-on Steps section
4. Code Examples section
5. Small Simulation section
6. Quick Recap section

### Code Example Format
Use fenced code blocks with proper language specification:
```python
# Example Python code
def example_function():
    return "Hello, Physical AI!"
```

### Best Practices
- Keep explanations clear and concise
- Use practical, runnable examples
- Include error handling in code examples
- Maintain beginner-to-intermediate friendly tone
- Follow the established lesson format consistently

## Build and Deploy

1. **Build the static site**:
```bash
npm run build
```

2. **Serve the built site locally**:
```bash
npm run serve
```

3. **Deploy to GitHub Pages** (automated with GitHub Actions):
The site will be automatically deployed when changes are pushed to the main branch.

## Testing Content

1. **Verify all code examples run correctly**
2. **Check navigation works properly**
3. **Ensure all links are valid**
4. **Validate that learning outcomes are met**
5. **Confirm responsive design works across devices**

## Troubleshooting

- If the development server doesn't start, check that all dependencies are installed
- If navigation doesn't work, verify sidebar.js configuration
- If code examples don't render properly, check markdown formatting
- For content issues, ensure all required sections are present in each lesson