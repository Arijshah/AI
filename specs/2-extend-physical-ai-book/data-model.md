# Data Model: Extend Physical AI Book

## Entities

### Lesson
- **Fields**:
  - title: string
  - slug: string
  - position: number
  - content: markdown text
  - learning_outcomes: array of strings
  - prerequisites: array of strings
  - code_examples: array of code blocks
  - hands_on_steps: array of step descriptions
- **Relationships**: belongs to Chapter
- **Validation**: title required, content required, position unique within chapter

### Chapter
- **Fields**:
  - title: string
  - slug: string
  - position: number
  - description: string
  - lessons: array of Lesson references
  - prerequisites: array of strings
- **Relationships**: contains many Lessons, belongs to Physical AI Book
- **Validation**: title required, lessons array not empty

### Code Example
- **Fields**:
  - language: string (python, bash, etc.)
  - code: string (the actual code)
  - description: string
  - lesson_id: reference to Lesson
  - purpose: string (what the code demonstrates)
- **Relationships**: belongs to Lesson
- **Validation**: language required, code required, lesson_id required

### Learning Outcome
- **Fields**:
  - text: string (the outcome statement)
  - lesson_id: reference to Lesson
  - measurable: boolean (can the outcome be verified)
- **Relationships**: belongs to Lesson
- **Validation**: text required, lesson_id required

### Docusaurus Navigation Item
- **Fields**:
  - type: string (category or doc)
  - label: string
  - items: array of references (for categories)
  - id: string (for docs)
  - position: number
- **Relationships**: represents navigation structure
- **Validation**: type required, label required for categories, id required for docs

## State Transitions

### Lesson State
- draft → review → published
- published → updated → published
- published → deprecated (when content becomes outdated)

### Chapter State
- planning → draft → review → published
- published → updating → published
- published → archived (when chapter is no longer relevant)

## Validation Rules

### Content Requirements
- Each Lesson must have 3-5 learning outcomes
- Each Lesson must contain at least one code example
- Each Lesson must include hands-on steps
- Each Lesson must follow the specified format: Overview → Hands-on steps → Code examples → Small simulation → Quick recap

### Navigation Requirements
- Each Chapter must appear in sidebar.js
- Each Lesson must be accessible through navigation
- Navigation structure must follow logical progression
- Sidebar must be organized by chapter with proper lesson nesting

### Quality Requirements
- All code examples must be functional
- All examples must include proper error handling
- Content must be beginner-to-intermediate friendly
- All technical concepts must be clearly explained