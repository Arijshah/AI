ID: 3
TITLE: fix-docusaurus-config
STAGE: misc
DATE_ISO: 2025-12-17
SURFACE: agent
MODEL: claude-3-5-sonnet
FEATURE: physical-ai
BRANCH: main
USER: user
COMMAND: fix-docusaurus-config
LABELS: ["docusaurus", "config", "fix"]
LINKS: null
FILES_YAML:
 - docusaurus.config.js
TESTS_YAML:
 - null

# Prompt
Fix this:
Object literal may only specify known properties, and 'onBrokenLinks' does not exist in type 'DeepPartial<MarkdownHooks>'.```javascript
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'warn',
      onBrokenLinks: 'warn',
    },
  },

```

# Response
Removed `onBrokenLinks` from `markdown.hooks` in `docusaurus.config.js` as it is not a valid property in `MarkdownHooks` and is already defined at the top level.
