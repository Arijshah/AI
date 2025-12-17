---
id: 2
title: "Fix Runtime Error - RAGChatbot DocProvider Issue"
stage: "refactor"
date: "2025-12-17"
model: "Claude Sonnet 4.5"
feature: "physical-ai"
branch: "1-physical-ai"
user: "arijh"
command: "sp.implement"
labels: ["bug-fix", "docusaurus", "component", "runtime-error"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - "/mnt/d/Claude Hackathon/ClaudeHackathon/src/components/RAGChatbot.js"
tests: []
outcome: "Successfully fixed the runtime error by removing the useDoc hook that was causing 'Hook useDocContext is called outside the <DocProvider>' error."
evaluation: "The component now works on all pages without crashing."
---

# Fix Runtime Error - RAGChatbot DocProvider Issue

## Issue
The RAGChatbot component was using the `useDoc` hook which can only be used within documentation pages (inside a DocProvider). When the component was rendered on non-documentation pages, it caused a runtime error: "Hook useDocContext is called outside the <DocProvider>".

## Solution
Removed the dependency on the `useDoc` hook entirely since the current document title was not actually being used in the component functionality.

## Changes Made
- Removed import of `useDoc` from `@docusaurus/plugin-content-docs/client`
- Removed the hook call that was causing the error
- Set a default value for `currentDocTitle` to maintain the same interface
- Updated comments to reflect the change

## Result
The RAGChatbot component now works on all pages without runtime errors while maintaining all functionality.