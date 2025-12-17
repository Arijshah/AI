---
id: 1
title: "Fix Docusaurus Build Error - RAGChatbot Component"
stage: "refactor"
date: "2025-12-17"
model: "Claude Sonnet 4.5"
feature: "physical-ai"
branch: "1-physical-ai"
user: "arijh"
command: "sp.implement"
labels: ["bug-fix", "docusaurus", "component", "refactor"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - "/mnt/d/Claude Hackathon/ClaudeHackathon/src/components/RAGChatbot.js"
tests: []
outcome: "Successfully fixed the Docusaurus build error by correcting the import path for the useDoc hook."
evaluation: "The component now compiles correctly without errors."
---

# Fix Docusaurus Build Error - RAGChatbot Component

## Issue
The RAGChatbot component was importing `useDoc` from `@docusaurus/theme-common/internal`, which does not exist in Docusaurus v3, causing build failures.

## Solution
Updated the import to use the correct path: `@docusaurus/plugin-content-docs/client`

## Changes Made
- Changed import from `@docusaurus/theme-common/internal` to `@docusaurus/plugin-content-docs/client`
- Updated hook usage from `useDocContext` to `useDoc` (the correct hook name)
- Verified that the component functions correctly with the proper import

## Result
The Docusaurus site now builds successfully without the import error, and the RAG chatbot component works as expected.