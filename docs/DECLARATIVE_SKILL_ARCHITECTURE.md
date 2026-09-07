# Declarative Skill Architecture

## Overview

The system introduces a declarative skill layer between Conversation Supervisor routing and execution workflows.

A Skill describes:

- usage scenarios
- workflow entry points
- required tools
- execution constraints
- expected outputs

## Design

User Query -> Conversation Supervisor -> Skill Selection -> Existing Workflow -> Tools/RAG

## Motivation

Skills avoid hard-coding every user intent in prompts and provide a scalable way to extend research capabilities without creating multiple autonomous agents.

The project keeps a single conversation-oriented decision entry. Skills represent reusable research capabilities rather than independent agents.
