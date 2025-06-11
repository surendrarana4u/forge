---
layout: default
title: Operation Modes
parent: Features
nav_order: 6
---

# Operation Modes

Forge uses different specialized agents to provide flexible assistance based on your needs. You can switch between these agents to get the right type of help for your current task:

## Acting Agent (Default)

The Acting Agent is the default when you start Forge and is empowered to directly implement changes to your codebase and execute commands:

- **Full Execution**: Forge can modify files, create new ones, and execute shell commands
- **Implementation**: Directly implements the solutions it proposes
- **Verification**: Performs verification steps to ensure changes work as intended
- **Best For**: When you want Forge to handle implementation details and fix issues directly

**Example**:

```bash
# Switch to Acting Agent within a Forge session
/act

# Or use the general agent command and then from drop down select `act` agent
/agent
```

## Planning Agent

The Planning Agent analyzes and plans but doesn't modify your codebase:

- **Read-Only Operations**: Can only read files and run non-destructive commands
- **Detailed Analysis**: Thoroughly examines code, identifies issues, and proposes solutions
- **Structured Planning**: Provides step-by-step action plans for implementing changes
- **Best For**: When you want to understand what changes are needed before implementing them yourself

**Example**:

```bash
# Switch to Planning Agent within a Forge session
/plan

# Or use the general agent command and then from drop down select `plan` agent
/agent 
```

You can easily switch between agents during a session using the `/act`, `/plan`, or `/agent` commands. The Planning Agent is especially useful for reviewing potential changes before they're implemented, while the Acting Agent streamlines the development process by handling implementation details for you.