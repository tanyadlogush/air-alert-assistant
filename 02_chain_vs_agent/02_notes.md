# CHAIN vs AGENT

| Feature | Chain (Deterministic) | Agent (Dynamic) |
| :--- | :--- | :--- |
| **Control Flow** | Hardcoded by developer | Decided by LLM |
| **Execution Path** | Fixed sequence (`A → B → C`) | Dynamic branching & iteration |
| **Tool Usage** | Explicitly invoked by code | Autonomous selection based on input |
| **Flexibility** | Low (rigid workflow) | High (adapts to request context) |

## Chain Concept
A deterministic workflow where the execution order (flow of control) is hardcoded in Python.

### Flow Control:
Program logic dictates when and how the LLM or external functions are called. 
The model does not choose the next step.

### LLM Role:
Acts as a specialized node (e.g., text extraction, output formatting, or semantic transformation).

### Single vs Multi-Tool Calls:
Combining sub-tasks into a single Python wrapper function reduces API latency, token usage, and execution drift.



## Agent Concept

An **Agent** is an AI design pattern where the **LLM controls the flow of execution**. 
Instead of following a hardcoded path, the model dynamically decides which tools to call, 
in what order, and when to terminate the execution loop.

### The Agent Loop Pattern
```text
User Request ──> [ LLM Call (with Tools Schema) ]
                      │
                      ├──> Case A: Model requests Tool Call
                      │       │
                      │       └──> Python executes function ──┐
                      │            & appends result to State ─┘
                      │
                      └──> Case B: Final Answer ──> [ Break Loop ]

```

### Key Components

- **Agent Loop (while True)**: 
The engine that feeds tool results back to the LLM until a stopping condition is met.

- **Tool Schemas**: 
JSON definitions enabling the model to construct structured argument payloads.

- **Agent State**: 
The structured array of messages/inputs capturing the full history of iterations and function execution outputs.

- **Stateless Nature**: 
Agents are inherently stateless by default. Persistent memory across independent sessions is an additive component, 
not a required agent trait.