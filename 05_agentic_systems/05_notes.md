# Agentic Systems

## Agent Loop
— a cycle of interaction between LLM, tools, and the application. 

It repeats until the model generates a final response or a stopping condition is reached.

**Key principle**: LLM decides what to do; the application executes tools and manages the loop.


## Tool selection & execution
- **selection**: the model selects a tool and forms arguments 
- **execution**: the application validates the request and executes the appropriate function
- **error handling**: if the tool is not available or the call is invalid, 
- the application handles the error 
- and returns it to the model if necessary
- **tool result**: the execution result is passed to the model fso that it can to continue working


## State
— data that the agent stores and updates during the execution of a task.

### State contains:
- `messages`: user requests, tool calls, and results.
- intermediate results and service data needed to control execution.
- step counter and control parameters.

### State vs History vs Memory 
- **State** — the current state of the agent's execution
- **Conversation history** — the history of messages and interactions
- **Memory** — information stored for use in future steps or conversations
- **Summary** — a concise summary of the history that can be used as part of the memory.

### Key principles
* For multi-step task, the agent should store data between steps.
* It is not necessary to pass the entire previous history for a new independent request.
* The memory can be organizes as a short summary: topic, result, time, or other important facts.
* Long-term Memory is not mandatory for every agent.


## Stopping Conditions
Basic conditions for agent termination:
- **final response**: the model reported that it had completed its work
- **step limit**: the agent has reached the maximum allowed number of iterations
- **time or resource limit**: for example, the allowable time or budget of the request has been exhausted
- **uncoverable error**: it is impossible or unsafe to continue the work


## Error handling & retries

- **Error handling** — controlled error handing during agent execution/
- **Retry** — retrying an operation after an error.
- Only errors that may be temporary should be retried.
- The number of retries should be limited.
- If all attempts fail, the application should report the error correctly.
- Incorrect arguments require correction, not just retry.
- The model can help resume execution, but the application controls the iterations and their limits.


## Human-in-the-loop
- HITL helps to control risky, irreversible or ambiguous actions.
- For critical actions, the application should block execution until confirmation is received.
- LLM proposes an action —> the application controls its execution —> 
—> the human (if necessary) confirms (rejects, corrects) it.

