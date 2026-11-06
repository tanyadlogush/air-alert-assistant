# LangGraph

## Core concepts
* **Node** — the function that performs a special operation.
* **Edge** — defines the next node to execute. It can be fixed or conditional.
* **State** - shared data that Nodes read and updat during graph execution.

### Work together
A node read data from State —> A node performs an operation —> returns updates to State —>
—> LangGraph applies the updates and follows the graph's transition rules.

### Key principles
- Nodes exchange results through the shared State.
- Conditional edges select the next node based on execution results.
- LangGraph manages transitions and State updates according to the defined graph logic.

## State and Nodes in LangGraph
**State** is shared data read and update by graph nodes.

- `TypeDict` can define the Sate schema.
- A node receives the current State and returns a dictionary of updates.
- LangGraph applies the updates; unchanged fields usually retain their values.
- A **reducer** (special update mechanism) defines how updates are applied, 
  such as appending items to a list instead of replacing it.


## Building a Graph in LangGraph
- StateGraph(AgentSystem) creates a graph with defined state schema.
- add_node() registers a node.
- add_edge() defines a regular transition.
- add_conditional_edges() selects the next node based on a routing function.
- START and END mark graph entry and completion.
- compile() prepares the graph for execution.
- invoke(initial_state) runs the graph and returns the final state.

## Conditional edges
- `add_conditional_edges()` selects the next node based on a routing function.
- The function reads the current State and returns a route name.
- A mapping connects route names and graph nodes.
- Conditional edges support branching and loops.
- Loops need explicit stopping conditions to prevent infinite execution.
- Check for successful result before checking the retry limit when success should take priority.
- END terminates graph execution.


## Loops & Reruns
- Conditional edges can create loops by routing back to the same node.
- `search_attempts` increments on every search execution.
- Check for a successful result first. If no result is found, stop after **N** attempts.
- Three attempts = one initial **search + (N-1)** retries.

## Tools & Agents in a LangGraph
- **Tool calling:** The `LLM` selects an available tool and provides arguments. 
`ToolNode` executes it and returns The result to the State for the LLM to use.
- **LangGraph**: `ToolNode` handles tool execution; 
`tools_condition` routes execution based on whether the LLM requests a tool or is ready to respond.
- An agent uses an `LLM` to choose its next action, including calling an available tool. 
`ToolNode `executes tools, and their results are added to `MessagesState.messages` 
so the `LLM` can use them in subsequent steps. 
`tools_condition` routes execution to a tool or ends the agent f? flow

Notes:
- `@tool` converts a Python function into a tool.
- `bind_tools()` provides tools to the model.
- `ToolNode` executes tool calls.
- `tools_condition` handles routing.
- `MessagesState` stores message history.
- `ToolMessage` contains tool execution results.
- The LLM–Tools loop supports multiple tool calls