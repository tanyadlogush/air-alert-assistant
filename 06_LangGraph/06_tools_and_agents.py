from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END, MessagesState

from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()


@tool
# simulation of receiving air alert status
def get_air_alert_status(region: str) -> str:
    """Get the current air alert status for a region."""
    statuses = {
        'Київ': 'ALERT',
        'Львів': 'NO ALERT',
    }

    return statuses.get(region, 'UNKNOWN')


tools = [get_air_alert_status, ]

# ToolNode will be able to execute the tools that the model requests.
tools_node = ToolNode(tools)

# model
model = ChatOpenAI(model='gpt-4o')
model_with_tools = model.bind_tools(tools)


# llm_node
def llm_node(state: MessagesState) -> dict:
    response = model_with_tools.invoke(state['messages'])  # passes the entire existing message history to the model
    return {'messages': [response]}  # returns a new model message to add to history


# LangGraph has a ready-made state type (structure) — MessagesState (instead of the class AgentState()...)
builder = StateGraph(MessagesState)

builder.add_node('llm', llm_node)
builder.add_node('tools', tools_node)

builder.add_edge(START, 'llm')
builder.add_conditional_edges(
    'llm',
    tools_condition,
)

builder.add_edge('tools', 'llm')

graph = builder.compile()

result = graph.invoke({'messages':['статус повітряної тривоги в Києві',]})
print(result)