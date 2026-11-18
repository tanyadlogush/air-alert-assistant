from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from langgraph.checkpoint.memory import InMemorySaver


class PracticeState(TypedDict):
    number: int
    message: str


def start_node(state: PracticeState) -> dict:
    res = state['number'] + 1
    return {'number': res}

def process_node(state: PracticeState):
    return {'number': state['number'] * 2}

def large_node(state: PracticeState):
    return {'message': 'Число велике'}

def small_node(state: PracticeState):
    return {'message': 'Число мале'}


def route_func(state: PracticeState):
    if state['number'] >= 10:
        return 'large_node'
    return 'small_node'


builder = StateGraph(PracticeState)

builder.add_node('start_node', start_node)
builder.add_node('process_node', process_node)
builder.add_node('large_node', large_node)
builder.add_node('small_node', small_node)

builder.add_edge(START, 'start_node')

builder.add_conditional_edges(  # for the conditional_edge
    'start_node',
    route_func,
    {
        'large_node': 'large_node',
        'small_node': 'small_node',
    }
)
builder.add_edge('large_node', END)
builder.add_edge('small_node', END)



memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)

config = {'configurable': {'thread_id': 'user_123'}}


input_data = {
    'number': 9,
    'message': 'Початкове повідомлення',
}

result = graph.invoke(input_data, config=config)

input_data_2 = {
    'number': 2,
    'message': 'Другий запит'
}

result_2 = graph.invoke(input_data_2, config=config)

history = list(graph.get_state_history(config))
print(len(history))

for snapshot in history:
    print(
        snapshot.values,
        'Next:', snapshot.next
    )

# s = graph.get_state(config)
# print(result_2)
# print(s.values)
dvuhv = 22-2

