from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# State description
class AgentState(TypedDict):
    question: str
    result: str

# creating two nodes
def first_node(state: AgentState) -> dict:
    return {'result': 'Запит отримано: ' + state['question']}

def second_node(state: AgentState) -> dict:
    return {'result': state['result'] + '. Обробку завершено.'}

# graph construction
builder = StateGraph(AgentState)

builder.add_node('first', first_node)
builder.add_node('second', second_node)

builder.add_edge(START, 'first')
builder.add_edge('first', 'second')
builder.add_edge('second', END)

graph = builder.compile()

# graph running
result = graph.invoke({
    'question': 'Який статус повітряної тривоги?',
    'result': ''
})

print(result)