from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt, Command

class PracticeState(TypedDict):
    report: str
    # human_decision: bool | None      # для випадку відповіді так/ні script 1
    human_decision: str | None                # для текстової відповіді script 2
    final_result: str | None


def prepare_node(state):
    text = 'Звіт готовий до перевірки'
    return {'report': text}

# script 1
# def human_review_node(state):
#     human_response = interrupt(state['report'])
#     return {'human_decision': human_response}

# script 2
def human_review_node(state):
    human_response = interrupt(
        f"Перевірте звіт: {state['report']}. Залиште коментар."
    )
    return {'human_decision': human_response}


# # script 1
# def final_node(state):
#     if state['human_decision']:
#         return {'final_result': 'Звіт підтверджено'}
#     return {'final_result': 'Звіт відхилено'}


#script 2
def final_node(state):
    comment = state['human_decision']
    if state['human_decision']:
        return {'final_result': f'Коментар отримано: {comment}'}
    return {'final_result': 'Коментар відсутній'}


builder = StateGraph(PracticeState)

builder.add_node('prepare_node', prepare_node)
builder.add_node('human_review_node', human_review_node)
builder.add_node('final_node', final_node)

builder.add_edge(START, 'prepare_node')
builder.add_edge('prepare_node', 'human_review_node')
builder.add_edge('human_review_node', 'final_node')
builder.add_edge('final_node', END)

memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)

# script 1
# сценарій 1/1 — людина схвалює звіт
config = {'configurable': {'thread_id': 'review_1'}}

result = graph.invoke(
    {'report': '', 'human_decision': None, 'final_result': ''},
    config=config
)

result_2 = graph.invoke(Command(resume=True), config=config)

print(result)
print(graph.get_state(config))



# сценарій 1/2 — людина відхиляє звіт
config_2 = {'configurable': {'thread_id': "review_2"}}

result_no = graph.invoke(
    {'report': '', 'human_decision': None, 'final_result': ''},
    config=config_2
)

result_no_2 = graph.invoke(Command(resume=False), config=config_2)

print(result_no_2)
print(graph.get_state(config_2))




# script 2

config_com = {'configurable': {'thread_id': 'review_3'}}

result_com  = graph.invoke(
    {'report': '', 'human_decision': None, 'final_result': ''},
    config=config_com
)

result_com_2 = graph.invoke(Command(resume='Потрібно уточнити дані за Києвом'), config=config_com )

print(result_com)
print(result_com_2)
print(graph.get_state(config_com ))










