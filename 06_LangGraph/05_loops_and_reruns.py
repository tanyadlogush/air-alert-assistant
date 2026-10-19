from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# 1. description of the State structure
class AgentState(TypedDict):
    question: str
    context: str
    answer: str
    search_attempts: int


# 2. document search simulation
# This is a training feature, not a real search service.
def search_documents(question: str) -> str:
    if 'правила' in question.lower():
        return 'Навчальний документ: правила безпеки під час тривоги.'

    return ''


# 3. search node
def search_node(state: AgentState) -> dict:
    context = search_documents(state['question'])

    return {
        'context': context,
        'search_attempts': state['search_attempts'] + 1,
    }


# 4. route function
def route_after_search(state: AgentState) -> str:
    if state['context']:
        return 'answer'

    if state['search_attempts'] >= 3:
        return 'no_info'

    return 'search'


# 5. answer formation node
def answer_node(state: AgentState) -> dict:
    return {
        'answer': f"Знайдена інформація: {state['context']}"
    }


# 6. node in case of lack of information
def no_info_node(state: AgentState) -> dict:
    return {
        'answer': 'Відповідної інформації не знайдено.'
    }


# 7. graph creation
builder = StateGraph(AgentState)

builder.add_node('search', search_node)
builder.add_node('answer_node', answer_node)
builder.add_node('no_info_node', no_info_node)

builder.add_edge(START, 'search')

builder.add_conditional_edges(
    'search',
    route_after_search,
    {
        'answer': 'answer_node',
        'no_info': 'no_info_node',
        'search': 'search',
    }
)

builder.add_edge('answer_node', END)
builder.add_edge('no_info_node', END)

# 8. graph compilation
graph = builder.compile()

# 9. graph launch
result = graph.invoke({
    'question': 'Які правила безпеки діють під час тривоги?',  # Який статус тривоги?
    'context': '',
    'answer': '',
    'search_attempts': 0,
})

print(result['answer'])
print('Кількість спроб пошуку:', result['search_attempts'])
