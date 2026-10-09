import os
from re import search

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import embeddings

load_dotenv()

# documents + retriever

docs = [
    Document(
        page_content="Під час оголошення повітряної тривоги негайно прямуйте до найближчого укриття.",
        metadata={"category": "air_alert"}
    ),
    Document(
        page_content="У приміщенні дотримуйтеся правила двох стін: перебувайте в коридорі або ванній.",
        metadata={"category": "safety_rules"}
    ),
    Document(
        page_content="Заборонено наближатися до вікон під час роботи ППО чи вибухів.",
        metadata={"category": "safety_rules"}
    ),
    Document(
        page_content="Якщо тривога застала на вулиці, негайно прямуйте до підземного переходу, метро або укриття. Уникайте висоток та скляних будівель.",
        metadata={"category": "street_rules"}
    )
]

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
vectorstore = QdrantVectorStore.from_documents(
    documents=docs,
    embedding=embeddings,
    location=':memory:',
    collection_name='emergency_rules_history'
)
retriever = vectorstore.as_retriever(search_kwargs={'k': 2})

def format_docs(documents: list) -> str:
    return '\n\n'.join(doc.page_content for doc in documents)

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)


# reformulating the query based on history (Contextualize Question)

contextualize_q_system_prompt = """Враховуючи історію чату та останнє запитання користувача, 
яке може посилатися на контекст з історії, сформулюй одне самостійне запитання, 
яке можна зрозуміти БЕЗ історії чату. НЕ відповідай на запитання, 
просто перефразуй його за потреби, або поверни як є, якщо воно вже самостійне."""

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

# a separate chain that only prepares the correct search query for the retriever
history_aware_retriever_query = (
        contextualize_q_prompt
        | llm
        | StrOutputParser()
)


# main RAG chain

qa_system_prompt = """Ти — помічник з безпеки. Відповідай на питання, використовуючи ТІЛЬКИ наданий контекст. 
Якщо у контексті немає відповіді, скажи "У мене немає інформації про це в інструкціях".

Контекст:
{context}"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])


# gathering context using a reformatted query
def get_retrieved_context(input_dict: dict) -> str:
    # if there is a story — rephrase the search query, if not — search directly
    if input_dict.get("chat_history"):
        search_query = history_aware_retriever_query.invoke(input_dict)
    else:
        search_query = input_dict["input"]

    docs = retriever.invoke(search_query)
    return format_docs(docs)


rag_chain = (
        RunnableParallel(
            context=get_retrieved_context,
            input=lambda x: x["input"],
            chat_history=lambda x: x.get("chat_history", [])
        )
        | qa_prompt
        | llm
        | StrOutputParser()
)


# saving dialogue history by session_id
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key=None
)

# real-time dialogue testing
if __name__ == "__main__":
    config = {"configurable": {"session_id": "user_session_1"}}

    print("Питання 1")
    q1 = "Які основні правила безпеки під час повітряної тривоги?"
    res1 = conversational_rag_chain.invoke({"input": q1}, config=config)
    print(f"Користувач: {q1}")
    print(f"Відповідь: {res1}\n")

    print("Питання 2 (з неявним посиланням)")
    q2 = "А якщо вона застала мене на вулиці?"
    res2 = conversational_rag_chain.invoke({"input": q2}, config=config)
    print(f"Користувач: {q2}")
    print(f"Відповідь: {res2}\n")