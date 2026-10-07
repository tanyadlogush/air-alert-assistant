import os
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore


load_dotenv()

# data Preparation and Retriever
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
    )
]

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

vectorstore = QdrantVectorStore.from_documents(
    documents=docs,
    embedding=embeddings,
    location=":memory:",
    collection_name="emergency_rules"
)

retriever = vectorstore.as_retriever(search_kwargs={'k': 2})

# helper function to merge found chunks into one continuous text for the prompt
def format_docs(documents):
    return '\n\n'.join(doc.page_content for doc in documents)

# creating a prompt template
system_prompt = """Ти - помічник з безпеки. Відповідай на питання, використовуючи ТІЛЬКИ наданий контекст. 
Якщо у контексті немає відповіді, скажи "У мене немає інформації про це в інструкціях".

Контекст:
{context}"""

prompt = ChatPromptTemplate.from_messages([
    ('system', system_prompt),
    ('user', '{question}')
])

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

# building a RAG pipeline via LCEL
rag_chain = (
    RunnableParallel(
        context=retriever | format_docs,
        question=RunnablePassthrough()
    )
    | prompt
    | llm
    | StrOutputParser()
)

# RAG-chain
if __name__ == '__main__':
    query = "Що робити, якщо під час тривоги я знаходжуся вдома?"  # for out-of-context control "Що робити, якщо на вулиці спека?"
    print(f"Запит: {query}")

    response = rag_chain.invoke(query)

    print("Відповідь RAG-системи:")
    print(response)