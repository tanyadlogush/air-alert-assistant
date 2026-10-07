import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore


load_dotenv()

# list of chunks (Document)
# (in a real project this list is returned by the file parser/splitter)
raw_chunks = [
    {
        "text": "Під час оголошення повітряної тривоги негайно прямуйте до найближчого укриття.",
        "meta": {"source": "civil_defense_manual.pdf", "page": 1, "category": "air_alert"}
    },
    {
        "text": "У приміщенні дотримуйтеся правила двох стін: перебувайте в коридорі або ванній.",
        "meta": {"source": "civil_defense_manual.pdf", "page": 2, "category": "safety_rules"}
    },
    {
        "text": "Заборонено наближатися до вікон під час роботи ППО чи вибухів.",
        "meta": {"source": "civil_defense_manual.pdf", "page": 2, "category": "safety_rules"}
    }
]

# wrap each chunk in a standard LangChain Document object
documents = [
    Document(page_content=item['text'], metadata=item['meta'])
    for item in raw_chunks
]

# embedding model initialization
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

# Creating an in-memory vectorstore (in-memory Qdrant) and loading documents
vectorstore = QdrantVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
    location=':memory:',    # # locally in memory for tests
    collection_name='emergency_rules'
)

# Retriever creation
retriever = vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs={'k':2}    # return only the top-2 similar chunks
)

query = "Що робити у квартирі під час вибухів?"

found_docs: list[Document] = retriever.invoke(query)

print("Знайдені документи (Top-2):")
for idx, doc in enumerate(found_docs, 1):
    print(f"\n[Документ {idx}]")
    print(f"Текст чанку: {doc.page_content}")
    # print(f"Метадані: {doc.metadata}")