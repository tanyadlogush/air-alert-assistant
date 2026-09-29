import os
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.models import Filter, FieldCondition, MatchValue
from rank_bm25 import BM25Okapi

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# initializing local Qdrant (in-memory)
qdrant = QdrantClient(':memory:')

COLLECTION_NAME = 'company_docs'
VECTOR_SIZE = 1536    # Dimension for text-embedding-3-small

# creating a collection (similar to a table)
qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
)

# documents preparation
documents = [
    {
        'id': 1,
        'text': "Якщо ви помітили пожежу, негайно повідомте рятувальну службу. "
                "Залиште приміщення через найближчий безпечний вихід. Не користуйтеся ліфтом та допомагайте "
                "іншим людям під час евакуації.",
        'category': 'Пожежа'
    },
    {
        'id': 2,
        'text': "Після сигналу повітряної тривоги необхідно негайно перейти в найближче укриття. "
                "Візьміть із собою документи, телефон, воду та необхідні ліки. Під час перебування в укритті "
                "слідкуйте за офіційними повідомленнями та не залишайте безпечне місце до сигналу відбою тривоги.",
        'category': 'Повітряна тривога'
    },
    {
        'id': 3,
        'text': "У разі відключення електроенергії використовуйте ліхтарики замість свічок. "
                "Перевірте заряд мобільного телефону та заощаджуйте заряд акумулятора. За можливості підготуйте "
                "запас води та продуктів.",
        'category': "Відключення електроенергії"
    }
]

# embedding function from OpenAI
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model='text-embedding-3-small'
    )
    return response.data[0].embedding

# inserting data (Points) into Qdrant
points = []
for doc in documents:
    vector = get_embedding(doc['text'])
    points.append(
        PointStruct(
            id=doc['id'],
            vector=vector,
            payload={
                'text': doc['text'],
                'category': doc['category']
            }
        )
    )

qdrant.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)



# ======== Initializing and launching BM25 Search ==============
# simple tokenization: lowercase and split by spaces
corpus_texts = [doc["text"] for doc in documents]
tokenized_corpus = [doc.lower().split() for doc in corpus_texts]

# initializing the BM25 index
bm25 = BM25Okapi(tokenized_corpus)

def search_bm25(query, top_k=3):
    tokenized_query = query.lower().split()
    # scores for each document
    scores = bm25.get_scores(tokenized_query)

    # sorting documents by BM25 scores
    ranked_docs = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc["id"] for doc, score in ranked_docs[:top_k]]


# ======== Getting two lists (Vector Search & BM25) ===========
def search_vector(query, top_k=3):
    query_vector = get_embedding(query)
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    ).points
    return [point.id for point in results]


# RRF (Reciprocal Rank Fusion)
def compute_rrf(vector_ids, bm25_ids, k=60):
    rrf_scores = {}

    # rank processing in vector search
    for rank, doc_id in enumerate(vector_ids, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    # rank processing in BM25
    for rank, doc_id in enumerate(bm25_ids, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    # sorting IDs by DESC total RRF score
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs


user_query = "Що робити з ліхтариками та свічками?"

# candidates from both sources
vector_ids = search_vector(user_query, 3)
bm25_ids = search_bm25(user_query,3)

# RRF results
rrf_results = compute_rrf(vector_ids, bm25_ids)
print(f"\nRRF Scores (doc_id, score): {rrf_results}")

# extracting the Top 2 documents after RRF for Reranker
top_candidates_ids = [doc_id for doc_id, score in rrf_results[:2]]
candidates_docs = [doc for doc in documents if doc["id"] in top_candidates_ids]

# LLM Reranking Step
def llm_rerank(query, docs):
    docs_text = "\n\n".join([f"ID {d['id']}: {d['text']}" for d in docs])

    prompt = f"""
Проаналізуй наступні документи та впорядкуй їх за релевантністю до запиту користувача.
    
    Запит: "{query}"
    
    Документи:
    {docs_text}
    
    Поверни ID НАЙБІЛЬШ релевантного документа та коротко поясни свій вибір.
    """

    response = client.responses.create(
        model="gpt-4o",
        input=prompt
    )
    return response.output_text

# final call
rerank_result = llm_rerank(user_query, candidates_docs)
print("\n  Результат Reranking")
print(rerank_result)