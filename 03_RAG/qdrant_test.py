import os
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.models import Filter, FieldCondition, MatchValue


load_dotenv()
client = OpenAI(api_key=os.getenv('OPEN_API_KEY'))

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

# ======= search without filter ==============
# # search in Qdrant (Similarity Search)
# user_query = 'Як поводитись під час повітряної тривоги?'
# query_vector = get_embedding(user_query)
#
# # search Top-1 nearest vector
# search_result = qdrant.query_points(
#     collection_name=COLLECTION_NAME,
#     query=query_vector,
#     limit=1
# ).points
#
# best_match = search_result[0]
#
# print(" Знайдено релевантний запис у Qdrant:")
# print(f"ID: {best_match.id}")
# print(f"Score схожості: {best_match.score:.4f}")
# print(f"Текст (з Payload): {best_match.payload['text']}")
# print(f"Категорія: {best_match.payload['category']}")


# ======================== add Filtration ================
# query that fits multiple categories
ambiguous_query = "Що робити з водою та телефонами?"
query_vector = get_embedding(ambiguous_query)

# search with filtration
filtered_search_result = qdrant.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="category",
                match=MatchValue(value="Повітряна тривога")
            )
        ]
    ),
    limit=1
).points

best_match = filtered_search_result[0]

print(" Знайдено запис із жорстким фільтром (категорія = 'Повітряна тривога'):")
print(f"Score схожості: {best_match.score:.4f}")
print(f"Категорія: {best_match.payload['category']}")
print(f"Текст: {best_match.payload['text']}")