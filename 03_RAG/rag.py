import os
import math
import json
from http.client import responses

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


# document Loading & Chunking by characters (+ overlap)
def chunk_text(text, chunk_size=1000, overlap=300):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

#  chunking by sentences / paragraphs
def chunk_text_by_lines(text):
    # by new lines
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    return lines

# embedding helper
def get_embedding(text, model='text-embedding-3-small'):
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding


# math: cosine similarity for comparing vextors
def cosine_similarity(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude_v1 = math.sqrt(sum(a * a for a in v1))
    magnitude_v2 = math.sqrt(sum(b * b for b in v2))
    if not magnitude_v1 or not magnitude_v2:
        return 0.0
    return dot_product / (magnitude_v1 * magnitude_v2)





raw_text = """
Інструкція з безпеки під час надзвичайних ситуацій

Пожежа
 Якщо ви помітили пожежу, негайно повідомте рятувальну службу. Залиште приміщення через найближчий безпечний вихід. 
 Не користуйтеся ліфтом та допомагайте іншим людям під час евакуації.

Повітряна тривога
 Після сигналу повітряної тривоги необхідно негайно перейти в найближче укриття. Візьміть із собою документи, телефон, 
 воду та необхідні ліки. Під час перебування в укритті слідкуйте за офіційними повідомленнями та не залишайте безпечне місце до сигналу відбою тривоги.

Відключення електроенергії
 У разі відключення електроенергії використовуйте ліхтарики замість свічок. Перевірте заряд мобільного телефону 
 та заощаджуйте заряд акумулятора. За можливості підготуйте запас води та продуктів.

Сильна негода
 Під час грози або сильного вітру залишайтеся в приміщенні. Закрийте вікна та тримайтеся подалі від дерев і рекламних 
 конструкцій. Слідкуйте за прогнозом погоди та попередженнями служб.
"""


# # chunking by characters
# chunks = chunk_text(raw_text, chunk_size=500, overlap=100)

# chunking by new lines
chunks = chunk_text_by_lines(raw_text)

# vectorization
vector_db = []

for i, chunk in enumerate(chunks):
    embedding = get_embedding(chunk)
    vector_db.append({
        'id': f'chunk_{i}',
        'text': chunk,
        'vector': embedding
    })

# similarity search
user_query = 'Як поводитись під час повітряної тривоги?'       # 'Як поводитись під час концерту?'
query_embedding = get_embedding(user_query)

# similarity of the question with each chunk
for item in vector_db:
    item['score'] = cosine_similarity(query_embedding, item['vector'])

# sort by descending similarity -> take the best chunk
vector_db.sort(key=lambda x: x['score'], reverse=True)
best_chunk = vector_db[0]

print(f"Знайдений релевантний чанк (Score: {best_chunk['score']:.4f}):")
print(f"'{best_chunk['text']}'\n")

# prompt generation + llm
prompt = f"""
Дай відповідь на питання користувача, використовуючи ТІЛЬКИ наданий контекст.
Якщо відповіді немає в контексті, скажи, що не знаєш.

Контекст:
{best_chunk['text']}

Питання: {user_query}
"""

response = client.responses.create(
    model='gpt-4o',
    input=prompt,
    instructions='Ти асистент з безпеки'
)

print('Фінальна відповідь RAG:')
print(response.output_text)
