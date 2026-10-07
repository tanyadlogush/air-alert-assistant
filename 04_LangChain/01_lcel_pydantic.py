import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


# Pydantic schema (determine the structure)

# # === a simple chain to get a simple string ===
# from langchain_core.output_parsers import StrOutputParser
#
# chain = prompt | llm | StrOutputParser()
# result = chain.invoke({"situation": "..."})


class SafetyInstruction(BaseModel):
    danger_level: str = Field(description="Рівень небезпеки: Low, Medium, High")
    action_items: list[str] = Field(description="Список із 3 конкретних кроків для безпеки")
    recommended_shelter_type: str = Field(description="Рекомендований тип укриття чи безпечного місця")


# model with forced structured inference
llm = ChatOpenAI(model='gpt-4o', temperature=0)
structured_llm = llm.with_structured_output(SafetyInstruction)

# prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "Ти фахівець із цивільного захисту. Проаналізуй ситуацію та надай структуровану інструкцію."),
    ("user", "Ситуація: {situation}")
])

# LCEL Chain: prompt | structured_llm
chain = prompt | structured_llm

# chain call
result: SafetyInstruction = chain.invoke({
    "situation": "У місті оголошено повітряну тривогу, чути роботу ППО."
})

print("--- Результат Structured Output (Pydantic) ---")
print(f"Тип об'єкта: {type(result)}")
print(f"Рівень небезпеки: {result.danger_level}")
print(f"Рекомендоване укриття: {result.recommended_shelter_type}")
print("Дії:")
for idx, step in enumerate(result.action_items, 1):
    print(f"  {idx}. {step}")

print()
print(result)
