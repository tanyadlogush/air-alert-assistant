# LangChain

Modern **LangChain** is used as an atomic constructor: 

**_Prompt + Model + Structured Output + Retriever_**
which are connected via **_LCEL_** (LangChain Expression). 
* Complex agent scenarios with loops are built higher up — on **LangGraph**.


## Block 1: Core Modules & Structured Output

### Key Components
- **Unified Models (`langchain-core`, `langchain-openai`)**: Standardized wrappers replacing raw API calls. 
Operate using `SystemMessage`, `HumanMessage`, and `AIMessage`.
- **Prompts (`ChatPromptTemplate`)**: Templating engine for role-based message construction with dynamic 
- variable injection (`{context}`, `{question}`).

### Output Types: Text vs. Structured Output
- **Unstructured Text (`str`)**:
  - Uses `StrOutputParser()` to extract plain string output from `AIMessage`.
  - *Pattern*: `chain = prompt | llm | StrOutputParser()`
- **Structured Output (`Pydantic` object)**:
  - Uses `.with_structured_output(BaseModelClass)` backed by OpenAI Tool/Function Calling under the hood.
  - Enforces deterministic output schema parsed directly into a Python object instance.
  - *Pattern*: `chain = prompt | llm.with_structured_output(MyPydanticClass)`


## Block 2: Document Abstraction, VectorStores & Retrievers

### Key Concepts
- **`Document` Object**: The core unit representing a single chunk of data.
  - `page_content`: Raw text content of the chunk.
  - `metadata`: Key-value dictionary (e.g., `source`, `page`, `category`) used for pre-filtering large collections.
- **`VectorStore` Wrapper**: Encapsulates the vector database (e.g., Qdrant) alongside the embedding model
(`OpenAIEmbeddings`). Responsible for document indexing and vector operations.
- **`Retriever` Abstraction**: A focused search interface derived via `vectorstore.as_retriever()`.
  - Hides vectorization and query handling behind a single unified method: `.invoke("query") -> list[Document]`.
  - Integrates seamlessly into LCEL execution chains.



## Block 3: LCEL RAG Pipeline

- RAG retrieves relevant context and gives it to the LLM together with the user's question.
- RunnableParallel creates multiple branches from the same input.
- context = retriever | format_docs retrieves and formats relevant documents.
- RunnablePassthrough() forwards the original input unchanged.
- RunnableParallel produces a dictionary containing context and question.
- ChatPromptTemplate maps these values to system and user messages.
- system and user are message roles; context and question are their values.
- StrOutputParser() converts the LLM output into a plain string.


**_Data Flow:_**

    question → ┬ → retriever → format_docs → context ─┐
    
               │                                        ├→ {context, question} → Prompt → LLM → StrOutputParser → answer
               
               └ → RunnablePassthrough → question ─────┘



## Block 4: Chat History & RAG Memory Mechanics

### Core Concepts
- **Stateless Nature of LLMs**: LLMs do not retain memory across API calls. 
Conversational context relies on sending the full message list (`[system, human, ai, human...]`) 
in every payload.
- **RAG Multi-Turn Challenge**: Standalone user queries with implicit references 
(e.g., *"What if I am outside?"*) fail in vector search because key domain keywords 
from previous turns are missing.

- **Two-Step Conversational RAG Pipeline**:
  1. **History-Aware Rewriting**: Uses an LLM to transform the chat history + latest user query into a single, 
  standalone search query.
  2. **RAG Execution**: Passes the reformulated query to the retriever to fetch context, 
  then generates the answer via the QA prompt.
- **LangChain Deprecation Note**: Legacy abstractions 
(`RunnableWithMessageHistory`, `InMemoryChatMessageHistory`) are deprecated in favor 
of **LangGraph state persistence (`Checkpointer`)**.


## Block 5: LangChain: Tools and Agents

### `@tool`

* Converts a Python function into a LangChain tool.
* Type hints define parameter types.
* The docstring describes the tool's purpose and helps the model decide when to use it.

### Providing tools to an agent

```python
tools = [search_documents, get_air_alert_status]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are an assistant."
)
```

The `tools` list contains all tools available to the agent.

### Tool-calling flow

1. The model receives tool descriptions and argument schemas.
2. The model selects a tool and generates a tool call with arguments.
3. The application executes the function through the agent mechanism.
4. The result is returned to the model, which can respond or request another tool.

**Key distinction:** The model selects the tool; the application executes it.

### Token usage

Tool names, descriptions, and argument schemas can increase input token usage. Keep descriptions concise, 
precise, and informative.
