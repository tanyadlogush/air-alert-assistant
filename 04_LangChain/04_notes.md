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


`pip install langchain-qdrant`