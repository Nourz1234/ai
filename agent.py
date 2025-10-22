import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_cpp import Llama

# --- Suppress Tokenizer Warning ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- CONFIGURATION ---
WORKSPACE_DIR = os.getcwd()
CLIENT_DIR_BASE = os.path.join(WORKSPACE_DIR, "client_requests")
GOV_DIR = os.path.join(WORKSPACE_DIR, "gov_procedures")
VECTOR_STORE_DIR = os.path.join(WORKSPACE_DIR, "vector_store")
LLM_SUMMARIES_DIR = os.path.join(WORKSPACE_DIR, "llm_summaries")

# MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
MAX_TOKENS = 4096
RETRIEVER_K = 5
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
MAX_CLIENT_CHARS = 1_000_000


# --- DIRECTORY SETUP ---
def setup_directories():
    os.makedirs(CLIENT_DIR_BASE, exist_ok=True)
    os.makedirs(GOV_DIR, exist_ok=True)
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    os.makedirs(LLM_SUMMARIES_DIR, exist_ok=True)
    print(f"\n📂 Workspace configured at: {WORKSPACE_DIR}")


# --- TEXT EXTRACTORS ---
def load_documents(directory_path):
    docs = []
    print(f"⏳ Loading documents from: {directory_path}")
    for root, _, filenames in os.walk(directory_path):
        for f_name in filenames:
            path = os.path.join(root, f_name)
            if f_name.startswith(".") or f_name.startswith("~$"):
                continue
            print(f"   -> Processing: {os.path.basename(path)}")
            loader = None
            try:
                if f_name.lower().endswith(".pdf"):
                    loader = PyPDFLoader(path)
                elif f_name.lower().endswith(".docx"):
                    loader = Docx2txtLoader(path)
                elif f_name.lower().endswith((".txt", ".md")):
                    loader = TextLoader(path, autodetect_encoding=True)
                else:
                    print(f"      [Skipping unsupported file: {os.path.basename(path)}]")
                    continue
                docs.extend(loader.load())
            except Exception as e:
                print(f"      [❌ Error loading {os.path.basename(path)}: {e}. Skipping.]")
    print(f"📄 Found and loaded {len(docs)} document sections.")
    if not docs:
        print(f"   Warning: No documents were successfully loaded from {directory_path}.")
    return docs


# --- VECTOR STORE MANAGEMENT ---
def create_vector_store(docs, embedding_model):
    if not docs:
        print("❌ No documents to index.")
        return None
    print("쪼 Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(docs)
    print(f"🧠 Creating/Updating vector store with {len(splits)} chunks...")
    vectorstore = Chroma.from_documents(
        documents=splits, embedding=embedding_model, persist_directory=VECTOR_STORE_DIR
    )
    print("✅ Vector store created/updated successfully.")
    return vectorstore


def load_vector_store(embedding_model):
    print(f"🧠 Loading existing vector store from: {VECTOR_STORE_DIR}")
    if not os.path.exists(VECTOR_STORE_DIR) or not os.listdir(VECTOR_STORE_DIR):
        print("❌ Vector store not found or empty. Please build it first.")
        return None
    try:
        vectorstore = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embedding_model)
        print("✅ Vector store loaded successfully.")
        return vectorstore
    except Exception as e:
        print(f"❌ Error loading vector store: {e}")
        return None


# --- RAG CHAIN SETUP ---
def setup_llm():
    from huggingface_hub import hf_hub_download

    # Download the GGUF file from Hugging Face
    model_path = hf_hub_download(
        repo_id="Qwen/Qwen2.5-72B-Instruct-GGUF",
        filename="qwen2.5-72b-instruct-q4_k_m.gguf"
    )

    # Initialize the model (this loads the quantized weights)
    return Llama(
        model_path=model_path,
        n_ctx=4096,  # context window
        n_threads=8,  # CPU threads (adjust to your CPU)
        n_gpu_layers=0,  # set >0 if using GPU acceleration build
    )

    # print("🔗 Connecting to local LLM via LM Studio...")
    # return ChatOpenAI(base_url=LLM_SERVER_URL, temperature=0.1)


def setup_rag_chain(vectorstore, llm):
    prompt_template = """
You are a Senior Conformity Engineer Assistant. Review the CLIENT REQUEST TEXT based ONLY on the provided GOVERNMENT PROCEDURE CONTEXT.

CLIENT REQUEST TEXT:
---
{question}
---

GOVERNMENT PROCEDURE CONTEXT:
---
{context}
---

Task: Provide ONLY the following output in English:

1. Overall Conformity Status:** [Choose ONE: **YES** / **NO** / **PENDING**]

2. Key Reasons / Missing Items:**
    * [List *only* the 3-5 most critical missing documents or major inconsistencies based *only* on the provided context. Be extremely brief.]

Do NOT provide any other details, summaries, introductions, or recommendations. Stick strictly to the format above.
"""
    # PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    # chain_type_kwargs = {"prompt": PROMPT}
    # qa_chain = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     chain_type="stuff",
    #     retriever=vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K}),
    #     chain_type_kwargs=chain_type_kwargs,
    #     verbose=False,
    # )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    qa_chain = (
        {
            "context": vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K}),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    print("✅ RAG chain is set up.")
    return qa_chain


# --- MAIN WORKFLOWS ---
def build_knowledge_base():
    print("\n--- 🏗️ Building Knowledge Base ---")
    gov_docs = load_documents(GOV_DIR)
    if gov_docs:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        create_vector_store(gov_docs, embeddings)
    else:
        print("❌ Could not build knowledge base - no government procedures found or loaded.")


def analyze_client_request():
    print("\n--- 🔍 Analyzing Client Request ---")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = load_vector_store(embeddings)
    if not vectorstore:
        return

    llm = setup_llm()
    qa_chain = setup_rag_chain(vectorstore, llm)

    try:
        client_folders = [
            d
            for d in os.listdir(CLIENT_DIR_BASE)
            if os.path.isdir(os.path.join(CLIENT_DIR_BASE, d))
        ]
    except FileNotFoundError:
        print(f"❌ Client directory not found: {CLIENT_DIR_BASE}")
        return

    if not client_folders:
        print(
            f"❌ No client request folders found. Please create a subfolder inside {CLIENT_DIR_BASE} for each client request."
        )
        return

    print("\n📁 Available Client Requests:")
    for i, folder in enumerate(client_folders):
        print(f"   {i + 1}. {folder}")

    selected_folder = None
    while not selected_folder:
        try:
            choice = int(input("\n👉 Enter the number of the client request to analyze: ").strip())
            if 1 <= choice <= len(client_folders):
                selected_folder = os.path.join(CLIENT_DIR_BASE, client_folders[choice - 1])
            else:
                print("⚠️ Invalid number.")
        except ValueError:
            print("⚠️ Please enter a single number.")

    print(f"\nProcessing request: {os.path.basename(selected_folder)}")
    client_docs = load_documents(selected_folder)
    if not client_docs:
        print("❌ Could not process client request - no documents loaded.")
        return

    client_query_text = f"Analyze the following client submission for EV Charger Model Approval based on SASO QMS-CR-10-92.\n\n"
    for doc in client_docs:
        client_query_text += (
            f"\n--- Client Doc: {os.path.basename(doc.metadata.get('source', 'Unknown'))} ---"
        )
        client_query_text += f"\n{doc.page_content}\n"

    # *** NEW: Truncate client text if it exceeds MAX_CLIENT_CHARS ***
    if len(client_query_text) > MAX_CLIENT_CHARS:
        print(
            f"⚠️ Client text VERY long ({len(client_query_text)} chars). Truncating to {MAX_CLIENT_CHARS} chars."
        )
        client_query_text = (
            client_query_text[:MAX_CLIENT_CHARS]
            + "\n\n[... CLIENT TEXT TRUNCATED DUE TO LENGTH ...]"
        )
    else:
        print(f"ℹ️ Client text length ({len(client_query_text)} chars) is within limits.")

    print("\n🤖 Sending request to LLM... (This may take a moment)")
    try:
        result = qa_chain.invoke(client_query_text)
        llm_output = result or "❌ No result received from LLM."
        print("\n--- 📊 AI Summary ---")
        print(llm_output)

        client_name = os.path.basename(selected_folder).replace(" ", "_").replace("/", "_")
        output_filename = os.path.join(LLM_SUMMARIES_DIR, f"conclusion_{client_name}.txt")
        try:
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(llm_output)
            print(f"✅ Summary saved to: {output_filename}")
        except Exception as e:
            print(f"⚠️ Error saving conclusion file: {e}")

    except Exception as e:
        print(f"❌ An error occurred during LLM interaction: {e}")
        print("\n   >>> PLEASE CHECK: <<<")
        print("   1. Is LM Studio running and server started?")
        print("   2. Is a model fully loaded?")
        print(
            "   3. Is the 'Context Length (n_ctx)' setting in LM Studio high enough (e.g., 8192 or more)?"
        )
        print(
            "   4. Consider removing non-essential files from the client folder to reduce input size."
        )


# --- MAIN MENU ---
def main_menu():
    setup_directories()
    while True:
        print("==============================")
        print("   Gulftic AI Assistant Menu")
        print("==============================")
        print("   1. Build/Update Knowledge Base (Gov Procedures)")
        print("   2. Analyze Client Request")
        print("   3. Exit")
        choice = input("👉 Enter your choice (1-3): ")
        if choice == "1":
            build_knowledge_base()
        elif choice == "2":
            analyze_client_request()
        elif choice == "3":
            print("👋 Exiting Gulftic AI Assistant. Goodbye!")
            break
        else:
            print("⚠️ Invalid choice. Please try again.")


# --- RUN ---
if __name__ == "__main__":
    main_menu()
