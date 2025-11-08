from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import os

# ----------------- Load Environment Variables -----------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API = os.getenv("GROQ_API")

if not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY in .env")
if not GROQ_API:
    raise ValueError("Missing GROQ_API in .env")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API  # required by ChatGroq


# ----------------- RAG Imports (LangChain v0.2) -----------------
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


# ----------------- Initialize Pinecone -----------------
index_name = "genativeai-encyclopedia"   # MUST match your add_data.py index

pc = Pinecone(api_key=PINECONE_API_KEY)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

db = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# ✅ Stronger retriever (better recall)
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 12}
)


# ----------------- Groq Model -----------------
chatmodel = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)


# ----------------- Prompt Template -----------------
system_prompt = """
You are a helpful AI assistant. Use ONLY the provided context to answer.
If the answer cannot be found in the context, say: "I don't know."

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])


# ----------------- Build RAG Chain -----------------
stuff_chain = create_stuff_documents_chain(
    llm=chatmodel,
    prompt=prompt
)

rag_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=stuff_chain
)


# ----------------- Flask App -----------------
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.json.get("question", "").strip()

    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    result = rag_chain.invoke({"input": user_question})
    answer = result.get("answer", "I don't know.")

    # DEBUG (safe encoding)
    print("\n--- Retrieved Context Chunks ---")
    for doc in result.get("context", []):
        try:
            print("SOURCE:", doc.metadata.get("source"))
            print("TEXT:", doc.page_content[:200].replace("\n", " ") + "...")
        except:
            pass

    return jsonify({"answer": answer})


# ----------------- Run Server -----------------
if __name__ == "__main__":
    print("Chatbot running at: http://127.0.0.1:8080")
    app.run(host="0.0.0.0", port=8080, debug=True)
