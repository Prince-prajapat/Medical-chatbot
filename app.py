from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from deep_translator import GoogleTranslator
import os

def format_as_bullets(text):
    sentences = text.split(". ")
    list_items = "".join([f"<li>{s.strip()}</li>" for s in sentences if s.strip()])
    return f"<ul>{list_items}</ul>"


# ------------------ Flask App ------------------ #
app = Flask(__name__)
load_dotenv()

# ------------------ Environment Keys ------------------ #
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY in .env")
if not OPENROUTER_API_KEY:
    raise ValueError("Missing OPENROUTER_API_KEY in .env")

# ------------------ Pinecone Setup ------------------ #
embeddings = download_hugging_face_embeddings()

index_name = "bot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ------------------ Chat Model (OpenRouter) ------------------ #
chatModel = ChatOpenAI(
    model="meta-llama/llama-3-8b-instruct",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.environ["OPENROUTER_API_KEY"]
)

# ------------------ Prompt Template ------------------ #
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# ------------------ Chains ------------------ #
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# ------------------ Routes ------------------ #
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User:", msg)

    # Detect if input contains Hindi characters (Unicode range \u0900–\u097F)
    if any("\u0900" <= ch <= "\u097F" for ch in msg):
        # Translate Hindi → English
        msg_translated = GoogleTranslator(source='hi', target='en').translate(msg)
        print("Translated to English:", msg_translated)

        # Run RAG pipeline in English
        response = rag_chain.invoke({"input": msg_translated})
        ans = response["answer"]
        print("Answer (EN):", ans)

        # Translate back to Hindi
        ans_hindi = GoogleTranslator(source='en', target='hi').translate(ans)
        print("Answer (HI):", ans_hindi)

        # Format as bullet points in Hindi
        formatted = format_as_bullets(ans_hindi)
        return formatted

    else:
        # Normal English flow
        response = rag_chain.invoke({"input": msg})
        print("Response:", response["answer"])

        formatted = format_as_bullets(response["answer"])
        return formatted


# ------------------ Run App ------------------ #
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
