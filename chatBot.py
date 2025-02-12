import os
import shutil
import logging
from dotenv import load_dotenv
from typing import Optional, List
import tkinter as tk
from tkinter import messagebox

from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms.base import BaseLLM
from langchain.schema import LLMResult, Generation
import google.generativeai as genai
from langdetect import detect
from deep_translator import GoogleTranslator
from pydantic import Field

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("document_chat_app")

_ = load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Gemini API key loaded successfully.")
else:
    logger.error("GEMINI_API_KEY is missing!")
    raise ValueError("GEMINI_API_KEY is missing!")

def enforce_english(text: str) -> str:
    try:
        if detect(text) != "en":
            logger.warning(f"Detected non-English text: {text}. Translating to English.")
            translated = GoogleTranslator(source="auto", target="en").translate(text)
            return translated
    except Exception as e:
        logger.error(f"Translation failed: {e}")
    return text

def sanitize_text(text: str) -> str:
    import unicodedata
    text = enforce_english(text)
    return ''.join(
        c for c in unicodedata.normalize('NFKD', text)
        if not unicodedata.combining(c)
    )

class CustomEmbedding(Embeddings):
    def __init__(self, model: SentenceTransformer):
        self.model = model
        logger.info("CustomEmbedding initialized.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        sanitized_texts = [sanitize_text(text) for text in texts]
        return self.model.encode(sanitized_texts, convert_to_tensor=False)

    def embed_query(self, text: str) -> List[float]:
        sanitized_text = sanitize_text(text)
        return self.model.encode([sanitized_text], convert_to_tensor=False)[0]


class GeminiLLM(BaseLLM):
    model_name: str = Field(default="models/gemini-pro")

    def __init__(self, model_name: str = "models/gemini-pro"):
        super().__init__(model_name=model_name)
        logger.info(f"Initialized GeminiLLM with model_name: {self.model_name}")

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            model = genai.GenerativeModel(self.model_name)
            api_response = model.generate_content(prompt)
            return sanitize_text(api_response.text.strip())
        except Exception as e:
            logger.error(f"Error with Gemini API: {e}")
            return "Error generating response."

    def _generate(
            self,
            prompts: List[str],
            stop: Optional[List[str]] = None,
            **kwargs
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            try:
                text = self._call(prompt, stop=stop)
                generations.append([Generation(text=text)])
            except Exception as e:
                logger.error(f"Error generating response for prompt '{prompt}': {e}")
                generations.append([Generation(text="Error generating response")])
        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "gemini_llm"


def build_chroma_vectorstore(pdf_paths: List[str], persist_directory: str = "chroma_db"):
    all_docs = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        all_docs.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)

    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    custom_embedding = CustomEmbedding(embedding_model)

    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=custom_embedding
    )
    doc_texts = [sanitize_text(chunk.page_content) for chunk in chunks]
    vectordb.add_texts(doc_texts)
    return vectordb


def rephrase_question(llm: GeminiLLM, chat_history: str, question: str) -> str:
    prompt = f"""
    Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question. 
    The output must be in English:

    Chat History:
    {chat_history}
    Follow-Up Input: {question}
    Standalone Question:
    """
    try:
        rephrased_question = llm._call(prompt)
        return sanitize_text(rephrased_question)
    except Exception as e:
        logger.error(f"Error in rephrasing question: {e}")
        return question


def build_conversational_chain(vectordb: Chroma, llm: GeminiLLM) -> ConversationalRetrievalChain:
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False,
        output_key="answer",
    )
    return qa_chain


def start_gui():
    try:
        pdf_paths = [
            "docs/pdf/Course1.pdf",
            "docs/pdf/Course2.pdf",
            "docs/pdf/Course3.pdf"
        ]
        vectordb = build_chroma_vectorstore(pdf_paths, persist_directory="docs/chroma/")
        gemini_llm = GeminiLLM(model_name="models/gemini-pro")
        conversational_chain = build_conversational_chain(vectordb, gemini_llm)
        chat_history = ""

        def handle_question():
            nonlocal chat_history
            user_input = user_question.get().strip()
            if not user_input:
                response_label.config(text="Bot: Please enter a question.")
                return

            try:
                rephrased_question = rephrase_question(gemini_llm, chat_history, user_input)
                response = conversational_chain.invoke({"question": rephrased_question})
                bot_response = response.get("answer", "No answer generated.")
                response_label.config(text=f"Bot: {bot_response}")
                chat_history += f"Human: {user_input}\nAssistant: {bot_response}\n"
            except Exception as e:
                logger.error(f"Error while handling question: {e}")
                response_label.config(text="Bot: An error occurred. Please try again.")
            finally:
                user_question.delete(0, tk.END)

        root = tk.Tk()
        root.title("Document Chat Interface")

        tk.Label(root, text="Enter your question below:").pack(pady=5)
        user_question = tk.Entry(root, width=70)
        user_question.pack(pady=5)
        response_label = tk.Label(root, text="", wraplength=500, justify="left")
        response_label.pack(pady=10)
        submit_button = tk.Button(root, text="Send", command=handle_question)
        submit_button.pack(pady=5)

        root.mainloop()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

if __name__ == "__main__":
    start_gui()
