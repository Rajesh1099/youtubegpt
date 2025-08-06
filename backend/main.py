from fastapi import FastAPI, Query, Body
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import os
import time
from dotenv import load_dotenv
load_dotenv()

# ---------------------
# ⚙️ FastAPI Setup
# ---------------------
app = FastAPI()

# ---------------------
# 🔠 Translator (for non-English transcripts)
# ---------------------
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en")

# ---------------------
# 📎 Chunk Splitter
# ---------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""]
)

# ---------------------
# 🧠 Embedding Model (MiniLM)
# ---------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------
# 📚 FAISS Vector Index (in-memory)
# ---------------------
dimension = 384  # for MiniLM
index = faiss.IndexFlatL2(dimension)
vector_id_to_chunk = []

# 🧠 Keep track of processed videos
video_memory = {}  # video_id: (chunks, np_embeddings)

# ---------------------
# 🔑 OpenRouter Setup
# ---------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# List of free models to try in fallback order
FREE_MODELS = [
    "openrouter/auto:free",
    "mistralai/mistral-7b-instruct:free",
    "meta-llama/llama-3-8b-instruct:free",
    "openchat/openchat-3.5-0106:free",
    "nousresearch/nous-capybara-7b:free",
    "gryphe/mythomax-l2-13b:free",
    "huggingfaceh4/zephyr-7b-beta:free",
    "undi95/toppy-m-7b:free",
    "gryphe/mythomist-7b:free",
    "deepseek/deepseek-r1t-chimera:free"
]

# ---------------------
# 📥 Extract transcript + translate + embed
# ---------------------
def process_video(video_url: str):
    if "shorts/" in video_url:
        video_id = video_url.split("shorts/")[-1].split("?")[0]
    else:
        video_id = video_url.split("v=")[-1].split("&")[0]

    # If video already processed, reuse it
    if video_id in video_memory:
        return video_id, None

    try:
        try:
            # Try to fetch English transcript
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
            lang_used = "en"
            
            # English transcripts are list of dicts
            translated_text = " ".join([item['text'] for item in transcript])

        except:
            # Fallback: Get any available transcript
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript_obj = list(transcript_list)[0]
            transcript = transcript_obj.fetch()
            lang_used = transcript_obj.language_code

            # Non-English transcripts are objects with .text
            full_text = " ".join([item.text for item in transcript])

            # Translate in chunks of 500 characters
            translated_chunks = []
            for i in range(0, len(full_text), 500):
                chunk = full_text[i:i+500]
                result = translator(chunk)[0]
                translated_chunks.append(result["translation_text"])
            
            # Final translated text in English
            translated_text = " ".join(translated_chunks)

        # Split translated text into chunks
        split_chunks = text_splitter.split_text(translated_text)

        # Generate vector embeddings
        embeddings = embedding_model.encode(split_chunks).tolist()
        np_embeddings = np.array(embeddings).astype("float32")

        # Store in FAISS index and memory
        index.add(np_embeddings)
        vector_id_to_chunk.extend(split_chunks)
        video_memory[video_id] = (split_chunks, np_embeddings)

        return video_id, {
            "language": lang_used,
            "chunks_stored": len(split_chunks)
        }

    except TranscriptsDisabled:
        raise Exception("❌ Subtitles are disabled for this video.")

    except Exception as e:
        raise e

# ---------------------
# 🤖 /ask_with_url - Main NotebookLM-style endpoint
# ---------------------
@app.post("/ask_with_url")
def ask_with_url(
    video_url: str = Body(..., embed=True, description="YouTube or Shorts URL"),
    question: str = Body(..., embed=True, description="Your question"),
    top_k: int = Body(5, embed=True, description="Top chunks to use from transcript")
):
    try:
        # Step 1: Ensure video is processed
        video_id, info = process_video(video_url)

        # Step 2: Semantic Search
        query_embedding = embedding_model.encode([question]).astype("float32")
        distances, indices = index.search(query_embedding, top_k)
        top_chunks = [vector_id_to_chunk[i] for i in indices[0]]
        context = "\n".join(top_chunks)

        # Step 3: Prompt
        prompt = f"""You are a helpful assistant. Use the transcript context below to answer the question.

Transcript Context:
{context}

Question: {question}
Answer:"""

        # Step 4: Try each model until success
        for model in FREE_MODELS:
            try:
                headers = {
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://yourdomain.com",
                    "X-Title": "YouTube NotebookLM"
                }

                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}]
                }

                start_time = time.time()
                res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
                res.raise_for_status()
                duration = round(time.time() - start_time, 2)
                reply = res.json()["choices"][0]["message"]["content"]

                return {
                    "model_used": model,
                    "latency_seconds": duration,
                    "video_id": video_id,
                    "question": question,
                    "answer": reply.strip(),
                    "context_chunks": top_chunks,
                    "transcript_info": info
                }

            except Exception:
                continue

        return {"error": "❌ All free models failed. Please try again later."}

    except Exception as e:
        return {"error": str(e)}


@app.get("/")
def root():
    return {"status": "✅ FastAPI running on Render"}
