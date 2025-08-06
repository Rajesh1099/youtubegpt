import streamlit as st
import requests
import time
from fpdf import FPDF
from io import BytesIO

st.set_page_config(page_title="YouTubeGPT", page_icon="🎥", layout="centered")

# --- Header (fixed) ---
with st.container():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🎥 YouTubeGPT")
        st.caption("Chat with YouTube videos like your personal notes!")
    with col2:
        export_format = st.selectbox("Export Chat", ["Export Chat as :", "PDF", "Text"], label_visibility="collapsed")

# --- Session state ---
if "video_url" not in st.session_state:
    st.session_state.video_url = ""
if "video_processed" not in st.session_state:
    st.session_state.video_processed = False
if "video_id" not in st.session_state:
    st.session_state.video_id = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Step 1: YouTube URL input ---
with st.container():
    youtube_url = st.text_input("🔗 Enter YouTube URL", value=st.session_state.video_url, placeholder="https://www.youtube.com/watch?v=...")
    process_btn = st.button("▶️ Process Video")

# --- Video processing ---
if process_btn and youtube_url:
    if youtube_url == st.session_state.video_url and st.session_state.video_processed:
        st.info("✅ This video is already processed. You can continue chatting below.")
    else:
        with st.spinner("📥 Processing ..."):
            try:
                payload = {
                    "video_url": youtube_url,
                    "question": "Summarize this video in short.",
                    "top_k": 3
                }
                response = requests.post("http://localhost:8000/ask_with_url", json=payload)
                res_data = response.json()

                if "error" not in res_data:
                    st.success("✅ Video processed successfully! Start chatting below.")
                    st.session_state.video_url = youtube_url
                    st.session_state.video_id = res_data.get("video_id")
                    st.session_state.video_processed = True
                    st.session_state.chat_history = []  # Reset chat
                else:
                    st.error(f"❌ Error: {res_data['error']}")
            except Exception as e:
                st.error(f"❌ Backend Error: {e}")

# --- Chat UI ---
if st.session_state.video_processed:
    st.markdown("### 💬 Chat with your video")

    # --- Display chat history (above input) ---
    for q, a in st.session_state.chat_history:
        with st.chat_message("user", avatar="🧑"):
            st.markdown(q)
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(a)

    # --- Ask input + Reset Chat + Export Chat ---
    with st.container():
        user_question = st.chat_input("Ask anything about the video...")
        if user_question:
            with st.chat_message("user", avatar="🧑"):
                st.markdown(user_question)

            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("🤖 Thinking..."):
                    try:
                        payload = {
                            "video_url": st.session_state.video_url,
                            "question": user_question,
                            "top_k": 5
                        }
                        response = requests.post("http://localhost:8000/ask_with_url", json=payload)
                        result = response.json()

                        if "answer" in result:
                            full_response = result["answer"]
                            animated_response = ""
                            response_placeholder = st.empty()
                            for char in full_response:
                                animated_response += char
                                response_placeholder.markdown(animated_response + "▌")
                                time.sleep(0.01)
                            response_placeholder.markdown(animated_response)

                            st.session_state.chat_history.append((user_question, full_response))
                        else:
                            st.error(f"❌ Failed: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"❌ Request failed: {e}")

        # --- Reset Chat Button (below input) ---
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        st.button("🧹 Reset Chat", use_container_width=True, on_click=lambda: st.session_state.update(chat_history=[]))

        # --- Export Chat (already handled in header) ---
        if export_format == "PDF" and st.session_state.chat_history:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            for q, a in st.session_state.chat_history:
                pdf.multi_cell(0, 10, f"You: {q}\nBot: {a}\n")
            pdf_bytes = pdf.output(dest='S').encode('latin-1', 'replace')
            buffer = BytesIO(pdf_bytes)
            st.download_button("⬇️ Download PDF", data=buffer, file_name="youtubegpt.pdf", mime="application/pdf")

        elif export_format == "Text" and st.session_state.chat_history:
            text_output = "\n\n".join([f"🧑 You: {q}\n🤖 Bot: {a}" for q, a in st.session_state.chat_history])
            st.download_button("⬇️ Download Text", data=text_output, file_name="youtubegpt.txt", mime="text/plain")
