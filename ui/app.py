# streamlit_app/ui.py
import streamlit as st
import requests
import json

# --- Configuration ---
API_URL = "http://127.0.0.1:8000" # URL of your FastAPI backend

# --- Page Configuration ---
st.set_page_config(
    page_title="Chat with Multiple Documents",
    page_icon="💬",
    layout="wide"
)

# --- HTML/CSS Templates ---
css = ''' 
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 15%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 85%;
  padding: 0 1.5rem;
  color: #fff;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEh_kzb_6j188hVsr5RHOjaG0geNq2tRqDwHH-bc60R1DRa-hjXzrLqQjP6Zr4dAixSqcnfsqKLuisOcvqDiFviKTpBM29IhGLpqpiF3mvqQ3GElJlik-VsfRRlROuStfzvFatiCVdUcPw8TxcmiMzwTWZGMPvDmZNkHtotcRhkz6H5CCAutlwDkxuHcotvB/s16000-rw/Robotics%20and%20Artificial%20Intelligence%20(AI).webp">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.pinimg.com/736x/8b/16/7a/8b167af653c2399dd93b952a48740620.jpg">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

st.write(css, unsafe_allow_html=True)

# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""

# --- Sidebar for File Upload and Control ---
with st.sidebar:
    st.subheader("Your Documents")
    uploaded_files = st.file_uploader(
        "Upload your documents (PDF, DOCX, CSV, TXT, etc.) and click 'Process'", 
        accept_multiple_files=True
    )

    if st.button("Process"):
        if uploaded_files:
            with st.spinner("Processing documents... This may take a moment."):
                files_to_upload = []
                for uploaded_file in uploaded_files:
                    files_to_upload.append(
                        ("files", (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type))
                    )
                
                try:
                    response = requests.post(f"{API_URL}/upload-and-process", files=files_to_upload)
                    if response.status_code == 200:
                        st.success("Documents processed successfully!")
                        # Store extracted text in session state
                        st.session_state.extracted_text = response.json().get("extracted_text", "")
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Could not connect to the backend: {e}")
        else:
            st.warning("Please upload at least one document.")
    
    st.subheader("Extracted Text")
    st.text_area(
        "Text from your documents will appear here",
        value=st.session_state.get("extracted_text", ""), 
        height=300,
        disabled=True
    )
    
    if st.button("Clear All Data & Chat"):
        with st.spinner("Clearing all data..."):
            try:
                response = requests.post(f"{API_URL}/clear")
                if response.status_code == 200:
                    st.success("All data and chat history cleared!")
                    # Clear session state
                    st.session_state.chat_history = []
                    st.session_state.extracted_text = ""
                else:
                    st.error(f"Error clearing data: {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to the backend to clear data: {e}")
        st.rerun()

# --- Main Chat Interface ---
st.header("Chat with your Documents ")

# Display chat history
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.write(user_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
    else:
        st.write(bot_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)

# Handle new user input
user_question = st.chat_input("Ask a question about your documents...")
if user_question:
    # Add user message to history and display it
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)
    
    # Get bot response
    with st.spinner("Thinking..."):
        try:
            response = requests.post(f"{API_URL}/query", json={"query": user_question})
            if response.status_code == 200:
                bot_response = response.json()["answer"]
                # Add bot response to history and display it
                st.session_state.chat_history.append({"role": "bot", "content": bot_response})
                with st.chat_message("bot"):
                    st.write(bot_template.replace("{{MSG}}", bot_response), unsafe_allow_html=True)
            else:
                error_msg = f"Error from API: {response.text}"
                st.error(error_msg)
                st.session_state.chat_history.append({"role": "bot", "content": error_msg})
        except requests.exceptions.RequestException as e:
            error_msg = f"Could not get a response from the backend: {e}"
            st.error(error_msg)
            st.session_state.chat_history.append({"role": "bot", "content": error_msg})