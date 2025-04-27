import asyncio
import sys
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
EMBEDDING_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')

# Fetch free models from OpenRouter
def get_free_openrouter_models():
    try:
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
        response = requests.get("https://openrouter.ai/api/v1/models", headers=headers)
        response.raise_for_status()
        models = [
            model['id'] for model in response.json()['data']
            if model.get('pricing', {}).get('completion', '0') == '0' and
            model.get('pricing', {}).get('prompt', '0') == '0'
        ]
        return models or ["No free models available."]
    except requests.RequestException as e:
        st.error(f"Failed to fetch models: {e}")
        return []

# Process PDF into text chunks
def process_pdf(pdf_file):
    try:
        pdf_reader = PdfReader(pdf_file)
        text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
        if not text.strip():
            raise ValueError("No text extracted from PDF.")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
        return text_splitter.split_text(text)
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return []

# Create FAISS index from text chunks
def create_embeddings(chunks):
    try:
        embeddings = EMBEDDING_MODEL.encode(chunks, show_progress_bar=False)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings.astype('float32'))
        return index
    except Exception as e:
        st.error(f"Error creating embeddings: {e}")
        return None

# Query a specific model with context
def query_model(prompt, context, model):
    try:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {OPENROUTER_API_KEY}"}
        data = {
            "messages": [
                {"role": "system", "content": f"Answer based on this context:\n{context}"},
                {"role": "user", "content": prompt}
            ],
            "model": model
        }
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        
        response_json = response.json()
        if 'choices' in response_json and len(response_json['choices']) > 0:
            return response_json['choices'][0]['message']['content']
        else:
            reset_time = response_json.get('error', {}).get('metadata', {}).get('headers', {}).get('X-RateLimit-Reset')
            if reset_time:
                wib_time = datetime.fromtimestamp(int(reset_time) / 1000) + timedelta(hours=7)
                formatted_time = wib_time.strftime('%Y-%m-%d %H:%M:%S WIB')
                return f"Rate limit exceeded. Please wait until {formatted_time} to try again."
            else:
                return "No reset time available."
            
    except requests.RequestException as e:
        return f"Error fetching answer: {e}"
    except KeyError as e:
        return f"Missing key in API response: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"

# Apply custom CSS for elegant UI with readable font
def set_custom_style():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        body {
            background: #0F172A;
            color: #E6E6FA;
            font-family: 'Inter', sans-serif;
            font-size: 16px;
            line-height: 1.5;
        }
        h1, h2, h3, h4 {
            color: #FFFFFF;
            font-family: 'Inter', sans-serif;
            font-weight: 700;
        }
        h1 { font-size: 28px; }
        h4 { font-size: 18px; }
        .stTextInput > div > div > input {
            background: #1E293B;
            color: #E6E6FA;
            border: 1px solid #334155;
            border-radius: 8px;
            font-family: 'Inter', sans-serif;
            font-size: 16px;
        }
        .stTextInput > label {
            color: #CBD5E1;
            font-family: 'Inter', sans-serif;
            font-size: 14px;
        }
        .stSelectbox > label {
            color: #CBD5E1;
            font-family: 'Inter', sans-serif;
            font-size: 14px;
        }
        .stFileUploader > div > div {
            background: #1E293B;
            border: 2px dashed #334155;
            border-radius: 8px;
            padding: 20px;
        }
        .stFileUploader > div > div > div > button {
            background: #0EA5E9;
            color: white;
            border-radius: 6px;
            padding: 6px 15px;
            font-family: 'Inter', sans-serif;
            transition: all 0.3s ease;
        }
        .stFileUploader > div > div > div > button:hover {
            background: #0284C7;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        .comparison-container {
            display: flex;
            gap: 20px;
            margin-top: 20px;
            flex-wrap: wrap;
        }
        .comparison-column {
            flex: 1;
            background: linear-gradient(145deg, #1E293B, #283447);
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
            border-left: 4px solid #0EA5E9;
            min-width: 300px;
        }
        .comparison-column p {
            font-family: 'Inter', sans-serif;
            font-size: 16px;
            color: #E6E6FA;
        }
        .model-badge {
            display: inline-block;
            background: linear-gradient(90deg, #0284C7, #0EA5E9);
            color: #FFFFFF;
            font-family: 'Inter', sans-serif;
            font-size: 12px;
            font-weight: 600;
            padding: 3px 12px;
            border-radius: 16px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        .info-box {
            background: rgba(14, 165, 233, 0.1);
            border-radius: 8px;
            padding: 16px;
            border-left: 4px solid #0EA5E9;
            font-family: 'Inter', sans-serif;
            font-size: 14px;
        }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="ChatPDF Assistant", layout="wide", page_icon="🔥")
    set_custom_style()
    
    # Header
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 20px;">
        <h1>🔥 ChatPDF Assistant</h1>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload PDF:", type="pdf")
        if uploaded_file and 'chunks' not in st.session_state:
            with st.spinner('Processing PDF...'):
                chunks = process_pdf(uploaded_file)
                if chunks:
                    st.session_state['chunks'] = chunks
                    st.session_state['index'] = create_embeddings(chunks)
                    st.markdown('<div class="info-box">✓ PDF processed successfully.</div>', unsafe_allow_html=True)
                else:
                    st.error("Failed to process PDF.")
        
        models = get_free_openrouter_models()
        st.subheader("Select Models")
        model1 = st.selectbox("⚙️ Model 1:", models, key="model1")
        model2 = st.selectbox("⚙️ Model 2:", models, key="model2")

    user_question = st.text_input("Ask a question:", placeholder="Type your question about the PDF...")
        
    if user_question:
        if not user_question.strip():
            st.warning("Please enter a question.")
        elif 'chunks' not in st.session_state:
            st.warning("Please upload a PDF first.")
        elif "No free models available." in [model1, model2]:
            st.warning("No free models available. Please try again later.")
        else:
            with st.spinner('Fetching answers...'):
                question_embedding = EMBEDDING_MODEL.encode([user_question])
                _, indices = st.session_state['index'].search(question_embedding.astype('float32'), 3)
                context = "\n".join(st.session_state['chunks'][i] for i in indices[0])
                
                answer1 = query_model(user_question, context, model1)
                answer2 = query_model(user_question, context, model2)
                
                st.markdown(f"""
                <div class="comparison-container">
                    <div class="comparison-column">
                        <p class="model-badge">⚙️ {model1}</p>
                        <p>{answer1}</p>
                    </div>
                    <div class="comparison-column">
                        <p class="model-badge">⚙️ {model2}</p>
                        <p>{answer2}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()