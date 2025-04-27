import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import requests
from dotenv import load_dotenv
import streamlit.components.v1 as components
import json

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
EMBEDDING_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

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
        return response.json()['choices'][0]['message']['content']
    except requests.RequestException as e:
        return f"Error fetching answer: {e}"

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
        .copy-button {
            background: #334155;
            color: #E6E6FA;
            border: none;
            border-radius: 6px;
            padding: 8px 16px;
            font-family: 'Inter', sans-serif;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .copy-button:hover {
            background: #475569;
            box-shadow: 0 4px 8px rgba(0, 0, 0,  Uy0.2);
        }
        .copy-button.copied {
            background: #14B8A6;
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
            with st.spinner('Processing document...'):
                chunks = process_pdf(uploaded_file)
                if chunks:
                    st.session_state['chunks'] = chunks
                    st.session_state['index'] = create_embeddings(chunks)
                    st.markdown('<div class="info-box">✓ Document processed successfully.</div>', unsafe_allow_html=True)
                else:
                    st.error("Failed to process document.")
        
        models = get_free_openrouter_models()
        st.subheader("Select Models")
        model1 = st.selectbox("⚙️ Model 1:", models, key="model1")
        model2 = st.selectbox("⚙️ Model 2:", models, key="model2")

    user_question = st.text_input("Ask a question:", placeholder="Type your question about the document...")
        
    if user_question:
        if not user_question.strip():
            st.warning("Please enter a question.")
        elif 'chunks' not in st.session_state:
            st.warning("Please upload a PDF document first.")
        elif "No free models" in [model1, model2]:
            st.warning("No free models available. Please try again later.")
        else:
            with st.spinner('Fetching answers...'):
                question_embedding = EMBEDDING_MODEL.encode([user_question])
                _, indices = st.session_state['index'].search(question_embedding.astype('float32'), 3)
                context = "\n".join(st.session_state['chunks'][i] for i in indices[0])
                
                answer1 = query_model(user_question, context, model1)
                answer2 = query_model(user_question, context, model2)
                
                # Escape answers for JavaScript by encoding to JSON
                answer1_escaped = json.dumps(answer1)
                answer2_escaped = json.dumps(answer2)
                
                # Generate unique IDs for copy buttons
                answer1_id = "answer1_copy"
                answer2_id = "answer2_copy"
                
                # Display answers with badges and copy buttons
                st.markdown(f"""
                <div class="comparison-container">
                    <div class="comparison-column">
                        <p class="model-badge">⚙️ {model1}</p>
                        <p>{answer1}</p>
                        <button class="copy-button" id="{answer1_id}_btn" onclick="copyToClipboard('{answer1_id}')">Copy Answer</button>
                    </div>
                    <div class="comparison-column">
                        <p class="model-badge">⚙️ {model2}</p>
                        <p>{answer2}</p>
                        <button class="copy-button" id="{answer2_id}_btn" onclick="copyToClipboard('{answer2_id}')">Copy Answer</button>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Inject JavaScript for copying text with textarea fallback
                components.html("""
                <script>
                    function copyToClipboard(id) {
                        var text;
                        if (id === '%s') {
                            text = %s;
                        } else if (id === '%s') {
                            text = %s;
                        }
                        
                        // Try modern Clipboard API
                        if (navigator.clipboard) {
                            navigator.clipboard.writeText(text).then(() => {
                                var button = document.getElementById(id + '_btn');
                                button.textContent = 'Copied!';
                                button.classList.add('copied');
                                setTimeout(() => {
                                    button.textContent = 'Copy Answer';
                                    button.classList.remove('copied');
                                }, 2000);
                            }).catch(err => {
                                fallbackCopy(text, id);
                            });
                        } else {
                            fallbackCopy(text, id);
                        }
                    }
                    function fallbackCopy(text, id) {
                        var textarea = document.createElement('textarea');
                        textarea.value = text;
                        document.body.appendChild(textarea);
                        textarea.select();
                        try {
                            document.execCommand('copy');
                            var button = document.getElementById(id + '_btn');
                            button.textContent = 'Copied!';
                            button.classList.add('copied');
                            setTimeout(() => {
                                button.textContent = 'Copy Answer';
                                button.classList.remove('copied');
                            }, 2000);
                        } catch (err) {
                            console.error('Fallback copy failed: ', err);
                            alert('Failed to copy text. Please select and copy manually.');
                        }
                        document.body.removeChild(textarea);
                    }
                </script>
                """ % (answer1_id, answer1_escaped, answer2_id, answer2_escaped), height=0)

if __name__ == "__main__":
    main()