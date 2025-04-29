import os
import streamlit as st
from dotenv import load_dotenv
import requests
import base64
from PyPDF2 import PdfReader
from datetime import datetime, timedelta

# Load environment
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY not found in .env file.")
    st.stop()

# Utilities
def get_free_openrouter_models(required_modality="text"):
    try:
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
        res = requests.get("https://openrouter.ai/api/v1/models", headers=headers)
        res.raise_for_status()
        data = res.json().get('data', [])
        free = []
        for m in data:
            pr = m.get('pricing', {})
            if all(pr.get(key, '0') == '0' for key in ['prompt', 'completion', 'request', 'image', 'web_search', 'internal_reasoning']):
                arch = m.get('architecture', {})
                input_modalities = arch.get('input_modalities', ['text'])
                if required_modality in input_modalities:
                    free.append({
                        'id': m['id'],
                        'name': m['name'],
                        'input_modalities': input_modalities
                    })
        return free
    except Exception as e:
        st.error(f"Error fetching models: {e}")
        return []

def encode_image(file):
    try:
        return base64.b64encode(file.read()).decode('utf-8')
    except Exception as e:
        st.error(f"Image encoding error: {e}")
        return None

def process_pdf(file):
    try:
        reader = PdfReader(file)
        text = ''.join(p.extract_text() or '' for p in reader.pages)
        if not text.strip():
            raise ValueError("No text extracted from PDF.")
        return text
    except Exception as e:
        st.error(f"PDF processing error: {e}")
        return None

def query_model(prompt, model_id, image_data=None):
    try:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {OPENROUTER_API_KEY}"}
        messages = []
        
        if image_data:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                ]
            })
        else:
            messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model_id,
            "messages": messages
        }
        resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data['choices'][0]['message']['content']
    except requests.HTTPError as he:
        err = he.response.json().get('error', {})
        reset = err.get('metadata', {}).get('headers', {}).get('X-RateLimit-Reset')
        if reset:
            t = datetime.fromtimestamp(int(reset) / 1000) + timedelta(hours=7)
            return f"Rate limit exceeded. Try after {t.strftime('%Y-%m-%d %H:%M:%S WIB')}"
        return f"Model API error: {err.get('message', str(he))}"
    except Exception as e:
        return f"Unexpected error: {e}"

def main():
    st.set_page_config(page_title="Zoro.ai", layout="wide", page_icon="🔥")
    st.sidebar.title("⚙️ Zoro's Settings")

    # File upload
    uploaded_file = st.sidebar.file_uploader("📂 Upload:", type=['pdf', 'png', 'jpg', 'jpeg'])

    # Determine input type and modality
    input_type = "text"
    file_data = None
    pdf_text = None
    if uploaded_file:
        file_extension = uploaded_file.name.lower().split('.')[-1]
        if file_extension in ['png', 'jpg', 'jpeg']:
            input_type = "image"
            file_data = encode_image(uploaded_file)
            if not file_data:
                st.error("Failed to encode image. Please try another file.")
                return
        elif file_extension == 'pdf':
            input_type = "text"
            pdf_text = process_pdf(uploaded_file)
            if not pdf_text:
                st.error("Failed to extract text from PDF. Please try another file.")
                return

    # Display extracted PDF text
    if pdf_text:
        st.sidebar.success("PDF extracted successfully.")

    # Model selection
    models = get_free_openrouter_models(required_modality=input_type)
    model_id = st.sidebar.selectbox("🤖 Model:", options=[m['id'] for m in models], format_func=lambda x: next(m['name'] for m in models if m['id'] == x))

    # Main interface
    st.header("🔥 Zoro.ai")
    
    # Use a form to enable Enter key submission
    with st.form(key="query_form"):
        prompt = st.text_area("Prompt:", placeholder="What do you want to know...")
        submit_button = st.form_submit_button("Get Answers", type="secondary")

        if submit_button:
            if not prompt:
                st.error("Please enter a prompt before submitting.")
            else:
                final_prompt = f"{pdf_text}\n\n{prompt}" if pdf_text else prompt
                with st.spinner("Wait a second..."):
                    ans = query_model(final_prompt, model_id, file_data if input_type == "image" else None)
                st.markdown(f'{ans}')

if __name__ == '__main__':
    main()