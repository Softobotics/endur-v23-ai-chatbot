
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
from docx import Document

# Path to your .docx folder
DOCX_FOLDER = "cleaned_docs"

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create FAISS index
index = None
metadata = []  # to store source filenames for retrieval

# Function to read docx file
def read_docx(path):
    doc = Document(path)
    return "\n".join([para.text for para in doc.paragraphs])


def chunk_text(text, chunk_size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # small overlap so context isn’t cut mid-sentence
    return chunks

# Loop over files
for filename in os.listdir(DOCX_FOLDER):
    if filename.endswith(".docx"):
        file_path = os.path.join(DOCX_FOLDER, filename)
        text = read_docx(file_path)

        # Split into chunks of ~500 chars
        chunks = chunk_text(text, chunk_size=3000, overlap=200)

        # Generate embedding for this doc
        vector = model.encode(chunks)  # returns list of embeddings

        if index is None:
            index = faiss.IndexFlatL2(vector.shape[1])  # initialize with embedding size

        index.add(np.array(vector))
        for i, chunk in enumerate(chunks):
            metadata.append({
                "source": filename,
                "chunk_id": i,
                "content": chunk
            })

print(f"Indexed {len(metadata)} chunks.")

# Save FAISS index
faiss.write_index(index, "doc_index.faiss")

# Save metadata for later retrieval
import pickle
with open("metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)



import requests
import numpy as np

def retrieve(query, k=3):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), k)

    results = []
    for idx in I[0]:
        results.append({
            "source": metadata[idx]["source"],
            "content": metadata[idx]["content"],
            "distance": float(D[0][list(I[0]).index(idx)])
        })
    return results


# Initialize DeepSeek API
DEEPSEEK_API_KEY = "sk-06800a756a184606a88395b24f436b95"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    "Content-Type": "application/json"
}


def answer_question(query):
    # Retrieve context
    context_results = retrieve(query)
    context = "\n".join([item["content"] for item in context_results])
    
    prompt = f"Use only the documentation below to answer the question.\n\nDocs:\n{context}\n\nQuestion: {query}\nAnswer:"
    payload = {
        "model": "deepseek-chat",  # or try "deepseek-coder" for code-related tasks
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 5000,
        "temperature": 0.1,
        "stream": False
    }

    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        answer = data["choices"][0]["message"]["content"]
        return answer.strip()
        
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    


import streamlit as st

# Streamlit app
st.title("Endur V23 Chatbot")

# Take user input
user_input = st.text_input("Ask a question:")

if st.button("Submit") and user_input:
    with st.spinner("Thinking..."):
        answer = answer_question(user_input)
    st.write("### Answer")
    st.write(answer)
