from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
import re
import requests
import pickle
import streamlit as st


# Path to your .txt folder
TXT_FOLDER = "flattened_chunk_txts"

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize DeepSeek API
DEEPSEEK_API_KEY = "sk-06800a756a184606a88395b24f436b95"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    "Content-Type": "application/json"
}

# Function to read txt file
def read_txt_file(file_path):
    file_contents = ''
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            file_contents = file.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return file_contents

def extract_paths(text):
    """Extract HTML path and ALL image paths from the text"""
    html_pattern = r'HTML file:\s*([^\.]+\.htm)'
    image_pattern = r'Image file:\s*([^\n]+)'
    
    html_match = re.search(html_pattern, text)
    image_matches = re.findall(image_pattern, text)
    
    html_path = html_match.group(1) if html_match else None
    image_paths = list(set(image_matches))
    
    return html_path, image_paths

# Check if index already exists, if not build it
def build_or_load_index():
    if os.path.exists("doc_index.faiss") and os.path.exists("metadata.pkl"):
        print("Loading existing index...")
        index = faiss.read_index("doc_index.faiss")
        with open("metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        
        # Check if metadata has content stored
        if metadata and "content" not in metadata[0]:
            print("Warning: Old metadata format detected. Rebuilding index with content...")
            return rebuild_index_with_content()
    else:
        return rebuild_index_with_content()
    
    return index, metadata

def rebuild_index_with_content():
    """Rebuild the index ensuring content is stored in metadata"""
    print("Building new index with content storage...")
    index = None
    metadata = []
    all_embeddings = []

    # Loop over files
    for filename in os.listdir(TXT_FOLDER):
        if filename.endswith(".txt"):
            file_path = os.path.join(TXT_FOLDER, filename)
            text = read_txt_file(file_path)
            if not text.strip():
                continue

            # Generate embedding for this doc
            embedding = model.encode(text, convert_to_numpy=True)
            all_embeddings.append(embedding)

            # Extract paths
            source_file, images = extract_paths(text)
            metadata.append({
                "source": source_file,
                "images": images,
                "filename": filename,
                "content": text  # CRITICAL: Store the actual content
            })
            
            print(f"Processed: {filename}")
            
            if filename == 'OLF_STAT_Help_Viewer_OLF_STAT_Help_Viewer.htm_0.txt':
                print(metadata)

    # Create FAISS index with all embeddings
    if all_embeddings:
        all_embeddings = np.array(all_embeddings)
        index = faiss.IndexFlatL2(all_embeddings.shape[1])
        index.add(all_embeddings)
        
        print(f"Indexed {len(metadata)} chunks.")
        
        # Save FAISS index and metadata
        faiss.write_index(index, "doc_index.faiss")
        with open("metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)
        
        print("Index and metadata saved successfully!")
        return index, metadata
    else:
        print("No documents were processed.")
        return None, None

def retrieve(query, model, index, metadata, k=3):
    """Retrieve top k most relevant documents"""
    query_vec = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vec, k)

    results = []
    for i, idx in enumerate(I[0]):
        if idx < len(metadata):
            results.append({
                "source": metadata[idx]["source"],
                "images": metadata[idx]["images"],
                "content": metadata[idx].get("content", ""),  # Get the stored content
                "filename": metadata[idx].get("filename", ""),
                "distance": float(D[0][i])
            })
    return results

def answer_question(query, model, index, metadata):
    if index is None or not metadata:
        return "Error: Index not properly loaded."
    
    # Retrieve top matching docs
    context_results = retrieve(query, model, index, metadata, k=3)
    # Debug: Check what content we retrieved
    # print(f"\nDebug - Retrieved content preview:")
    for i, result in enumerate(context_results):
        content_preview = result.get('content', '')[:500] + "..." if result.get('content') else "EMPTY CONTENT"
        # print(f"Doc {i+1}: {content_preview}")
    
    # Build context from actual content
    context_parts = []
    for i, item in enumerate(context_results):
        if item.get('content'):
            context_parts.append(f"--- Document {i+1} from {item['source']} ---")
            context_parts.append(f"Source: {item['source']}")
            context_parts.append(f"Filename: {item['filename']}")
            context_parts.append(f"Images: {', '.join(item['images']) if item['images'] else 'No images'}")
            context_parts.append("Content:")
            context_parts.append(item['content'])
            context_parts.append("")  # Empty line between documents

    
    if not context_parts:
        return "No relevant content found in the documents to answer this question."
    
    context = "\n".join(context_parts)
    
    prompt = f"""Based EXCLUSIVELY on the following documentation, answer the user's question. If the answer cannot be found in this documentation, say so.

CRITICAL INSTRUCTIONS:
1. Answer using ONLY the information from the documentation provided
2. When mentioning any images, you MUST include the complete image file paths exactly as shown in the documentation
3. Do not modify or shorten the image paths in any way

Documentation:
{context}

Question: {query}

Answer based only on the documentation above. Always include full image paths:\n\n\n\n"""

    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 5000,
        "temperature": 0.1,
        "stream": False
    }

    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        return f"Error: Failed to get response from API - {e}"

index, metadata = build_or_load_index()


# Streamlit app
st.title("Endur V23 Chatbot")

user_input = st.text_input("Ask a question:")

if st.button("Submit") and user_input:
    with st.spinner("Thinking..."):
        answer = answer_question(user_input, model, index, metadata)

    try:
        script_dir = os.getcwd()
    except:
        script_dir = os.path.dirname(os.path.abspath(__file__))

    onlinehelp_path = os.path.join(script_dir, "OnlineHelp_test\\OLF")

    # Find and display images
    image_pattern = r"([^\s]+?\.(?:jpg|jpeg|png|gif))"
    image_matches = re.findall(image_pattern, answer)


    for img_path in image_matches:
        img_path = img_path.replace('`', '')
        full_path = os.path.join(onlinehelp_path, img_path)
        
        # Normalize the path (handles any path inconsistencies)
        full_path = os.path.normpath(full_path)
        st.image(full_path)

    # Display text without image markers
    clean_text = re.sub(image_pattern, '', answer)
    st.write(clean_text)
