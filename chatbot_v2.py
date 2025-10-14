import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import re
import streamlit as st
import requests


def create_embeddings_from_chunks(chunks_folder, output_folder=None, model_name='all-MiniLM-L6-v2'):
    
    if output_folder is None:
        output_folder = chunks_folder
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize the model
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Get all txt files
    txt_files = [f for f in os.listdir(chunks_folder) if f.endswith('.txt')]
    print(f"Found {len(txt_files)} text files to process")
    
    texts = []
    metadatas = []
    filenames = []
    
    # Process each txt file
    for txt_file in txt_files:
        txt_path = os.path.join(chunks_folder, txt_file)
        
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                text_content = f.read().strip()
            
            if not text_content:
                print(f"Skipping empty file: {txt_file}")
                continue
            
            images_list = re.findall(r'!\[.*?\]\((.*?)\)', text_content)

            file_paths_list = re.findall(r'\b[\w\-]+\.htm[l]?\b', text_content)
            
            source_html_path = None
            base_name = os.path.splitext(txt_file)[0]
            
            # Check for possible source HTML files
            possible_html_files = [
                f"{base_name}.htm",
                f"{base_name}.html", 
                os.path.basename(base_name) + ".htm",
                os.path.basename(base_name) + ".html"
            ]
            
            for html_file in possible_html_files:
                possible_path = os.path.join(chunks_folder, html_file)
                if os.path.exists(possible_path):
                    source_html_path = possible_path
                    break
            
            if not source_html_path:
                source_html_path = txt_path.replace('.txt', '.htm')
                if not os.path.exists(source_html_path):
                    source_html_path = txt_path.replace('.txt', '.html')
                    if not os.path.exists(source_html_path):
                        source_html_path = txt_file.replace('.txt', '.htm') 
            
            texts.append(text_content)
            filenames.append(txt_file)
            
            metadata = {
                "images": images_list,
                "source_filepath": source_html_path,
                "linked_filepaths": file_paths_list,
                "chunk_filename": txt_file
            }
            metadatas.append(metadata)
            
            print(f"Processed: {txt_file} (images: {len(images_list)}, links: {len(file_paths_list)})")
            
        except Exception as e:
            print(f"Error processing {txt_file}: {str(e)}")
            continue
    
    if not texts:
        print("No valid text content found to process!")
        return None, None
    
    # Create embeddings
    print("Creating embeddings...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    
    # Create FAISS index
    print("Creating FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    # Save FAISS index
    index_path = os.path.join(output_folder, "faiss_index.index")
    faiss.write_index(index, index_path)
    
    # Save metadata
    metadata_path = os.path.join(output_folder, "metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump({
            'metadatas': metadatas,
            'filenames': filenames,
            'texts': texts,
            'dimension': dimension,
            'model_name': model_name
        }, f)
    
    print(f"Embeddings created successfully!")
    print(f"FAISS index saved to: {index_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Total documents processed: {len(texts)}")
    print(f"Embedding dimension: {dimension}")
    
    return index_path, metadata_path

def load_embeddings(index_path, metadata_path):
    index = faiss.read_index(index_path)
    
    with open(metadata_path, 'rb') as f:
        metadata_dict = pickle.load(f)
    
    return index, metadata_dict


chunks_folder = "chunks" 

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions" 
DEEPSEEK_API_KEY = "sk-06800a756a184606a88395b24f436b95" 

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
}

def retrieve(query, model, index, metadata_dict, k=3):
    query_vec = model.encode([query], convert_to_numpy=True)    
    faiss.normalize_L2(query_vec)
    D, I = index.search(query_vec, k)
    
    results = []
    for i, idx in enumerate(I[0]):
        if idx < len(metadata_dict['metadatas']):
            results.append({
                "source": metadata_dict['metadatas'][idx]["source_filepath"],
                "images": metadata_dict['metadatas'][idx]["images"],
                "content": metadata_dict['texts'][idx],
                "filename": metadata_dict['metadatas'][idx]["chunk_filename"],
                "linked_files": metadata_dict['metadatas'][idx]["linked_filepaths"],
                "distance": float(D[0][i])
            })
    return results

def answer_question(query, model, index, metadata_dict):
    """Answer question using retrieved context and DeepSek API"""
    if index is None or not metadata_dict:
        return "Error: Index not properly loaded."
    
    # Retrieve top matching docs
    context_results = retrieve(query, model, index, metadata_dict, k=3)
    
    # Build context from actual content
    context_parts = []
    for i, item in enumerate(context_results):
        if item.get('content'):
            context_parts.append(f"--- Document {i+1} ---")
            context_parts.append(f"Source File: {item['source']}")
            context_parts.append(f"Filename: {item['filename']}")
            context_parts.append(f"Images: {', '.join(item['images']) if item['images'] else 'No images'}")
            context_parts.append(f"Linked Files: {', '.join(item['linked_files']) if item['linked_files'] else 'No linked files'}")
            context_parts.append("Content:")
            context_parts.append(item['content'])
            context_parts.append("")  # Empty line between documents

    if not context_parts:
        return "No relevant content found in the documents to answer this question."
    
    context = "\n".join(context_parts)
    
    prompt = f"""Based EXCLUSIVELY on the following documentation, answer the user's question. If the answer cannot be found in this documentation, say so.

CRITICAL INSTRUCTIONS:
1. Be precise and factual - do not add information not present in the documentation
2. When mentioning any images, you MUST include the complete image file paths exactly as shown in the documentation
3. Try to include image file paths wherever possible so that user can understand better
4. If you encounter an HTML table code anywhere please convert to a visual table. 
5. Answer using ONLY the information from the documentation provided

Documentation:
{context}

Question: {query}

Answer based only on the documentation above. Always include full image paths when relevant, and provide detailed explanations for beginners:\n\n"""

    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 5000,
        "temperature": 0.1,
        "stream": False
    }

    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        return f"Error: Failed to get response from API - {e}"
    except KeyError as e:
        return f"Error: Unexpected API response format - {e}"

def initialize_system(chunks_folder="chunks", model_name='all-MiniLM-L6-v2'):
    
    """Initialize the complete system - create or load embeddings"""    
    index_path = os.path.join(chunks_folder, "faiss_index.index")
    metadata_path = os.path.join(chunks_folder, "metadata.pkl")
    
    # Check if embeddings already exist
    if os.path.exists(index_path) and os.path.exists(metadata_path):
        print("Loading existing embeddings...")
        index, metadata_dict = load_embeddings(index_path, metadata_path)
    else:
        print("Creating new embeddings...")
        index_path, metadata_path = create_embeddings_from_chunks(chunks_folder)
        index, metadata_dict = load_embeddings(index_path, metadata_path)
    
    # Load the sentence transformer model
    model = SentenceTransformer(model_name)
    
    return model, index, metadata_dict



# Streamlit app
st.title("Endur V23 Chatbot")
model, index, metadata_dict = initialize_system("chunks")
user_input = st.text_input("Ask a question:")

if st.button("Submit") and user_input:
    with st.spinner("Thinking..."):
        answer = answer_question(user_input, model, index, metadata_dict)
    
    try:
        script_dir = os.getcwd()
    except:
        script_dir = os.path.dirname(os.path.abspath(__file__))

    onlinehelp_path = os.path.join(script_dir, "OnlineHelp_test\\OLF")

    # Improved regex to capture complete image paths and remove unwanted characters
    image_pattern = r'!\[.*?\]\(([^)]+\.(?:jpg|jpeg|png|gif|bmp))\)|([a-zA-Z0-9_\-\\/]+\.(?:jpg|jpeg|png|gif|bmp))'
    image_matches = re.findall(image_pattern, answer)

    # Extract and clean image paths
    clean_image_paths = []
    for match in image_matches:
        img_path = match[0] if match[0] else match[1]
        if img_path:
            img_path = re.sub(r'[`"\'<>]', '', img_path)
            img_path = img_path.replace('\\', '/')
            img_path = img_path.strip()
            clean_image_paths.append(img_path)

    clean_text = re.sub(r'[a-zA-Z0-9_\-\\/]+\.(?:jpg|jpeg|png|gif|bmp)', 'KELLER999', answer)
    text_parts = clean_text.split('KELLER999')
    image_index = 0
    for i, text_part in enumerate(text_parts):
        # Display the text part
        if text_part.strip():
            st.write(text_part)
        
        # Display corresponding image after each text part (except the last one)
        if i < len(text_parts) - 1 and image_index < len(clean_image_paths):
            img_path = clean_image_paths[image_index]
            full_path = os.path.join(onlinehelp_path, img_path)
            full_path = os.path.normpath(full_path)
            
            if os.path.exists(full_path):
                st.image(full_path, caption=os.path.basename(img_path))
            else:
                st.warning(f"Image not found: {full_path}")
            
            image_index += 1