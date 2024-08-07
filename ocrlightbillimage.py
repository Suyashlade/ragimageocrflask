from flask import Flask, request, render_template, jsonify
from paddleocr import PaddleOCR
from PIL import Image
from io import BytesIO
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
import os

app = Flask(__name__)

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Function to extract text from PNG image using PaddleOCR
def extract_text_from_image(file):
    try:
        # Load the image
        img = Image.open(file)
        img = img.convert('RGB')
        
        # Convert PIL image to bytes
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()
        
        # Apply OCR
        result = ocr.ocr(img_bytes)
        text_lines = [line[1][0] for line in result[0]]
        text = "\n".join(text_lines)
        print(f"Extracted Text: {text}")  # Debug: Print extracted text
        return text, text_lines
    except Exception as e:
        print(f"Error processing image: {e}")
        return "", []

# Function to split text into chunks
def split_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    print(f"Text Chunks: {chunks}")  # Debug: Print text chunks
    return chunks

# Store chat history
chat_history = []

# Path for data storage
data_storage_path = "C:\\Users\\Suyash\\Desktop\\chromaafter\\aaj24624"

# Ensure the folder specified in data_storage_path exists
if not os.path.exists(data_storage_path):
    os.makedirs(data_storage_path)

# Initialize Ollama embeddings
embeddings = OllamaEmbeddings()

# Initialize Chroma vector store
vectorstore = Chroma(collection_name="documents", persist_directory=data_storage_path, embedding_function=embeddings)

@app.route('/', methods=['GET', 'POST'])
def index():
    global vectorstore
    global chat_history
    show_chat = False
    if request.method == 'POST':
        if 'file' in request.files:
            uploaded_file = request.files['file']
            try:
                print("Extracting text from PNG image...")
                image_text, text_lines = extract_text_from_image(uploaded_file)
                if not image_text:
                    raise ValueError("No text extracted from image.")
                
                print("Text extraction complete. Splitting text into chunks...")
                chunks = split_text(image_text)
                print("Text splitting complete. Saving chunks...")

                 # Save chunks as text files and add to Chroma vector store
                for idx, chunk in enumerate(chunks):
                    doc = Document(page_content=chunk)
                    vectorstore.add_documents([doc])
                    with open(os.path.join(data_storage_path, f"chunk_{idx}.txt"), "w", encoding="utf-8") as f:
                        f.write(chunk)

                message = "PNG successfully uploaded to ChromaDB."
                show_chat = True  # Set flag to show chat input box

            except Exception as e:
                message = f"An error occurred: {str(e)}"
                print(message)

            return render_template('index2.html', message=message, chat_history=chat_history, show_chat=show_chat)

    return render_template('index2.html', chat_history=chat_history, show_chat=show_chat)

@app.route('/query', methods=['POST'])
def query():
    global vectorstore
    global chat_history
    user_query = request.form.get('query')
    try:
        print(f"User Query: {user_query}")
        # Initialize Ollama model
        llm = Ollama(model="llama2", base_url="http://localhost:11434")

        # Create retriever from vector store
        retriever = vectorstore.as_retriever()

        # Retrieve relevant documents
        relevant_docs = retriever.get_relevant_documents(user_query)
        print(f"Relevant Documents: {relevant_docs}")

        # Check if relevant documents are found
        if relevant_docs:
            relevant_texts = " ".join([doc.page_content for doc in relevant_docs])

            # Create prompt
            prompt = f"User query: {user_query}\n\nMost relevant document content:\n{relevant_texts}\n\nAnswer the query based on the document content above. If the document content does not have information related to the query then just print the message 'I don't know', do not print any other information."
            print(f"Prompt: {prompt}")

            # Generate response
            response = llm(prompt)
            print(f"Model Response: {response}")

            # Check if the response is relevant by verifying if it contains part of the document content
            if not any(relevant_text in response for relevant_text in relevant_texts.split()):
                response = "I don't know"
        else:
            response = "I don't know"
        
        chat_history.append((user_query, response))
        return jsonify({'query': user_query, 'response': response})

    except Exception as e:
        response = "I don't know"
        print(f"An error occurred during query processing: {str(e)}")
        chat_history.append((user_query, response))
        return jsonify({'query': user_query, 'response': response})

if __name__ == '__main__':
    app.run()

