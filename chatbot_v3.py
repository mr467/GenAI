import os
import fitz  # PyMuPDF for PDF processing
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI
import numpy as np

# Set API keys and environment configurations
os.environ["OPENAI_API_KEY"] = "My Key"  # Replace with your actual key
client_openAI = OpenAI()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_FORCE_CPU"] = "True"

# Initialize Sentence Transformer and FAISS
# model = SentenceTransformer('all-MiniLM-L6-v2')
# Change to qa model for better performance in QA tasks of chatbot
model = SentenceTransformer('multi-qa-distilbert-cos-v1')
index = faiss.IndexFlatL2(model.get_sentence_embedding_dimension())

# function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    texts = [page.get_text() for page in doc]
    doc.close()
    return " ".join(texts)


# Function to chunk text
def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]
    return chunks


pdf_path = "Module Descriptions.pdf"
text = extract_text_from_pdf(pdf_path)
chunks = chunk_text(text)

for chunk in chunks:
    embedding = model.encode(chunk, convert_to_tensor=True).cpu().numpy()
    index.add(np.array([embedding]))


# Define a simple conversation memory to store history
class SimpleConversationMemory:
    def __init__(self):
        self.history = []

    def add_to_history(self, query, response):
        self.history.append({"query": query, "response": response})

    def get_formatted_history(self):
        history_text = ""
        for exchange in self.history:
            history_text += f"Q: {exchange['query']}\nA: {exchange['response']}\n"
        return history_text


# Function to query the index and generate a response
def query_index_and_generate_response(query, conversation_memory, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True).cpu().numpy()
    _, indices = index.search(query_embedding.reshape(1, -1), top_k)
    context = ' '.join([chunks[i] for i in indices.flatten()])
    history_context = conversation_memory.get_formatted_history()

    prompt = f"{history_context}Given the context: {context}\nQuestion: {query}\nAnswer: "

    response = client_openAI.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.5,  # Adjust temperature to 0.5 for a balance between creativity and coherence,
        # ideal for the chatbot
        max_tokens=300,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    # Return the generated response or a default message
    return response.choices[0].message.content if response.choices else "No response generated."


if __name__ == "__main__":
    # Initialize the conversation memory
    conversation_memory = SimpleConversationMemory()

    while True:
        query = input("Ask a question about the HSLU courses (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        response = query_index_and_generate_response(query, conversation_memory)
        print("Response:", response)
        conversation_memory.add_to_history(query, response)
        print("Conversation History:")
        print(conversation_memory.get_formatted_history())

