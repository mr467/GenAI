import os
import fitz  # PyMuPDF for PDF processing
from sentence_transformers import SentenceTransformer
import faiss
from transformers import RagTokenizer, RagTokenForGeneration
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cdist
from openai import OpenAI


# Option 1: Set API key as environment variable (recommended)
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"  # Replace with your actual key
client = OpenAI()


# Use an environment variable for the OpenAI API key for better security
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_FORCE_CPU"] = "True"

# Initialize Sentence Transformer and FAISS
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.IndexFlatL2(model.get_sentence_embedding_dimension())

# Initialize RAG components
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
rag_model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq")

# Context preface for the chatbot
CONTEXT_PREFACE = (
    "As a knowledgeable study guide, you're equipped with comprehensive information from the 'Module Descriptions' document. "
    "Your role is to assist students in navigating their course selections with insightful guidance. "
    "When students inquire about their coursework, you'll offer detailed advice on: \n"
    "- The specific programming languages each module entails, highlighting the applicability of Python, Java, or any other language.\n"
    "- Identifying which modules are compulsory and which are elective, providing clarity on the curriculum structure.\n"
    "- Elaborating on the assessment methods used in each module, such as written exams, project work, or continuous assessment, to help students prepare effectively.\n"
    "- Any prerequisites for modules, ensuring students meet necessary competencies before enrollment.\n"
    "Your responses should be concise, informative, and directly relevant to the queries posed, empowering students to make informed decisions about their academic journey."
)

# Global variable for conversation history
conversation_history = []

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    texts = [page.get_text() for page in doc]
    doc.close()
    return " ".join(texts)


# Function to chunk text
def chunk_text(text, chunk_size=300, overlap=100):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]
    return chunks


# Function to calculate TF-IDF weights for chunks
def calculate_tfidf(chunks):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(chunks)
    return {chunk: tfidf_vector for chunk, tfidf_vector in zip(chunks, tfidf_matrix.toarray())}


# Function to add chunks to FAISS index
def add_to_index(chunks, tfidf_weights):
    all_text_chunks = []
    index_to_chunk_map = {}
    for i, chunk in enumerate(chunks):
        embedding = model.encode([chunk], convert_to_tensor=True).cpu().numpy()
        index.add(embedding)
        all_text_chunks.append(chunk)
        index_to_chunk_map[i] = chunk
    return all_text_chunks, index_to_chunk_map


# Query function
def query_index(query, top_k=5):
    query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()  # Add .cpu() before .numpy()
    distances, indices = index.search(query_embedding, top_k)
    return indices[0]

# Generate response
def generate_response(input_text):
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    output = rag_model.generate(input_ids)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


# Process PDFs and populate the index
pdf_paths = ["Module Descriptions.pdf"]
all_text_chunks = []
index_to_chunk_map = {}
for pdf_path in pdf_paths:
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    tfidf_weights = calculate_tfidf(chunks)
    chunk_list, chunk_map = add_to_index(chunks, tfidf_weights)
    all_text_chunks.extend(chunk_list)
    index_to_chunk_map.update(chunk_map)


# Function to handle user query
def handle_query(query):
    global conversation_history

    indices = query_index(query)
    retrieved_chunks = [all_text_chunks[i] for i in indices]

    tfidf_weights = calculate_tfidf([query] + retrieved_chunks)  # Recalculate TF-IDF for query and chunks

    query_tfidf = tfidf_weights[query]  # Access query's TF-IDF vector
    chunk_tfidf = [tfidf_weights[chunk] for chunk in retrieved_chunks]  # Extract TF-IDF vectors for retrieved chunks
    cosine_similarities = 1 - cdist(query_tfidf.reshape(1, -1), chunk_tfidf,
                                    metric='cosine').flatten()  # Calculate cosine similarities

    ranked_chunks = sorted(zip(retrieved_chunks, cosine_similarities), key=lambda item: item[1],
                           reverse=True)  # Sort by similarity

    # Use top-ranked chunks for generating a response
    input_text = " ".join([chunk for chunk, _ in ranked_chunks[:3]])  # Join top 3 ranked chunks

    response = generate_response_with_chatgpt(input_text)
    conversation_history.append({"query": query, "response": response})

    return response


def preprocess_query_with_context(query):
    # Safely concatenate recent queries and responses
    recent_interactions = " ".join([
        f'Q: {interact.get("query", "")} A: {interact.get("response", "")}'
        for interact in conversation_history[-5:]
    ])
    return f"{CONTEXT_PREFACE} Recent Interactions: {recent_interactions} \n\nYour Question: {query}"


# Function to generate response using ChatGPT
def generate_response_with_chatgpt(input_text):
    # Preprocess query to include context
    contextual_query = preprocess_query_with_context(input_text)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Adjust based on the available models
        messages=[{"role": "system", "content": contextual_query}],
        temperature=0.8,
        max_tokens=500,  # Adjust based on needs
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    # Assuming the structure includes 'choices' with messages containing 'content'
    if response.choices and len(response.choices) > 0:
        generated_text = response.choices[0].message.content
        return generated_text
    else:
        return "No response generated."


if __name__ == "__main__":
    while True:
        query = input("Ask a question about the courses (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        response = handle_query(query)
        print("Response:", response)
