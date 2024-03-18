from simple_ollama_rag import SimpleOllamaRag

# Membuat instance dari SimpleOllamaRag
so_rag = SimpleOllamaRag(
    inference_model="phi",
    embeddings_model="nomic-embed-text",
    tokenizer_semantic_chunk="bert-base-uncased",
    persist_directory="db",
    rag_data_directory="rag_data",
    max_tokens_embeddings=100,
    inference_config={"stop": ["\n"]},
)

# Memuat vector store
so_rag.load_vectorstore()

# Bertanya pertanyaan
question = 'What are not true salmon?'
# Mendapatkan respons dari RAG
response = so_rag.rag_chain(question)

# Mencetak konten pesan dari respons
print(response)