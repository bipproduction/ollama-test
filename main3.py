from simple_ollama_rag import SimpleOllamaRag
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

def main():
    # Membuat instance dari SimpleOllamaRag
    

    print("Welcome to Simple Chat!")
    print("You can start chatting by typing your message. Enter 'exit' to quit.")

    while True:
        # Meminta pengguna untuk memasukkan pesan
        user_input = input("You: ")

        # Keluar dari loop jika pengguna memasukkan 'exit'
        if user_input.lower() == 'exit':
            print("Exiting Simple Chat. Goodbye!")
            break

        # Mendapatkan respons dari RAG berdasarkan masukan pengguna
        response = so_rag.rag_chain(user_input)

        # Mencetak konten pesan dari respons
        print("Bot:", response[0]["message"]["content"])

if __name__ == "__main__":
    main()
