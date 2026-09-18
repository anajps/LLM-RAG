import PyPDF2
from openai import OpenAI
import chromadb
import uuid
import os

openai_client = OpenAI(api_key="OPENAI_API_KEY")

#chromadb_path = "C:/Users/psana/OneDrive - Universidade Federal de Uberlândia/IC/llM_IC" # CONFIG YOUR PATH
#chromadb_path = "C:/Users/Ana Júlia Peixoto/Downloads/LLM_IC/LLM_IC"
# Altere para:
chromadb_path = "C:\\Users\\Ana Júlia Peixoto\\Downloads\\LLM_IC\\LLM_IC"#
chroma_client = chromadb.PersistentClient(path=chromadb_path)
collection = chroma_client.get_collection("my_collection")

#chroma_client.delete_collection(name="my_collection")
#collection = chroma_client.create_collection(name="my_collection")
chroma_client = chromadb.PersistentClient(path=chromadb_path)
# Apenas carrega a coleção existente (que foi criada pelo prep_docs.py)
collection = chroma_client.get_or_create_collection(name="my_collection")
# prompt_template = """Você é um assistente de IA que responde as dúvidas dos usuários com bases nos documentos a baixo.
# Os documentos abaixo apresentam as fontes atualizadas e devem ser consideradas como verdade.
# Cite a fonte quando fornecer a informação. 
# Documentos:
# {documents}
# """

def get_embedding(text):
    """Transform a text in a vector using embedding model"""
    embedding = openai_client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    
    return embedding.data[0].embedding

def search_document(question):
    """Search for relevant documents in ChromaDB according to the question"""
    query_embedding = get_embedding(question)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    return results

def format_search_result(relevant_documents):
    """Auxiliary function to format a list of documents into a string"""
    formatted_list = []
    
    for i, doc in enumerate(relevant_documents["documents"][0]):
        formatted_list.append("[{}]: {}".format(relevant_documents["metadatas"][0][i]["source"], doc))
    
    documents_str = "\n".join(formatted_list)
    return documents_str



prompt_template = """Você é um assistente de IA que responde as dúvidas dos usuários com bases nos documentos a baixo.
Os documentos abaixo apresentam as fontes atualizadas e devem ser consideradas como verdade.
Cite a fonte quando fornecer a informação. 
Documentos:
{documents}
"""




def execute_llm(prompt, question):
    """Execute a call to LLM using prompt for system and question for user message"""
    chat_completion = openai_client.chat.completions.create(
        messages=[
            {"role": "system","content": f"{prompt}"},
            {"role": "user","content": f"{question}"}
        ],
        model="gpt-4o",
        max_tokens=500,
        temperature=0
    )

    return chat_completion.choices[0].message.content

def run():
    while True:
        question = input(">> Digite sua pergunta (ou 'sair' para encerrar): ").strip()
        if not question:
            print("Você precisa digitar uma pergunta.")
            continue
        if question.lower() in ["sair", "exit", "quit"]:
            print("Encerrando...")
            break

        relevant_documents = search_document(question)
        documents_str = format_search_result(relevant_documents)
        prompt = prompt_template.format(documents=documents_str)
        answer = execute_llm(prompt, question)

        print("\nResposta:\n")
        print(answer)


if __name__ == "__main__":
    run()