import os
import chromadb
from sentence_transformers import SentenceTransformer

docs_path = "support_assistant/docs"
vector_path = "support_assistant/vector_store"

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=vector_path)

collection = client.get_or_create_collection(
    name="zepto_policies"
)

documents = []
ids = []
metadatas = []

chunk_size = 300
overlap = 50

for filename in sorted(os.listdir(docs_path)):
    if filename.endswith(".txt"):
        file_path = os.path.join(docs_path, filename)

        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read().strip()

        doc_id = filename.replace(".txt", "")

        start = 0
        chunk_number = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()

            if chunk:
                documents.append(chunk)
                ids.append(f"{doc_id}_chunk_{chunk_number}")
                metadatas.append({"source": doc_id})

            chunk_number += 1
            start += chunk_size - overlap

embeddings = model.encode(documents).tolist()

collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)

print("Documents loaded:", len(set(item["source"] for item in metadatas)))
print("Chunks created:", len(documents))
print("Embeddings created:", len(embeddings))
print("ChromaDB collection:", collection.name)
print("Documents stored:", collection.count())