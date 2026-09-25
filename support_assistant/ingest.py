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

for filename in sorted(os.listdir(docs_path)):
    if filename.endswith(".txt"):
        file_path = os.path.join(docs_path, filename)

        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read().strip()

        documents.append(text)
        ids.append(filename.replace(".txt", ""))

embeddings = model.encode(documents).tolist()

collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings
)

print("Documents loaded:", len(documents))
print("Embeddings created:", len(embeddings))
print("ChromaDB collection:", collection.name)
print("Documents stored:", collection.count())