from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-V2"
)


# load vector DB
db = FAISS.load_local(
    "data/vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)

# check index size
print("Index size:", db.index.ntotal)

# test retrieval
docs = db.similarity_search("AI assignment", k=3)

print("Docs retrieved:", len(docs))

for i, d in enumerate(docs):
    print(f"\nDoc {i+1}:")
    print(d.page_content[:200])