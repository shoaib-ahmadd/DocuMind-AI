import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_PATH = "data/uploads"
DB_PATH = "data/vector_db"

# ---------------------------------------------------
# Create folders if missing
# ---------------------------------------------------
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(DB_PATH, exist_ok=True)

# ---------------------------------------------------
# Load PDFs
# ---------------------------------------------------
documents = []

pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]

if not pdf_files:
    print("⚠️ No PDF files found in uploads folder.")
    print("➡️ Skipping vector DB creation safely.")
    exit()

for file in pdf_files:
    try:
        pdf_path = os.path.join(DATA_PATH, file)

        print(f"📄 Loading: {file}")

        loader = PyPDFLoader(pdf_path)
        loaded_docs = loader.load()

        if loaded_docs:
            documents.extend(loaded_docs)
            print(f"✅ Loaded {len(loaded_docs)} pages")
        else:
            print(f"⚠️ No text extracted from: {file}")

    except Exception as e:
        print(f"❌ Error loading {file}: {e}")

# ---------------------------------------------------
# Final document check
# ---------------------------------------------------
if not documents:
    print("❌ No valid document content found.")
    exit()

print(f"\n📚 Total pages loaded: {len(documents)}")

# ---------------------------------------------------
# Split documents
# ---------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)

if not docs:
    print("❌ No chunks created from documents.")
    exit()

print(f"✂️ Created {len(docs)} chunks")

# ---------------------------------------------------
# Embeddings
# ---------------------------------------------------
print("🧠 Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------------------------------------
# Create FAISS DB
# ---------------------------------------------------
print("⚡ Creating FAISS vector database...")

db = FAISS.from_documents(docs, embeddings)

# ---------------------------------------------------
# Save DB
# ---------------------------------------------------
db.save_local(DB_PATH)

print(f"\n✅ Vector DB created successfully at: {DB_PATH}")