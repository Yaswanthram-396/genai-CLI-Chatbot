from pypdf import PdfReader
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb


model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"  )


def createChunks(text,chunk_size):
    chunks=[]
    start=0
    
    while(start<len(text)):
        chunk=text[start:start+chunk_size]

        chunks.append(chunk)
        start+=chunk_size
    
    return chunks


def chunkDocuements(documents,chunk_size):
    all_chunks=[]
    for doc in documents:
        text=doc["text"]
        chunks=createChunks(text,chunk_size)
        for index,chunk in enumerate(chunks):
            all_chunks.append({
                "text":chunk,
                "source":doc["source"],
                "page":doc["page"],
                "chunk_index":index

            })

    return all_chunks




pdf_path=Path("documents").glob("*.pdf")
documents=[]


for pdf in pdf_path:
    reader = PdfReader(str(pdf))

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            documents.append({
                "text": text,
                "source": str(pdf),
                "page": page_number
            })

smallchunks=chunkDocuements(documents,500)
largechunks=chunkDocuements(documents,1000)


texts = [chunk["text"] for chunk in largechunks]

embeddings = model.encode(texts)

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="documents_1000"
)

collection.add(
    ids=[f"chunk_{i}" for i in range(len(largechunks))],
    documents=texts,
    embeddings=embeddings.tolist(),
    metadatas=[
        {
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk_index": chunk["chunk_index"],
            "chunk_size": 1000
        }
        for chunk in largechunks
    ]
)

print(collection.count())

query = "What is the employee leave policy?"

query_embedding = model.encode(query)

print(len(query_embedding))
results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=3
)

print(results["documents"][0])