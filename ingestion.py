from qdrant_client import QdrantClient, models
from jproperties import Properties
import asyncio
from langchain_community.document_loaders import PyPDFLoader
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from ollama import Client as OllamaClient
from langchain_ollama import OllamaEmbeddings
import uuid


configs = Properties()
with open('app_config.properties', 'rb') as config_file:
    configs.load(config_file)

qdrant_client = QdrantClient(
    url=configs.get("QDRANT_URL").data,
    api_key=configs.get("QDRANT_API_KEY").data
)

# Delete the collection if it exists
try:
    qdrant_client.delete_collection(collection_name=configs.get("QDRANT_COLLECTION_NAME").data)
    print("Collection deleted successfully.")
except Exception as e:
    if "not found" in str(e):
        print("Collection does not exist, skipping deletion.")
    else:
        print(f"Error deleting collection: {e}")

# Create a new collection
try:
    qdrant_client.create_collection(
        collection_name=configs.get("QDRANT_COLLECTION_NAME").data,
        vectors_config=models.VectorParams(
            size=4096,  # Ollama embeddings has 4096 dimensions
            distance=models.Distance.COSINE,
            # on_disk_payload=True,
            # shard_number=9,
        ),
    )
    print("Collection created successfully.")
except Exception as e:
    if "already exists" in str(e):
        print("Collection already exists, skipping creation.")
    else:
        print(f"Error creating collection: {e}")

df = pd.read_csv(configs.get("FILE_PATH").data)

df.insert(len(df.columns),"Merge","")

for i in range(len(df)):
    print(i)
    merge = str(df['Title'][i])+ " " + str(df['Abstract'][i])+ " " +str(df['Content'][i])
    df.loc[i,'Merge'] = merge

doc_lst =  []
for item in range(len(df)):
    page = Document(page_content=df['Merge'][item],
                    metadata={
                        "Arvix_id": str(df['ArXiv ID'][item]),
                        "author": str(df['Authors'][item]),
                        "published_date": str(df['Published Date'][item]),
                        "link": str(df['PDF Link'][item]),
                    })
    doc_lst.append(page)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
texts = text_splitter.split_documents(doc_lst)
print (f'Now you have {len(texts)} documents')

# Initialize Ollama client
Ollama_Client = OllamaClient(host="http://localhost:11434")

def get_embedding(text, model="llama3"):
    return Ollama_Client.embeddings( prompt=text,model=model).embedding

collectionName = configs.get("QDRANT_COLLECTION_NAME").data

def process_embeddings(documents):
    """Process embeddings in batches"""
    batch_size = 50  # Adjust based on your requirements
    not_processed = 0
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        points = []
        
        for doc in batch:
            try:
                # Generate embedding
                embedding = get_embedding(doc.page_content)
    
                # Create a unique ID
                point_id = str(uuid.uuid4())
                # Create a PointStruct for Qdrant
                point = models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "paper_content": doc.page_content,
                        "arxiv_id": doc.metadata.get("Arvix_id", ""),
                        "author": doc.metadata.get("author", ""),
                        "published_date": doc.metadata.get("published_date", ""),
                        "link": doc.metadata.get("link", ""),
                        "pc": "Waseem",
                    }
                )
                points.append(point)
                
            except Exception as e:
                print(f"Error processing document: {e}")
                not_processed += 1
        
        # Upsert batch of points
        if points:
            qdrant_client.upsert(
                collection_name=collectionName,
                points=points
            )
            print(f"Inserted {len(points)} documents into Qdrant")
    
    return not_processed

# Process embeddings
not_app = process_embeddings(texts)
print(f"Number of documents not processed: {not_app}")