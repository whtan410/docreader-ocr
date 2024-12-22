import os
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
embeddings = GoogleGenerativeAIEmbeddings (model="models/embedding-001", google_api_key=GEMINI_API_KEY)
client = MongoClient(os.getenv("MONGODB_ATLAS_CLUSTER_URI"))

#configure db and collection
DB_NAME = "creditcard_db"
COLLECTION_NAME = "creditcard_collection"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"
MONGODB_COLLECTION = client [DB_NAME][COLLECTION_NAME]

vector_store = MongoDBAtlasVectorSearch(
    collection=MONGODB_COLLECTION,
    embedding=embeddings,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    relevance_score_fn="cosine",
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 100
)

#load pdf
pdf_files = [
    "C:/Users/pa662/Documents/GitHub/docreader-ocr/assets/M2U Bill 0272 Mar 2024.pdf",
    "C:/Users/pa662/Documents/GitHub/docreader-ocr/assets/M2U Bill 0272 Apr 2024.pdf",
    "C:/Users/pa662/Documents/GitHub/docreader-ocr/assets/M2U Bill 0272 May 2024.pdf",
    "C:/Users/pa662/Documents/GitHub/docreader-ocr/assets/M2U Bill 0272 Jun 2024.pdf",
    "C:/Users/pa662/Documents/GitHub/docreader-ocr/assets/M2U Bill 0272 Jul 2024.pdf",
    "C:/Users/pa662/Documents/GitHub/docreader-ocr/assets/M2U Bill 0272 Aug 2024.pdf",
    "C:/Users/pa662/Documents/GitHub/docreader-ocr/assets/M2U Bill 0272 Sep 2024.pdf",
    "C:/Users/pa662/Documents/GitHub/docreader-ocr/assets/M2U Bill 0272 Oct 2024.pdf",
    "C:/Users/pa662/Documents/GitHub/docreader-ocr/assets/M2U Bill 0272 Nov 2024.pdf",
]

#process each pdf
for pdf_file in pdf_files:
    try: 
        print(f"Processing {pdf_file}")

        #Load and split documents
        loader = PyPDFLoader(pdf_file)
        docs = loader.load_and_split(text_splitter)

        # add documents to vector store
        vector_store.add_documents(docs)
        print(f"Processed {pdf_file} and added {len(docs)} documents to vector store")

    except Exception as e:
        print(f"Error processing {pdf_file}: {e}")
        continue

print("All PDFs processed and added to vector store")
client.close()