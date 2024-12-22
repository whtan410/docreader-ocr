from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from fastapi import FastAPI, Query, HTTPException
from typing import Optional
from pydantic import BaseModel, Field

import os
import uvicorn
import fastapi

from dotenv import load_dotenv
load_dotenv()

class QueryRequest(BaseModel):
    query: str = Field(description="The query to ask the AI")

class QueryResponse(BaseModel):
    status: str = Field(description="Response status")
    question: str = Field(description="The question asked by the user")
    answer: str = Field(description="AI generated answer")

class ErrorResponse(BaseModel):
    status: str = Field(description="Response status")
    error: str = Field(description="Error message")

app = FastAPI(
    title="Document QA System",
    description="API for querying PDF documents using RAG",
    version="1.0.0"
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
llm_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro",api_key=GEMINI_API_KEY)
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

@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    try:
        retriever = vector_store.as_retriever()
        chain = RetrievalQA.from_chain_type( 
            llm = llm_model,
            retriever = retriever, 
            chain_type = "stuff"
        )
        response = chain.invoke(request.query)
        return QueryResponse(question=request.query, answer=response["result"], status="success")
    except Exception as e:
       raise HTTPException(
           status_code=400,
           detail=str(e)
       )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)


