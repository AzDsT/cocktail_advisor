from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from langchain_openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import openai
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app = FastAPI()

# Load the dataset
df = pd.read_csv("data/cocktails.csv")  

# Initialize OpenAI LLM
llm = OpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    openai_api_key="api-key"  
)

# Initialize Pinecone
pc = Pinecone(api_key="api-key")  
index_name = "cocktails"

# Create or connect to the Pinecone index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  
        metric="cosine", 
        spec=ServerlessSpec(
            cloud="aws",  #AWS as the cloud provider
            region="us-west-2"  # Specify the region
        )
    )

# Connect to the index
index = pc.Index(index_name)

# Function to generate embeddings using OpenAI
def generate_embedding(text: str):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

# Preprocess the dataset and store embeddings in Pinecone
def store_cocktail_embeddings():
    for idx, row in df.iterrows():
        cocktail_name = row['name']
        cocktail_ingredients = row['ingredients']
        
        # Generate embedding for the cocktail
        embedding = generate_embedding(cocktail_ingredients)
        
        # Store the embedding in Pinecone
        index.upsert([(cocktail_name, embedding)])

# Call this function once to store embeddings (e.g., during initialization)
store_cocktail_embeddings()

class UserQuery(BaseModel):
    question: str

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Root endpoint to serve the chat interface
@app.get("/")
async def root():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

# Ask endpoint
@app.post("/ask")
async def ask(query: UserQuery):
    question = query.question.lower()  

    
    if "lemon" in question:
        # Fetch cocktails containing lemon from the dataset
        lemon_cocktails = df[df['ingredients'].str.contains('lemon', case=False, na=False)].head(5)
        response = lemon_cocktails.to_dict(orient='records')
    elif "non-alcoholic" in question and "sugar" in question:
        # Fetch non-alcoholic cocktails containing sugar
        non_alcoholic_sugar_cocktails = df[(df['alcoholic'] == False) & (df['ingredients'].str.contains('sugar', case=False, na=False))].head(5)
        response = non_alcoholic_sugar_cocktails.to_dict(orient='records')
    else:
        # Use RAG for other questions
        try:
            # Generate embedding for the question
            question_embedding = generate_embedding(question)

            # Query Pinecone for similar cocktails
            results = index.query(question_embedding, top_k=5, include_metadata=True)

            # Extract relevant cocktail names from Pinecone results
            similar_cocktails = [match['id'] for match in results['matches']]

            # Fetch details of similar cocktails from the dataset
            similar_cocktails_details = df[df['name'].isin(similar_cocktails)].to_dict(orient='records')

            # Use OpenAI to generate a response based on retrieved data
            prompt = f"Answer this question about cocktails: {question}. Relevant cocktails: {similar_cocktails_details}"
            response = llm(prompt)
        except Exception as e:
            # Handle errors
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    return {"response": response}
































