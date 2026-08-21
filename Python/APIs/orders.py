from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
from bson import ObjectId

app = FastAPI(
    title="Ecommerce Order Management",
    description="Simple CRUD operations using API and MongoDB",
    version="1.0"
)

# Establish connection with MongoDB
client = MongoClient("mongodb://localhost:27017/")

db = client["training_db"]
collection = db["orders"]


@app.post("/orders")
def create_order(order: dict):
    result = collection.insert_one(order)

    return {
        "message": "Ecommerce Order data has been created",
        "_id": str(result.inserted_id)
    }

