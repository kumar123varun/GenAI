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

@app.get("/orders/get")
def get_orders():
    orders=[]
    for order in collection.find():
        order["_id"] = str(order["_id"])
        orders.append(order)

    return(orders)


@app.put("/orders/{order_id}")
def update_order(order_id:str, order:dict):
    result = collection.update_one(
        {"_id":ObjectId(order_id)},
        {"$set":order }
    )

    return{
            "message": "Ecommerce Order details updated"
}

@app.delete("/orders/{order_id}")
def delete_order(order_id:str):
    result = collection.delete_one(
      {"_id":ObjectId(order_id)}
    )
        
    return{
        "message": "Ecommerce order is deleted"
    }
