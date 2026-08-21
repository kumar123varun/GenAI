from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from bson import ObjectId

app = FastAPI(
    title="Student CRUD Operation",
    description="Simple CRUD API using FastAPI and Mongo",
    version="1.0"
)

# Establish connection with the db
client = MongoClient("mongodb://localhost:27017/")
db = client["training_db"]
collection = db["students"]

# 1. Define the student structure using Pydantic
class Student(BaseModel):
    name: str
    age: int
    course: str
    email: str

@app.post("/students")
def create_student(student: Student):
    # .model_dump() turns the Pydantic model into a dictionary for MongoDB
    result = collection.insert_one(student.model_dump())

    return {
        "message": "student data has been created",
        "_id": str(result.inserted_id)
    }

@app.get("/students/get")
def get_students():
    students = []
    for student in collection.find():
        student["_id"] = str(student["_id"])
        students.append(student)

    return students    

@app.put("/students/{student_id}")
def update_student(student_id: str, student: Student):
    result = collection.update_one(
        {"_id": ObjectId(student_id)},
        {"$set": student.model_dump()}
    )

    return {
        "message": "student updated"
    }

@app.delete("/students/{student_id}")
def delete_student(student_id: str):
    result = collection.delete_one(
       {"_id": ObjectId(student_id)}
    )
    return {
        "message": "Student data deleted"
    }

@app.patch("/students/{student_id}/field/{field_name}")
def delete_student_field(student_id: str, field_name: str):
    result = collection.update_one(
        {"_id": ObjectId(student_id)},
        {"$unset": {field_name: ""}}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Student not found")

    return {
        "message": f"Field '{field_name}' has been successfully deleted from the student document."
    }