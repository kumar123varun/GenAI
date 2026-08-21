from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
from bson import ObjectId

app = FastAPI(
    title="Student CRUD Operation",
    description="Simple CRUD operations using API and MongoDB",
    version="1.0"
)

# Establish connection with MongoDB
client = MongoClient("mongodb://localhost:27017/")

db = client["training_db"]
collection = db["students"]


@app.post("/students")
def create_student(student: dict):
    result = collection.insert_one(student)

    return {
        "message": "Student data has been created",
        "_id": str(result.inserted_id)
    }

@app.get("/students/get")
def get_students():
    students=[]
    for student in collection.find():
        student["_id"] = str(student["_id"])
        students.append(student)

    return(students)


@app.put("/students/{student_id}")
def update_student(student_id:str, student:dict):
    result = collection.update_one(
        {"_id":ObjectId(student_id)},
        {"$set":student }
    )

    return{
            "message": "Student details updated"
}

@app.delete("/students/{student_id}")
def delete_student(student_id:str):
    result = collection.delete_one(
      {"_id":ObjectId(student_id)}
    )
        
    return{
        "message": "Student data deleted"
    }
