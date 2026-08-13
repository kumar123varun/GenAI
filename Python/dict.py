#dictionary stores values in the form of key-value pair inside a flower bracket
# keys are always unique and whereas values can be unique

#Properties
#mutable
#Keys are unique
#Values can be duplicate
# you will use key instead of indexes

student = {
    "Name" : ["Varun", "kartik", "shruthi"],
    "Age" : "33",
    "Location": "Bengaluru"
}

print (student["Name"][2])

student.update({"dob":"01031993"})
print(student)
student.pop("Location")
print(student)