Salary = float(input("Enter your salary: "))

if Salary < 5000:
    print("You are in the low salary bracket.")
elif Salary >= 5000 and Salary < 10000:
    print("You are in the middle salary bracket.")
else:
    print("You are in the high salary bracket.")
    