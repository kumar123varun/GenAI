num1 = int(input("Enter the first number: "))
num2 = float(input("Enter the second number: "))        
operator = input("Enter the operator (+, -, *, /): ")

match operator:
    case "+":
        print("The result of the addition is:", num1 + num2)
    case "-":
        print("The result of the subtraction is:", num1 - num2)
    case "*":
        print("The result of the multiplication is:", num1 * num2)
    case "/":
        if num2 != 0:
            print("The result of the division is:", num1 / num2)
        else:
            print("Error: Division by zero is not allowed.")