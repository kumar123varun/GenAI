exp = int(input("Please enter your years of experience: "))

if exp <= 5:
    
    print("you are a junior consultant.")

elif exp > 5 and exp < 10:
    print("you are a consultant.")

elif exp >= 10 and exp < 15:
    print("you are a senior consultant.")

else :

    print("you are a Manager.")