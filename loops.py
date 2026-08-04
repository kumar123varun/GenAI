# #for loop
# for i in range(5):
#     print("Hello World")

# # for i in range(1, 6):
# #    print(i)

# for 0 in range(5):
#     print(0)

# for 1 in range(5):
#     print(1)

# for 2 in range(5):
#     print(2)    

# for 3 in range(5):
#     print(3)

# for 4 in range(5):
#     print(4)


# for i in range(12, 121, 12):
#     print(i)


# #while loop
# i = 1
# while i <= 5:
#     print("Hello World")
#     i = i + 1

# nested loop

# for i in range(3):
#     for j in range(2):
#         print(i, j)


pwd = input("Enter your password: ")
while pwd != "python123":
    print("Incorrect password. Please try again.")
    pwd = input("Enter your password: ")
print("Password correct. Access granted.")