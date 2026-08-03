# color = input("Enter the color of the traffic signal (red, yellow, green): ")

# if color == "red":
#     print("Stop! The traffic signal is red.")
# elif color == "yellow":
#     print("Caution! The traffic signal is yellow.")
# elif color == "green":
#     print("Go! The traffic signal is green.")
# else:
#     print("Invalid color entered.")

traffic_signal = input ("Enter the color of the traffic signal (red, yellow, green): ")

match traffic_signal:
    case "red":
        print("Stop! The traffic signal is red.")
    case "yellow":
        print("Caution! The traffic signal is yellow.")
    case "green":
        print("Go! The traffic signal is green.")
    case _:
        print("Invalid color entered.") 