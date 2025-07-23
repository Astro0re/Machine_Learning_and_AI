# Should you Save or Spend
def Save_or_Spend(__init__):
    balance= input("How Much do you have: ")
    want= input("How much do you want to save: ")
    if balance % want == 10:
        print("You are free to save")
    else:
        print("Please Save!!!")


# Lagos Movement
def Lagos_Movement(__init__):
    print("So we want to go out?")
    print("In this Economy!!!")
    print("Alright, Let's see if we can.")
    print("What day are we heading out?")
    day = input("Enter the day of the week: ")
    if day.lower() in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
        print("We can go out on a weekday, but let's be cautious.")
        weekday = +1
    if day.lower() in ["saturday", "sunday"]:
        print("Great! We can go out on the weekend.")
        weekend =+1
    day_time= input("What time of the day are we leaving?")
    if weekday == 1 & day_time.lower() in ["morning", "noon"]:
        print("We probably should not go out")
    elif weekday == 1 & day_time.lower() in ["night"]:
        print("Ok we can go but we have to be back on time")