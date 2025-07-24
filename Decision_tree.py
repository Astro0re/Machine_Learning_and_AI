# Should you Save or Spend
def Save_or_Spend():
    balance= int(input("How Much do you have: "))
    want= int(input("How much do you want to save: "))
    remain = balance - want
    if remain > (10 % balance) :
        print("You are free to spend")
    else:
        print("Please Save!!!")


# Lagos Movement
def Lagos_Movement():
    print("So we want to go out?")
    print("In this Economy!!!")
    print("Alright, Let's see if we can.")
    print("What day are we heading out?")
    
    weekday= 0
    weekend= 0
    day = input("Enter the day of the week? (monday-sunday): ")
    if day.lower() in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
        print("We can go out on a weekday, but let's be cautious.")
        weekday = 1
    elif day.lower() in ["saturday", "sunday"]:
        print("Great! We can go out on the weekend.")
        weekend = 1
    else:
        print("No matching scenario found. Please check your inputs.")
        print(day)
    
    weekday_calm= 0
    weekend_fun= 0
    day_time= input("What time of the day are we leaving? (morning/noon/night): ")
    if weekday == 1 and day_time.lower() in ["morning", "noon"]:
        print("We probably should not go out")
    elif weekday == 1 and day_time.lower() in ["night"]:
        print("Ok we can go but we have to be back on time")
        weekday_calm= 1
    elif weekend == 1 and day_time.lower() in ["morning", "noon", "night"]:
        print("We can go out and have fun")
        weekend_fun = 1
    else:
        print("No matching scenario found. Please check your inputs.")
        print(day_time)

    indoor= 0
    outdoor= 0    
    location=input("Is it an indoor or outdoor event? (indoor/outdoor): ").strip().lower()
    if day_time == "indoor":
        indoor= 1
    elif day_time == "outdoor":
        outdoor= 1
    else:
        print("No matching scenario found. Please check your inputs.")
        print(location)
    
    weather=input("How is the weather looking? (clear/raining)").strip().lower()
    if weekday_calm == 1 and indoor == 1 &weather == "clear":
        print("Alright have fun, and be careful in the dark")
    elif weekday_calm == 1 and outdoor == 1 &weather == "clear":
        print("Alright have fun, and be careful in the dark")
    elif weekend_fun == 1 and indoor == 1 &weather == "clear":
        print("Alright have fun")
    elif weekend_fun == 1 and outdoor == 1 &weather == "clear":
        print("Alright have fun, do not forget your shades")
    elif weekday_calm == 1 and indoor == 1 &weather == "raining":
        print("Alright have fun, take an umbreall with you")
    elif weekday_calm == 1 and outdoor == 1 &weather == "raining":
        print("Yeah just stay indors for this one")
    elif weekend_fun == 1 and indoor == 1 &weather == "raining":
        print("Alright have fun")
    elif weekend_fun == 1 and outdoor == 1 &weather == "raining":
        print("Yeah just stay indors for this one")
    else:
        print("No matching scenario found. Please check your inputs.")
        print(weather)    

Lagos_Movement()
