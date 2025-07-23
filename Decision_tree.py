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
    day = input("Enter the day of the week? (monday-sunday): ")
    if day.lower() in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
        print("We can go out on a weekday, but let's be cautious.")
        weekday = +1
    if day.lower() in ["saturday", "sunday"]:
        print("Great! We can go out on the weekend.")
        weekend = +1
    day_time= input("What time of the day are we leaving? (morning/noon/night): ")
    if weekday == 1 & day_time.lower() in ["morning", "noon"]:
        print("We probably should not go out")
    elif weekday == 1 & day_time.lower() in ["night"]:
        print("Ok we can go but we have to be back on time")
        weekday_calm= +1
    elif weekend == 1 & day_time.lower() in ["morning", "noon", "night"]:
        print("We can go out and have fun")
        weekend_fun = +1
    input("Is it an indoor or outdoor event? (indoor/outdoor): ")
    if day_time.lower() == "indoor":
        indoor= +1
    elif day_time.lower() == "outdoor":
        outdoor= +1
    weather=input("How is the weather looking? (clear/raining)")
    if weekday_calm == 1 & indoor == 1 &weather.lower() in ["clear"]:
        print("Alright have fun, and be careful in the dark")
    elif weekday_calm == 1 & outdoor == 1 &weather.lower() in ["clear"]:
        print("Alright have fun, and be careful in the dark")
    elif weekend_fun == 1 & indoor == 1 &weather.lower() in ["clear"]:
        print("Alright have fun")
    elif weekend_fun == 1 & outdoor == 1 &weather.lower() in ["clear"]:
        print("Alright have fun, do not forget your shades")
    elif weekday_calm == 1 & indoor == 1 &weather.lower() in ["raining"]:
        print("Alright have fun, take an umbreall with you")
    elif weekday_calm == 1 & outdoor == 1 &weather.lower() in ["raining"]:
        print("Yeah just stay indors for this one")
    elif weekend_fun == 1 & indoor == 1 &weather.lower() in ["raining"]:
        print("Alright have fun")
    elif weekend_fun == 1 & outdoor == 1 &weather.lower() in ["raining"]:
        print("Yeah just stay indors for this one")    
    