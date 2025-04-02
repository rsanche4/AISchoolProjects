import matplotlib.pyplot as plt

# Uncomment the variables to see the scenarios. I also wrote a report that summirzes my findings. I uploaded it here: https://docs.google.com/document/d/1KSbqro3AQn9ezXdkR_4lBAu2f1Dy9hfuAFBcmcNUFxo/edit?usp=sharing

# Scenarios
# 1: Balanced Consumption
# The world's birthrates remain above replacement level (in our world that simply means above death rate), and resource replenishment is the same as consumption. So everybody is sort of mindful of what they consume. In this outcome, society actually first starts at the amount given, and then it rises at a steady rate, given resources are used and recycled effectively. Birthrates remain at a steady pace, and then of course you reach a point where you cant output more resources as you might have reached the world limit, in which case population evens out and remains constant forever. Note: as more people are being born, although we might think resources will go down faster, this is not the case as in our model, everybody outputs the same as they consume. Essentially, in that case your resources never rise, and never fall below the initial amount.
# Variables for this simulation:
# years = 100 
# population = 8000000000      
# resources = 9000000000     
# max_resources = 10000000000
# birth_rate = 18.5/1000    
# death_rate = 56000000/population     
# resource_output_per_person = 0.5 
# resource_consumption_per_person = 0.5

# 2: Limited Prosperity
# In this world, we use the same as in world A, but now everybody consumes just slightly more than they produce. So its a society of small consumers (but not that much, just a small amount), but enough to offset the system and not allow for equilibrium. Again, birthrates remain above the death rate, (or above replacement), in which case, society eventually within a century reaches record lows. This makes sense, because you can't keep up with replacement, so your population slowly vanishes. Notice that resources also drop constantly since the beginning, and this will starve even more people because there is no resources to go around, and that means less work output, which reduces this even further. Notice also that we start with more people being born. This makes sense! Because there is more resource for everybody at first, so society reaches a small period of prosperity where we havent seen the full depletion yet. Then, we reach the peak, and then we go back down. Thats why we see a rise at first, and then a steady decline. Also, birthrates being bigger doesnt contribute because more people means more mouths to feed, but no one is working at a pace higher than what they consume, so this means resource depletion. Notice, making the differences in these numbers bigger doesnt change the outcome, just speeds up the collapse.
# Variables for this simulation:
# years = 100 
# population = 8000000000      
# resources = 9000000000     
# max_resources = 10000000000
# birth_rate = 18.5/1000    
# death_rate = 56000000/population     
# resource_output_per_person = 0.25 
# resource_consumption_per_person = 0.3

# 3: Inevitable Collapse
# In this world, we are getting a bit closer to our world. In which case, birthrates basically drop below replacement, and society consumes more than they produce, or doesnt figure out how to effectively use resources and prevent their complete depletion over time. So they dont invent some way of fixing this. The outcome is catastrophic. In our experiment, when the moment of "lower replacement birthrates" hits, society actually doesnt see much a difference for a while. It is once it reaches a certain point in time where you just dont have enough workers to replace the ones dying with, your resources are no longer being produced at the rate to feed everyone, so 2 things lead to the collapse immidiately after: falling birthrates and falling resources. Both equally contributing to the spiral downwards of each other. And there is a very sharp fall after that. Now couple of notes for this: You can play around with the resource output and consumption rates, and the result is always the same. It is irrelevant whether a society is productive or not, once birthrates fall, it leads to its collapse.
# Variables for this simulation:
years = 100 
population = 8000000000      
resources = 9000000000     
max_resources = 10000000000
birth_rate = 6/1000    
death_rate = 56000000/population     
resource_output_per_person = 0.25 
resource_consumption_per_person = 0.3

# Arrays to store results
time = []
population_history = []
resources_history = []

# Simulation loop
t = 0
while True:
    if t>years:
        break
    
    # Update population
    deaths = population * death_rate
    births = population * birth_rate
    population += (births - deaths)
    
    # Now after that, we must also remove any person that was not able to consume a resource cuz they need that to live otherwise they die
    if resource_consumption_per_person* population > resources:
        # this means that the amount we need to feed everyone isn't there, thus we need to kill off anyone who couldnt feed
        sustainable_population = resources / resource_consumption_per_person
        starvation_deaths = population - sustainable_population
        population -= starvation_deaths

    population = int(population)
    if population <= 0 :
        population=0
        break # all people died, no chance of recovery, done with the sim
    
    # Update resources (finite, deplete over time, and with more people more resources need to be used)
    resources -= resource_consumption_per_person* population
    if resources < 0:
        resources = 0  # No negative resources
    
    # and each person contributes in increasing the resources available by working thus contributing to the system
    resources += resource_output_per_person * population
    
    if resources > max_resources:
        resources = max_resources # cap at a point, the limit of Earth
    
    resources = int(resources)
    # Store results
    time.append(t)
    population_history.append(population)
    resources_history.append(resources)

    t+=1

# Create 3 subplots (vertically stacked)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

# Plot Population (Billions)
ax1.plot(time, population_history, 'b-', label="Population")
ax1.set_ylabel("Population")
ax1.grid()
ax1.legend()

# Plot Resources (Trillions)
ax2.plot(time, resources_history, 'r-', label="Resources")
ax2.set_ylabel("Resources")
ax2.grid()
ax2.legend()

plt.suptitle("Simplified World Dynamics Model")
plt.tight_layout()
plt.show()