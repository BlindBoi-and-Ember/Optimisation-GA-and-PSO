import numpy as np
import random as rnd
from optproblems import *
from optproblems import cec2005 as cec2005

#Assesses the Fitness of a particular solution within the population with regards to the problem being optimized
#This acts as a measure for evaluating different particles, to adjust the velocity of less optimal particles
def assessFitness(position, problem):
    solution = Individual(position)
    problem.evaluate(solution)
    fitness = solution.objective_values
    return fitness

#Problem Definitions
problem = Problem(cec2005.F4(problem_length)) #this is the Function being optimized
problem_length = 10 #number of input variables for the function defining the problem
range_min = -100 #Minimum value allowed on each axis as defined by the problem
range_max = 100 #Maximum value allowed on each axis as defined by the problem

#Hyperperameters
informant_number = 5 #defines the number of other particles each particle evaluates, when adjusting its velocity
swarm_size = 100 #Number of particles making up the swarm
generations = 250 #TODO: good comment for generations.


time_step = 0.5 #scales the change in position for each particle for each generation.
vel_scalar = 0.9 #weights the contribution the previous velocity contributes to the new velocity, for each particle
personal_scalar = 0.1 #weights the contribution of the previous personal best to the new velocity, for each particle
informant_scalar = 0.1 #weights the contribution of the best informant particle to the new velocity, for each particle
global_scalar = 0.1 #weights the contribution of the global best particle to the new velocity, for each particle

#stores stuff
particle_swarm_pos = list() #stores the current position of each particle in the swarm
particle_swarm_vel = list() #stores the current velocity of each particle in the swarm
particle_swarm_informants = list() #stores a list of informant indexs for each particle in the swarm
particle_swarm_bests = list() #stores the best position each particle in the swarm has visited
particle_swarm_best_fitness = list() #stores the current best fitness value each particle has been evaluated to

#Creates "swarm size" number of particles
#the information on each "particle" is distributed across the list, referenced by a common index 
for idx_1 in range(swarm_size):
    #TODO David to show Aidan readible rand array function
    particle_pos = ((np.random.rand(problem_length))*200-100).tolist() #creates a list of length "problem_length" with postion values between -100 and 100
    particle_vel = ((np.random.rand(problem_length))*20-10).tolist() #creates a list of length "problem_length" with velocity values between -100 and 100
    #particle_vel = ((np.random.rand(problem_length))*2-1).tolist()
    
    particle_swarm_pos.append(particle_pos) #TODO: appropriate comment for this
    particle_swarm_bests.append(particle_pos) #best swarm positon at t=0 is initial position
    particle_swarm_vel.append(particle_vel) #TODO: appropriate comment for this
    particle_swarm_best_fitness.append(np.inf) #particle is yet to be evaluated so best fitness is initialised to infinity

    #creates a list of "informant number" amount of random informants for each particle
    informants = list()
    for idx_2 in range(informant_number):
        informants.append(rnd.randint(0,(swarm_size-1)))
    #end
    
    particle_swarm_informants.append(informants) #assigns the informant list to each particle
#end



best_particle_idx = 0
best_particle_fitness = np.inf

for gen_idx in range(generations):
    print("generation: ",gen_idx+1)
    for particle_idx in range(swarm_size):
        particle_pos = particle_swarm_pos[particle_idx]
        particle_fitness = assessFitness(particle_pos,problem)
        particle_historic_best_fitness = particle_swarm_best_fitness[particle_idx]
        if (particle_fitness < particle_historic_best_fitness):
            particle_swarm_bests[particle_idx] = particle_pos
            particle_swarm_best_fitness[particle_idx] = particle_fitness
        #end
        if (particle_fitness < best_particle_fitness):
            best_particle_fitness = particle_fitness
            best_particle_idx = particle_idx
        #end

    #end

    for particle_idx in range(swarm_size):

        best_informant = 0
        best_informant_fitness = np.inf
        particle_informants = particle_swarm_informants[particle_idx]
        for informants_idx in range(len(particle_swarm_informants[particle_idx])):
            informant = particle_informants[informants_idx]
            informant_best_pos = particle_swarm_bests[informant]
            informant_fitness = assessFitness(informant_best_pos,problem)
            if (informant_fitness < best_informant_fitness):
                best_informant_fitness = informant_fitness
                best_informant = informant
            #end
        #end
    
        retained_velocity_comp = vel_scalar*np.array(particle_swarm_vel[particle_idx])
        personal_best_velocity_comp = personal_scalar*(np.array(particle_swarm_bests[particle_idx]) - np.array(particle_swarm_pos[particle_idx]))
        informant_best_velocity_comp = informant_scalar*(np.array(particle_swarm_bests[best_informant]) - np.array(particle_swarm_pos[particle_idx]))
        global_best_velocity_comp = global_scalar*(np.array(particle_swarm_bests[best_particle_idx]) - np.array(particle_swarm_pos[particle_idx]))

        velocity = (retained_velocity_comp + personal_best_velocity_comp + informant_best_velocity_comp + global_best_velocity_comp).tolist()
        particle_swarm_vel[particle_idx] = velocity

        #if(particle_idx == swarm_size-1):
            #print("evaluated and updated all particles")
    print("best fitness idx: ", best_particle_idx)
    print("best fitness: ", best_particle_fitness)
    print("generation: ",gen_idx+1, "Fitness: ",best_particle_fitness)

    for particle_idx in range(swarm_size):
        new_pos = np.array(particle_swarm_pos[particle_idx]) + time_step*np.array(particle_swarm_vel[particle_idx])
        new_pos = np.clip(new_pos,range_min,range_max)
        particle_swarm_pos[particle_idx] = new_pos.tolist()
    #end

print("Best Fitness: ", best_particle_fitness, " at index: ", best_particle_idx)
print(particle_swarm_best_fitness)