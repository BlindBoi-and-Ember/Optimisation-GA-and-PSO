import numpy as np
import random as rnd
from optproblems import *
from optproblems import cec2005 as cec2005


def assessFitness(position, problem):
    solution = Individual(position)
    problem.evaluate(solution)
    fitness = solution.objective_values
    return fitness

informant_number = 5
problem_length = 10
range_min = -100
range_max = 100
swarm_size = 100
generations = 250

time_step = 0.5
vel_scalar = 0.9
personal_scalar = 0.1
informant_scalar = 0.1
global_scalar = 0.1

particle_swarm_pos = list()
particle_swarm_vel = list()
particle_swarm_informants = list()
particle_swarm_bests = list()
particle_swarm_best_fitness = list()

problem = Problem(cec2005.F4(problem_length))

for idx_1 in range(swarm_size):
    particle_pos = ((np.random.rand(problem_length))*200-100).tolist()
    particle_vel = ((np.random.rand(problem_length))*20-10).tolist()
    #particle_vel = ((np.random.rand(problem_length))*2-1).tolist()
    
    particle_swarm_bests.append(particle_pos)
    particle_swarm_pos.append(particle_pos)
    particle_swarm_vel.append(particle_vel)
    particle_swarm_best_fitness.append(np.inf)

    informants = list()
    for idx_2 in range(informant_number):
        informants.append(rnd.randint(0,(swarm_size-1)))
    #end

    particle_swarm_informants.append(informants)
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