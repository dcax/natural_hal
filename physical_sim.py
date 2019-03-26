
from scipy.constants import G
import numpy as np

#This is where the bulk of the physical projection code will be. 

epsilon = 1e-4 #This is the softness paramater, which is used to avoid singularities in the potential
dt = 1e-3 #Time fineness paramater
number_particles = 3

state_particle_length = 1 + 3 + 3 #length one particle takes in the state vector that describes the system
state_len = number_particles * state_particle_length

mean_mass   = 1000.
mass_spread = 15.
mean_x   = 0.0
x_spread = 2.
mean_y   = 0.0
y_spread = 2.
mean_z   = 0.0
z_spread = 2.
mean_v_x   = 0.0
v_x_spread = 100.0
mean_v_y   = 0.0
v_y_spread = 100.0
mean_v_z   = 0.0
v_z_spread = 100.0

def mk_particle():
    #returns a random particle state according to some specification

    return np.random.normal(loc=[mean_mass,mean_x,mean_y,mean_z,mean_v_x,mean_v_y,mean_v_z],
        scale=[mass_spread,x_spread,y_spread,z_spread,v_x_spread,v_y_spread,v_z_spread,])

def mk_rand_state():
    result_state = np.zeros(state_len) #system state to return
    for i in range(number_particles):
        #state has structure: [m1 x1 ... v_x1 ... particle2 ...]
        #get particle
        p = mk_particle()
        while p[0] >= 0.:
            p = mk_particle() #ensures mass is positive
        result_state[i*number_particles:(i+1)*number_particles] = p #set slice of array
    return result_state



