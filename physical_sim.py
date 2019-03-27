
from scipy.constants import G
import numpy as np
import pandas as pd
import seaborn as sns #Seaborn allows better looking ploys
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.add_subplot(110, projection='3d')
#sns.set()

#This is where the bulk of the physical projection code will be. 

epsilon = 1.e-8 #This is the softness paramater, which is used to avoid singularities in the potential
#Softness param not useful to spring potentials

dt = 1.e-3 #Time fineness paramater
number_particles = 2

spatial_dim = 1
state_particle_length = 1 + spatial_dim + spatial_dim #length one particle takes in the state vector that describes the system
state_len = number_particles * state_particle_length

mean_mass   = 1.
mass_spread = .2
mean_x   = 0.0
x_spread = 1.
mean_y   = 0.0
y_spread = 20.
mean_z   = 0.0
z_spread = 20.
mean_v_x   = 0.0
v_x_spread = 1.0e-1
mean_v_y   = 0.0
v_y_spread = 1.0e-6
mean_v_z   = 0.0
v_z_spread = 1.0e-6

spatial_dim = 1
end_time = 100.

spring_constant = 10.
spring_length = 1. #Should let these be paramaters to learn, but need to get data fast more

mean_k = spring_constant
k_spread = 2.


def mk_particle():
    #returns a random particle state according to some specification
    #particles have no charge and are point masses
    if spatial_dim == 1:
        return np.random.normal(loc=[mean_mass,mean_x,mean_v_x])
    elif spatial_dim == 3:
        return np.random.normal(loc=[mean_mass,mean_x,mean_y,mean_z,mean_v_x,mean_v_y,mean_v_z],
        scale=[mass_spread,x_spread,y_spread,z_spread,v_x_spread,v_y_spread,v_z_spread,])
    else:
        exit("Unpreped for this")

def mk_rand_state():
    result_state = np.zeros(state_len) #system state to return
    for i in range(number_particles):
        #state has structure: [m1 x1 ... v_x1 ... particle2 ...]
        #get particle
        p = mk_particle()
        #print("Made particle {}.".format(p))
        while p[0] < 0.:
            p = mk_particle() #ensures mass is positive

        result_state[i*state_particle_length: state_particle_length + i*state_particle_length] = p #set slice of array
    return result_state

def mk_oscillator_state():
    #Makes a random ish oscillator state for 2-body coupled oscillator
    assert spatial_dim == 1 and number_particles == 2



def get_gravitational_forces(state):
    forces = np.zeros(spatial_dim*number_particles)
    #Each spatial dimension gets a component of the forces for each particle
    #The forces are total and work of the softened gravitational force
    #F = Gm1m2/(r^2+eps^2)^3/2*r

    #should have (n-1)(n-2) force terms to compute total where n is num particles
    for subject in range(number_particles):
        for agent in range(subject+1, number_particles):
            #agent does force on subject
            #First [0] index at loc is the mass
            r = state[agent * state_particle_length + 1:
                agent * state_particle_length+1+spatial_dim] - state[subject * state_particle_length+1:subject * state_particle_length +1+spatial_dim]   #radial between particles
            f = G*state[subject*state_particle_length]*state[agent*state_particle_length] * r /np.power(np.dot(r,r) + epsilon**2,3/2)
            forces[subject:subject + spatial_dim] += f
            forces[agent:agent + spatial_dim]     += -f
            assert f is not np.nan

    return forces

def get_springs_force(state):
    #returns force on system where each body is attatched by means of springs
    forces = np.zeros(spatial_dim*number_particles)
    #Each spatial dimension gets a component of the forces for each particle
    #The forces are total and work of the hooke force

    #should have (n-1)(n-2) force terms to compute total where n is num particles
    for subject in range(number_particles):
        for agent in range(subject+1, number_particles):
            #agent does force on subject
            #First [0] index at loc is the mass
            r = state[agent * state_particle_length + 1:
                agent * state_particle_length+1+spatial_dim] - state[subject * state_particle_length+1:subject * state_particle_length +1+spatial_dim]   #radial between particles
            f = - spring_constant*(spring_length - np.linalg.norm(r))*r
            forces[subject:subject + spatial_dim] += f
            forces[agent:agent + spatial_dim]     += -f
            assert f is not np.nan

    return forces

def get_forces(state): 
    forces = np.zeros(spatial_dim*number_particles)

    #Does the forces on a system
    if False: #gravity off
        forces += get_gravitational_forces(state)
    if True:
        forces += get_springs_force(state)

    #print(forces)

    return forces


def state_advance(state):
    #Advances the state 1 leapfrog time step, returning the new state.
    #https://en.wikipedia.org/wiki/Leapfrog_integration#cite_note-Yoshida1990-6
    mutated_state = np.array(state, copy=True)
    #This uses the yoshida integrator method for fourth order accuracy
    #and conserved hamiltonian
    w_0 = - np.cbrt(2.)/(2-np.cbrt(2.)) #Auxillary values
    w_1 = 1/(2-np.cbrt(2.))
    c = np.array([w_1/2, (w_1 + w_0)/2, (w_1 + w_0)/2, w_1/2]) #Time step adjustment for each intreval
    d = np.array([w_1, w_0, w_1])
    
    for step in range(4): #Four steps in Yashida integrator
    
        for i in range(number_particles):
            v_0 = mutated_state[i*state_particle_length + 1 + spatial_dim: i*state_particle_length + 1 + spatial_dim + spatial_dim]
            
            #x_0 = state[i*state_particle_length + 1 : i*state_particle_length + 1 + spatial_dim]
            #Updates position of the particle in space, but only the virtual first component
            mutated_state[i*state_particle_length + 1: i*state_particle_length + 1 + spatial_dim] += c[step] * v_0 * dt
        
        if step == spatial_dim:
            break
            #fourth step does not change velocity further

        forces = get_forces(mutated_state) #Gets the gravitational forces on the current state

        for i in range(number_particles):
            #Now we update the velocity stored in the updated state
            m = mutated_state[i*state_particle_length] #Does not really matter whether we use original or updated for mass
            if (m == 0.0):
                continue #skip for nil masses (for padding purposes)
            f = forces[i*spatial_dim: spatial_dim + i*spatial_dim] #force on particle i
            mutated_state[i*state_particle_length + 1 + spatial_dim: i*state_particle_length + 1 + spatial_dim + spatial_dim] += d[step]*f/m*dt
    
    return mutated_state

def do_evolution(state, n=1):
    #does state evelotion of the system a number of time steps
    #returns series in time
    history = np.zeros((n, state_len))
    for instant in range(n-1):
        history[instant] = state
        state = state_advance(state)
    history[instant] = state #record history with no redundencies

    return history

def dep_plot_history(history, discretion=1):
    #plots a system's history
    #discretion controls how fine the points are taken, including whether to drop some
    #1 is do all
    #first we take every discretion value
    indices_to_keep = discretion*np.arange(history.shape[0]//discretion)   
    history = history[indices_to_keep,:]
    print(history.shape)
    for i in range(number_particles):
        x = history[:,i*state_particle_length+1:i*state_particle_length+1+spatial_dim] #Location of particle i
        print(x.shape)
        print(x[:,0])
        #plt.plot(x[:,0],x[:,1],x[:,2]) #plot the ith particle's trajectory
        c = np.linspace(0, 1/history.shape[0], 100) #color gradient
        ax.scatter(x[:,0],x[:,1],x[:,2],c=c)
    
    #print(history) #prints history to give full model parameters


    plt.title("Actual Trajectories")
    plt.xlabel("x")
    plt.ylabel("y")
    
    plt.show()

def mk_random_oscillator_specification():
    #returns a random oscillator specification
    #oscillator is [k m1 m2 x1 x2 v1 v2]
    #position in origin of particle basis
    spec = np.random.normal(loc=[mean_k,mean_mass,mean_mass,mean_x,mean_x,
        mean_v_x,mean_v_x], spread=[k_spread,mass_spread,mass_spread,x_spread,x_spread,v_x_spread,v_x_spread])
    while spec[0] < 0.0 or spec[1] < 0.0 or spec[2] < 0.0:
        #checks that spec is realistic
        spec = np.random.normal(loc=[mean_k,mean_mass,mean_mass,mean_x,mean_x,
        mean_v_x,mean_v_x], spread=[k_spread,mass_spread,mass_spread,x_spread,x_spread,v_x_spread,v_x_spread])

    return spec
    

def plot_in_time_old(system, limit=end_time, time_intreval=1/end_time/1000):
    #plots a systems 1 dimensional history as a function of time
    assert spatial_dim == 1
    t = np.linspace(0,time_intrevel,limit)

    x1 = np.apply_along_axis(lambda a: oscillator_position(a,system),0,t)
    x2 = np.apply_along_axis(lambda a: oscillator_position(a,system),0,t)

    plt.plot(t,x1)
    plt.plot(t,x2)

    
    #print(history) #prints history to give full model parameters


    plt.title("Actual Trajectories")
    plt.xlabel("t")
    plt.ylabel("x")
    
    plt.show()

def oscillator_position(t, system): 
    #project oscillator forward in position to time t.
    #oscillator is [k m1 m2 x1 x2 v1 v2]
    w = sqrt(system[0]*system) #observed angular frequency


