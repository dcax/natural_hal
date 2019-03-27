

from scipy.constants import G
import numpy as np
import pandas as pd
import seaborn as sns #Seaborn allows better looking ploys
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
#ax = fig.add_subplot(110, projection='2d')
sns.set()

#This is where the bulk of the physical projection code will be. 



dt = 1.e-3 #Time fineness paramater

spatial_dim = 1
state_particle_length = 1 + spatial_dim + spatial_dim #length one particle takes in the state vector that describes the system
state_len = 1 * state_particle_length

mean_mass   = 1.
mass_spread = .2
mean_x   = 0.0
x_spread = 1.
z_spread = 20.
mean_v   = 0.0
v_spread = 1.0e-1

end_time = 1000.

spring_constant = 10.
spring_length = 1. #Should let these be paramaters to learn, but need to get data fast more

mean_k = spring_constant
k_spread = 2.

#This implements the actual model info for a simple harmonic oscillator
#This is useful because such as system can easily be studied
#System specified by k,m,x,v



def mk_random_oscillator_specification():
    #returns a random oscillator specification
    #oscillator is [k m x v]
    #position in origin of particle basis
    spec = np.random.normal(loc=[mean_k,mean_mass,mean_x,
        mean_v], spread=[k_spread,mass_spread,x_spread,v_spread])
    while spec[0] < 0.0 or spec[1] < 0.0 or spec[2] < 0.0:
        #checks that spec is realistic
        spec = np.random.normal(loc=[mean_k,mean_mass,mean_x,
        mean_v], spread=[k_spread,mass_spread,x_spread,v_spread])

    return spec
    

def plot_in_time(k,m,x0,v0, limit=end_time, time_intreval=1/end_time/1000):
    #plots a systems 1 dimensional history as a function of time
    assert spatial_dim == 1
    t = np.linspace(0,time_intreval,limit) #discretised time increments
    f_x = np.vectorize(lambda tau: x(tau,k,m,x0,v0))
    f_v = np.vectorize(lambda tau: v(tau,k,m,x0,v0))

    xs = f_x(t)
    vs = f_v(t)

    plt.scatter(t,xs)
    plt.scatter(t,vs)

    
    #print(history) #prints history to give full model parameters


    plt.title("Actual Trajectories")
    plt.xlabel("t")
    plt.ylabel("X and V")
    plt.show()

def x(t,k,m,x,v):
    #find x at later time given initial conditions
    w = np.sqrt(k/m) #angualr frequency
    return x*np.cos(w*t) + v/w*np.sin(w*t)
    
def v(t,k,m,x,v):
    #find v at later time given initial conditions
    w = np.sqrt(k/m) #angualr frequency
    return -x*w*np.sin(w*t) + v*np.cos(w*t)


