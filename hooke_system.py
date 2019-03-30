

from scipy.constants import G
import numpy as np
import pandas as pd
import seaborn as sns #Seaborn allows better looking ploys
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
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

#mean_omega   =
#omega_spread = 
time_max = 5.0E1 #Max time for uniform distribution
#period_max = 1.0E3 #Similar for period (related to omega)
#period_min = .01 #to Avoid singularities
omega_max = 1.5 #would prefer larger but need focus on small

mean_x   = 3.0 #not really a mean, since it's bimodal
x_spread = 3. #just uses spread now

mean_v   = 3.0
v_spread = 2.

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
    f_x = np.vectorize(lambda tau: x_old(tau,k,m,x0,v0))
    f_v = np.vectorize(lambda tau: v_old(tau,k,m,x0,v0))

    xs = f_x(t)
    vs = f_v(t)

    plt.scatter(t,xs)
    plt.scatter(t,vs)

    
    #print(history) #prints history to give full model parameters


    plt.title("Actual Trajectories")
    plt.xlabel("t")
    plt.ylabel("X and V")
    plt.show()

def x_old(t,k,m,x,v):
    #find x at later time given initial conditions
    w = np.sqrt(k/m) #angualr frequency
    return x*np.cos(w*t) + v/w*np.sin(w*t)
    
def v_old(t,k,m,x,v):
    #find v at later time given initial conditions
    w = np.sqrt(k/m) #angualr frequency
    return -x*w*np.sin(w*t) + v*np.cos(w*t)

def x(t,w,x,v):
    #find x at later time given initial conditions
    #w = np.sqrt(k/m) #angualr frequency
    return x*np.cos(w*t) + v/w*np.sin(w*t)
    
def v(t,w,x,v):
    #find v at later time given initial conditions
    #w = np.sqrt(k/m) #angualr frequency
    return -x*w*np.sin(w*t) + v*np.cos(w*t)

def project(p):
    #projects a particle to the future
    return np.array([x(p[0],p[1],p[2],p[3]),v(p[0],p[1],p[2],p[3])])

mk_time = lambda: np.random.random()*time_max
mk_omega = lambda: np.random.random()*omega_max #avoids 0
def mk_x():
    #bimodal x distribution
    """if np.random.random() > .5:
        return np.random.normal(mean_x,x_spread)
    else:
        return np.random.normal(-mean_x,x_spread)
    """
    return np.random.random()*(2*x_spread) - x_spread


def mk_v():
    #bimodal x distribution

    """
    if np.random.random() > .5:
        return np.random.normal(mean_v,v_spread)
    else:
        return np.random.normal(-mean_v,v_spread)
        """ #Old attempt
    return np.random.random()*(2*v_spread) - v_spread



def mk_potential_point():
    #returns particles in some params in (length,NUM_INPUTS)
    #points = np.zeros((length,4))
    t = mk_time()
    w = mk_omega()
    x = mk_x()
    v = mk_v()
    assert t >= 0. and w > 0.
    return np.array([t,w,x,v,t*w])

def mk_potential_points(length):
    #returns particles in some params in (length,NUM_INPUTS)
    points = np.zeros((length,5))
    for i in range(length):
        points[i,:] = mk_potential_point()
    
    return points

def get_hooke_data(length=1):
    #returns fied hooke' physical modelling

    x = mk_potential_points(length)

    #Now we use x to find y
    y = np.apply_along_axis(project,1,x)

    return x, y

def energy(w,pos,vel): #Standard hamiltonian for harmonic oscillator with w = sqrt(k/m)
    return 1/2*((vel**2) + (pos**2)*(w**2))

def check_hooke_data(x,y):
    #confirms that the energy is not truly lost in systems.
    #uses 1/2(v^2/w + x^2*w) as energy.
    #energies stores energy lost in systems
    assert x.shape[0] == y.shape[0]
    energies = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        energies[i] = energy(x[i,1],y[i,0],y[i,1]) - energy(x[i,1],x[i,2],x[i,3])
    print("Energy profile: {}\n.".format(scipy.stats.describe(energies)))


