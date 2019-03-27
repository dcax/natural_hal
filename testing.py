
from pprint import pprint
import matplotlib as mpl
import matplotlib.pyplot as plt
#Seaborn?

from physical_sim import *
from hooke_system import *


#This is where the bulk of the testing code should go
num_trials = 20000






def test_batch_0():
    #First test to see if interface works:
    pprint(mk_rand_state())
    st = mk_rand_state()

    history = do_evolution(st, n=num_trials)
    #We how have the data from the model, we just have to plot it
    pprint(history.shape)
    plot_history(history=history)

def kepler_test():
    #DEPRECATED IN MOVE TO 2 SPING MASS
    #simple two body keplerian test
    #first body is incredibly heavy
    M = 10000000.
    R = 1000.
    v = np.sqrt(G*M/R)
    st = np.array([M,0.,0.,0., 0.,0.,0., 1., R, 0, 0., 0., v, 0., 0.,0.,0.,0.,0.,0.,0.])
    history = do_evolution(st, n=num_trials)
    plot_history(history, discretion=50)

def test_batch_1():
    #testing based more strongly on the simple harmonic oscillator of 1 degrees of freedom

    #Working
    plot_in_time(200.,1.,-1.,1.,limit=100000.,time_intreval=1)



def main():
    #test_batch_0()
    #kepler_test()
    test_batch_1()




if __name__ == "__main__":
    main()

