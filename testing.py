
from pprint import pprint
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
#Seaborn?

from physical_sim import *
from hooke_system import *
from hal_model_0  import *


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
    #test_batch_1()
    print("Choose 1=train, 2=plot in time, 3=overfitt, 4=improve, 5=controlled train ")
    choice = int(input())
    if choice == 1:
        hal_main_maker() #truncate=15,epochs=64*1024,batch=10)
    elif choice == 2:
        print("Which file do you want? ")
        f = input()
        f = os.path.join("./saved_models",f)
        #f = input()
        test_hal_in_time(f)
    elif choice == 3:
        #over fits the data to develop somehwat workable model
        hal_main_maker(truncate=20,epochs=36*1024,batch=15)
    elif choice == 4:
        #improve network
        print("Which file do you want? ")
        f = input()
        f = os.path.join("./saved_models",f)
        hal_improve_model(f)
    elif choice == 5:
        #train with parameters accepted
        print("What truncation of data do you want? ")
        truncate = input()
        try:
            truncate = int(truncate)
        except Exception as err:
            #not an int
            truncate = None
        print("What batch size do you want? ")
        batch = int(input())
        print("How many epochs do you want? ")
        epochs = int(input())
        print()
        hal_main_maker(truncate=truncate,epochs=epochs,batch=batch)




if __name__ == "__main__":
    main()

