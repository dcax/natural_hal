


#Hal Model SHM: Neural network for simple harmonic motion
import numpy as np
import pandas as pd
import tensorflow as tf
import time
from pprint import pprint
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #Stops superlfuous err message
#Eager on default for tensorflow 2

from hooke_system import *

#Neural network spec
NUM_INPUTS = 4 #T, w, x, v
NUM_HIDDEN = 32 #Inc? 
HIDDEN_LAYER_SPECS = [NUM_HIDDEN,NUM_HIDDEN]
NUM_OUPUTS = 2 #Position and velocity
#Network is 4 layers deep

EPOCHS     = 2**12
BATCH      = 2**8 #Inc?
DATA_FETCH_LENGTH = EPOCHS*BATCH #Unused
LEARNING_RATE   = .001 #Maybe start this out large then trim it down.
LEAKY_RELU_RATE = .0001 #Used for the leaky ReLU to prevent dead ReLUs


#file prep method
#ensures new file for each run




CHECKPOINT_PATH = "training_1/checkpoint.ckpt"
CHECKPOINT_DIR  = os.path.dirname(CHECKPOINT_PATH)
# Create checkpoint callback
#ceckpoint_callback = tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, 
#    save_weights_only=False,
#    verbose=1) #Saves model during and after training
#Eager does not work well with callbacks


UPPDER_TIME = 1000
dt = .5 #specs used when doing plot comparison

def leaky_relu(x): #Encapsulates the leaky ReLU to avoid dead ReLUs
    return tf.keras.activations.relu(x,LEAKY_RELU_RATE)
#lambda x: tf.keras.activations.relu(x,LEAKY_RELU_RATE)

act = tf.keras.activations.relu

def model(hidden_layers):
    #Makes neural network model given hidden layer spec

    #Now we create the layers to accept the data
    #Layers given orthogonal weights
    layers = []
    layers.append(tf.keras.layers.Dense(hidden_layers[0],input_dim=NUM_INPUTS, 
        activation=act))
    for layer in hidden_layers[1:]: #Consider dropout for versatility
        layers.append(tf.keras.layers.Dense(layer, activation=act))
    #Output linear for the purpose of outputing a real value
    layers.append(tf.keras.layers.Dense(NUM_OUPUTS, 
        activation=tf.keras.activations.linear))


    m = tf.keras.Sequential(layers)

    #accuracy is a bad continuos metric since it is discreteish
    m.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), 
        loss=tf.keras.losses.MSE, 
        metrics=['accuracy','mae', 'mape'])
    #m.optimizer.lr = LEARNING_RATE

    return m

def time_str(sec):
    #Given ellapsed seconds, makes ellapsed time string
    r = ""
    minuites = sec//60
    hours = minuites//60
    days = hours//24
    sec %= 60 #normalise the data
    minuites %= 60
    hours %= 24
    if days != 0:
        return "{} days, {} hours, {} min, {} sec".format(days,hours,minuites,sec)
    elif hours != 0:
        return "{} hours, {} min, {} sec".format(hours,minuites,sec)
    elif minuites != 0:
        return "{} min, {} sec".format(minuites,sec)
    else:
        return str(sec) + " sec"


def hal_main_maker(truncate=None, batch=BATCH, epochs=EPOCHS):
    
    m = model(HIDDEN_LAYER_SPECS)
    #checkpoint = tf.train.Checkpoint(model=m)

    #The point of intrest is the loss to this experiment
    m.summary()
    start_time = time.time()
    if truncate is not None:       
        #truncation allows overfitting on restriction of data
        x, y = get_hooke_data(truncate)
    else:
        x, y = get_hooke_data(batch*epochs)
    time_data_made = time.time()
    
    check_hooke_data(x,y)

    time_data_checked = time.time()


    m.fit(x, y, epochs=epochs, batch_size=batch)
    time_data_fitted = time.time()

    #print(m.weights)
    print()
    #pprint(m.weights)

    #print("File {}.".format(FILE))
    #tf.keras.models.save_model(model=m, filepath=FILE)
    
    #Now we note the next model done
    #with open("sys_params") as f:
    #    f.write(int(i)+1)

    saved_model_path = "./checkoint/{}".format(int(time.time()))
    #Checkpoints are the eager way to save models
    #checkpoint.save(saved_model_path)
    #Better file saving
    m.save(saved_model_path)
    
    #Now we cat the time info to shell
    print()
    print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
    print("Took {} to generate data.".format(time_str(time_data_made - start_time)))
    print("Took {} to check data.".format(time_str(time_data_checked - time_data_made)))
    print("Took {} to fit data.".format(time_str(time_data_fitted - time_data_checked)))
    print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")

    print()

    #Now we do predicitve test
    print(m.predict(np.array([np.array([10.,1.,1.,0.])])))

    #m.save(saved_model_path, include_optimizer=False)

    print("PASSED MODEL FORMING")


def plot_hal_model_in_time(m):
    #plots the test results from the hal model
    #plots a systems 1 dimensional history as a function of time
    #plots velocity and position of real and projected systems
    t = np.linspace(0,dt,UPPDER_TIME) #discretised time increments
    w = mk_omega()
    x0 = mk_x() #geting parameters of system to try
    y0 = mk_v()
    f_x = np.vectorize(lambda tau: x(tau,w,x0,v0))
    f_v = np.vectorize(lambda tau: v(tau,w,x0,v0))

    xs = f_x(t)
    vs = f_v(t)

    plt.scatter(t,xs)
    plt.scatter(t,vs)

    data_inputs = np.transpose(np.array([t,w,x0,y0]))

    #Now we get the projected values
    projected_phase = m.predict()

    plt.scatter(t,projected_phase[0,:])
    plt.scatter(t,projected_phase[1,:])

    
    #print(history) #prints history to give full model parameters


    plt.title("Actual Trajectories")
    plt.xlabel("t")
    plt.ylabel("X and V")
    plt.show()

def test_hal_in_time(f):
    m = tf.keras.models.load_model(f)

    plot_hal_model_in_time(m)


