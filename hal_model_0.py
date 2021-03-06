


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
NUM_INPUTS = 3 #T, x, v, T*w
NUM_HIDDEN = 100 #Inc? 
HIDDEN_LAYER_SPECS = [NUM_HIDDEN,NUM_HIDDEN]
NUM_OUPUTS = 2 #Position and velocity
#Network is 4 layers deep

EPOCHS     = 2**10
BATCH      = 2**6 #Inc?
DATA_FETCH_LENGTH = 10*100000 #EPOCHS*BATCH #Used for testing
LEARNING_RATE   = .005 #Maybe start this out large then trim it down.
LEAKY_RELU_RATE = .01 #Used for the leaky ReLU to prevent dead ReLUs.
DEFAULT_PHYSICAL_IMPORTANCE = 0.#6.#1.5 #1. #Param that describes the importance of the physical learning check
REGULARISATION_RATE = .00001

#file prep method
#ensures new file for each run


DATA_OUTPUT_FILE = "data_noted.csv"

CHECKPOINT_PATH = "training_1/checkpoint.ckpt"
CHECKPOINT_DIR  = os.path.dirname(CHECKPOINT_PATH)
# Create checkpoint callback
#ceckpoint_callback = tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, 
#    save_weights_only=False,
#    verbose=1) #Saves model during and after training
#Eager does not work well with callbacks

UPPDER_TIME = 48.
dt = .01 #specs used when doing plot comparison

def leaky_relu(x): #Encapsulates the leaky ReLU to avoid dead ReLUs
    return tf.keras.activations.relu(x,LEAKY_RELU_RATE)
#lambda x: tf.keras.activations.relu(x,LEAKY_RELU_RATE)

act = tf.keras.activations.linear#tf.keras.activations.relu
#Leaky relu accomplished as a layer

#Physical loss is in the form of a regulariser
def loss_fun(model=None, physical_importance=DEFAULT_PHYSICAL_IMPORTANCE):
    #This add the phyiscal energy term and extracts the inputs to put energy
    inputs = model.get_input_at(0)
    #[input_layer.get_input_at(index) for index in range(NUM_INPUTS)]
    def loss_interior(y_observed,y_predicted):
        #Done in this pattern to give loss access to inputs
        #physical term does not care about y_observed
        #abs (L1) would zero out the coefficients
        physical_term = physical_importance*tf.reduce_mean(tf.abs(
                energy(y_predicted[:,0],y_predicted[:,1]) - energy(inputs[:,1],inputs[:,2])))
        return physical_term + tf.keras.losses.mse(y_predicted,y_observed)
    
    return loss_interior

#metrics to list:

def energy_metric(m):
    #metric that outputs energy expended squared
    inputs = m.get_input_at(0)
    def energy_error(y_observed,y_predicted):
        #Done in this pattern to give loss access to inputs
        #physical term does not care about y_observed
        physical_term = tf.reduce_mean(tf.abs(
                energy(y_predicted[:,0],y_predicted[:,1]) - energy(inputs[:,1],inputs[:,2])))
        return physical_term
    
    return energy_error

metrics = ['accuracy','mae', 'mape', "mse"]

def model(hidden_layers, physical_importance=DEFAULT_PHYSICAL_IMPORTANCE):
    #Makes neural network model given hidden layer spec

    #Now we create the layers to accept the data
    #Layers given orthogonal weights
    layers = []
    layers.append(tf.keras.layers.Dense(hidden_layers[0],input_dim=NUM_INPUTS, 
        activation=act, kernel_regularizer=tf.keras.regularizers.l2(REGULARISATION_RATE)))
    #input_layer = layers[0]
    for layer in hidden_layers[1:]: #Consider dropout for versatility
        layers.append(tf.keras.layers.Dense(layer, activation=act,
            kernel_regularizer=tf.keras.regularizers.l2(REGULARISATION_RATE)))
        layers.append(tf.keras.layers.LeakyReLU(LEAKY_RELU_RATE))

    #Output linear for the purpose of outputing a real value
    layers.append(tf.keras.layers.Dense(NUM_OUPUTS, 
        activation=tf.keras.activations.linear, ))#kernel_regularizer=tf.keras.regularizers.l2(REGULARISATION_RATE)))


    m = tf.keras.Sequential(layers)

    #accuracy is a bad continuos metric since it is discreteish
    m.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), 
        loss=loss_fun(model=m, physical_importance=physical_importance), 
        metrics=metrics + [energy_metric(m)])
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


def hal_main_maker(truncate=None, batch=BATCH, epochs=EPOCHS, v_kill=False, physical_importance=DEFAULT_PHYSICAL_IMPORTANCE):
    
    m = model(HIDDEN_LAYER_SPECS,physical_importance=physical_importance)
    #checkpoint = tf.train.Checkpoint(model=m)

    #The point of intrest is the loss to this experiment
    m.summary()
    start_time = time.time()
    if truncate is not None:       
        #truncation allows overfitting on restriction of data
        x, y = get_hooke_data(truncate, v_kill)
    else:
        x, y = get_hooke_data(DATA_FETCH_LENGTH, v_kill)
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
    timestr = time.strftime("%-Y%m-%d-%H-%M-%S")


    saved_model_path = "./saved_models/{}+{}".format(timestr,physical_importance)
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
    print(m.predict(np.array([np.array([10.,1.,0.])])))

    #m.save(saved_model_path, include_optimizer=False)

    print("PASSED MODEL FORMING")
    return saved_model_path


def hal_improve_model(f, truncate=None, batch=BATCH, epochs=EPOCHS, save_data=True, v_kill=False,trial=0, physical_importance=DEFAULT_PHYSICAL_IMPORTANCE):
    #improves model
    #m = model(HIDDEN_LAYER_SPECS)
    #checkpoint = tf.train.Checkpoint(model=m)
    m = get_model(f,physical_importance=physical_importance)
    """tf.keras.models.load_model(f, compile=False)
    m.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), 
        loss=loss_fun(m), 
        metrics=metrics + [energy_metric(m)]) #compile to get loss func"""

    #The point of intrest is the loss to this experiment
    m.summary()
    start_time = time.time()
    if truncate is not None:       
        #truncation allows overfitting on restriction of data
        x, y = get_hooke_data(truncate, v_kill)
    else:
        x, y = get_hooke_data(DATA_FETCH_LENGTH, v_kill)
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
    timestr = time.strftime("%-Y%m-%d-%H-%M-%S")

    saved_model_path = "./saved_models/{}+{}:{}".format(timestr,physical_importance,trial)
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
    print(m.predict(np.array([np.array([10.,1.,0.])])))
    print("x, v = {}.".format(project([10.,1.,0.])))

    return saved_model_path

def plot_hal_model_in_time(m):
    #plots the test results from the hal model
    #plots a systems 1 dimensional history as a function of time
    #plots velocity and position of real and projected systems
    t = np.linspace(0,UPPDER_TIME,num=UPPDER_TIME//dt) #discretised time increments
    w = mk_omega()
    x0 = mk_x() #geting parameters of system to try
    v0 = mk_v()
    f_x = np.vectorize(lambda tau: x(tau,w,x0,v0))
    f_v = np.vectorize(lambda tau: v(tau,w,x0,v0))
    
    #For sanity
    print()
    print(m.predict(np.array([np.array([10.,1.,0.])])))


    xs = f_x(t)
    vs = f_v(t)

    plt.scatter(t,xs)
    plt.scatter(t,vs)

    data_inputs = np.array([np.array([tau,x0,v0]) for tau in t])

    #Now we get the projected values
    print(data_inputs.shape)
    projected_phase = m.predict(data_inputs)

    plt.scatter(t,projected_phase[:,0])
    plt.scatter(t,projected_phase[:,1])

    
    #print(history) #prints history to give full model parameters


    plt.title("Actual Trajectories")
    plt.xlabel("t")
    plt.ylabel("X and V")
    plt.show()

def get_model(f,physical_importance=DEFAULT_PHYSICAL_IMPORTANCE):
    m = tf.keras.models.load_model(f, compile=False)
    m.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), 
        loss=loss_fun(physical_importance=physical_importance,model=m.layers[0]), 
        metrics=metrics + [energy_metric(m)]) #compile to get loss func
    return m

def test_hal_in_time(f):
    m = get_model(f)

    plot_hal_model_in_time(m)

def do_model_test(m,f_name,trial=0, epochs=None, batch=None, physical_importance=DEFAULT_PHYSICAL_IMPORTANCE):
    #does one model test
    x, y = get_hooke_data(DATA_FETCH_LENGTH)
    evaluation = evaluate(m,x,y)
    f_out = "{}_{}_{}".format(epochs,batch,DATA_OUTPUT_FILE) if epochs is not None and batch is not None else DATA_OUTPUT_FILE
    with open(f_out, 'a') as f:
        f.write("{}, {}, {}, {}\n".format(f_name, physical_importance, trial,str(list(evaluation))[1:-1]))
        print("{}, {}".format(physical_importance,str(list(evaluation))[1:-1]))


def evaluate(model,x,y):
    #evaluates the model
    #returns evaluation data as tuple
    #record = {'physical_validity_factor': PHYSICAL_IMPORTANCE, 'loss': None, 'acc': None, 
    #    'mae': None, 'energy_error': energy_metric(model.layers[0]),' mape': None, "mse": None}
    evaluation = model.evaluate(x,y)
    return evaluation

def plot_experiment_summary(f):
    #plots experiment summary
    #fields = "file_name,physical_importance,trial,loss,{},energy_metric".format(str(metrics)[1:-1]).split(",")
    fields = ["file_name","physical_importance","trial","loss"] + metrics + ["energy_metric"]
    #assumes that file lacks first fields line
    records = pd.read_csv(f,names=fields)
    #print(records)
    #print(list(records))
    #records = records[records.physical_importance <= .2]
    plt.scatter(records.physical_importance,records.energy_metric)
    plt.ylabel("Mean Absolute Energy Deviation")
    plt.xlabel("Physical Importance")
    plt.show()

    plt.scatter(records.physical_importance,records.mse)
    print()
    print(records[["physical_importance","mse"]])
    print()
    processed_data = records[["physical_importance","mse"]]

    #now we compute averages
    lambdas = processed_data["physical_importance"].unique()
    with open("average_proccessed_data.csv","w") as f_out: 
        for rate in lambdas:
            f_out.write("{}, {}\n".format(rate,records[records["mse"] == rate]["mse"].mean()))
    

    processed_data.to_csv("processed_data.csv") #puts proccessed data into a file
    plt.ylabel("Mean Sqaure Error")
    plt.xlabel("Physical Importance")
    plt.title("Physical Weight vs MSE")
    m = tf.keras.Sequential([#tf.keras.layers.Dense(units=1,input_dim=1),
        tf.keras.layers.Lambda(lambda x:tf.sqrt(x), input_shape=(1,)),
        tf.keras.layers.Dense(units=1)])
    m.compile(optimizer="sgd",loss='mse')
    m.fit(np.array(records.physical_importance), np.array(records.mse), epochs=4096, batch_size=32, verbose=0)
    print(m.weights)
    threshold = .08 #To cut off outliers
    m_regular = tf.keras.Sequential([#tf.keras.layers.Dense(units=1,input_dim=1),
        tf.keras.layers.Lambda(lambda x:tf.sqrt(x), input_shape=(1,)),
        tf.keras.layers.Dense(units=1)])
    m_regular.compile(optimizer="sgd",loss='mse')
    m_regular.fit(np.array(records.physical_importance[records.mse<=threshold]), 
        np.array(records.mse[records.mse<=threshold]), epochs=4096, batch_size=32, verbose=0)
    #print(m.weights)
    print(m_regular.weights)
    x_ticks = np.linspace(0.0,max(records.physical_importance),num=3000)
    y_ticks = m.predict(x_ticks)
    y_ticks_regular = m_regular.predict(x_ticks)
    plt.scatter(x_ticks,y_ticks)
    plt.scatter(x_ticks,y_ticks_regular)
    plt.show()
    #Model appears to be of form: mse = a*sqrt(physical_importance) + b
