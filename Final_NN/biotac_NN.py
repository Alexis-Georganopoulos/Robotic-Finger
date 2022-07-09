#! /usr/bin/env python

import rospy
import message_filters

from biotac_sensors.msg import BioTacHand #for biotac sensor

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

###### GLOBAL VARIABLES  ##########################
coeff_space = ["coeff0.csv","coeff1.csv","coeff2.csv",
               "coeff3.csv","coeff4.csv","coeff5.csv",
               "coeff6.csv","coeff7.csv"]

bias_space = ["bias0.csv","bias1.csv","bias2.csv",
              "bias3.csv","bias4.csv","bias5.csv",
              "bias6.csv","bias7.csv",]

RNN1 = MLPRegressor(hidden_layer_sizes=(5,)*7,max_iter=500,
                    learning_rate_init=0.01,alpha = 0.0001,
                    learning_rate = 'adaptive')

callback_state = 0
calibration_counter = 0
pdc_offset = 0
electrodes_offset = np.reshape(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),(1,19)) 
###################################################

##### FUNCTION DEFINITIONS ########################

def csv_to_arr_np(name_space):
    
    array_of_numpy = []
    j = 0
    for i in name_space:
        temp = pd.read_csv(i)
        temp2 = temp.to_numpy()
        temp3 = temp2[:,1:]
        
        array_of_numpy.append(temp3)
        if(array_of_numpy[j].shape[1] == 1):
             array_of_numpy[j] =  array_of_numpy[j].flatten()
             
        j = j + 1
        
    return array_of_numpy

def initialise_nn():
    
    global RNN1, coeff_space,bias_space
    
    coeff_NN = csv_to_arr_np(coeff_space)
    bias_NN = csv_to_arr_np(bias_space)
    
    RNN1.coefs_ = coeff_NN
    RNN1.intercepts_ = bias_NN
    RNN1.n_layers_ = 9
    RNN1.n_outputs_ = 2
    RNN1.out_activation_ = 'identity'
    
    
def nn_predict(electrode_data):
    global RNN1
    
    x_z = RNN1.predict(electrode_data)
    x = x_z[0][0]
    z = x_z[0][1]
    
    ydir = np.sign(x)*np.sign(z)
    check_if_real = 1 - x**x - z**z 
    
    if( check_if_real < 0):
        ymag = 0
    else:
        ymag =  check_if_real**0.5
    normal_dir = [x, ydir*ymag, z]
    return normal_dir
##################################################


##### CALLBACK FUNCTION ##########################
def total_callback(dataS):
    
    global callback_state,calibration_counter,pdc_offset, electrodes_offset
    
    pdc = dataS.bt_data[0].pdc_data
    electrode = dataS.bt_data[0].electrode_data
    electrode = np.reshape(np.array(list(electrode)),(1,19))
    
    if (callback_state == 0):
        initialise_nn()
        rospy.loginfo('Nerual Network Succesefully initialised \n Now entering calibration mode \n')
        callback_state = 1
        
    if (callback_state == 1):
        pdc_offset = pdc_offset + pdc
        electrodes_offset = electrodes_offset + electrode
        calibration_counter = calibration_counter + 1
        if(calibration_counter >= 500):
            pdc_offset = pdc_offset/500
            electrodes_offset = electrodes_offset/500
            rospy.loginfo('Calibration Succesefull \n')
            callback_state = 2
        
    if(callback_state == 2):
        if(pdc <= 1.05*pdc_offset):
            normal_direction = [0,0,0]
        else:
            normal_direction = nn_predict(electrode - electrodes_offset)
            
        rospy.loginfo('Normal Direction [x,y,z]: {}\n'.format(normal_direction))
#################################################



##### MAIN CALLBACK INITIALISER #################
def listener():
	rospy.init_node('biochip_listener',anonymous = True)

	biotac_sub = message_filters.Subscriber("/biotac_pub", BioTacHand,total_callback)

	rospy.spin()
#################################################

if __name__ == '__main__':
	listener()