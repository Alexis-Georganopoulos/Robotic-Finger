#! /usr/bin/env python

import rospy
import message_filters

from biotac_sensors.msg import BioTacHand #for biotac sensor
from geometry_msgs.msg import PoseStamped #for normal and hand frames

import numpy as np

##### GLOBAL VARIABLES-CONSIDER REMOVING THEM #####
T_n = np.zeros((3,3))
T_b = np.zeros((3,3))
z_dir = np.zeros((3,1))

ROTMAT = np.array([(1,0,0),(0,0,-1),(0,1,0)])
###################################################



##### LOCAL FUNCTIONS #####
def norm_quat_to_rot(quaternions):

	Qi = quaternions.pose.orientation.x
	Qj = quaternions.pose.orientation.y
	Qk = quaternions.pose.orientation.z
	Qr = quaternions.pose.orientation.w

	s = 1/(Qi**2 + Qj**2 + Qk**2 + Qr**2) # normalisation factor

	Rq = np.array([[1-2*s*(Qj**2 + Qk**2),   2*s*(Qi*Qj - Qk*Qr),   2*s*(Qi*Qk + Qj*Qr)],

				   [2*s*(Qi*Qj + Qk*Qr),    1-2*s*(Qi**2 + Qk**2),  2*s*(Qj*Qk - Qi*Qr)],  

				   [2*s*(Qi*Qk - Qj*Qr),     2*s*(Qj*Qk + Qi*Qr),   1-2*s*(Qi**2 + Qj**2)]])

	return Rq
##########################



#####CALLBACK FUNCTIONS#####
# def listener_callback_sensors(dataS):
# 	pdc = dataS.bt_data[0].pdc_data
# 	electrode = dataS.bt_data[0].electrode_data
# 	rospy.loginfo('\npdc: \n{}\nelectrode: \n{}'.format(pdc,electrode))


# def listener_callback_rotN(dataR):

# 	global T_n, z_dir

# 	T_n = norm_quat_to_rot(dataR)
# 	#rospy.loginfo('T_n: \n {} \n'.format(T_n))

# 	z_dir = ((T_b.T).dot(T_n))[:,[2]]
# 	rospy.loginfo('\nz_dir: \n{}'.format(z_dir))

# def listener_callback_rotB(dataR):

# 	global T_b, ROTMAT

# 	T_b = (norm_quat_to_rot(dataR)).dot(ROTMAT)
	#rospy.loginfo('T_b: \n {} \n'.format(T_b))

# def callback_distributor(dataS,dataRN,dataRB):
# 	listener_callback_sensors(dataS)
# 	listener_callback_rotB(dataRB)
# 	listener_callback_rotN(dataRN)
# 	rospy.loginfo('im in distributor\n')

###Remove below after debugging is complete###

# def listener_callback_robot(dataR):
# 	rospy.loginfo('im in robot callback\n')

# def callback_distributor(dataS,dataR):
# 	listener_callback_sensors(dataS)
# 	listener_callback_robot(dataR)
# 	rospy.loginfo('im in distributor\n')

def total_callback(dataS,dataRN,dataRB):

	pdc = dataS.bt_data[0].pdc_data
	electrode = dataS.bt_data[0].electrode_data

	global T_n,T_b,ROTMAT, z_dir

	T_b = (norm_quat_to_rot(dataRB)).dot(ROTMAT)
	T_n = norm_quat_to_rot(dataRN)
	z_dir = ((T_b.T).dot(T_n))[:,[2]]

	rospy.loginfo('\npdc: \n{}\nelectrode: \n{}\nz_dir: \n{}'.format(pdc,electrode,z_dir))
#############################



##### MAIN CALLBACK INITIALISER #####
def listener():
	rospy.init_node('biochip_listener',anonymous = True)
	# rospy.Subscriber("/biotac_pub", BioTacHand, listener_callback_sensors)
	# rospy.Subscriber("/vrpn_client_node/normalSurface/pose", PoseStamped, listener_callback_rotN)
	# rospy.Subscriber("/vrpn_client_node/baseHand/pose", PoseStamped, listener_callback_rotB)
	biotac_sub = message_filters.Subscriber("/biotac_pub", BioTacHand)
	# normal_sub = message_filters.Subscriber("/vrpn_client_node/normalSurface/pose", PoseStamped)
	# base_sub = message_filters.Subscriber("/vrpn_client_node/baseHand/pose", PoseStamped)
	normal_sub = message_filters.Subscriber("/vrpn_client_node/normalSurface/pose", PoseStamped)
	base_sub = message_filters.Subscriber("/vrpn_client_node/baseHand/pose", PoseStamped)
	synchronizer = message_filters.ApproximateTimeSynchronizer([biotac_sub,normal_sub,base_sub],500,0.002)
	synchronizer.registerCallback(total_callback)
	rospy.spin()
#####################################

if __name__ == '__main__':
	listener()