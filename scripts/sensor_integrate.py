#!/usr/bin/python

import rospy

# Custom messages
from leg_tracker.msg import Person, PersonArray, Leg, LegArray, PoseWithCov, PoseWithCovArray 

# ROS messages
from visualization_msgs.msg import Marker
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid

# Standard python modules
import numpy as np
import random
import math
from scipy.optimize import linear_sum_assignment
import scipy.stats
import scipy.spatial
from geometry_msgs.msg import PointStamped, Point, Pose, PoseArray
from std_msgs.msg import Time, String
import tf
import copy
import timeit
import message_filters
import sys


# External modules
from pykalman import KalmanFilter # To install: http://pykalman.github.io/#installation


class sensor_integration(object):
    
    def __init__(self):
    
        rospy.init_node('sensor_integration', anonymous=True)
    	# ROS publishers
        self.people_tracking_pub = rospy.Publisher('people_tracking', PoseWithCovArray, queue_size=1000)

        # ROS subscribers         
        rospy.Subscriber('/leg_pose', PoseWithCovArray, self.leg_callback)
        rospy.Subscriber('/yolo_pose', PoseWithCovArray, self.yolo_callback)
        
        self.tracked_pairs = []
        self.tracked_pair_index = 0
        self.legPose = PoseWithCovArray()
        self.yoloPose = PoseWithCovArray()
        self.prev_person_marker_id = 0
        self.marker_id = 0
        self.colour = (random.random(), random.random(), random.random())
        self.frame_id_global = String()
        self.stamp_global = Time()

        rospy.spin() # So the node doesn't immediately shut down
        
        
    def leg_callback(self, data): 
        self.legPose = data
        tracked_people = PoseWithCovArray()
        count_result = 0
        tracked_people.header = data.header
        
        for j in range(len(self.legPose.poses)):
                
            if (math.sqrt(self.legPose.poses[j].pose.orientation.x**2 + self.legPose.poses[j].pose.orientation.y**2)) > 0.05:
                yaw = math.atan2(self.legPose.poses[j].pose.orientation.y, self.legPose.poses[j].pose.orientation.x)
                self.legPose.poses[j].pose.orientation.x = 0.05*math.cos(yaw)
                self.legPose.poses[j].pose.orientation.y = 0.05*math.sin(yaw)
            if abs(self.legPose.poses[j].pose.orientation.x) < 0.01 or abs(self.legPose.poses[j].pose.orientation.y) < 0.01:
                continue
            result_pose = PoseWithCov()
            result_pose = self.legPose.poses[j]
            tracked_people.poses.append(result_pose)
            count_result = count_result + 1
            
        if count_result > 0:
            self.people_tracking_pub.publish(tracked_people)
            
        
        
    def yolo_callback(self, data): # if yolo and leg, then integrate; if only yolo then take the yolo
        self.yoloPose = data
        tracked_people = PoseWithCovArray()
        tracked_people.header = data.header
        yoloArray = []
        legArray = []
        
        
        for i in range(len(self.yoloPose.poses)):
            x_yolo = self.yoloPose.poses[i].pose.position.x #0
            y_yolo = self.yoloPose.poses[i].pose.position.y #1
            vx_yolo = self.yoloPose.poses[i].pose.orientation.x #2
            vy_yolo = self.yoloPose.poses[i].pose.orientation.y #3
            xCov_yolo = self.yoloPose.poses[i].xCov #4
            yCov_yolo = self.yoloPose.poses[i].yCov #5
            vxCov_yolo = self.yoloPose.poses[i].vxCov #6
            vyCov_yolo = self.yoloPose.poses[i].vyCov #7
            id_yolo = self.yoloPose.poses[i].id #8
            orientation_z = self.yoloPose.poses[i].pose.orientation.z #9
            orientation_w = self.yoloPose.poses[i].pose.orientation.w #10
            yoloArray.append([x_yolo, y_yolo, vx_yolo, vy_yolo, xCov_yolo, yCov_yolo, vxCov_yolo, vyCov_yolo, id_yolo, orientation_z, orientation_w])
            
        if len(self.legPose.poses):
            for i in range(len(self.legPose.poses)):   
                x_leg = self.legPose.poses[i].pose.position.x #0
                y_leg = self.legPose.poses[i].pose.position.y #1
                vx_leg = self.legPose.poses[i].pose.orientation.x #2
                vy_leg = self.legPose.poses[i].pose.orientation.y #3
                xCov_leg = self.legPose.poses[i].xCov #4
                yCov_leg = self.legPose.poses[i].yCov #5
                vxCov_leg = self.legPose.poses[i].vxCov #6
                vyCov_leg = self.legPose.poses[i].vyCov #7
                id_leg = self.legPose.poses[i].id #8
                orientation_z = self.legPose.poses[i].pose.orientation.z #9
                orientation_w = self.legPose.poses[i].pose.orientation.w #10
                legArray.append([x_leg, y_leg, vx_leg, vy_leg, xCov_leg, yCov_leg, vxCov_leg, vyCov_leg, id_leg, orientation_z, orientation_w])
        
        
                    
        for i in range(len(self.yoloPose.poses)):
            result_pose = PoseWithCov()
            min_dist = 2.0
            id_yolo = -1
            id_leg = -1
            found_leg = False
            self.marker_id = 0
            count_result = 0
            for j in range(len(self.legPose.poses)):
                if (self.yoloPose.poses[i].pose.position.x != 0 and self.yoloPose.poses[i].pose.position.y != 0): 
                  
                    if abs(yoloArray[i][2]) < 0.02 and abs(yoloArray[i][3]) < 0.02:
                        result_pose = self.yoloPose.poses[i]
                        found_leg = True
                    
                    elif math.sqrt(pow(yoloArray[i][0] - legArray[j][0],2) + pow(yoloArray[i][1] - legArray[j][1],2)) < min_dist and math.sqrt(legArray[i][3]**2 + legArray[i][2]**2) > 0.01:
                    
	                    min_dist = math.sqrt(pow(yoloArray[i][0] - legArray[j][0],2) + pow(yoloArray[i][1] - legArray[j][1],2))
	                    weightx = legArray[j][4] / (legArray[j][4] + yoloArray[i][4])
	                    weighty = legArray[j][5] / (legArray[j][5] + yoloArray[i][5])
	                    weightvx = legArray[j][6] / (legArray[j][6] + yoloArray[i][6])
	                    weightvy = legArray[j][7] / (legArray[j][7] + yoloArray[i][7])
	                    
	                    yaw = math.atan2(legArray[j][3], legArray[j][2])
	                    vel = math.sqrt(yoloArray[i][3]**2 + yoloArray[i][2]**2)
	                    
	                    
	                    result_pose.pose.position.x = 1.0 * legArray[j][0] + (0.0) * yoloArray[i][0]
	                    result_pose.pose.position.y = 1.0 * legArray[j][1] + (0.0) * yoloArray[i][1]
	                    result_pose.pose.orientation.x = vel * math.cos(yaw)
	                    result_pose.pose.orientation.y = vel * math.sin(yaw)
	                    
	                    result_pose.id = yoloArray[i][8]
	                    
	                    quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
	                    result_pose.pose.orientation.z = quaternion[2]
	                    result_pose.pose.orientation.w = quaternion[3]
	                    result_pose.pose.position.z = 0
	                    result_pose.xCov = legArray[j][4]
	                    result_pose.yCov = legArray[j][5]
	                    result_pose.vxCov = legArray[j][6]
	                    result_pose.vyCov = legArray[j][7]
	                    id_yolo = yoloArray[i][8]
	                    id_leg = legArray[i][8]
	                    found_leg = True
	                    
            if found_leg:
                tracked_people.poses.append(result_pose)
            else:
                result_pose = self.yoloPose.poses[i]
                tracked_people.poses.append(result_pose)
            count_result = count_result + 1

        if count_result > 0:
            self.people_tracking_pub.publish(tracked_people)

 
if __name__ == '__main__':
    talker = sensor_integration()




