#!/usr/bin/env python
# coding:utf-8
import rospy
from people_msgs.msg import People, Person
from leg_tracker.msg import Leg, LegArray, PoseWithCov, PoseWithCovArray
from std_msgs.msg import String
from std_msgs.msg import Header
from geometry_msgs.msg import Point, PoseWithCovarianceStamped, PoseWithCovariance, Pose, PoseArray
#from my_people_tracker_pkg.msg import Pose
from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox
import math
import tf

class people_from_yolo(object):
    def __init__(self):
        rospy.init_node('people_from_yolov3', anonymous=True)
        rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, self.callback_yolo)
        #rospy.Subscriber('detected_leg_clusters', LegArray, self.callback_leg)


        self.pub_clusters = rospy.Publisher('detected_leg_clusters', LegArray, queue_size = 1000)
	self.pub_people_pose = rospy.Publisher('people_pose', PoseArray, queue_size = 1000)
	self.legArray = LegArray()
        
        rospy.spin()
        
    def callback_leg(self, data):
        self.legArray = data
        if len(self.legArray.legs) > 0:
            self.pub_clusters.publish(self.legArray)
        
         
    def callback_yolo(self, data):
    
        my_list = []
        for i in data.bounding_boxes:
            my_list.append(i)
            
        self.peoplePose = PoseArray()
        
        yoloTracked = LegArray()
        
        people_r = People()
        self.peoplePose.header = data.image_header
        self.peoplePose.header.frame_id = "camera_link"
        
        yoloTracked.header = data.image_header
        yoloTracked.header.frame_id = "camera_link"
        
        for i in my_list:
            
            self.peoplePose.poses.append(Pose())
            yoloTracked.legs.append(Leg())

        count = 0
        for i in my_list:
        
            if not math.isnan(i.Z) and not math.isnan(i.X):
            
		        self.peoplePose.poses[count].position.x = i.Z + 0
		        self.peoplePose.poses[count].position.y = - i.X + 0.25
		        self.peoplePose.poses[count].position.z = 0
		        
		        yoloTracked.legs[count].position.x = i.Z + 0
		        yoloTracked.legs[count].position.y = - i.X + 0.25
		        yoloTracked.legs[count].position.z = 0
		        yoloTracked.legs[count].confidence = 0.99
		        
		        yoloTracked.legs.append(yoloTracked.legs[count])
		        
		        count = count + 1
		            
		        count_angle = 0
		        
		        for i in my_list:
		            self.peoplePose.poses[count_angle].orientation.x = 0
		            self.peoplePose.poses[count_angle].orientation.y = 0
		            self.peoplePose.poses[count_angle].orientation.z = 1
		            self.peoplePose.poses[count_angle].orientation.w = 0
		            
		            count_angle = count_angle + 1  


        count_result = 0
        for i in range(len(yoloTracked.legs)):
            clusters = LegArray()
            clusters.header = yoloTracked.header
            min_dist = 0.05
            clusters.legs.append(yoloTracked.legs[i])
            clusters.legs.append(yoloTracked.legs[i])
            count_result = count_result + 1
            '''for j in range(len(self.legArray.legs)):
                if (yoloTracked.legs[i].position.x != 0 and yoloTracked.legs[i].position.y != 0): 
             
                    if math.sqrt((yoloTracked.legs[i].position.x - self.legArray.legs[j].position.x)**2 + (yoloTracked.legs[i].position.y - self.legArray.legs[j].position.y)**2) > min_dist:'''


        if count_result:
            self.pub_clusters.publish(clusters)
            
        self.pub_people_pose.publish(self.peoplePose)

if __name__ == '__main__':
    talker = people_from_yolo()

