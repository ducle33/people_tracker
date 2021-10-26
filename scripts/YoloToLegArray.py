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
        rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, self.callback_person)

        self.pub_people_pose = rospy.Publisher('people_pose', PoseArray, queue_size = 1000)

        self.pub_yolo_track = rospy.Publisher('yolo_tracked', LegArray, queue_size = 1000)
        #self.yolo_pose_pub = rospy.Publisher('yolo_pose', PoseWithCovArray, queue_size=1000)
        self.turn = 0
        self.last_people = People()
        self.time_vel = 0
        self.yoloCount = 0
        
        rospy.spin()
        
    def ToQuaternion(self, yaw, pitch, roll):
        cy = math.cos(yaw * 0.5);
        sy = math.sin(yaw * 0.5);
        cp = math.cos(pitch * 0.5);
        sp = math.sin(pitch * 0.5);
        cr = math.cos(roll * 0.5);
        sr = math.sin(roll * 0.5);
        
        w = cr * cp * cy + sr * sp * sy;
        x = sr * cp * cy - cr * sp * sy;
        y = cr * sp * cy + sr * cp * sy;
        z = cr * cp * sy - sr * sp * cy;

        return [x, y, z, w];
        
         
              
    def callback_person(self, data):
    
        my_list = []
        for i in data.bounding_boxes:
            my_list.append(i)
            
        self.peoplePose = PoseArray()
        
        yoloTracked = LegArray()
        
        #yoloPose = PoseWithCovArray()
        
        velocity = []
        angles = []
        init_vel = [0,0,0]
        init_angle = [0,0,0,0]
        
        people_r = People()
        self.peoplePose.header = data.image_header
        self.peoplePose.header.frame_id = "camera_link"
        
        yoloTracked.header = data.image_header
        yoloTracked.header.frame_id = "camera_link"
        
        #yoloPose.header = data.image_header
        #yoloPose.header.frame_id = "camera_link"
        
        #people_r.header = data.image_header
        #people_r.header.frame_id = "camera_link"
        
        for i in my_list:
            #people_r.people.append(Person())
            #velocity.append(init_vel)
            #angles.append(init_angle)
            
            self.peoplePose.poses.append(Pose())
            yoloTracked.legs.append(Leg())
            #yoloPose.poses.append(PoseWithCov())

        count = 0
        for i in my_list:
            #people_r.people[count].position.x = i.X
            #people_r.people[count].position.y = i.Y
            #people_r.people[count].position.z = i.Z
            #people_r.people[count].reliability = i.probability
            #people_r.people[count].name = i.Class + " " + str(count + 1)
            #people_r.people[count].tagnames = [people_r.people[count].name]
            #people_r.people[count].tags = ["uuids"]
            
            self.peoplePose.poses[count].position.x = i.Z + 0
            self.peoplePose.poses[count].position.y = - i.X + 0.25
            self.peoplePose.poses[count].position.z = 0
            
            yoloTracked.legs[count].position.x = i.Z + 0
            yoloTracked.legs[count].position.y = - i.X + 0.25
            yoloTracked.legs[count].position.z = 0
            yoloTracked.legs[count].confidence = 0.99
            
            yoloTracked.legs.append(yoloTracked.legs[count])
            
            #yoloPose.poses[count].pose.position.x = i.Z + 0 
            #yoloPose.poses[count].pose.position.y = - i.X + 0.25
            #yoloPose.poses[count].pose.position.z = 0
            #yaw = 0
            #vel = 0.2
            #quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
            #yoloPose.poses[count].pose.orientation.x = vel * math.cos(yaw)
            #yoloPose.poses[count].pose.orientation.y = vel * math.sin(yaw)
            #yoloPose.poses[count].pose.orientation.z = quaternion[2]
            #yoloPose.poses[count].pose.orientation.w = quaternion[3] 
            #yoloPose.poses[count].id = self.yoloCount
            #yoloPose.poses[count].xCov = 0.7
            #yoloPose.poses[count].yCov = 0.7
            #yoloPose.poses[count].vxCov = 0.7
            #yoloPose.poses[count].vyCov = 0.7
            
            self.yoloCount = self.yoloCount + 1
            
            count = count + 1
            
        
        
        #count_vel = 0

        #self.turn = self.turn + 1
        #if self.turn != 1:
        #    for i in people_r.people:
        #        for j in self.last_people.people:
        #            if (i.name == j.name):
        #                if people_r.header.stamp != self.last_people.header.stamp:
        #                    velocity[count_vel][0] = (i.position.x - j.position.x)/0.05
        #                    velocity[count_vel][1] = (i.position.y - j.position.y)/0.05
        #                    velocity[count_vel][2] = (i.position.z - j.position.z)/0.05
        #                    
        #                    angles[count_vel] = (self.ToQuaternion(0,0,math.atan2(velocity[count_vel][2],velocity[count_vel][0])))
                            
        #        count_vel = count_vel + 1
                
            count_angle = 0
            
            for i in my_list:
                self.peoplePose.poses[count_angle].orientation.x = 0
                self.peoplePose.poses[count_angle].orientation.y = 0
                self.peoplePose.poses[count_angle].orientation.z = 1
                self.peoplePose.poses[count_angle].orientation.w = 0
                
                count_angle = count_angle + 1  
           
            
        self.pub_people_pose.publish(self.peoplePose)
        self.pub_yolo_track.publish(yoloTracked)
        #self.yolo_pose_pub.publish(yoloPose)
        
        
        #if (self.turn == 1) and (self.last_people.people == []):
        #    self.last_people = people_r
        #elif ((people_r.header.stamp - self.last_people.header.stamp) != 0) and (self.last_people.people != []):
        #    self.last_people = people_r



if __name__ == '__main__':
    talker = people_from_yolo()

