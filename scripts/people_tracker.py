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
from geometry_msgs.msg import PointStamped, Point
import tf
import copy
import timeit
import message_filters
import sys

# External modules
from pykalman import KalmanFilter # To install: http://pykalman.github.io/#installation


class DetectedCluster:
    """
    A detected scan cluster. Not yet associated to an existing track.
    """
    def __init__(self, pos_x, pos_y, confidence, vx, vy):
        """
        Constructor
        """
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.confidence = confidence
        self.vel_x = vx
        self.vel_y = vy


class ObjectTracked:
    """
    A tracked object. Could be a person leg, entire person or any arbitrary object in the laser scan.
    """
    new_leg_id_num = 1

    def __init__(self, x, y, now, confidence, vx, vy): 
        """
        Constructor
        """        
        self.id_num = ObjectTracked.new_leg_id_num
        ObjectTracked.new_leg_id_num += 1
        self.colour = (random.random(), random.random(), random.random())
        self.last_seen = now
        self.seen_in_current_scan = True
        self.times_seen = 1
        self.confidence = confidence
        self.dist_travelled = 0.
        self.deleted = False

        # People are tracked via a constant-velocity Kalman filter with a Gaussian acceleration distrubtion
        # Kalman filter params were found by hand-tuning. 
        # A better method would be to use data-driven EM find the params. 
        # The important part is that the observations are "weighted" higher than the motion model 
        # because they're more trustworthy and the motion model kinda sucks
        scan_frequency = rospy.get_param("scan_frequency", 7.5)
        delta_t = 1./scan_frequency
        if scan_frequency > 7.49 and scan_frequency < 7.51:
            std_process_noise = 0.06666
        elif scan_frequency > 9.99 and scan_frequency < 10.01:
            std_process_noise = 0.05
        elif scan_frequency > 14.99 and scan_frequency < 15.01:
            std_process_noise = 0.03333
        else:
            print "Scan frequency needs to be either 7.5, 10 or 15 or the standard deviation of the process noise needs to be tuned to your scanner frequency"
        std_pos = std_process_noise
        std_vel = std_process_noise
        std_obs = 0.1
        var_pos = std_pos**2
        var_vel = std_vel**2
        # The observation noise is assumed to be different when updating the Kalman filter than when doing data association
        var_obs_local = std_obs**2 
        self.var_obs = (std_obs + 0.4)**2

        self.filtered_state_means = np.array([x, y, vx, vy])
        self.pos_x = x
        self.pos_y = y
        self.vel_x = vx
        self.vel_y = vy
        self.pos_x_cov = 0
        self.pos_y_cov = 0
        self.vel_x_cov = 0
        self.vel_y_cov = 0

        self.filtered_state_covariances = 0.5*np.eye(4) 
        
        self.estimate_state_covariances = np.zeros((4,4))
        self.innovation_matrix = np.zeros((2,2))
        self.kalmanGain = np.zeros((4,2))
        
        

        # Constant velocity motion model
        transition_matrix = np.array([[1, 0, delta_t,        0],
                                      [0, 1,       0,  delta_t],
                                      [0, 0,       1,        0],
                                      [0, 0,       0,        1]])

        # Oberservation model. Can observe pos_x and pos_y (unless person is occluded). 
        self.observation_matrix = np.array([[1, 0, 0, 0],
                                       [0, 1, 0, 0]])

        transition_covariance = np.array([[var_pos,       0,       0,       0],
                                          [      0, var_pos,       0,       0],
                                          [      0,       0, var_vel,       0],
                                          [      0,       0,       0, var_vel]])

        observation_covariance =  var_obs_local*np.eye(2)

        self.kf = KalmanFilter(
            transition_matrices=transition_matrix,
            observation_matrices=self.observation_matrix,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
        )


    def update(self, observations):
        """
        Update our tracked object with new observations
        """
        self.filtered_state_means, self.filtered_state_covariances = (
            self.kf.filter_update(
                self.filtered_state_means,
                self.filtered_state_covariances,
                observations
            )
        )

        # Keep track of the distance it's travelled 
        # We include an "if" structure to exclude small distance changes, 
        # which are likely to have been caused by changes in observation angle
        # or other similar factors, and not due to the object actually moving
        delta_dist_travelled = ((self.pos_x - self.filtered_state_means[0])**2 + (self.pos_y - self.filtered_state_means[1])**2)**(1./2.) 
        if delta_dist_travelled > 0.01: 
            self.dist_travelled += delta_dist_travelled

        self.pos_x = self.filtered_state_means[0]
        self.pos_y = self.filtered_state_means[1]
        self.vel_x = self.filtered_state_means[2]
        self.vel_y = self.filtered_state_means[3]

        
        self.innovation_matrix = self.observation_matrix.dot(self.filtered_state_covariances.dot(self.observation_matrix.transpose())) + self.var_obs*np.ones((2,2)) # 2x4 * 4x4 * 4x2 + 2x2 = 2x2
        
        self.kalmanGain = self.filtered_state_covariances.dot(self.observation_matrix.transpose().dot(np.linalg.inv(self.innovation_matrix))) # 4x4 * 4x2 * 2x2 = 4x2
        
        self.estimate_state_covariances = np.eye(4) - self.kalmanGain.dot(self.observation_matrix.dot(self.filtered_state_covariances)) # 4x4 - 4x2 * 2x4 * 4x4 = 4x4
        
        
        self.pos_x_cov = self.estimate_state_covariances[0][0]
        self.pos_y_cov = self.estimate_state_covariances[1][1]
        self.vel_x_cov = self.estimate_state_covariances[2][2]
        self.vel_y_cov = self.estimate_state_covariances[3][3]
    


class KalmanMultiTracker:    
    """
    Tracker for tracking all the people and objects
    """
    max_cost = 9999999

    def __init__(self):      
        """
        Constructor
        """
        self.objects_tracked = []
        self.people_tracked = []
        self.prev_track_marker_id = 0
        self.prev_person_marker_id = 0
        self.prev_time = None
        self.listener = tf.TransformListener()
        random.seed(1) 

        # Get ROS params
        self.fixed_frame = rospy.get_param("fixed_frame", "odom")
        self.confidence_threshold_to_maintain_track = rospy.get_param("confidence_threshold_to_maintain_track", 0.1)
        self.publish_occluded = rospy.get_param("publish_occluded", True)
        self.publish_people_frame = rospy.get_param("publish_people_frame", self.fixed_frame)
        self.use_scan_header_stamp_for_tfs = rospy.get_param("use_scan_header_stamp_for_tfs", False)
        self.publish_detected_people = rospy.get_param("display_detected_people", False)        
        self.dist_travelled_together_to_initiate_leg_pair = rospy.get_param("dist_travelled_together_to_initiate_leg_pair", 0.5)
        scan_topic = rospy.get_param("scan_topic", "scan");
        self.scan_frequency = rospy.get_param("scan_frequency", 7.5)
        self.confidence_percentile = rospy.get_param("confidence_percentile", 0.90)
        self.max_std = rospy.get_param("max_std", 0.9)

        self.mahalanobis_dist_gate = scipy.stats.norm.ppf(1.0 - (1.0-self.confidence_percentile)/2., 0, 1.0)
        self.max_cov = self.max_std**2
        self.latest_scan_header_stamp_with_tf_available = rospy.get_rostime()

    	# ROS publishers
        self.people_pose_pub = rospy.Publisher('people_pose', PoseWithCovArray, queue_size=100)
        
        self.marker_pub = rospy.Publisher('people_pose_marker', Marker, queue_size=100)

        # ROS subscribers         
        self.detected_clusters_sub = rospy.Subscriber('people_tracking', PoseWithCovArray, self.detected_clusters_callback)

        rospy.spin() # So the node doesn't immediately shut down

        
    def match_detections_to_tracks_GNN(self, objects_tracked, objects_detected):
        """
        Match detected objects to existing object tracks using a global nearest neighbour data association
        """
        matched_tracks = {}

        # Populate match_dist matrix of mahalanobis_dist between every detection and every track
        match_dist = [] # matrix of probability of matching between all people and all detections.   
        eligable_detections = [] # Only include detections in match_dist matrix if they're in range of at least one track to speed up munkres
        for detect in objects_detected: 
            at_least_one_track_in_range = False
            new_row = []
            for track in objects_tracked:
                # Use mahalanobis dist to do matching
                cov = track.filtered_state_covariances[0][0] + track.var_obs # cov_xx == cov_yy == cov
                mahalanobis_dist = math.sqrt(((detect.pos_x-track.pos_x)**2 + (detect.pos_y-track.pos_y)**2)/cov) # = scipy.spatial.distance.mahalanobis(u,v,inv_cov)**2
                if mahalanobis_dist < self.mahalanobis_dist_gate:
                    cost = mahalanobis_dist
                    at_least_one_track_in_range = True
                else:
                    cost = self.max_cost 
                new_row.append(cost)                    
            # If the detection is within range of at least one track, add it as an eligable detection in the munkres matching 
            if at_least_one_track_in_range: 
                match_dist.append(new_row)
                eligable_detections.append(detect)

        # Run munkres on match_dist to get the lowest cost assignment
        if match_dist:
            elig_detect_indexes, track_indexes = linear_sum_assignment(match_dist)
            for elig_detect_idx, track_idx in zip(elig_detect_indexes, track_indexes):
                if match_dist[elig_detect_idx][track_idx] < self.mahalanobis_dist_gate:
                    detect = eligable_detections[elig_detect_idx]
                    track = objects_tracked[track_idx]
                    matched_tracks[track] = detect

        return matched_tracks

      
    def detected_clusters_callback(self, detected_clusters_msg):    
        """
        Callback for every time detect_leg_clusters publishes new sets of detected clusters. 
        It will try to match the newly detected clusters with tracked clusters from previous frames.
        """
        # Waiting for the local map to be published before proceeding. This is ONLY needed so the benchmarks are consistent every iteration

        now = detected_clusters_msg.header.stamp
       
        detected_clusters = []
        detected_clusters_set = set()
        for cluster in detected_clusters_msg.poses:
            new_detected_cluster = DetectedCluster(
                cluster.pose.position.x, 
                cluster.pose.position.y, 
                0.9,
                cluster.pose.orientation.x,
                cluster.pose.orientation.y
            )
            detected_clusters.append(new_detected_cluster)
            detected_clusters_set.add(new_detected_cluster)  
      
        # Propogate existing tracks
        to_duplicate = set()
        propogated = copy.deepcopy(self.objects_tracked)
        for propogated_track in propogated:
            propogated_track.update(np.ma.masked_array(np.array([0, 0]), mask=[1,1])) 
            to_duplicate.add(propogated_track)
       
        # Duplicate tracks of people so they can be matched twice in the matching
        duplicates = {}
        for propogated_track in to_duplicate:
            propogated.append(copy.deepcopy(propogated_track))
            duplicates[propogated_track] = propogated[-1]

        # Match detected objects to existing tracks
        matched_tracks = self.match_detections_to_tracks_GNN(propogated, detected_clusters)  
  

        # Update all tracks with new oberservations 
        tracks_to_delete = set()   
        for idx, track in enumerate(self.objects_tracked):
            propogated_track = propogated[idx] # Get the corresponding propogated track
            if propogated_track in matched_tracks and duplicates[propogated_track] in matched_tracks:
                # Two matched legs for this person. Create a new detected cluster which is the average of the two
                md_1 = matched_tracks[propogated_track]
                md_2 = matched_tracks[duplicates[propogated_track]]
                matched_detection = DetectedCluster((md_1.pos_x+md_2.pos_x)/2., (md_1.pos_y+md_2.pos_y)/2., (md_1.confidence+md_2.confidence)/2.,(md_1.vel_x+md_2.vel_x)/2.,(md_1.vel_y+md_2.vel_y)/2.)
            elif propogated_track in matched_tracks:
                # Only one matched leg for this person
                md_1 = matched_tracks[propogated_track]
                md_2 = duplicates[propogated_track]
                matched_detection = DetectedCluster((md_1.pos_x+md_2.pos_x)/2., (md_1.pos_y+md_2.pos_y)/2., md_1.confidence,(md_1.vel_x+md_2.vel_x)/2.,(md_1.vel_y+md_2.vel_y)/2.)                    
            elif duplicates[propogated_track] in matched_tracks:
                # Only one matched leg for this person 
                md_1 = matched_tracks[duplicates[propogated_track]]
                md_2 = propogated_track
                matched_detection = DetectedCluster((md_1.pos_x+md_2.pos_x)/2., (md_1.pos_y+md_2.pos_y)/2., md_1.confidence, (md_1.vel_x+md_2.vel_x)/2., (md_1.vel_y+md_2.vel_y)/2.)                                        
            else:      
                # No legs matched for this person 
                matched_detection = None  

            if matched_detection:
                observations = np.array([matched_detection.pos_x, 
                                         matched_detection.pos_y]) 
                track.confidence = 0.95*track.confidence + 0.05*matched_detection.confidence                                       
                track.times_seen += 1
                track.last_seen = now
                track.seen_in_current_scan = True
            else: # propogated_track not matched to a detection
                # don't provide a measurement update for Kalman filter 
                # so send it a masked_array for its observations
                observations = np.ma.masked_array(np.array([0, 0]), mask=[1,1]) 
                track.seen_in_current_scan = False
                        
            # Input observations to Kalman filter
            track.update(observations)

            # Check track for deletion           
            if  track.confidence < self.confidence_threshold_to_maintain_track:
                tracks_to_delete.add(track)
                # rospy.loginfo("deleting due to low confidence")
            else:
                # Check track for deletion because covariance is too large
                cov = track.filtered_state_covariances[0][0] + track.var_obs # cov_xx == cov_yy == cov
                if cov > self.max_cov:
                    tracks_to_delete.add(track)
                    # rospy.loginfo("deleting because unseen for %.2f", (now - track.last_seen).to_sec())

        # Delete tracks that have been set for deletion
        for track in tracks_to_delete:         
            track.deleted = True # Because the tracks are also pointed to in self.potential_leg_pairs, we have to mark them deleted so they can deleted from that set too
            self.objects_tracked.remove(track)
            
        # If detections were not matched, create a new track  
        for detect in detected_clusters:      
            if not detect in matched_tracks.values():
                self.objects_tracked.append(ObjectTracked(detect.pos_x, detect.pos_y, now, detect.confidence, detect.vel_x, detect.vel_y))
 

        # Publish to rviz and /people_tracked topic.
        self.publish_tracked_people(now)
            

    def publish_tracked_people(self, now):
        """
        Publish markers of tracked people to Rviz and to <people_tracked> topic
        """  
        
        pose_msg = PoseWithCovArray()
        pose_msg.header.stamp = now
        pose_msg.header.frame_id = self.publish_people_frame        
        marker_id = 0

        # Make sure we can get the required transform first:
        if self.use_scan_header_stamp_for_tfs:
            tf_time = now
            try:
                self.listener.waitForTransform(self.publish_people_frame, self.fixed_frame, tf_time, rospy.Duration(1.0))
                transform_available = True
            except:
                transform_available = False
        else:
            tf_time = rospy.Time(0)
            transform_available = self.listener.canTransform(self.publish_people_frame, self.fixed_frame, tf_time)

        marker_id = 0
        if not transform_available:
            rospy.loginfo("Person tracker: tf not avaiable. Not publishing people")
        else:
            for person in self.objects_tracked:
                if self.publish_occluded or person.seen_in_current_scan: # Only publish people who have been seen in current scan, unless we want to publish occluded people
                    # Get position in the <self.publish_people_frame> frame 
                    ps = PointStamped()
                    ps.header.frame_id = self.fixed_frame
                    ps.header.stamp = tf_time
                    ps.point.x = person.pos_x
                    ps.point.y = person.pos_y
                    try:
                        ps = self.listener.transformPoint(self.publish_people_frame, ps)
                    except:
                        rospy.logerr("Not publishing people due to no transform from fixed_frame-->publish_people_frame")                                                
                        continue
                    
                    # publish to leg_pose topic
                    new_pose = PoseWithCov() 
                    new_pose.pose.position.x = ps.point.x 
                    new_pose.pose.position.y = ps.point.y 
                    yaw = math.atan2(person.vel_y, person.vel_x)
                    quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
                    new_pose.pose.orientation.x = person.vel_x # vel_x
                    new_pose.pose.orientation.y = person.vel_y # vel_y
                    new_pose.pose.orientation.z = quaternion[2]
                    new_pose.pose.orientation.w = quaternion[3] 
                    new_pose.id = person.id_num
                    new_pose.xCov = person.pos_x_cov
                    new_pose.yCov = person.pos_y_cov
                    new_pose.vxCov = person.vel_x_cov
                    new_pose.vyCov = person.vel_y_cov
                    pose_msg.poses.append(new_pose)
                    # publish rviz markers
                    # Cylinder for body 
                    marker = Marker()
                    marker.header.frame_id = self.publish_people_frame
                    marker.header.stamp = now
                    marker.ns = "People_tracked"
                    marker.color.r = person.colour[0]
                    marker.color.g = person.colour[1]
                    marker.color.b = person.colour[2]
                    marker.color.a = (rospy.Duration(3) - (rospy.get_rostime() - person.last_seen)).to_sec()/rospy.Duration(3).to_sec() + 0.1
                    marker.pose.position.x = ps.point.x 
                    marker.pose.position.y = ps.point.y
                    marker.id = marker_id 
                    marker_id += 1
                    marker.type = Marker.CYLINDER
                    marker.scale.x = 0.2
                    marker.scale.y = 0.2
                    marker.scale.z = 1.2
                    marker.pose.position.z = 0.8
                    self.marker_pub.publish(marker)  
                    
                    # Sphere for head shape
                    marker.type = Marker.SPHERE
                    marker.scale.x = 0.2
                    marker.scale.y = 0.2
                    marker.scale.z = 0.2                
                    marker.pose.position.z = 1.5
                    marker.id = marker_id 
                    marker_id += 1                        
                    self.marker_pub.publish(marker)     
                    
                    # Text showing person's ID number 
                    marker.color.r = 1.0
                    marker.color.g = 1.0
                    marker.color.b = 1.0
                    marker.color.a = 1.0
                    marker.id = marker_id
                    marker_id += 1
                    marker.type = Marker.TEXT_VIEW_FACING
                    marker.text = str(person.id_num)
                    marker.scale.z = 0.2         
                    marker.pose.position.z = 1.7
                    self.marker_pub.publish(marker)
                    
                    # Arrow pointing in direction they're facing with magnitude proportional to speed
                    marker.color.r = person.colour[0]
                    marker.color.g = person.colour[1]
                    marker.color.b = person.colour[2]          
                    marker.color.a = (rospy.Duration(3) - (rospy.get_rostime() - person.last_seen)).to_sec()/rospy.Duration(3).to_sec() + 0.1                        
                    start_point = Point()
                    end_point = Point()
                    start_point.x = marker.pose.position.x 
                    start_point.y = marker.pose.position.y 
                    end_point.x = start_point.x + 0.5*person.vel_x
                    end_point.y = start_point.y + 0.5*person.vel_y
                    marker.pose.position.x = 0.
                    marker.pose.position.y = 0.
                    marker.pose.position.z = 0.1
                    marker.id = marker_id
                    marker_id += 1
                    marker.type = Marker.ARROW
                    marker.points.append(start_point)
                    marker.points.append(end_point)
                    marker.scale.x = 0.05
                    marker.scale.y = 0.1
                    marker.scale.z = 0.2
                    self.marker_pub.publish(marker)                           
                    
                    # <self.confidence_percentile>% confidence bounds of person's position as an ellipse:
                    cov = person.filtered_state_covariances[0][0] + person.var_obs # cov_xx == cov_yy == cov
                    std = cov**(1./2.)
                    gate_dist_euclid = scipy.stats.norm.ppf(1.0 - (1.0-self.confidence_percentile)/2., 0, std)
                    marker.pose.position.x = ps.point.x 
                    marker.pose.position.y = ps.point.y                    
                    marker.type = Marker.SPHERE
                    marker.scale.x = 2*gate_dist_euclid
                    marker.scale.y = 2*gate_dist_euclid
                    marker.scale.z = 0.01   
                    marker.color.r = person.colour[0]
                    marker.color.g = person.colour[1]
                    marker.color.b = person.colour[2]            
                    marker.color.a = 0.1
                    marker.pose.position.z = 0.0
                    marker.id = marker_id 
                    marker_id += 1                    
                    self.marker_pub.publish(marker)    
                    
                    # Hall Model Visualization
                    marker.pose.position.x = ps.point.x 
                    marker.pose.position.y = ps.point.y                    
                    marker.type = Marker.SPHERE
                    marker.scale.x = 0.5
                    marker.scale.y = 0.5
                    marker.scale.z = 0.02   
                    marker.color.r = 0.75
                    marker.color.g = 0
                    marker.color.b = 0            
                    marker.color.a = 0.9
                    marker.pose.position.z = 0.0
                    marker.id = marker_id 
                    marker_id += 1                    
                    self.marker_pub.publish(marker)
                    marker.pose.position.x = ps.point.x 
                    marker.pose.position.y = ps.point.y                    
                    marker.type = Marker.SPHERE
                    marker.scale.x = 1
                    marker.scale.y = 1
                    marker.scale.z = 0.01   
                    marker.color.r = 0.15
                    marker.color.g = 0.5
                    marker.color.b = 0
                    marker.color.a = 0.5
                    marker.pose.position.z = 0.0
                    marker.id = marker_id 
                    marker_id += 1                    
                    self.marker_pub.publish(marker)   
                    marker.pose.position.x = ps.point.x 
                    marker.pose.position.y = ps.point.y                    
                    marker.type = Marker.SPHERE
                    marker.scale.x = 4
                    marker.scale.y = 4
                    marker.scale.z = 0.01   
                    marker.color.r = 0.15
                    marker.color.g = 0.5
                    marker.color.b = 0            
                    marker.color.a = 0.2
                    marker.pose.position.z = 0.0
                    marker.id = marker_id 
                    marker_id += 1
                    self.marker_pub.publish(marker)
                    print("Number of People Detection: ")
                    print(len(pose_msg.poses)) 

        # Clear previously published people markers
        for m_id in xrange(marker_id, self.prev_person_marker_id):
            marker = Marker()
            marker.header.stamp = now                
            marker.header.frame_id = self.publish_people_frame
            marker.ns = "People_tracked"
            marker.id = m_id
            marker.action = marker.DELETE   
            self.marker_pub.publish(marker)
        self.prev_person_marker_id = marker_id           
        
        # Publish leg_pose message
        self.people_pose_pub.publish(pose_msg)           


if __name__ == '__main__':
    rospy.init_node('multi_people_tracker', anonymous=True)
    kmt = KalmanMultiTracker()





