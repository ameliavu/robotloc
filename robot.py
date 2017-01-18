#!/usr/bin/env python
import rospy
from read_config import read_config
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from map_utils import *
from helper_functions import * 
import random
from numpy import sqrt, pi, exp
from math import *
import numpy as np
from copy import deepcopy
from sklearn.neighbors import KDTree

class Particle():
	def __init__(self, x, y, theta, weight):
		self.x = x
		self.y = y
		self.theta = theta
		self.weight = weight

class Robot():
	def __init__(self):
		self.config = read_config()
		self.n = self.config['num_particles']
#		self.n = 100
		self.seed = self.config['seed']
		self.sigma = self.config['laser_sigma_hit']
		self.moves = self.config['move_list']
		self.move_index = 0
		self.width = 0
		self.height = 0
		self.min_weight = 0
		self.particle_array = []
		self.true_map = 0
		self.likelihood_map = 0
		self.latest_scan = None
		rospy.init_node("robot")
		self.pose_publisher = rospy.Publisher(
			"/particlecloud",
			PoseArray,
			queue_size = 10,
			latch = True
		)
		self.likelihood_publisher = rospy.Publisher(
			"/likelihood_field",
			OccupancyGrid,
			queue_size = 10,
			latch = True
		)
		self.result_publisher = rospy.Publisher(
			"/result_update",
			Bool,
			queue_size = 10
		)
		self.complete_publisher = rospy.Publisher(
			"/sim_complete",
			Bool,
			queue_size = 10
		)
		self.map_subscriber = rospy.Subscriber(
			"/map",
			OccupancyGrid,
			self.get_map
		)
		self.scan_subscriber = rospy.Subscriber(
			"/base_scan",
			LaserScan,
			self.save_scan
		)
		rospy.spin()
	def get_map(self, msg):
		self.width = msg.info.width
		self.height = msg.info.height
		pose_array = PoseArray()
		pose_array.header.stamp = rospy.Time.now()
		pose_array.header.frame_id = 'map'
		pose_array.poses =[]
#		random.seed(self.seed)
		weight = float(1.0/self.n)
		self.min_weight = weight
		for i in range (self.n):
			x = random.uniform(0, self.width)
			y = random.uniform(0, self.height)
			theta = random.uniform(0, 2*pi)
#			x = self.width / 2
#			y = self.height / 2
#			theta = 0
			pose = get_pose(x, y, theta)
			pose_array.poses.append(pose)
			particle = Particle(x, y, theta, weight)
			self.particle_array.append(particle)
		rospy.sleep(1)
		self.pose_publisher.publish(pose_array)
		self.true_map = Map(msg)
		self.likelihood_map = deepcopy(self.true_map)
		self.construct_likelihood()
		self.move()
		self.complete_publisher.publish(True)
		rospy.sleep(1)
		rospy.signal_shutdown("shutting down")
	def construct_likelihood(self):
		obstacles = []
		for i in range (self.width):
			for j in range (self.height):
				(x, y) = self.true_map.cell_position(j, i)
				val = self.true_map.get_cell(x, y)
				if val == 1:
					point = [x, y]
					obstacles.append(point)
		obstacles = np.array(obstacles)
		tree = KDTree(obstacles, leaf_size=2)
		for i in range (self.width):
			for j in range (self.height):
				(x,y) = self.likelihood_map.cell_position(j, i)
				dist, index = tree.query((x,y), 1)
				prob = exp(-(dist**2/2/(self.sigma**2)))
				self.likelihood_map.set_cell(x,y,prob)
		msg = self.likelihood_map.to_message()
		self.likelihood_publisher.publish(msg)
	def move(self):
		for move in self.moves:
			pose_array = PoseArray()
			pose_array.header.stamp = rospy.Time.now()
			pose_array.header.frame_id = 'map'
			pose_array.poses =[]
			move_function(move[0], 0)
			for particle in self.particle_array:
				new_theta = particle.theta + (move[0]*pi/180)
				if self.move_index == 0:
					particle.theta = new_theta + random.gauss(0, self.config['first_move_sigma_angle'])
				else:
					particle.theta = new_theta
				pose = get_pose(particle.x, particle.y, particle.theta) 
				pose_array.poses.append(pose)
			self.pose_publisher.publish(pose_array)
			self.update_weight()
			self.resample()
			for k in range (move[2]):
				move_function(0, move[1])
				pose_array = PoseArray()
				pose_array.header.stamp = rospy.Time.now()
				pose_array.header.frame_id = 'map'
				pose_array.poses =[]
				for particle in self.particle_array:
					new_x = particle.x + move[1] * cos(particle.theta)
					new_y = particle.y + move[1] * sin(particle.theta)
					if self.move_index == 0:
						particle.x = new_x + random.gauss(0, self.config['first_move_sigma_x'])
						particle.y = new_y + random.gauss(0, self.config['first_move_sigma_y'])
					else:
						particle.x = new_x
						particle.y = new_y
					pose = get_pose(particle.x, particle.y, particle.theta) 
					pose_array.poses.append(pose)
				self.pose_publisher.publish(pose_array)
				self.update_weight()
				self.resample()
			self.move_index = self.move_index + 1
#			self.update_weight()
#			self.resample()
			self.result_publisher.publish(True)
	def save_scan(self, msg):
		self.latest_scan = msg
	def update_weight(self):
		ranges = self.latest_scan.ranges
		angle_min = self.latest_scan.angle_min
		angle_max = self.latest_scan.angle_max
		angle_incr = self.latest_scan.angle_increment
		k = 0.0
		for particle in self.particle_array:
			p = self.likelihood_map.get_cell(particle.x, particle.y)
			# put them back to the map
			if isnan(p) or p == 1:
#				x = random.uniform(0, self.width)
#				y = random.uniform(0, self.height)
#				theta = random.uniform(0, 2*pi)
#				particle.weight = self.min_weight
#				k = k + particle.weight
				particle.weight = 0.0
				continue
			p_tot = 0.0
			angle = angle_min
			for measurement in ranges:
				end_theta = particle.theta + angle
				end_x = measurement * cos(end_theta) + particle.x
				end_y = measurement * sin(end_theta) + particle.y
				prob = self.likelihood_map.get_cell(end_x, end_y)
				if isnan(prob):
					continue
				new_prob = self.config['laser_z_hit'] * prob + self.config['laser_z_rand']
				p_tot = p_tot + new_prob**3
				angle = angle + angle_incr
			particle.weight = particle.weight * (1.0/(1.0+exp(-p_tot))) 
			k = k + particle.weight
		for particle in self.particle_array:
			particle.weight = float(particle.weight / k)
#			if particle.weight < self.min_weight:
#				self.min_weight = particle.weight
	def resample(self):

		spin = random.uniform(0, 1.0/self.n-0.000001)
		
		bound_map = []
		p_sum = 0.0
		index = 0
		for i in range(len(self.particle_array)):
			p_sum += self.particle_array[i].weight
			print "p_sum " + str(p_sum)
			bound_map.append(p_sum)
		new_particles = []
		k = 0.0
		pose_array = PoseArray()
		pose_array.header.stamp = rospy.Time.now()
		pose_array.header.frame_id = 'map'
		pose_array.poses =[]
		for i in range(len(self.particle_array)):
			rand_num = i * 1.0/self.n + spin
			# rand_num = random.uniform(0,1)
			for j in range(len(self.particle_array)):
				if bound_map[j] >= rand_num:
					index = j
					break
			picked = self.particle_array[index]
			new_x = picked.x + random.gauss(0, self.config['resample_sigma_x'])	
			new_y = picked.y + random.gauss(0, self.config['resample_sigma_y'])	
			new_theta = picked.theta + random.gauss(0, self.config['resample_sigma_angle'])
			new_particle = Particle(new_x, new_y, new_theta, picked.weight)
			k = k + picked.weight
			new_particles.append(new_particle)
			pose = get_pose(new_x, new_y, new_theta)
			pose_array.poses.append(pose)
		for p in new_particles:
			p.weight = float(p.weight/k)
		self.particle_array = deepcopy(new_particles)	
		# publish to particlecloud
		self.pose_publisher.publish(pose_array)
if __name__ == '__main__':
	robot = Robot()
