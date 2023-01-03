# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 20:27:04 2020

@author: Tejas
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.linear_model import LinearRegression
#%%
class neuron:
    def __init__(self, preferred_direction, sigma):
        self.base_firing = np.random.uniform(0, 0.001)
        self.preferred_direction = preferred_direction
        self.sigma = sigma
        
    def get_firing_rate(self, direction_to_fire):
        firing = scipy.stats.norm(self.preferred_direction, self.sigma).pdf(direction_to_fire) + self.base_firing
        
        return firing

    def get_firing_rate_cartesian(self, direction, weight):
        r = weight*self.get_firing_rate(direction)
        theta = self.preferred_direction
        return np.array([r*np.cos(theta*np.pi/180), r*np.sin(theta*np.pi/180)])
    
#%%
neuron_population = np.zeros(360, dtype = object)
for i in range(360):
    neuron_population[i] = neuron(i, 30)
#%%    
weights = np.random.uniform(0, 1, 360)
results = np.zeros((360, 2))
for neuron in range(len(neuron_population)):
    results[neuron] = neuron_population[neuron].get_firing_rate_cartesian(60, weights[neuron]) 
origin = np.zeros(np.shape(results[::20]))
plt.quiver(origin[:, 0], origin[:, 1], results[::20,0], results[::20,1])
net_output = np.sum(results, axis = 0)
plt.quiver(0, 0, net_output[0], net_output[1], color = 'r')
#%%
required_mag = 2
required_direction = 60
required_movement = np.array([required_mag*np.cos(required_direction*np.pi/100), required_mag*np.sin(required_direction*np.pi/100)]) 
aiming_direction = 60

def produced_movement(neuron_population, aiming_direction, weights):
    for neuron in range(len(neuron_population)):
        results[neuron] = neuron_population[neuron].get_firing_rate_cartesian(aiming_direction, weights[neuron]) 
    return net_output
def error_produced_movement_required_movement(required_movement, neuron_population, aiming_direction, weights):
    actual_movement = produced_movement(neuron_population, aiming_direction, weights)
    return np.sum((required_movement - actual_movement)**2)


print(error_produced_movement_required_movement(required_movement, neuron_population, aiming_direction, weights))


