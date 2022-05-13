#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 15:22:54 2022

@author: dp
"""
import matplotlib.pyplot as plt
import numpy as np
def cameraOrientation(R, t):
    t = t.reshape(3)
    fig = plt.figure(figsize=(30,30))
    ax = fig.add_subplot(projection = '3d')
    
    
    VecStart_x1 = np.array([0,0,0])
    VecStart_y1 = np.array([0,0,0])
    VecStart_z1 = np.array([0,0,0])
    VecEnd_x1 = np.array(np.array([1,0,0]) * 20 + VecStart_x1)
    VecEnd_y1 = np.array(np.array([0,1,0]) * 20 + VecStart_y1)
    VecEnd_z1 = np.array(np.array([0,0,1]) * 20 + VecStart_z1)
    
    label = 'CAML'
    ax.scatter(VecStart_x1[0],VecStart_y1[0],VecStart_z1[0], color = 'g')
    ax.text(VecStart_x1[0],VecStart_y1[0],VecStart_z1[0], '%s' % (label), size = 20, zorder = 1, color = 'k')
    for i in range(3):
        ax.plot([VecStart_x1[i], VecEnd_x1[i]], [VecStart_y1[i],VecEnd_y1[i]],zs=[VecStart_z1[i],VecEnd_z1[i]])
        
    VecStart_x2 = np.array([t[2], t[2], t[2]]) + VecStart_x1
    VecStart_y2 = np.array([t[0], t[0], t[0]]) + VecStart_y1
    VecStart_z2 = np.array([t[1], t[1], t[1]]) + VecStart_z1
    VecEnd_x2 = np.dot(R,np.array([[20],[0],[0]])) + VecStart_x2.reshape((3,1))
    VecEnd_y2 = np.dot(R,np.array([[0],[20],[0]])) + VecStart_y2.reshape((3,1))
    VecEnd_z2 = np.dot(R,np.array([[0],[0],[20]])) + VecStart_z2.reshape((3,1))
    VecEnd_x2 = VecEnd_x2.reshape(3)
    VecEnd_y2 = VecEnd_y2.reshape(3)
    VecEnd_z2 = VecEnd_z2.reshape(3)
    
    
    label = 'CAMR'
    ax.scatter(VecStart_x2[0],VecStart_y2[0],VecStart_z2[0], color = 'g')
    ax.text(VecStart_x2[0],VecStart_y2[0],VecStart_z2[0], '%s' % (label), label = 'CAMR', size = 20, zorder = 1, color = 'k')
    
    for i in range(3):
        ax.plot([VecStart_x2[i], VecEnd_x2[i]], [VecStart_y2[i],VecEnd_y2[i]],zs=[VecStart_z2[i],VecEnd_z2[i]])
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')
    plt.show()