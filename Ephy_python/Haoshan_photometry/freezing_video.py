# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 12:12:00 2019

@author: Qixin
"""
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
from draw_arena import set_arena
from scipy import interpolate
import os
import pickle
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.animation as animation
#%%
fileindex=0
pathname=r'Y:\ChenHaoshan\11. fiber photometry\video\20181021'
extensions=('*.asf','*.avi','*.mp4')
videos=[]
for extension in extensions:
    videos.append(glob.glob(pathname+'/'+extension))          
videofile=sorted(videos)[-1][fileindex]   
csvfile=os.path.join(pathname,os.path.split(videofile)[1][:-4]+'_freezing.csv')
#%%   
i=130
ttemp=ts[i]
freeze_data=pd.read_csv(csvfile)
fig, ax1 = plt.subplots()
left, bottom, width, height = [0.2, 0.5, 0.3,0.3]
ax2 = fig.add_axes([left, bottom, width, height])

ts=np.extract(~np.isnan(freeze_data['percentage']),freeze_data['ts(s)'])
freeze_sig=np.extract(~np.isnan(freeze_data['percentage']),freeze_data['percentage'])

xwindow=100
batch=i//xwindow
cap=cv2.VideoCapture(videofile)
rect,frame=cap.read()
ax1.imshow(frame)

ax2.plot(ts,freeze_sig,alpha=0.7)
ax2.axvline(ttemp,color='r')
ax2.set_xlim([batch*xwindow,(batch+1)*xwindow])
ax1.set_axis_off()
#ax2.set_axis_off()
plt.show()
#%%

#%%
# Setup figure and subplots
fig,ax=plt.subplots(1,2,figsize=(10,5))
ax[0].set_ylim(0,2)
ax[0].set_xlim(0,xwindow)
# Turn on grids
ax[0].grid(True)
# set label names
ax[0].set_xlabel("x")
ax[0].set_ylabel("freezing")

# Data Placeholders
yp1=np.zeros(0)
t=np.zeros(0)
# set plots
p,= ax[0].plot(t,yp1,'b-')


# Data Update
xmin = 0.0
xmax = xwindow

def updateData(i):
	global yp1
	global t    
	yp1=np.append(yp1,freeze_sig[i])
	t=np.append(t,ts[i])
    
	p.set_data(t,yp1)   
	if ts[i] >= xmax-1.00:
		p.axes.set_xlim(ts[i]-xmax+1.0,ts[i]+1.0)
	return p

simulation = animation.FuncAnimation(fig, updateData, blit=False, frames=2000, interval=30, repeat=False)

plt.show()
