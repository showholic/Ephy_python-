# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 18:20:02 2019

@author: Qixin
"""
import pandas as pd
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
#%%
pathname=r'Y:\ChenHaoshan\11. fiber photometry\Qixin Yang'
sig_df=pd.read_csv(os.path.join(pathname,'#191113_conditioning.csv'))
behav_df=pd.read_csv(os.path.join(pathname,'#191113_conditioning_XY.csv')) 
extensions=('*.asf','*.avi','*.mp4')
videos=[]
for extension in extensions:
    videos.append(glob.glob(pathname+'/'+extension))          
videofile=sorted(videos)[-1][0]      
                  
#%%
sigz=zscore(sig_df['RawF_1'])
plt.figure()
plt.plot(sigz)
#%%
behav_df=behav_df[:-1]
output_video=os.path.join(pathname,'sig.avi')        
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video,fourcc,30,(640,480))        
cap=cv2.VideoCapture(videofile)
rect,frame=cap.read()
engram=np.zeros(np.shape(frame))
cap.release()
cap=cv2.VideoCapture(videofile)
i=0
ratio=0.12
animal_size=10
while(cap.isOpened()):
    rect,frame=cap.read()
    if rect==True:
        if sigz[i]>=2:
            frame=cv2.circle(frame,(int(behav_df['X'][i]/ratio), int(behav_df['Y'][i]/ratio)),int(sigz[i]),(0,255,0), -1)
            engram[int(behav_df['Y'][i]/ratio)-animal_size:int(behav_df['Y'][i]/ratio)+animal_size,
                    int(behav_df['X'][i]/ratio)-animal_size:int(behav_df['X'][i]/ratio)+animal_size,
                        0]+=int(sigz[i])*5
        frame=cv2.addWeighted(frame,1,engram.astype('uint8'),1,0)
        out.write(frame)
            
        cv2.imshow('frame',frame)
        i+=1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
out.release()
cv2.destroyAllWindows()         