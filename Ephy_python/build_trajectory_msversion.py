# -*- coding: utf-8 -*-
"""
Created on Sun May  5 12:25:54 2019

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
from matplotlib.widgets import RectangleSelector
# %%
def remove_outlier(postemp,tstemp,arena,interp=True):  
    #coord_ll is the lower left coordinate
    x=arena['x']
    y=arena['y']
    if interp:
        posfilt=postemp[(postemp['x']>=x[0]) & (postemp['x']<=x[1]) &  (postemp['y']>=y[0]) &(postemp['y']<=y[1])]
        tsfilt=tstemp[(postemp['x']>=x[0]) & (postemp['x']<=x[1]) &  (postemp['y']>=y[0]) &(postemp['y']<=y[1])]
        fx=interpolate.interp1d(tsfilt.values,posfilt['x'].values,fill_value="extrapolate")
        fy=interpolate.interp1d(tsfilt.values,posfilt['y'].values,fill_value="extrapolate")
        xnew=fx(tstemp)
        ynew=fy(tstemp)
        postemp2=pd.DataFrame(data={'x':xnew,'y':ynew})
        tstemp2=tstemp
        num_outlier=(len(postemp)-len(posfilt))/len(postemp)
    else:
        postemp2=postemp[(postemp['x']>=x[0]) & (postemp['x']<=x[1]) &  (postemp['y']>=y[0]) &(postemp['y']<=y[1])]
        tstemp2=tstemp[(postemp['x']>=x[0]) & (postemp['x']<=x[1]) &  (postemp['y']>=y[0]) &(postemp['y']<=y[1])]
        num_outlier=(len(postemp)-len(postemp2))/len(postemp)
    
    
    return postemp2,tstemp2,num_outlier

def check_tracking(pos):
    f, ax = plt.subplots(1,len(pos['trial']), sharey=True, sharex=True)
    f.set_size_inches(12,2)
    for i,postemp in enumerate(pos['trial']):
        ax[i].set_aspect(1)
        ax[i].plot(postemp['x'],postemp['y'],'.-',markersize=0.1) 
        ax[i].autoscale(enable=True, axis='both', tight=True)

#main function 
def build_pos(filepath,body_part='Body',reset_arena=False):       
    #filepath=r'C:\Users\Qixin\XuChunLab\Ephy_python\nexdata\192043'
    frame_start=[60,141,39,30,28,27,19,23]
    
    filelist=sorted(glob.glob(filepath+'\*h5'))
    videotslist=sorted(glob.glob(filepath+'\*txt'))
    extensions=('*.asf','*.avi','*.mp4')
    if len(videotslist)==0:
        videotslist=sorted(glob.glob(filepath+'\*dat'))
        tstype='dat'
    else:
        tstype='txt'
    samplevideo=[]
    videos=[]
    for extension in extensions:
        videos.append(glob.glob(filepath+'/'+extension))
    samplevideo=sorted(videos)[-1][0]
    vidcap=cv2.VideoCapture(samplevideo)
    success,img = vidcap.read()  
    #default arena:  x is the minimal and maximal range of x, same with the y
    coord_ll=(50,60)
    r_width=110
    r_height=268
    if reset_arena:
        coordll,r_height,r_width=set_arena_by_trajectory(filepath)
    ratio=r_width/20 #unit in pixel/cm
    arena={'x':(coord_ll[0],coord_ll[0]+r_width),'y':(coord_ll[1],coord_ll[1]+r_height),'ratio':ratio}
    pos={'trial':[],'all':[],'arena':arena,'FPS':None}
    dt_raw=[]
    for pos_file,ts_file,fstart in zip(filelist,videotslist,frame_start):
        pos_df=pd.read_hdf(pos_file)
        postemp_raw=pos_df[pos_df.columns.levels[0][0]][body_part]
        #add video timestamp
        #vid_ts=pd.read_csv(ts_file, sep=" ", names='t')
        if tstype=='txt':
            vid_ts=pd.read_csv(ts_file,encoding='utf-16',header=None,names='t')
            tstemp=vid_ts['t']
        elif tstype=='dat':
            vid_ts=pd.read_csv(ts_file,header=0,sep='\s+')['sysClock']
            vid_ts=vid_ts/1000
            vid_ts[0]=0
            tstemp=vid_ts
        dt_raw.append(np.diff(tstemp.values))
        #remove the points outside of the defined arena

        #postemp,ts_outlier_filtered,num_outlier=remove_outlier(postemp_raw,tstemp,arena)
        tstemp2=tstemp.iloc[fstart+1:]
        postemp_raw2=postemp_raw.iloc[fstart+1:]
        postemp,ts_outlier_filtered,num_outlier=remove_outlier(postemp_raw2,tstemp2,arena)
        postemp['ts']=ts_outlier_filtered
        postemp['speed']=np.concatenate([[0],np.linalg.norm(np.stack((np.diff(postemp['x']),np.diff(postemp['y'])),axis=1),axis=1)/np.diff(postemp['ts'])/ratio],axis=0)
        postemp['hd']=np.concatenate([[0],np.arctan2(np.diff(postemp['y']),np.diff(postemp['x']))* 180 / np.pi])   
        #postemp['hd']=np.concatenate([[0],np.arctan2(np.diff(postemp['y']),np.diff(postemp['x']))])  
        pos['trial'].append(postemp)
       # print('{:.1%}'.format(num_outlier)+ ' position outliers have been removed')
    pos['all']=pd.concat(pos['trial'],ignore_index=True)
    pos['FPS']=int(1/np.median(np.concatenate(np.array(dt_raw),axis=0)))
    return pos 


        
def set_arena_by_trajectory(filepath,body_part='Body'):
    filelist=sorted(glob.glob(filepath+'\*h5'))
    pos_list=[]
    for pos_file in filelist:
        pos_df=pd.read_hdf(pos_file)
        postemp_raw=pos_df[pos_df.columns.levels[0][0]][body_part]
        pos_list.append(postemp_raw)
    pos_all=pd.concat(pos_list,ignore_index=True)
    fig=plt.figure()
    ax=plt.axes([0,0,1,1],frameon=False)
    ax.scatter(pos_all['x'],pos_all['y'])
    plt.show()
    fig.savefig(os.path.join(filepath,'arena.png'))
    plt.close()
    img=cv2.imread(os.path.join(filepath,'arena.png'))
    coordll,r_height,r_width=set_arena(img)
    return coordll,r_height,r_width
    
    


        
if __name__=='__main__':
    filepath=r'D:\EPHY_Qiushou\Ephy_data_organized\M028\20181112'
    mypos=build_pos(filepath,reset_arena=False,body_part='Body')
    check_tracking(mypos)


        

#frame_start for 20181112: [60,141,39,30,28,27,19,23]

#%%
#X=mypos['trial'][0]['x'][:5].values
#Y=mypos['trial'][0]['y'][:5].values
#T=mypos['trial'][0]['hd'][:5].values
#S=mypos['trial'][0]['speed'][:5].values
#plt.plot(X,Y)