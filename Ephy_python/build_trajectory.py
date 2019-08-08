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
import pickle
from scipy.ndimage.filters import gaussian_filter1d

# %%
def remove_outlier(postemp,tstemp,arena,interp=True):  
    #coord_ll is the lower left coordinate
    x=arena['x']
    y=arena['y']
    if interp:
        posfilt=postemp[(postemp['x']>=x[0]) & (postemp['x']<=x[1]) &  (postemp['y']>=y[0]) &(postemp['y']<=y[1])]
        tsfilt=tstemp[(postemp['x']>=x[0]) & (postemp['x']<=x[1]) &  (postemp['y']>=y[0]) &(postemp['y']<=y[1])]
        fx=interpolate.interp1d(tsfilt.values,posfilt['x'].values,fill_value="extrapolate",kind='nearest')
        fy=interpolate.interp1d(tsfilt.values,posfilt['y'].values,fill_value="extrapolate",kind='nearest')
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

def speed_filt(postemp,ratio,sigma_t=0.25):
    s=np.concatenate([[0],np.linalg.norm(np.stack((np.diff(postemp['x']),np.diff(postemp['y'])),axis=1),axis=1)/np.diff(postemp['ts'])/ratio],axis=0)
    fps=int(1/np.median(np.diff(postemp['ts'].values)))
    s_filt=gaussian_filter1d(s,sigma_t/(1/fps))
    return s_filt
    
    
def check_tracking(pos):
    f, ax = plt.subplots(1,len(pos['trial']), sharey=True, sharex=True)
    f.set_size_inches(12,2)
    for i,postemp in enumerate(pos['trial']):
        ax[i].set_aspect(1)
        ax[i].plot(postemp['x'],postemp['y'],'.-',markersize=0.1) 
        ax[i].autoscale(enable=True, axis='both', tight=True)
        
def check_HD_tracking(poshead,posbody,postemp,samplevideo,pathname,nframe=10):
    framestart=3000
    step=50
    
    new_dir=os.path.join(pathname,'temp_img')
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
        
    vidcap=cv2.VideoCapture(samplevideo)   

    for f in range(nframe):
        fig,ax=plt.subplots()
        i=framestart+f*step
        vidcap.set(1,i)
        #fig.clf()
        flag,frame=vidcap.read()
        ax.imshow(frame,origin='lower')
        ax.scatter(poshead['x'][i],poshead['y'][i],s=50,c='g')
        ax.scatter(posbody['x'][i],posbody['y'][i],s=50,c='g')
        ax.set_title('HD = %.3f' %(postemp['hd'][i])) 
        img_name=os.path.join(new_dir, str(f) + '.png')
        plt.show()
        fig.savefig(img_name)
    plt.close('all')
        
    vidcap.release()
    cv2.destroyAllWindows() 
    
def check_speed(mypos,pathname,trial_index=0):
    #default to evaluate the first trial position with the corresponding video 
    postemp=mypos['trial'][trial_index]
    extensions=('*.asf','*.avi','*.mp4')
    samplevideo=[]
    videos=[]
    for extension in extensions:
        videos.append(glob.glob(pathname+'/'+extension))
    samplevideo=sorted(videos)[-1][trial_index]
    
    new_dir=os.path.join(pathname,'temp_speed_img')
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    output_video=os.path.join(new_dir,'speed_video.avi')
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video,fourcc,30,(640,480))    
    
    cap=cv2.VideoCapture(samplevideo)
    i=0       
    font = cv2.FONT_HERSHEY_SIMPLEX
    while(cap.isOpened()):
        rect,frame=cap.read()
        if rect==True:
            frame=cv2.circle(frame,(int(postemp['x'][i]),int(postemp['y'][i])),int(postemp['speed'][i]),(0,255,0), -1)            
            cv2.putText(frame,str(round(postemp['speed'][i],2)),(int(postemp['x'][i])+5,int(postemp['y'][i])+5), font,1,(255,0,0),2)
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


#main function 
def build_pos(filepath,body_part='Body',reset_arena=False,reset_arena_method='video',interp=True,filter_speed=True,check_HD=False):       
    #filepath=r'C:\Users\Qixin\XuChunLab\Ephy_python\nexdata\192043'
    
    filelist=sorted(glob.glob(filepath+'\*h5'))
    videotslist=sorted(glob.glob(filepath+'\*ts.txt'))
    if len(videotslist)==0:
        videotslist=sorted(glob.glob(filepath+'\*dat'))
        tstype='dat'
    else:
        tstype='txt'
    

    extensions=('*.asf','*.avi','*.mp4')
    samplevideo=[]
    videos=[]
    for extension in extensions:
        videos.append(glob.glob(filepath+'/'+extension))
    samplevideo=sorted(videos)[-1][0]
    vidcap=cv2.VideoCapture(samplevideo)
    success,img = vidcap.read()  
  
    if reset_arena:
        if reset_arena_method=='video':
            coord_ll,r_height,r_width=set_arena(img)
        elif reset_arena_method=='trajectory':    
            coord_ll,r_height,r_width=set_arena_by_trajectory(filepath,img,body_part=body_part)
        elif reset_arena_method=='manual':
            answer=input('Enter lower left coordinate and top right coordinate (e.g: 180,410,400,410) :').split(',')
            x1=int(answer[0])
            y1=int(answer[1])
            x2=int(answer[2])
            y2=int(answer[3])
            coord_ll=(x1,y1)
            r_height=abs(x2-x1)
            r_width=abs(y2-y1)
    else:
        try:
            arena_file=os.path.join(filepath,'arena_info.pkl')
            f = open(arena_file, 'rb')
            [coord_ll,r_height,r_width] = pickle.load(f)
            f.close()
        except:
            print('No existing arena information')
        
    ratio=r_width/20 #unit in pixel/cm
    arena={'x':(coord_ll[0],coord_ll[0]+r_width),'y':(coord_ll[1],coord_ll[1]+r_height),'ratio':ratio}
    pos={'trial':[],'all':[],'arena':arena,'FPS':None}
    dt_raw=[]
    for i,(pos_file,ts_file) in enumerate(zip(filelist,videotslist)):
        pos_df=pd.read_hdf(pos_file)
        postemp_raw=pos_df[pos_df.columns.levels[0][0]][body_part]
        pos_head=pos_df[pos_df.columns.levels[0][0]]['Head']
        pos_body=pos_df[pos_df.columns.levels[0][0]]['Body']
        #add video timestamp
        #vid_ts=pd.read_csv(ts_file, sep=" ", names='t')
        if tstype=='txt':
            try:
                vid_ts=pd.read_csv(ts_file,encoding='utf-16',header=None,names='t')
            except:
                vid_ts=pd.read_csv(ts_file,header=None,names='t')
            tstemp=vid_ts['t']
        elif tstype=='dat':
            vid_ts=pd.read_csv(ts_file,header=0,sep='\s+')['sysClock']
            vid_ts=vid_ts/1000
            vid_ts[0]=0
            tstemp=vid_ts
        dt_raw.append(np.diff(tstemp.values))
        #remove the points outside of the defined arena

        postemp,ts_outlier_filtered,num_outlier=remove_outlier(postemp_raw,tstemp,arena,interp=interp)
        
        postemp['ts']=ts_outlier_filtered
        if filter_speed:
            postemp['speed']=speed_filt(postemp,ratio)
            
        else:
            postemp['speed']=np.concatenate([[0],np.linalg.norm(np.stack((np.diff(postemp['x']),np.diff(postemp['y'])),axis=1),axis=1)/np.diff(postemp['ts'])/ratio],axis=0)
            
        postemp['hd']=np.arctan2(np.subtract(pos_head['y'],pos_body['y']),np.subtract(pos_head['x'],pos_body['x']))* 180 / np.pi   
        #postemp['hd']=np.concatenate([[0],np.arctan2(np.diff(postemp['y']),np.diff(postemp['x']))])  
        pos['trial'].append(postemp)
        print('{:.1%}'.format(num_outlier)+ ' position outliers have been removed')
        
        if check_HD:
            if i==0:
                check_HD_tracking(pos_head,pos_body,postemp,samplevideo,filepath)
    pos['all']=pd.concat(pos['trial'],ignore_index=True)
    pos['FPS']=int(1/np.median(np.concatenate(np.array(dt_raw),axis=0)))
    return pos 


        
def set_arena_by_trajectory(filepath,img,body_part='Body'):
    filelist=sorted(glob.glob(filepath+'\*h5'))
    pos_list=[]
    for pos_file in filelist:
        pos_df=pd.read_hdf(pos_file)
        postemp_raw=pos_df[pos_df.columns.levels[0][0]][body_part]
        pos_list.append(postemp_raw)
    pos_all=pd.concat(pos_list,ignore_index=True)
    fig=plt.figure()
    ax=plt.axes([0,0,1,1],frameon=False)
    ax.imshow(img)
    ax.scatter(pos_all['x'],pos_all['y'])
    plt.show()
    fig.savefig(os.path.join(filepath,'arena_img.png'))
    plt.close()
    img=cv2.imread(os.path.join(filepath,'arena_img.png'))
    coord_ll,r_height,r_width=set_arena(img)
    arena_file=os.path.join(filepath,'arena_info.pckl')
    f = open(arena_file, 'wb')
    pickle.dump([coord_ll,r_height,r_width], f)
    f.close()
    return coord_ll,r_height,r_width
        
if __name__=='__main__':
    pathname=r'C:\Users\Qixin\XuChunLab\nexdata\192043'
    mypos=build_pos(pathname,reset_arena=True,reset_arena_method='trajectory',body_part='Body',interp=True,check_HD=False)
    check_tracking(mypos)
    #%%
    #check_speed(mypos,pathname,2)

        

#frame_start for 20181112: [60,141,39,30,28,27,19,23]

#%%
#X=mypos['trial'][0]['x'][:5].values
#Y=mypos['trial'][0]['y'][:5].values
#T=mypos['trial'][0]['hd'][:5].values
#S=mypos['trial'][0]['speed'][:5].values
#plt.plot(X,Y)