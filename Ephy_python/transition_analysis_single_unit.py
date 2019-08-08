# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 14:07:49 2019

@author: Qixin
"""

import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from build_unit import *
from scipy.stats import zscore
from draw_arena import set_arena
from scipy.signal import find_peaks
from scipy import stats

#%%
pathname=r'C:\Users\Qixin\OneDrive\Lab_stuff\EPHY_Qiushou\Ctx_discrimination_data\192043\20190311'
filelist=glob.glob(os.path.join(pathname,'*.pkl'))

#%%
ensemble=[]
for filename in filelist:
    ensemble.append(pd.read_pickle(filename)) 
#%% delete interneuron
ptdur=[unit.waveform.peak_trough_dur for unit in ensemble]
maxfr=[np.max(np.stack([unit.ctx[0]['fr'] ,unit.ctx[1]['fr']],axis=0)) for unit in ensemble]
plt.figure()
delete_ind=[]
plt.scatter(ptdur,maxfr) 
for n,unit in enumerate(ensemble):
    if (ptdur[n]<=0.4) & (maxfr[n]>=5):
        delete_ind.append(n)
for ind in delete_ind:
    del ensemble[ind]

#%% make videos to visualize transition period (one unit at a time)
extensions=('*.asf','*.avi','*.mp4')
videos=[]
for extension in extensions:
    videos.append(glob.glob(pathname+'/'+extension))
new_dir=os.path.join(os.path.join(pathname,'temp_spike_img'),myUnit.name)
if not os.path.exists(new_dir):
    os.makedirs(new_dir)
for num_trial in range(8):    
    videofile=sorted(videos)[-1][num_trial]
    output_video=os.path.join(new_dir,myUnit.name+'_%d.avi'%(num_trial+1))        
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video,fourcc,30,(640,480))        
    cap=cv2.VideoCapture(videofile)
    rect,frame=cap.read()
    engram=np.zeros(np.shape(frame))
    cap.release()
    cap=cv2.VideoCapture(videofile)
    i=0       
    
    sigz=zscore(trial[num_trial]['fr_bs'])
    
    animal_size=10
    while(cap.isOpened()):
        rect,frame=cap.read()
    #    frame=cv2.flip(frame,flipCode=1)
        if rect==True:
            if sigz[i]>=1:
                frame=cv2.circle(frame,(int(myUnit.pos['trial'][num_trial]['x'][i]), int(myUnit.pos['trial'][num_trial]['y'][i])),int(sigz[i])*5,(0,255,0), -1)
                engram[int(myUnit.pos['trial'][num_trial]['y'][i])-animal_size:int(myUnit.pos['trial'][num_trial]['y'][i])+animal_size,
                    int(myUnit.pos['trial'][num_trial]['x'][i])-animal_size:int(myUnit.pos['trial'][num_trial]['x'][i])+animal_size,
                        0]+=int(sigz[i])*5
            frame=cv2.addWeighted(frame,1,engram.astype('uint8'),1,0)
            out.write(frame)
            
            #cv2.imshow('frame',frame)
            i+=1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break         
    cap.release()
    out.release()
    cv2.destroyAllWindows()

arena=myUnit.pos['arena']
plt.figure()
plt.imshow(engram.astype('uint8')[arena['y'][0]:arena['y'][1],arena['x'][0]:arena['x'][1],:])

#%% set door area 
extensions=('*.asf','*.avi','*.mp4')
videos=[]
for extension in extensions:
    videos.append(glob.glob(pathname+'/'+extension))
videofile=sorted(videos)[-1][0]
cap=cv2.VideoCapture(videofile)
rect,frame=cap.read()
coord_ll,r_height,r_width=set_arena(frame)
door={'x':(coord_ll[0],coord_ll[0]+r_width),'y':(coord_ll[1],coord_ll[1]+r_height)}
plt.figure()
plt.imshow(frame[door['y'][0]:door['y'][1],door['x'][0]:door['x'][1],:])
np.sum(frame[door['y'][0]:door['y'][1],door['x'][0]:door['x'][1],:])
cap.release()
#%% detect door open and door close

lum_all=[]
for num_trial in range(8):    
    videofile=sorted(videos)[-1][num_trial]
    cap=cv2.VideoCapture(videofile)
    lum=[]
    while(cap.isOpened()):
        rect,frame=cap.read()
        if rect==True:
            lum.append(np.sum(frame[door['y'][0]:door['y'][1],door['x'][0]:door['x'][1],:]))
        else:
            break
    
    #plt.figure()
    #plt.plot(gaussian_filter1d(lum,11))    
    lum_all.append(lum)
#%%
door_marker={'open':[],'close':[]} 
thres_all=[]
for lum in lum_all:
    lum=gaussian_filter1d(lum,11)
    thres=np.median(lum)-2*np.std(lum)
    cross=np.where(lum<=thres)[0]
    door_marker['open'].append(np.extract(cross>=1800,cross)[0])
    door_marker['close'].append(cross[np.where(cross<=len(lum)-5*60*30)[0][-1]])
    thres_all.append(thres)   
def verify_doormarker(door_marker,videos,lum_all,thres_all):
    fig,ax=plt.subplots(3,8)
    for num_trial in range(8):
        door_open=door_marker['open'][num_trial]
        door_close=door_marker['close'][num_trial]
        videofile=sorted(videos)[-1][num_trial]
        cap=cv2.VideoCapture(videofile)
        cap.set(1,door_open)
        rect,frame=cap.read()
        ax[0,num_trial].imshow(frame,origin='lower')
        ax[0,num_trial].set_title('frame %d'%door_open)
            
        cap.set(1,door_close)
        rect,frame=cap.read()
        ax[1,num_trial].imshow(frame,origin='lower')
        ax[1,num_trial].set_title('frame %d'%door_close)
            
        ax[2,num_trial].plot(gaussian_filter1d(lum_all[num_trial],11))
        ax[2,num_trial].axhline(thres_all[num_trial])
verify_doormarker(door_marker,videos,lum_all,thres_all)
door_marker['open']=np.multiply(door_marker['open'],(1/30))
door_marker['close']=np.multiply(door_marker['close'],(1/30))
door_marker['duration']=door_marker['close']-door_marker['open']
#%% detect enter context 
#1. extract positions from door open to door close 
def detect_enter_context(pos,pathname):
    extensions=('*.asf','*.avi','*.mp4')
    videos=[]
    for extension in extensions:
        videos.append(glob.glob(pathname+'/'+extension))
    videofile=sorted(videos)[-1][0]
    cap=cv2.VideoCapture(videofile)
    rect,frame=cap.read()
    
    fig=plt.figure()
    ax=plt.axes([0,0,1,1],frameon=False)
    ax.imshow(frame)
    ax.plot(pos['trial'][0]['x'],pos['trial'][0]['y'])
    fig.savefig(os.path.join(pathname,'transition.png'))
    plt.close()
    img=cv2.imread(os.path.join(pathname,'transition.png'))
    coord_ll,r_height,r_width=set_arena(img)
    buffer={'x':(coord_ll[0],coord_ll[0]+r_width),'y':(coord_ll[1],coord_ll[1]+r_height)}    
    enter_context={'ts':[],
                   'frame':[]}
    for t,trial in enumerate(pos['trial']):
        xtemp=trial['x']
        ytemp=trial['y']
        ttemp=trial['ts']
        enter_context['frame'].append(np.where((xtemp>=buffer['x'][0]) & (xtemp<=buffer['x'][1]) & (ytemp>=buffer['y'][0]) & (ytemp<=buffer['y'][1]))[0][-1])
        enter_context['ts'].append(ttemp[enter_context['frame'][t]])
    return enter_context

enter_context=detect_enter_context(myUnit.pos,pathname)
#%% 
for unit in ensemble:
    unit.marker.door=door_marker
save_all_units(ensemble,pathname)

#%%
def transition_response_delay(myUnit,postdoor=10,threshold=1.65,isfig=False):
    trial=myUnit.split_trial()
    res_delay={'A':[],'B':[]}
    if isfig:
        f, ax = plt.subplots(2,4)
    ia=0
    ib=0
    for num_trial in range(len(trial)):
        spkt=trial[num_trial]['spkt']
        spkt=np.extract(spkt<=door_marker['close'][num_trial]+postdoor,spkt)
        ifr,ts=binned_FR(spkt,door_marker['close'][num_trial]+postdoor,0.1,sigma=5)
        zfr=zscore(ifr)
        peaks,properties = find_peaks(zfr, height=threshold)
        tsig=ts[peaks]
        tsig_trans=np.extract((tsig>=door_marker['open'][num_trial]),tsig)
        if tsig_trans.any():
            delay=round(tsig_trans[0]-door_marker['open'][num_trial],2)
        else:
            delay=[]
        if trial[num_trial]['name']=='A':
            res_delay['A'].append(delay)
            if isfig:
                ax[0,ia].plot(ts,zfr)
                ax[0,ia].axvline(door_marker['open'][num_trial],color='g')
                ax[0,ia].axvline(door_marker['close'][num_trial],color='g')
                ax[0,ia].plot(ts[peaks], zfr[peaks], "x")
                ax[0,ia].eventplot(spkt,color='k',lineoffsets=-3)
            ia+=1
            
        elif trial[num_trial]['name']=='B':
            res_delay['B'].append(delay)
            if isfig:
                ax[1,ib].plot(ts,zfr)
                ax[1,ib].axvline(door_marker['open'][num_trial],color='g')
                ax[1,ib].axvline(door_marker['close'][num_trial],color='g')
                ax[1,ib].plot(ts[peaks], zfr[peaks], "x")
                ax[1,ib].eventplot(spkt,color='k',lineoffsets=-3)
            ib+=1
    return res_delay

delay=[]
for myUnit in ensemble:
    delay.append(transition_response_delay(myUnit,threshold=2.56))
             
    
#%% 
#myUnit=ensemble[9]    
#postdoor=10
#postdoor_open=10
#predoor_open=20
#trial=myUnit.split_trial()
#tbin=0.1
#z_postdooropen={'A':[],
#                'B':[],
#                'Aspkt':[],
#                'Bspkt':[],
#                'Aspkc':0,
#                'Bspkc':0}
#
#
#rasteroffset=-2
#linelength=0.5
#maxfr_threshold=20
#maxfr_overall=maxfr_threshold #threshold=20
#for num_trial in range(len(trial)):
#    spkt=trial[num_trial]['spkt']
#    spkt=np.extract(spkt<door_marker['close'][num_trial]+postdoor,spkt)
#    maxfr=np.max(binned_FR(spkt,door_marker['close'][num_trial]+postdoor,tbin,filtered=False)[0])
#    if maxfr>maxfr_overall:
#        maxfr_overall=maxfr
#    ifr,ts=binned_FR(spkt,door_marker['close'][num_trial]+postdoor,tbin,sigma=5)
#    zfr=zscore(ifr)
#    z_dooropen=np.extract((ts>door_marker['open'][num_trial]-predoor_open),zfr)[:int((predoor_open+postdoor_open)/tbin)+1]
#    spkt_dooropen=np.extract((spkt>door_marker['open'][num_trial]-predoor_open)&(spkt<=door_marker['open'][num_trial]+postdoor_open),spkt)-door_marker['open'][num_trial]
#    if trial[num_trial]['name']=='A':
#        z_postdooropen['A'].append(np.array(z_dooropen))
#        z_postdooropen['Aspkt'].append(spkt_dooropen)
#        z_postdooropen['Aspkc']+=np.size(spkt_dooropen)
#
#    elif trial[num_trial]['name']=='B':
#        z_postdooropen['B'].append(np.array(z_dooropen))
#        z_postdooropen['Bspkt'].append(spkt_dooropen)
#        z_postdooropen['Bspkc']+=np.size(spkt_dooropen)
#if maxfr_overall>maxfr_threshold:        
#    plt.figure()
#    z_postdooropen['ts']=np.arange(-predoor_open,postdoor_open+tbin,tbin)
#    z_postdooropen['A_mean']=np.nanmean(z_postdooropen['A'],0)
#    z_postdooropen['A_error']=stats.sem(z_postdooropen['A'],0,nan_policy='omit')
#    z_postdooropen['B_mean']=np.nanmean(z_postdooropen['B'],0)
#    z_postdooropen['B_error']=stats.sem(z_postdooropen['B'],0,nan_policy='omit')
#    
#    plt.plot(z_postdooropen['ts'],z_postdooropen['A_mean'],color='red')
#    plt.fill_between(z_postdooropen['ts'],z_postdooropen['A_mean']-z_postdooropen['A_error'],z_postdooropen['A_mean']+z_postdooropen['A_error'],alpha=0.5,facecolor='red')
#    plt.plot(z_postdooropen['ts'],z_postdooropen['B_mean'],color='green')
#    plt.fill_between(z_postdooropen['ts'],z_postdooropen['B_mean']-z_postdooropen['B_error'],z_postdooropen['B_mean']+z_postdooropen['B_error'],alpha=0.5,facecolor='green')
#    for i in range(4):
#        plt.eventplot(z_postdooropen['Aspkt'][i],color='red',lineoffsets=rasteroffset-i*linelength,linelengths=linelength)
#        plt.eventplot(z_postdooropen['Bspkt'][i],color='green',lineoffsets=rasteroffset-4*linelength-i*linelength,linelengths=linelength)
#        
#%%
new_dir=os.path.join(pathname,'transition_plot')
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

for myUnit in ensemble: 
    postdoor=20
    event=door_marker['open']
    postevent=5
    preevent=60
    trial=myUnit.split_trial()
    tbin=0.1
    z_perievent={'A':[],
                'B':[],
                'Aspkt':[],
                'Bspkt':[],
                'Aspkc':0,
                'Bspkc':0}
    
    rasteroffset=-2
    linelength=0.5
    maxfr_threshold=20
    maxfr_overall=maxfr_threshold #threshold=20
    for num_trial in range(len(trial)):
        spkt=trial[num_trial]['spkt']
        spkt=np.extract(spkt<door_marker['close'][num_trial]+postdoor,spkt)
        maxfr=np.max(binned_FR(spkt,door_marker['close'][num_trial]+postdoor,tbin,filtered=False)[0])
        if maxfr>maxfr_overall:
            maxfr_overall=maxfr
        ifr,ts=binned_FR(spkt,door_marker['close'][num_trial]+postdoor,tbin,sigma=5)
        ifr_baseline=np.extract((ts>=event[num_trial]-preevent)&(ts<event[num_trial]),ifr)
        z_baseline=zscore(ifr_baseline)
        mean_baseline=np.nanmean(ifr_baseline)
        ts_baseline=np.extract((ts>=event[num_trial]-preevent)&(ts<event[num_trial]),ts)-event[num_trial]
        std_baseline=np.nanstd(ifr_baseline)
        ifr_event=np.extract((ts>=event[num_trial]),ifr)[:int((postevent)/tbin)+1]
        z_event=np.divide((ifr_event-mean_baseline),std_baseline)
        ts_event=np.extract((ts>=event[num_trial]),ts)[:int((postevent)/tbin)+1]-event[num_trial]
        spkt_event=np.extract((spkt>event[num_trial]-preevent)&(spkt<=event[num_trial]+postevent),spkt)-event[num_trial]
        if trial[num_trial]['name']=='A':
            z_perievent['A'].append(np.concatenate((z_baseline,z_event)))
            z_perievent['Aspkt'].append(spkt_event)
            z_perievent['Aspkc']+=np.size(spkt_event)
    
        elif trial[num_trial]['name']=='B':
            z_perievent['B'].append(np.concatenate((z_baseline,z_event)))
            z_perievent['Bspkt'].append(spkt_event)
            z_perievent['Bspkc']+=np.size(spkt_event)
    
           
    fig, ax = plt.subplots()
    z_perievent['ts']=np.arange(-preevent,postevent+tbin,tbin)
    z_perievent['A_mean']=np.nanmean(z_perievent['A'],axis=0)
    z_perievent['A_error']=stats.sem(z_perievent['A'],axis=0,nan_policy='omit')
    z_perievent['B_mean']=np.nanmean(z_perievent['B'],axis=0)
    z_perievent['B_error']=stats.sem(z_perievent['B'],axis=0,nan_policy='omit')
    
    plt.plot(z_perievent['ts'],z_perievent['A_mean'],color='red')
    plt.fill_between(z_perievent['ts'],z_perievent['A_mean']-z_perievent['A_error'],z_perievent['A_mean']+z_perievent['A_error'],alpha=0.5,facecolor='red')
    plt.plot(z_perievent['ts'],z_perievent['B_mean'],color='green')
    plt.fill_between(z_perievent['ts'],z_perievent['B_mean']-z_perievent['B_error'],z_perievent['B_mean']+z_perievent['B_error'],alpha=0.5,facecolor='green')
    for i in range(4):
        plt.eventplot(z_perievent['Aspkt'][i],color='red',lineoffsets=rasteroffset-i*linelength,linelengths=linelength)
        plt.eventplot(z_perievent['Bspkt'][i],color='green',lineoffsets=rasteroffset-4*linelength-i*linelength,linelengths=linelength)
    plt.axvline(x=0,color='k',linestyle='--',alpha=0.5)   
    postevent1=postevent
    #plt.axvline(x=postevent1)
    plt.axhline(y=1.65,alpha=0.5)
    
    #
    postdoor=20
    event=door_marker['close']
    postevent=10
    preevent=5
    trial=myUnit.split_trial()
    tbin=0.1
    z_perievent={'A':[],
                'B':[],
                'Abaseline':[],
                'Bbaseline':[],
                'Aspkt':[],
                'Bspkt':[],
                'Aspkc':0,
                'Bspkc':0}
    
    rasteroffset=-2
    linelength=0.5
    maxfr_threshold=20
    maxfr_overall=maxfr_threshold #threshold=20
    for num_trial in range(len(trial)):
        spkt=trial[num_trial]['spkt']
        spkt=np.extract(spkt<door_marker['close'][num_trial]+postdoor,spkt)
        maxfr=np.max(binned_FR(spkt,door_marker['close'][num_trial]+postdoor,tbin,filtered=False)[0])
        if maxfr>maxfr_overall:
            maxfr_overall=maxfr
        ifr,ts=binned_FR(spkt,door_marker['close'][num_trial]+postdoor,tbin,sigma=5)
        ifr_baseline=np.extract((ts>=door_marker['open'][num_trial]-60)&(ts<door_marker['open'][num_trial]),ifr)
        z_baseline=zscore(ifr_baseline)
        mean_baseline=np.nanmean(ifr_baseline)
        std_baseline=np.nanstd(ifr_baseline)
        
        ifr_event=np.extract((ts>=event[num_trial]-preevent),ifr)[:int((preevent+postevent)/tbin)+1]
        z_event=np.divide((ifr_event-mean_baseline),std_baseline)
        ts_event=np.extract((ts>=event[num_trial]-preevent),ts)[:int((preevent+postevent)/tbin)+1]-event[num_trial]
        spkt_event=np.extract((spkt>event[num_trial]-preevent)&(spkt<=event[num_trial]+postevent),spkt)-event[num_trial]
        if trial[num_trial]['name']=='A':
            z_perievent['A'].append(z_event)
            z_perievent['Aspkt'].append(spkt_event)
            z_perievent['Aspkc']+=np.size(spkt_event)
    
        elif trial[num_trial]['name']=='B':
            z_perievent['B'].append(z_event)
            z_perievent['Bspkt'].append(spkt_event)
            z_perievent['Bspkc']+=np.size(spkt_event)
    gap=20
           
    z_perievent['ts']=np.arange(-preevent,postevent+tbin,tbin)+gap
    z_perievent['A_mean']=np.nanmean(z_perievent['A'],axis=0)
    z_perievent['A_error']=stats.sem(z_perievent['A'],axis=0,nan_policy='omit')
    z_perievent['B_mean']=np.nanmean(z_perievent['B'],axis=0)
    z_perievent['B_error']=stats.sem(z_perievent['B'],axis=0,nan_policy='omit')
    
    plt.plot(z_perievent['ts'],z_perievent['A_mean'],color='red')
    plt.fill_between(z_perievent['ts'],z_perievent['A_mean']-z_perievent['A_error'],z_perievent['A_mean']+z_perievent['A_error'],alpha=0.5,facecolor='red')
    plt.plot(z_perievent['ts'],z_perievent['B_mean'],color='green')
    plt.fill_between(z_perievent['ts'],z_perievent['B_mean']-z_perievent['B_error'],z_perievent['B_mean']+z_perievent['B_error'],alpha=0.5,facecolor='green')
    for i in range(4):
        plt.eventplot(z_perievent['Aspkt'][i]+gap,color='red',lineoffsets=rasteroffset-i*linelength,linelengths=linelength)
        plt.eventplot(z_perievent['Bspkt'][i]+gap,color='green',lineoffsets=rasteroffset-4*linelength-i*linelength,linelengths=linelength)
    #plt.axvline(x=gap-preevent)
    plt.axvline(x=gap,color='k',linestyle='--',alpha=0.5)
    
    ax.set_yticks([0,1.65])
    ax.set_xticks([-60,0,postevent1,gap-preevent,gap,gap+postevent])
    ax.set_xticklabels([-60,'open',postevent1,-preevent,'close',postevent])
    ax.set_ylim([-6,5])
    plt.savefig(os.path.join(new_dir,myUnit.animal_id+'_'+myUnit.name))

#%% 
    
