import nexfile as nex
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm 
from scipy.stats import norm
from scipy.stats import zscore
import build_trajectory as bt
from scipy import interpolate
from scipy.ndimage.filters import gaussian_filter1d
from scipy.ndimage import gaussian_filter
import copy
import glob
import pickle
import scipy.io
import cv2
from draw_arena import set_arena
# %%Class definition
class Marker:
    def __init__(self,record,door,protocol,enter_context):
        self.record=record
        self.door=door
        self.protocol=protocol
        self.enter_context=enter_context
class Waveform:
    def __init__(self,timestamps,values,sr):
        self.timestamps=timestamps
        self.values=values
        self.sr=sr
        self.peak_trough_dur=FindPTdur(self.values,self.sr)
    def Plotwf(self):
        avgwf=np.mean(self.values,axis=0)
        sdwf=np.std(self.values,axis=0)
        x=np.linspace(1,len(avgwf),len(avgwf))  #*(1/self.sr)*1000
        f,ax=plt.subplots()
        f.set_size_inches(8,2)
        ax.plot(x,avgwf)
        ax.fill_between(x,avgwf+sdwf,avgwf-sdwf,alpha=0.3)
        plt.show()
        plt.title('Peak-Trough duration: %.3f ms' %self.peak_trough_dur)

#Calculate the peak trough duration 
def FindPTdur(wfvalue,samplingrate):
    avgwf=np.mean(wfvalue,axis=0)
    avgquad=avgwf.reshape(4,-1)
    return (avgquad.min(axis=0).argmax()-avgquad.min(axis=0).argmin())*(1/samplingrate)*1000 #in ms
    
class Unit:
    def __init__(self,spktrain,marker,waveform,pos,name,date,animal_id,bootstrap): #marker and waveform are class as well 
        if pos:
            self.arena_ratio=pos['arena']['ratio']
            self.pos=pos
            
        self.spktrain=spktrain
        self.marker=marker
        self.waveform=waveform 
        self.ctx=self.split_context(pos)
        self.cdi=[calc_cdi(self.ctx),calc_cdi(self.ctx,method=1)]
        self.max_spkc_ctx=np.max([np.max(self.ctx[0]['spkc']),np.max(self.ctx[1]['spkc'])])
        self.name=name
        self.date=date
        self.animal_id=animal_id
        self.num_ctx=len(self.ctx[:-1]) #how many contexts
        self.num_max_trial=max([len(c['spkc']) for c in self.ctx][:-1])
        self.context_selectivity=[]
        self.FR_map=[]
        self.SI=[]
        self.pSI=[]
        self.placefield=[]
        if bootstrap:
            self.context_selectivity=bootstrap_context(self)       
            
                
                    
            
    def split_context(self,pos,plot_ctx=False,plot_buffer=False):
        context=[{} for x in range(len(np.unique(self.marker.protocol))+1)]

        #construct the explored context 
        for c,ctx in enumerate(np.unique(self.marker.protocol)):
            context[c]['name']=ctx
            context[c]['index']=np.where(np.asarray(self.marker.protocol)==ctx)[0] #find the indices of specific contexts 
            context[c]['spkt']=[]
            context[c]['spkc']=[]
            context[c]['dur']=[]
            context[c]['fr']=[]   
            context[c]['ctx_start']=[]
            context[c]['ctx_end']=[]
            
            if pos:
                context[c]['pos']=[]
                context[c]['spkpos']=[]
                context[c]['spkt_raw']=[]
                context[c]['pos2']=[]
            for n,t in enumerate(context[c]['index']):
                ctxstart=self.marker.door['close'][t]  
                context[c]['spkt'].append(self.spktrain[(self.spktrain>=ctxstart) &(self.spktrain<=self.marker.record[1][t])]-ctxstart)
                context[c]['spkc'].append(context[c]['spkt'][n].size)
                context[c]['dur'].append(self.marker.record[1][t]-ctxstart)
                context[c]['fr'].append(context[c]['spkt'][n].size/context[c]['dur'][n])
                context[c]['ctx_start'].append(ctxstart-self.marker.record[0][t])
                context[c]['ctx_end'].append(self.marker.record[1][t]-self.marker.record[0][t])
                if pos:
                    if context[c]['spkc'][n]!=0:
                        spktemp=self.spktrain[(self.spktrain>=self.marker.record[0][t]) &(self.spktrain<=self.marker.record[1][t])]-self.marker.record[0][t] #the spikes from record start to record end 
                        spk_postemp=align_spike_pos(pos['trial'][t],spktemp,self.pos['FPS'])
                        context[c]['spkt_raw'].append(spktemp[spktemp>=(ctxstart-self.marker.record[0][t])])
                        context[c]['spkpos'].append(spk_postemp[spk_postemp['ts']>=(ctxstart-self.marker.record[0][t])])                   
                    else:
                        
                        context[c]['spkpos'].append([])
                        context[c]['spkt_raw'].append([])
                    context[c]['pos'].append(pos['trial'][t].loc[(pos['trial'][t]['ts']>=(ctxstart-self.marker.record[0][t]))])
                    context[c]['pos2'].append(context[c]['pos'][n].drop(labels='ts',axis=1))
                    newts=context[c]['pos'][n]['ts']-(ctxstart-self.marker.record[0][t])
                    context[c]['pos2'][n]['ts']=newts                  
                    context[c]['pos2'][n]['ts']=newts.values
                    context[c]['travel_distance']=np.sum(np.linalg.norm(np.stack((np.diff(context[c]['pos'][n]['x']),np.diff(context[c]['pos'][n]['y'])),axis=1),axis=1))
        #construct the buffer context 
        context[-1]['name']='buffer'
        context[-1]['index']=np.asarray([int(i) for i in range(len(self.marker.protocol))])
        context[-1]['spkt']=[]
        context[-1]['spkc']=[]
        context[-1]['dur']=[]
        context[-1]['fr']=[]
        for n,t in enumerate(context[-1]['index']):
            context[-1]['spkt'].append(self.spktrain[(self.spktrain>=self.marker.record[0][t])&(self.spktrain<=self.marker.door['open'][t])]-self.marker.record[0][t])
            context[-1]['spkc'].append(context[-1]['spkt'][n].size)
            context[-1]['dur'].append(self.marker.door['open'][t]-self.marker.record[0][t])
            context[-1]['fr'].append(context[-1]['spkt'][n].size/context[-1]['dur'][n])
            #we define here from record start to door open 
        
        #Plot the spike histogram for context A and B
        if plot_ctx:
            print('CDI:' + str(self.cdi))
            i=0;
            f, ax = plt.subplots(2,4, sharey=True, sharex=True)
            f.set_size_inches(8,2)
            for c in range(2):    
                for t,spk in enumerate(context[c]['spkt']):                                         
                    plot_spkhist(spk,ax[c,t])
                    i+=1
            plt.suptitle('CDI:  %.3f' %self.cdi[0])
                    
        if plot_buffer:
            i=0;
            f2, ax2 = plt.subplots(1,8, sharey=True, sharex=True)
            f2.set_size_inches(8,1)
            for t,spk in enumerate(context[-1]['spkt']): 
                plot_spkhist(spk,ax2[t],tlim=(0,60))
                i+=1

        return context
    
    #def split_context_pos(self,pos):
        
    def plot_ctx(self):
        i=0;
        f, ax = plt.subplots(2,4, sharey=True, sharex=True)
        f.set_size_inches(8,2)
        for c in range(2):    
            for t,spk in enumerate(self.ctx[c]['spkt']):                                         
                plot_spkhist(spk,ax[c,t])
                i+=1
                
    def split_trial(self): #split by trial, which contain both       
        trial=[{} for x in range(len(self.marker.record[0]))]
        for t,(tstart,tend) in enumerate(zip(self.marker.record[0], self.marker.record[1])):            
            trial[t]['spkt']=(self.spktrain[(self.spktrain>=tstart) & (self.spktrain<=tend)]-tstart)
            trial[t]['name']=self.marker.protocol[t]
            trial[t]['end']=self.marker.record[1][t]-self.marker.record[0][t]
            trial[t]['door_open']=self.marker.door['open'][t]-self.marker.record[0][t]
            trial[t]['door_close']=self.marker.door['close'][t]-self.marker.record[0][t]
            trial[t]['ifr'],trial[t]['ifr_ts']=binned_FR(trial[t]['spkt'], np.max(self.pos['trial'][t]['ts']),1/30,filtered=False)
            ft=interpolate.interp1d(trial[t]['ifr_ts'],trial[t]['ifr'],fill_value="extrapolate")
            trial[t]['fr_bs']=gaussian_filter1d(ft(self.pos['trial'][t]['ts']),7)
            trial[t]['fr_bs_ts']=self.pos['trial'][t]['ts'].values
        return trial
    

# %%The primary function for builidng a unit 
def buildneurons(pathname=r'C:\Users\Qixin\XuChunLab\nexdata\192043',file_type='nex',build_tracking=False,arena_reset=False,body_part='Body',bootstrap=False,filter_speed=True):
    #import nex file with a GUI window 
    experiment_date=os.path.split(pathname)[1]
    animal_id=os.path.split(os.path.split(pathname)[0])[1]  
    if build_tracking:
        #build position 
        pos=bt.build_pos(pathname,reset_arena=arena_reset,body_part=body_part,filter_speed=filter_speed)
    else:
        pos=[]
    ensemble=[]
    
    if file_type=='nex':
        filepath=glob.glob(os.path.join(pathname,'*.nex'))[0]
        nexin=nex.Reader(useNumpy=True).ReadNexFile(filepath)
        neurons=[]
        waveforms=[]
        events=[]
        markers=[]
        for var in nexin['Variables']:
            if var['Header']['Type']==0:
                neurons.append(var)
                #print('neuron',len(neurons))
            if var['Header']['Type']==1:
                events.append(var)
                #print('events',len(events))
            if var['Header']['Type']==3:
                waveforms.append(var)
                #print('waveforms',len(waveforms))
            if var['Header']['Type'] == 6 and len(var['Timestamps']) != 0:
                markers.append(var)
                #print('markers',len(markers))        
        #ask for user input of context protocol

        suppm=pd.read_csv(glob.glob(os.path.join(pathname,'*protocol.txt'))[0],header=None,delimiter=' ')
        input_protocol=suppm.values[0]
#        except:
#            input_protocol=[str(x) for x in input('Enter the order of context protocol: ').split() or 'A B A B A B B A'.split()] 
#            print(input_protocol)    
        record_marker=[events[0]['Timestamps'],events[1]['Timestamps']]
        door_marker={'open':[],
                     'close':[],
                     'duration':[]}
        for mrker in markers:
            if mrker['Header']['Name']=='KBD1':
                door_marker['open']=(mrker['Timestamps'].tolist())
            elif mrker['Header']['Name']=='KBD3':
                door_marker['close']=(mrker['Timestamps'].tolist())
        door_marker['duration']=[a-b for a,b in zip(door_marker['close'],door_marker['open'])]
#        door_marker[1]=np.delete(door_marker[1],0) #remove first door open marker
        enter_context=detect_enter_context(pos,pathname)
        allmarker=Marker(record_marker,door_marker,input_protocol,enter_context)
        for i in range(1,len(neurons)):
            ensemble.append(Unit(neurons[i]['Timestamps'],
                                 allmarker,
                                 Waveform(waveforms[i]['Timestamps'],
                                          waveforms[i]['WaveformValues'],
                                          waveforms[i]['Header']['SamplingRate']),
                                 pos,
                                 neurons[i]['Header']['Name'],
                                 experiment_date,
                                 animal_id,
                                 bootstrap
                                 ))
    elif file_type=='mat':
        filepath=sorted(glob.glob(os.path.join(pathname,'*.mat')))
        for matfile in filepath:
            data = scipy.io.loadmat(matfile)    
            record_marker=[data['record_start'].T[0],data['record_end'].T[0]]
            door_marker=[data['door_open'].T[0],data['door_close'].T[0]]
            input_protocol=[]
            for c in data['input_protocol'][0]:
                if c==1:        
                    input_protocol.append('A')
                elif c==2:
                    input_protocol.append('B')
            allmarker=Marker(record_marker,door_marker,input_protocol)
            ensemble.append(Unit(data['Timestamps'],
                                     allmarker,
                                     Waveform(data['Timestamps'],
                                              data['WaveformValues'],         
                                              data['SamplingRate']),
                                     pos,
                                     data['name'][0],
                                     bootstrap,
                                     experiment_date,
                                     animal_id
                                     ))
                
    return ensemble,pos
#%%
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

# %%Some general plot functions are defined here 
#1. raster
def plot_raster(neuralData,linecolor=[0,0,0],linesize=0.5,x='trial#',title_name='Spike raster plot'):
    plt.eventplot(neuralData,color=linecolor,linelengths=linesize)
    plt.title(title_name)
    plt.ylabel(x)

#2. spike histogram
def plot_spkhist(neuralData,ax,dt=5,tlim=[0,300]):
    spkc_hist=np.histogram(neuralData,bins=tlim[1]//dt,range=tlim)
    fr_hist=spkc_hist[0]/dt
    ax.bar(spkc_hist[1][:-1],fr_hist,width=dt)
    plt.xlabel('Time (s)')
    plt.ylabel('Firing rate (Hz)')
    return 

def plot_spkhist2(spk,ax,time_window,tbin=0.5,fit=True,sigma=2):
    bin_edges=np.arange(time_window[0],time_window[1]+tbin,tbin)
    ts_bin=np.arange(bin_edges[0]+tbin/2,bin_edges[-1],tbin)
    fr_binned=np.divide(np.histogram(spk,bin_edges)[0],tbin)
    ax.bar(ts_bin,fr_binned,width=tbin)  
    if fit:
        fr_filt=gaussian_filter1d(fr_binned,sigma)
        ax.plot(ts_bin,fr_filt,color='orange')
    
def spkhist_wholetrial(unit,tbin=0.5,fit=True,sigma=2):
    trial=unit.split_trial()
    ctx_name=np.unique([ctx['name'] for ctx in trial])
    num_ctx=ctx_name.size
    context={}
    for i in range(num_ctx):
        context[i]={}
        context[i]=[ctx for ctx in trial if ctx['name']==ctx_name[i]]
    max_num_trial=np.max([len(ctx) for ctx in context.values()])
    
    f, ax = plt.subplots(num_ctx,max_num_trial, sharey=True, sharex=True)
    f.set_size_inches(8,2)
    
    for c,ctx in enumerate(context.values()):
        for t,trl in enumerate(ctx):
            plot_spkhist2(trl['spkt'],ax[c,t],[0,trl['end']],tbin=tbin,fit=True,sigma=sigma)
            ax[c,t].axvline(trl['door_open'],color='green')
            ax[c,t].axvline(trl['door_close'],color='green')

#3. spike map
def plot_spikemap(Unit,speed_threshold=5,hd_lim=[],show_hd=False):
    
    f, ax = plt.subplots(Unit.num_ctx,Unit.num_max_trial, sharey=True, sharex=True)
    f.set_size_inches(Unit.num_max_trial*2,Unit.num_ctx*2)
    spkhd=[[] for i in range(Unit.num_ctx)]
    hd=[[] for i in range(Unit.num_ctx)]
    for c in range(Unit.num_ctx):    
        for t,(pos,spkpos) in enumerate(zip(Unit.ctx[c]['pos'],Unit.ctx[c]['spkpos'])):
            if np.any(hd_lim):
                x=pos['x'][(pos['speed']>=speed_threshold) & (pos['hd']>=hd_lim[0]) &  (pos['hd']<=hd_lim[1])]
                y=pos['y'][(pos['speed']>=speed_threshold) & (pos['hd']>=hd_lim[0]) &  (pos['hd']<=hd_lim[1])]
                h=pos['hd'][(pos['speed']>=speed_threshold) & (pos['hd']>=hd_lim[0]) &  (pos['hd']<=hd_lim[1])]
            else:                     
                x=pos['x'][pos['speed']>=speed_threshold]
                y=pos['y'][pos['speed']>=speed_threshold]
                h=pos['hd'][pos['speed']>=speed_threshold]
            try:
                if np.any(hd_lim):
                    spkposx=spkpos['x'].loc[(spkpos['speed']>=speed_threshold) & (spkpos['hd']>=hd_lim[0]) &  (spkpos['hd']<=hd_lim[1])]
                    spkposy=spkpos['y'].loc[(spkpos['speed']>=speed_threshold) & (spkpos['hd']>=hd_lim[0]) &  (spkpos['hd']<=hd_lim[1])]
                    spkposs=spkpos['speed'].loc[(spkpos['speed']>=speed_threshold) & (spkpos['hd']>=hd_lim[0]) &  (spkpos['hd']<=hd_lim[1])]
                    spkposhd=spkpos['hd'].loc[(spkpos['speed']>=speed_threshold) & (spkpos['hd']>=hd_lim[0]) &  (spkpos['hd']<=hd_lim[1])]
                else:
                    spkposx=spkpos['x'].loc[spkpos['speed']>=speed_threshold]
                    spkposy=spkpos['y'].loc[spkpos['speed']>=speed_threshold]
                    spkposs=spkpos['speed'].loc[spkpos['speed']>=speed_threshold]
                    spkposhd=spkpos['hd'].loc[spkpos['speed']>=speed_threshold]
                spkhd[c].append(spkposhd)
                hd[c].append(h)
            except:
                spkposx=[]
                spkposy=[]
                spkposs=[]
                spkposhd=[]
            try:
                ax[c,t].set_aspect(1)
                ax[c,t].plot(x,y,alpha=0.5) 
                 #ax[c,t].scatter(spkpos['x'],spkpos['y'],s=5,color='r')             
                ax[c,t].scatter(spkposx,spkposy,s=spkposs,color='r',alpha=0.4)
                ax[c,t].set_title('spkc='+str(len(spkposx)))
            except:
                ax[c].set_aspect(1)
                ax[c].plot(x,y,alpha=0.5) 
                 #ax[c,t].scatter(spkpos['x'],spkpos['y'],s=5,color='r')             
                ax[c].scatter(spkposx,spkposy,s=spkposs,color='r',alpha=0.2)
                ax[c].set_title('spkc='+str(len(spkposx)))
                
    if show_hd:
        f2, ax2 = plt.subplots(Unit.num_ctx,Unit.num_max_trial, sharey=True, sharex=True)
        f2.set_size_inches(Unit.num_max_trial*2,Unit.num_ctx*2)
        for c,(spkhd_ctx,hd_ctx) in enumerate(zip(spkhd,hd)):    
            for t,(spkhd_trl,hd_trl) in enumerate(zip(spkhd_ctx,hd_ctx)):
                try:
                    ax2[c,t].hist(hd_trl,bins=360//45,range=(-180,180),facecolor='blue', alpha=0.5,density=True)
                    ax2[c,t].hist(spkhd_trl,bins=360//45,range=(-180,180),facecolor='red', alpha=0.5,density=True)
                except:
                    ax2[c].hist(hd_trl,bins=360//45,range=(-180,180),facecolor='blue', alpha=0.5,density=True)
                    ax2[c].hist(spkhd_trl,bins=360//45,range=(-180,180),facecolor='red', alpha=0.5,density=True)
    else:
        pass
             
def plot_context_summary(unit,tbin=2):
    trial=unit.split_trial()
    ctx_name=np.unique([ctx['name'] for ctx in trial])
    num_ctx=ctx_name.size
    context=unit.ctx[:-1]
    min_exploration_time=np.floor(np.min([np.min(cx['dur']) for cx in context]))
    
    if num_ctx==2:
        f, ax = plt.subplots(3,num_ctx, sharey='row', sharex='all')
        f.set_size_inches(8,6)
    else:
        f, ax = plt.subplots(2,num_ctx, sharey='row', sharex=True)
        f.set_size_inches(6,2)
    colors=['red','green','blue','yellow']
    bin_edges=np.arange(0,min_exploration_time+tbin,tbin)
    ts_bin=np.arange(bin_edges[0]+tbin/2,bin_edges[-1],tbin)
    z_score=[]
    for c,ctx in enumerate(context):
        fr_binned=[]
        for t,spkt in enumerate(ctx['spkt']):
            spkt=spkt[spkt<=min_exploration_time]
            ax[0,c].eventplot(spkt,lineoffsets=t+1,color=colors[c],linelengths=0.5) #plot rasters          
            fr_binned.append(np.divide(np.histogram(spkt,bin_edges)[0],tbin))
        fr_mean=np.mean(fr_binned,axis=0)
        z_score.append(zscore(fr_mean))
        ax[1,c].bar(ts_bin,fr_mean,width=tbin,color=colors[c])
        
    if num_ctx==2:
        f.add_subplot(313)
        f.delaxes(ax[2,0])
        f.delaxes(ax[2,1])
        plt.bar(ts_bin,z_score[0],width=tbin,color=colors[0],alpha=0.7)
        plt.bar(ts_bin,z_score[1],width=tbin,color=colors[1],alpha=0.7)
        plt.ylabel('z score')
        plt.xlabel('time (s)')
    try:
        plt.suptitle('CDI = %.3f' %unit.cdi[0])
    except:
        print('no valid CDI')
             
def binned_FR(spk,dur,tbin,filtered=True,sigma=1):
    bin_edges=np.arange(0,dur,tbin)
    ts_bin=np.arange(bin_edges[0]+tbin/2,bin_edges[-1],tbin)
    if filtered:
        fr_binned=gaussian_filter1d(np.divide(np.histogram(spk,bin_edges)[0],tbin),sigma)
    else:
        fr_binned=np.divide(np.histogram(spk,bin_edges)[0],tbin)
    return fr_binned,ts_bin
# %% Numeric computation functions are defined here ###
def ctx_ranksum(unit):
    from scipy.stats import ranksums 
    fra=unit.ctx[0]['fr']
    frb=unit.ctx[1]['fr']
    t,p=ranksums(fra,frb)
    if (p<=0.05) & (t>0):
        ctxid='A'
    elif (p<=0.05) & (t<0):
        ctxid='B'
    else:
        ctxid='O'
    return ctxid

def calc_cdi(context,method=0):
    if method==0:
        avg_fr=[]
        for i in range(2):
            avg_fr.append(sum(context[i]['spkc'])/sum(context[i]['dur']))
        cdi=(avg_fr[0]-avg_fr[1])/(avg_fr[0]+avg_fr[1])
    elif method==1:
        cdi_singletrial=[]
        for fr1,fr2 in zip(context[0]['fr'],context[1]['fr']):
            cdi_singletrial.append((fr1-fr2)/(fr1+fr2))
        cdi=np.mean(np.asarray(cdi_singletrial))
        
    return cdi

def bootstrap_context(Unit,eval_method='cdi',shuffle_num=10000,spkthreshold=100,isfig=False):
    init_t=0
    spkt_ob=[]
    tstart=[]
    tend=[]
    dur=[]
    ttemp=0
    #create a long spiketrain containing only spikes from context 
    for i,record_end in enumerate(Unit.marker.record[1]):
        spkctx=Unit.spktrain[(Unit.spktrain>=Unit.marker.door['open'][i])&(Unit.spktrain<=record_end)]-Unit.marker.door['open'][i]
        spkt_ob.append(spkctx+init_t)
        dur.append((record_end-Unit.marker.door['open'][i]))
        init_t+=dur[i]
        tstart.append(ttemp)
        tend.append(ttemp+dur[i])
        ttemp+=dur[i]        
    spkt_observed=np.concatenate(spkt_ob)
    if spkt_observed.size>=spkthreshold:
        #keep the ISI the same but shuffled 
        ISI=np.insert(np.diff(spkt_observed),0,Unit.spktrain[0])  
        #create pseudo spiketrains
        spk_shuffle=[]
    
        for i in range(shuffle_num):
            spk_new=[]
            currentspk=0
            new_ISI=np.random.permutation(ISI)
            for isi in new_ISI:
                spk_new.append(currentspk+isi)
                currentspk+=isi
            spk_shuffle.append(np.array(spk_new))
        
        
    
        #1. compare the cdi between observed and shuffled (not a good measure)
        if eval_method=='cdi':
            thres=1.96 #99% zscore value
            cdi_observed=cal_ctx_cdi(spkt_observed,Unit,tstart,tend,dur)[1]
            cdi_shuffle=[]
            for spk_s in spk_shuffle:
                cdi_shuffle.append(cal_ctx_cdi(spk_s,Unit,tstart,tend,dur)[1])
            mu,sigma=norm.fit(cdi_shuffle)
            CI1=thres*sigma+mu 
            CI2=mu-thres*sigma
            if cdi_observed>CI1:
                cell_identity='A'
            elif cdi_observed<CI2:
                cell_identity='B'
            else:
                cell_identity='others'
            if isfig:            
                plt.figure()
                n,bins,patches=plt.hist(cdi_shuffle,bins=100)
                plt.axvline(cdi_observed,color='g')
                CI=thres*sigma+mu 
                plt.axvline(CI,color='r')
                plt.show()
            
        
        #2. we compare the firing rate of each trial to its shuffled results 
     
        if eval_method=='fr1':
            thres=1.96 #99% zscore value
            ctx_ob=cal_ctx_cdi(spkt_observed,Unit,tstart,tend,dur)[0]
            ctx_num=len(np.unique(Unit.marker.protocol))
            trl_num=np.unique(Unit.marker.protocol,return_counts=True)[1][0]
            ctx_shuffle_fr=np.zeros((ctx_num,trl_num,shuffle_num)) 
            for s,spk_s in enumerate(spk_shuffle):
                ctx_temp=cal_ctx_cdi(spk_s,Unit,tstart,tend,dur)[0]
                for c in range(ctx_num):                   
                    for t in range(trl_num):
                        ctx_shuffle_fr[c][t][s]=ctx_temp[c]['fr'][t]
            #plot the shuffled distribution 
            if isfig: 
                f, ax = plt.subplots(ctx_num, trl_num, sharey=True, sharex=True)
                for c in range(ctx_num):
                    for t in range(trl_num):
                        n,bins,patches=ax[c,t].hist(ctx_shuffle_fr[c][t],60,density=True,alpha=0.75)
                        ax[c,t].axvline(ctx_ob[c]['fr'][t],color='g')
                        #we use 99% confidence interval as threshold
                        mu,sigma=norm.fit(ctx_shuffle_fr[c][t])
                        y = plt.mlab.normpdf(bins, mu, sigma)
                        ax[c,t].plot(bins, y, 'y--', linewidth=2)
                        CI=thres*sigma+mu 
                        ax[c,t].axvline(CI,color='r')
            #show the context preference of each trial
            ctx_pref=np.zeros((ctx_num,trl_num))
            for c in range(ctx_num):
                for t in range(trl_num):
                    mu,sigma=norm.fit(ctx_shuffle_fr[c][t])
                    zobserved=(ctx_ob[c]['fr'][t]-mu)/sigma
                    if abs(zobserved)<thres:
                        ctx_pref[c][t]=0
                    elif zobserved>thres:
                        ctx_pref[c][t]=1
                    elif zobserved<-thres:
                        ctx_pref[c][t]=-1
            #decide which context this unit prefers 
            if np.sum(ctx_pref[0]==1)>=2:
                cell_identity=ctx_ob[0]['name']
            elif np.sum(ctx_pref[1]==1)>=2:
                cell_identity=ctx_ob[1]['name']
            else:
                cell_identity='others'
            print('This unit prefers ' + cell_identity )
            
    else:
        cell_identity=[]
    return cell_identity 
            
                    
                
def cal_ctx_cdi(spk,Unit,tstart,tend,dur):
    ctx_bs=[{} for x in range(len(np.unique(Unit.marker.protocol)))]       
    for c,ctx in enumerate(np.unique(Unit.marker.protocol)):
        ctx_bs[c]['name']=ctx
        ctx_bs[c]['index']=np.where(np.asarray(Unit.marker.protocol)==ctx)[0]
        ctx_bs[c]['spkt']=[]
        ctx_bs[c]['spkc']=[]
        ctx_bs[c]['dur']=[]
        ctx_bs[c]['fr']=[]
        for n,t in enumerate(ctx_bs[c]['index']):
            ctx_bs[c]['spkt'].append(spk[(spk>=tstart[t]) & (spk<=tend[t])])
            ctx_bs[c]['dur'].append(dur[t])
            ctx_bs[c]['spkc'].append(ctx_bs[c]['spkt'][n].size)
            ctx_bs[c]['fr'].append(ctx_bs[c]['spkc'][n]/dur[t])
    avg_fr=[]
    for i in range(2):
        avg_fr.append(sum(ctx_bs[i]['spkc'])/sum(ctx_bs[i]['dur']))
    cdi=(avg_fr[0]-avg_fr[1])/(avg_fr[0]+avg_fr[1])
    return ctx_bs, cdi 

def align_spike_pos(postemp,spktemp,fps,isfig=False,ts_filt=0.25):
    
    pos_t=postemp['ts']
    fx=interpolate.interp1d(pos_t,postemp['x'],fill_value="extrapolate",kind='nearest')
    fy=interpolate.interp1d(pos_t,postemp['y'],fill_value="extrapolate",kind='nearest')
    #note here we filtered the speed and head-direction with a 1D gaussian kernel of sigma around 250ms (7 datapoints)
    fs=interpolate.interp1d(pos_t,gaussian_filter1d(postemp['speed'],ts_filt/(1/fps)),fill_value="extrapolate",kind='nearest')
    fhd=interpolate.interp1d(pos_t,postemp['hd'],fill_value="extrapolate",kind='nearest')
    #fhd=interpolate.interp1d(pos_t,gaussian_filter1d(postemp['hd'],7),fill_value="extrapolate",kind='nearest')
    spktemp=spktemp[spktemp<=np.max(pos_t)] #make sure all spktemp are within pos_t
    xnew=fx(spktemp)
    ynew=fy(spktemp)
    snew=fs(spktemp)
    hdnew=fhd(spktemp)
    pos_spk=pd.DataFrame({'x':xnew,'y':ynew,'ts':spktemp,'speed':snew,'hd':hdnew})
    if isfig:  #plot spike map of one single trial      
        fig, ax = plt.subplots()
        ax.set_aspect(1)            
        ax.plot(postemp['x'],postemp['y'],'.-',markersize=0.1,alpha=0.5)
        ax.scatter(xnew,ynew,s=10,color='r')
        plt.show()
    return pos_spk
# %% Place cell analysis 
def evaluate_spatial(myUnit,pathname,filtered=True,real_binsize=2,sigma=2,occupancy_threshold=3,isfig=True,speed_threshold=5):
    arena_new=refine_arena(myUnit,real_binsize)
    binsize=real_binsize*np.floor(myUnit.arena_ratio) #2cm per bin 
    xbincount=int((np.ceil(arena_new['x'][1]-arena_new['x'][0])/binsize))
    ybincount=int(np.ceil((arena_new['y'][1]-arena_new['y'][0])/binsize))
    xnudge=(binsize*xbincount-(arena_new['x'][1]-arena_new['x'][0]))/2
    ynudge=(binsize*ybincount-(arena_new['y'][1]-arena_new['y'][0]))/2
    x_new=(arena_new['x'][0]-xnudge,arena_new['x'][1]+xnudge)
    y_new=(arena_new['y'][0]-ynudge,arena_new['y'][1]+ynudge)
    yrange=np.arange(y_new[0],y_new[1]+binsize,binsize)
    xrange=np.arange(x_new[0],x_new[1]+binsize,binsize)
    occupancy=[[] for i in range(myUnit.num_ctx)]
    occupancy_map=[[] for i in range(myUnit.num_ctx)]
    for c in range(myUnit.num_ctx):
        for t,postemp in enumerate(myUnit.ctx[c]['pos']):
            x=postemp['x'][postemp['speed']>=speed_threshold]
            y=postemp['y'][postemp['speed']>=speed_threshold]            
            H,xedges,yedges=np.histogram2d(x,y,bins=(xrange,yrange))
            H=H.T 
            if filtered:
                H=gaussian_filter(H,sigma=sigma)
            occupancy[c].append(H[H>occupancy_threshold].size/H.size)
            occupancy_map[c].append(H)
            
    extensions=('*.asf','*.avi','*.mp4')
    samplevideo=[]
    videos=[]
    for extension in extensions:
        videos.append(glob.glob(pathname+'/'+extension))
    samplevideo=sorted(videos)[-1][0]
    vidcap=cv2.VideoCapture(samplevideo)
    vidcap.set(1,3000)
    success,img = vidcap.read() 
    
    
    fig,ax=plt.subplots()
    ax.imshow(img,origin='lower')
    ax.plot(x,y,alpha=0.2)
    for xl,yl in zip(xedges,yedges):
        ax.axvline(xl,color='g')
        ax.axhline(yl,color='g')

def refine_arena(Unit,real_binsize,pad_bin=1):
    xmin=[]
    xmax=[]
    ymin=[]
    ymax=[]
    for ctx in Unit.ctx[:-1]:
        for postrial in ctx['pos']:
            xmin.append(np.min(postrial['x']))
            xmax.append(np.max(postrial['x']))
            ymin.append(np.min(postrial['y']))
            ymax.append(np.max(postrial['y']))
    padding=Unit.pos['arena']['ratio']*real_binsize*pad_bin
    arena_new={'x':(np.floor(np.min(xmin)-padding),np.ceil(np.max(xmax)+padding)),'y':(np.floor(np.min(ymin)-padding),np.ceil(np.max(ymax)+padding))}
    return arena_new

def plot_occupancy_map(Unit,filtered=True,real_binsize=2,sigma=2,occupancy_threshold=3,isfig=True,speed_threshold=5):
    #occupancy_threshold: the animal has to pass the bin more than this many times 
    if isfig:
        f, ax = plt.subplots(Unit.num_ctx,Unit.num_max_trial, sharey=True, sharex=True)
        f.set_size_inches(Unit.num_max_trial*2,Unit.num_ctx*2)
    arena_new=refine_arena(Unit,real_binsize)
    binsize=real_binsize*np.floor(Unit.arena_ratio) #2cm per bin 
    xbincount=int((np.ceil(arena_new['x'][1]-arena_new['x'][0])/binsize))
    ybincount=int(np.ceil((arena_new['y'][1]-arena_new['y'][0])/binsize))
    xnudge=(binsize*xbincount-(arena_new['x'][1]-arena_new['x'][0]))/2
    ynudge=(binsize*ybincount-(arena_new['y'][1]-arena_new['y'][0]))/2
    x_new=(arena_new['x'][0]-xnudge,arena_new['x'][1]+xnudge)
    y_new=(arena_new['y'][0]-ynudge,arena_new['y'][1]+ynudge)
    yrange=np.arange(y_new[0],y_new[1]+binsize,binsize)
    xrange=np.arange(x_new[0],x_new[1]+binsize,binsize)
    occupancy=[[] for i in range(Unit.num_ctx)]
    occupancy_map=[[] for i in range(Unit.num_ctx)]
    for c in range(Unit.num_ctx):
        for t,postemp in enumerate(Unit.ctx[c]['pos']):
            x=postemp['x'][postemp['speed']>=speed_threshold]
            y=postemp['y'][postemp['speed']>=speed_threshold]            
            H,xedges,yedges=np.histogram2d(x,y,bins=(xrange,yrange))
            H=H.T 
            if filtered:
                H=gaussian_filter(H,sigma=sigma)
            if isfig:
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                ax[c,t].imshow(H, extent=extent, cmap='jet',origin='lower')
            occupancy[c].append(H[H>occupancy_threshold].size/H.size)
            occupancy_map[c].append(H)
    return occupancy,occupancy_map,xrange,yrange

def make_FRmap(spkpostemp,speed_threshold,occupancy,xrange,yrange,sigma,filtered=True):
    if np.any(spkpostemp):
        x=spkpostemp['x'][spkpostemp['speed']>=speed_threshold]
        y=spkpostemp['y'][spkpostemp['speed']>=speed_threshold]
    else:
        x=[]
        y=[]
    try:
        H,xedges,yedges=np.histogram2d(x,y,bins=(xrange,yrange))
        H=H.T
        if filtered:
            H=gaussian_filter(H,sigma=sigma)
    except:
        H=np.zeros(np.shape(occupancy[0][0]))
    return H
        
    
def plot_place_field(Unit,filtered=True,real_binsize=1,sigma=2,isfig=True,speed_threshold=3,min_spkc_threshold=20,min_occup_threshold=2,num_shuffle=1000):
    fps=Unit.pos['FPS']
    if isfig:
        f, ax = plt.subplots(Unit.num_ctx,Unit.num_max_trial, sharey=True, sharex=True)
        f.set_size_inches(Unit.num_max_trial*2,Unit.num_ctx*2)
    occupancy,occupancy_map,xrange,yrange=plot_occupancy_map(Unit,filtered=True,sigma=sigma,real_binsize=real_binsize,isfig=False,speed_threshold=speed_threshold)
    FR_map=[[] for i in range(Unit.num_ctx)]
    spkc=[[] for i in range(Unit.num_ctx)]
    SI=[[] for i in range(Unit.num_ctx)]
    for c in range(Unit.num_ctx):
        for t,spkpostemp in enumerate(Unit.ctx[c]['spkpos']):            
            if np.any(spkpostemp):
                x=spkpostemp['x'][spkpostemp['speed']>=speed_threshold]
                y=spkpostemp['y'][spkpostemp['speed']>=speed_threshold]
            else:
                x=[]
                y=[]
            spkc[c].append(len(x))
            try:
                H,xedges,yedges=np.histogram2d(x,y,bins=(xrange,yrange))
                H=H.T
                if filtered:
                    H=gaussian_filter(H,sigma=sigma)
            except:
                H=np.zeros(np.shape(occupancy[0][0]))
            pbin=occupancy_map[c][t]/np.sum(occupancy_map[c][t])
            SI[c].append(calc_SI(H,pbin))
            FR_map[c].append(H/(occupancy_map[c][t]*(1/fps)))
            
                    
    SI_sig=[[] for i in range(Unit.num_ctx)]       
    if isfig:
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        cmap=cm.jet
        cmap.set_bad('white',1.)        
        for c in range(Unit.num_ctx):
            for t,spkpostemp in enumerate(Unit.ctx[c]['spkpos']):
                if spkc[c][t]>=min_spkc_threshold:
                    frmap=FR_map[c][t]
                    frmap[occupancy_map[c][t]<=min_occup_threshold]=np.nan
                    masked_frmap=np.ma.array(frmap,mask=np.isnan(frmap)) 
                    try:
                        im=ax[c,t].imshow(masked_frmap,extent=extent,vmin=0,vmax=np.nanmax(FR_map)*0.9,cmap=cmap, origin='lower')
                    except:
                        im=ax[c].imshow(masked_frmap,extent=extent,vmin=0,vmax=np.nanmax(FR_map)*0.9,cmap=cmap, origin='lower')
                    try:
                        ax[c,t].set_title('SI = %.3f' %SI[c][t])
                    except:
                        ax[c].set_title('SI = %.3f' %SI[c][t])
                    
                    if num_shuffle:
                        if occupancy[c][t]>=0.7:
                            pseudoSI=[]
                            for i in range(num_shuffle):
                                try:
                                    pseudospk=np.array(make_pseudospktrian(Unit.ctx[c]['spkt'][t]))
                                    pseudo_spkpos=align_spike_pos(Unit.ctx[c]['pos2'][t],pseudospk,fps)
                                    pseudoH=make_FRmap(pseudo_spkpos,speed_threshold,occupancy,xrange,yrange,sigma)
                                    pseudoSI.append(calc_SI(pseudoH,pbin))
                                except:
                                    i-=1
                            if SI[c][t]>=np.mean(pseudoSI)+2.58*np.std(pseudoSI):
                                try:
                                    ax[c,t].set_title('SI = %.3f *' %SI[c][t])
                                except:
                                    ax[c].set_title('SI = %.3f *' %SI[c][t])
                            pseudoSI=np.array(pseudoSI)
                            SI_sig[c].append(len(pseudoSI[pseudoSI>SI[c][t]])/num_shuffle)
                        else:
                            SI_sig[c].append(np.nan)

                                      
                        
        f.subplots_adjust(right=0.8)
        cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
        try:
            f.colorbar(im,cax=cbar_ax)
        except:
            pass
        plt.show()
        try:
            if (True in [sis<0.01 for sis in SI_sig[0]]) & (True in [sis<0.01 for sis in SI_sig[1]]):
                placefield='Both'
            if True in [sis<0.01 for sis in SI_sig[0]]:
                placefield='A'
            elif True in [sis<0.01 for sis in SI_sig[1]]:
                placefield='B'
            else:
                placefield='none'
        except:
            placefield='none'
    return FR_map,SI,SI_sig,placefield


def plot_place_field_withnan(Unit,filtered=True,real_binsize=2,sigma=2,isfig=True,speed_threshold=5,min_spkc_threshold=20,min_occup_threshold=2):
    fps=Unit.pos['FPS']
    if isfig:
        f, ax = plt.subplots(Unit.num_ctx,Unit.num_max_trial, sharey=True, sharex=True)
        f.set_size_inches(Unit.num_max_trial*2,Unit.num_ctx*2)
    occupancy,occupancy_map,xrange,yrange=plot_occupancy_map(Unit,filtered=True,sigma=sigma,real_binsize=real_binsize,isfig=False,speed_threshold=speed_threshold)
    FR_map=[[] for i in range(Unit.num_ctx)]
    FR_map_nan=[[] for i in range(Unit.num_ctx)]
    spkc=[[] for i in range(Unit.num_ctx)]
    for c in range(Unit.num_ctx):
        for t,spkpostemp in enumerate(Unit.ctx[c]['spkpos']):            
            if np.any(spkpostemp):
                x=spkpostemp['x'][spkpostemp['speed']>=speed_threshold]
                y=spkpostemp['y'][spkpostemp['speed']>=speed_threshold]
            else:
                x=[]
                y=[]
            spkc[c].append(len(x))
            H,xedges,yedges=np.histogram2d(x,y,bins=(xrange,yrange)) #spike map 
            H=H.T
            if filtered:
                H=gaussian_filter(H,sigma=sigma)
                H[occupancy_map[c][t]<=min_occup_threshold]=np.nan
                V=H.copy()
                V[np.isnan(H)]=0
                VV=gaussian_filter(V,sigma=sigma)
                W=0*H.copy()+1
                W[np.isnan(H)]=0
                WW=gaussian_filter(W,sigma=sigma)
                Z=VV/WW
                Z_nan=Z.copy()
                Z_nan[np.isnan(H)]=np.nan
            
            FR_map[c].append(Z/(occupancy_map[c][t]*(1/fps)))
            FR_map_nan[c].append(Z_nan/(occupancy_map[c][t]*(1/fps)))
            
    if isfig:
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        cmap=cm.jet
        cmap.set_bad('white',1.)        
        for c in range(Unit.num_ctx):
            for t,spkpostemp in enumerate(Unit.ctx[c]['spkpos']):
                if spkc[c][t]>=min_spkc_threshold:
                    frmap=FR_map[c][t]
                    frmap[occupancy_map[c][t]<=min_occup_threshold]=np.nan
                    masked_frmap=np.ma.array(frmap,mask=np.isnan(frmap)) 
                    im=ax[c,t].imshow(masked_frmap,extent=extent,vmin=0,vmax=np.nanmax(FR_map_nan)*0.9,cmap=cmap, origin='lower')

        f.subplots_adjust(right=0.8)
        cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
        f.colorbar(im,cax=cbar_ax)
        plt.show()
    return FR_map


def plot_ctxwise_PF(Unit,filtered=True,real_binsize=2,sigma=2,isfig=True,speed_threshold=5,min_spkc_threshold=20,min_occup_threshold=2):
    fps=Unit.pos['FPS']
    if isfig:
        f, ax = plt.subplots(Unit.num_ctx,1, sharey=True, sharex=True)
        f.set_size_inches(2*2,Unit.num_ctx*2)
    occupancy,occupancy_map,xrange,yrange=plot_occupancy_map(Unit,filtered=False,real_binsize=real_binsize,isfig=False,speed_threshold=speed_threshold) #Do not filter just yet 
    occupancy_ctx=[[] for i in range(Unit.num_ctx)]
    P_occup=[[] for i in range(Unit.num_ctx)]
    dur_ctx=[[] for i in range(Unit.num_ctx)] #the total duration in a context excluding the quiet periods
    for c,ctx_om in enumerate(occupancy_map):
        occupancy_ctx[c]=gaussian_filter(np.sum(ctx_om,axis=0),sigma=sigma)
        P_occup[c]=occupancy_ctx[c]/np.sum(occupancy_ctx[c])
        dur_ctx[c]=np.sum(np.sum(ctx_om,axis=0))*(1/fps) #total time spent in each context (moving epoches only)

    spk_map=[[] for i in range(Unit.num_ctx)]
    FR_ctx=[[] for i in range(Unit.num_ctx)]
    spkc_ctx=[0 for i in range(Unit.num_ctx)] #the total spike count in a context 
    for c in range(Unit.num_ctx):
        for t,spkpostemp in enumerate(Unit.ctx[c]['spkpos']):
            if np.any(spkpostemp):
                x=spkpostemp['x'][spkpostemp['speed']>=speed_threshold]
                y=spkpostemp['y'][spkpostemp['speed']>=speed_threshold]
            else:
                x=[]
                y=[]
            spkc_ctx[c]+=np.size(x)
            try:
                H,xedges,yedges=np.histogram2d(x,y,bins=(xrange,yrange))
                H=H.T
            except:
                H=np.zeros(np.shape(occupancy[0][0]))
            spk_map[c].append(H)    
    FR_map_ctx=[[] for i in range(Unit.num_ctx)]
    for c,ctx_spkmap in enumerate(spk_map):
        FR_map_ctx[c]=gaussian_filter(((np.sum(ctx_spkmap,axis=0))/((occupancy_ctx[c])*(1/fps))),sigma=sigma)
        FR_ctx[c]=np.sum(np.multiply(FR_map_ctx[c],P_occup[c]))
        
    if isfig:
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        cmap=cm.jet
        cmap.set_bad('white',1.)
        for c,FRmap in enumerate(FR_map_ctx):
            if spkc_ctx[c]>=min_spkc_threshold:   
                FRmap[occupancy_ctx[c]<=min_occup_threshold]=np.nan
                masked_FRmap=np.ma.array(FRmap,mask=np.isnan(FRmap)) 
                im=ax[c].imshow(masked_FRmap,extent=extent, vmin=0,vmax=np.max(FR_map_ctx)*0.9,cmap=cmap, origin='lower')
        
        f.subplots_adjust(right=0.8)
        cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
                
       

    
    spatial_information=[0 for i in range(Unit.num_ctx)]
    for i,(pbin,FR_bin,meanFR) in enumerate(zip(P_occup,FR_map_ctx,FR_ctx)):
        for pbiny,FR_biny in zip(pbin,FR_bin):
            for pbinyx,FR_binyx in zip(pbiny,FR_biny):
                if pbinyx==0:
                    spatial_information[i]+=0 
                else:
                    spatial_information[i]+=pbinyx*FR_binyx*(np.log2(FR_binyx/meanFR))
        spatial_information[i]=spatial_information[i]/meanFR

    if isfig:
        try:
            f.colorbar(im,cax=cbar_ax)
            ax[0].set_title('SI = %.2f' %spatial_information[0])
            ax[1].set_title('SI = %.2f' %spatial_information[1])
        except:
            print('no CSI yet')
        plt.show()
        
    return spatial_information

def calc_SI(FR_bin,pbin):
    FR_bin[np.isnan(FR_bin)]=0
    pbin[np.isnan(pbin)]=0
    meanFR=np.sum(np.multiply(FR_bin,pbin))
    spatial_information=0
    for pbiny,FR_biny in zip(pbin,FR_bin):
        for pbinyx,FR_binyx in zip(pbiny,FR_biny):
            if pbinyx==0:
                spatial_information+=0 
            else:
                temp=np.log2(FR_binyx/meanFR)
                if temp==-np.inf:
                    spatial_information+=0
                else:
                    spatial_information+=pbinyx*FR_binyx*(temp)
    spatial_information=spatial_information/meanFR
    return spatial_information

def make_pseudospktrian(spktemp):
    pseudospk=[]
    currentspk=0
    if np.any(spktemp):
        ISI=np.insert(np.diff(spktemp),0,spktemp[0])
        ISI_new=np.random.permutation(ISI)
        for isi in ISI_new:
            pseudospk.append(currentspk+isi) #insert the shuffled spikes
            currentspk+=isi 
    return pseudospk

#def calc_pseudoFR_bin(pseudospk,postemp):
#    
    
    
def make_pseudoUnit(Unit):
    pseudoUnit=copy.deepcopy(Unit)
    #We need to change the 'spkpos' in ctx: that means the x, y, and the speed
    #First let's shuffle the spikes within trial 
    for c,ctx in enumerate(Unit.ctx[:-1]):
        for n,(spkpostemp,spktemp) in enumerate(zip(ctx['spkpos'],ctx['spkt'])):
            if ctx['spkc'][n]>=0:
                t=ctx['index'][n]
                ctxstart=Unit.marker.door['close'][t]-Unit.marker.record[0][t]
                postemp=Unit.pos['trial'][t]          
                pseudoUnit.ctx[c]['spkt'][n]=[]
                pseudoUnit.ctx[c]['spkpos'][n]=[]
                pseudospk=[] 
                currentspk=0
                if np.any(spktemp):
                    ISI=np.insert(np.diff(spktemp),0,spktemp[0]) #calc ISI and insert the first spike at the beginning 
                    ISI_new=np.random.permutation(ISI)
                    for isi in ISI_new:
                        pseudospk.append(currentspk+isi) #insert the shuffled spikes
                        currentspk+=isi
                else:
                    pseudospk=[]
                pseudoUnit.ctx[c]['spkt'][n]=pseudospk 
                pseudoUnit.ctx[c]['spkpos'][n]=align_spike_pos(postemp,pseudospk+ctxstart,Unit.pos['FPS']) 
    
    return pseudoUnit

def spatial_information_bootstrap(Unit,shuffle_num=100,isfig=True,real_binsize=1,sigma=2,speed_threshold=3):
    spatial_information_observed = plot_ctxwise_PF(Unit,filtered=True,isfig=False,real_binsize=real_binsize,sigma=sigma,speed_threshold=speed_threshold)
    spatial_information_shuffle=[]
    for s in range(shuffle_num):
        spatial_information_shuffle.append(plot_ctxwise_PF(make_pseudoUnit(Unit),filtered=True,real_binsize=real_binsize,sigma=sigma,isfig=False,speed_threshold=speed_threshold))
    

    if isfig:
        f, ax = plt.subplots(1,2, sharey=True, sharex=True)
        for i,(SI_ob,SI_shufle_dist) in enumerate(zip(spatial_information_observed,np.asarray(spatial_information_shuffle).T)):
            n,bins,patches=ax[i].hist(SI_shufle_dist,60,density=True,alpha=0.75)
            ax[i].axvline(SI_ob,color='g')
    
    
    return spatial_information_observed,spatial_information_shuffle
#%% Head Direction analysis
def get_quad_lim(quadrant,xedges,yedges):
    quad_lim={'y':yedges[quadrant[0]:quadrant[0]+2],'x':xedges[quadrant[1]:quadrant[1]+2]}
    return quad_lim

def make_spider(hd_count,ts_bin,ax,color='r'):
    from math import pi
    N=len(ts_bin)
    ts_bin=np.flip(ts_bin)
    angles=[n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    #plt.xticks(angles[0:-1:18], ts_bin[0:-1:18], color='grey', size=8)
    plt.xticks(angles[0:-1:18], [0,90,180,-90], color='grey', size=8)
#    plt.ylim(0,40)
#    
    values=np.flip(hd_count).tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.4)
    ax.get_yaxis().set_ticks([])
    return 

def hd_tuning(Unit,quadrant=[],speed_threshold=5,binsize=5,sigma=2,spkc_threshold=20,plot_style=2):   
    #1. Divide the arena into 4 quadrant
    if quadrant:
        arena_new=refine_arena(Unit,real_binsize=2)
        quadrant_occup,xedges,yedges=np.histogram2d(Unit.pos['all']['x'],Unit.pos['all']['y'],bins=[2,2],range=[arena_new['x'],arena_new['y']])
        quadlim=get_quad_lim(quadrant,xedges,yedges)
        
    hd_count=[[] for i in range(Unit.num_ctx)]
    f, ax = plt.subplots(Unit.num_ctx,Unit.num_max_trial, sharey=True, sharex=True,subplot_kw=dict(polar=True))
    f.set_size_inches(Unit.num_max_trial*2,Unit.num_ctx*2)
    colors=['r','g','b']
    if plot_style==1:
        for c in range(Unit.num_ctx):
            for t,(postemp,spkpostemp) in enumerate(zip(Unit.ctx[c]['pos'],Unit.ctx[c]['spkpos'])):
                try:
                    if quadrant:
                        hd=postemp['hd'][(postemp['speed']>=speed_threshold) & 
                                  (postemp['x']>=quadlim['x'][0]) & (postemp['x']<=quadlim['x'][1]) & 
                                  (postemp['y']>=quadlim['y'][0]) & (postemp['y']<=quadlim['y'][1])]
                        spkhd=spkpostemp['hd'][(spkpostemp['speed']>=speed_threshold) & 
                                  (spkpostemp['x']>=quadlim['x'][0]) & (spkpostemp['x']<=quadlim['x'][1]) & 
                                  (spkpostemp['y']>=quadlim['y'][0]) & (spkpostemp['y']<=quadlim['y'][1])]
                    else:
                        hd=postemp['hd'][(postemp['speed']>=speed_threshold)]
                        spkhd=spkpostemp['hd'][(spkpostemp['speed']>=speed_threshold)]
                        
                    if len(spkhd)>=spkc_threshold:   
                        hdc,hd_binedges=np.histogram(hd,bins=360//binsize,range=(-180,180),density=True)
                        spkhdc,hd_binedges=np.histogram(spkhd,bins=360//binsize,range=(-180,180),density=True)
                        
                        hdc_filt=gaussian_filter1d(hdc,sigma=sigma,mode='wrap')
                        spkhdc_filt=gaussian_filter1d(spkhdc,sigma=sigma,mode='wrap')
                        #ts_bin=np.arange(hd_binedges[0]+binsize/2,hd_binedges[-1],binsize)
                        ts_bin=hd_binedges[:-1]
                        try:
                            make_spider(hdc_filt,ts_bin,ax[c][t],color=colors[2]) 
                            make_spider(spkhdc_filt*3,ts_bin,ax[c][t],color=colors[c]) 
                            ax[c][t].set_title('spkc = %d' %len(spkhd),fontsize=8,loc='left')
                        except:
                            make_spider(hdc_filt,ts_bin,ax[c],color=colors[2]) 
                            make_spider(spkhdc_filt*3,ts_bin,ax[c],color=colors[c]) 
                            ax[c].set_title('spkc = %d' %len(spkhd),fontsize=8,loc='left')
                        hd_count[c].append(hdc)  
                except:
                    print('not enough spikes')
    elif plot_style==2:
        for c in range(Unit.num_ctx):
            for t,(postemp,spkpostemp) in enumerate(zip(Unit.ctx[c]['pos'],Unit.ctx[c]['spkpos'])):
                try:
                    if quadrant:
                        hd=postemp['hd'][(postemp['speed']>=speed_threshold) & 
                                  (postemp['x']>=quadlim['x'][0]) & (postemp['x']<=quadlim['x'][1]) & 
                                  (postemp['y']>=quadlim['y'][0]) & (postemp['y']<=quadlim['y'][1])]
                        spkhd=spkpostemp['hd'][(spkpostemp['speed']>=speed_threshold) & 
                                  (spkpostemp['x']>=quadlim['x'][0]) & (spkpostemp['x']<=quadlim['x'][1]) & 
                                  (spkpostemp['y']>=quadlim['y'][0]) & (spkpostemp['y']<=quadlim['y'][1])]
                    else:
                        hd=postemp['hd'][(postemp['speed']>=speed_threshold)]
                        spkhd=spkpostemp['hd'][(spkpostemp['speed']>=speed_threshold)]
                    if len(spkhd)>=spkc_threshold:   
                        hdc,hd_binedges=np.histogram(hd,bins=360//binsize,range=(-180,180),density=True)
                        spkhdc,hd_binedges=np.histogram(spkhd,bins=360//binsize,range=(-180,180),density=True)
                        
                        hdc_filt=gaussian_filter1d(hdc,sigma=sigma,mode='wrap')
                        spkhdc_filt=gaussian_filter1d(spkhdc,sigma=sigma,mode='wrap')
                        
                        spkhdc_norm=spkhdc_filt/hdc_filt
                        spkhdc_norm[np.isnan(spkhdc_norm)]=0
                        #ts_bin=np.arange(hd_binedges[0]+binsize/2,hd_binedges[-1],binsize)
                        ts_bin=hd_binedges[:-1]
                        try:
                            make_spider(spkhdc_norm,ts_bin,ax[c][t],color=colors[2]) 
                            ax[c][t].set_title('spkc = %d' %len(spkhd),fontsize=8,loc='left')
                        except:
                            make_spider(spkhdc_norm,ts_bin,ax[c],color=colors[2]) 
                            ax[c].set_title('spkc = %d' %len(spkhd),fontsize=8,loc='left')
                        hd_count[c].append(hdc)                    
                except:
                    print('not enough spikes')
    
                
    if quadrant==[1,0]:
        f.suptitle('Upper left quadrant')
    elif quadrant==[1,1]:
        f.suptitle('Upper right quadrant')
    elif quadrant==[0,0]:
        f.suptitle('Lower left quadrant')
    elif quadrant==[0,1]:
        f.suptitle('Lower right quadrant')
    elif not quadrant:
        f.suptitle('Whole arena')
        
def hd_tuning_all_quadrant(Unit,pathname,speed_threshold=5,binsize=5,sigma=2,spkc_threshold=20):
    import matplotlib.backends.backend_pdf
    import matplotlib._pylab_helpers
    plt.close('all') #make sure everything's closed
    hd_tuning(Unit,speed_threshold=speed_threshold,binsize=binsize,sigma=sigma,spkc_threshold=spkc_threshold) 
    hd_tuning(Unit,quadrant=[0,0],speed_threshold=speed_threshold,binsize=binsize,sigma=sigma,spkc_threshold=spkc_threshold) #lower left
    hd_tuning(Unit,quadrant=[0,1],speed_threshold=speed_threshold,binsize=binsize,sigma=sigma,spkc_threshold=spkc_threshold) #lower right
    hd_tuning(Unit,quadrant=[1,0],speed_threshold=speed_threshold,binsize=binsize,sigma=sigma,spkc_threshold=spkc_threshold) #upper left
    hd_tuning(Unit,quadrant=[1,1],speed_threshold=speed_threshold,binsize=binsize,sigma=sigma,spkc_threshold=spkc_threshold) #upper right
    figures=[manager.canvas.figure 
                 for manager in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
    pdf_filename=os.path.join(pathname,'HD_tuning_'+Unit.animal_id+'_'+Unit.date+'_'+Unit.name  +'.pdf')
    pdf=matplotlib.backends.backend_pdf.PdfPages(pdf_filename)
    for fig in figures: 
        pdf.savefig(fig)
    pdf.close()
    plt.close('all')
#%% Save results

def save_all_units(ensemble,pathname):
    for i,unit in enumerate(ensemble):
        if unit.max_spkc_ctx>=50:
            filename=os.path.join(pathname, unit.date +'_'+ unit.animal_id +'_'+ str(i)+ '.pkl')
            with open(filename,'wb') as output:
                pickle.dump(unit,output,pickle.HIGHEST_PROTOCOL)
                
def save_result_as_pdf(ensemble,pathname,real_binsize=1,sigma=2,speed_threshold=3):
    import matplotlib.backends.backend_pdf
    import matplotlib._pylab_helpers
    plt.close('all') #make sure everything's closed
    for myUnit in ensemble: 
        
        myUnit.waveform.Plotwf() 
        spkhist_wholetrial(myUnit,tbin=1,sigma=1)
        plot_context_summary(myUnit)      

        #Plot the spike map 
        plot_spikemap(myUnit,speed_threshold=speed_threshold)       
        #occupancy,occupancy_map,xrange,yrange=plot_occupancy_map(myUnit,real_binsize=1,sigma=3,occupancy_threshold=10)
        try:
            plot_place_field(myUnit,real_binsize=real_binsize,sigma=sigma,speed_threshold=speed_threshold)
        except:
            print('no place field')
        try:
            plot_ctxwise_PF(myUnit,real_binsize=real_binsize,sigma=sigma,speed_threshold=speed_threshold)
        except:
            print('no place field')
        figures=[manager.canvas.figure 
                     for manager in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
        pdf_filename=os.path.join(pathname,myUnit.animal_id+'_'+myUnit.date+'_'+myUnit.name  +'_SummaryFigs.pdf')
        pdf=matplotlib.backends.backend_pdf.PdfPages(pdf_filename)
        for fig in figures: 
            pdf.savefig(fig)
        pdf.close()
        plt.close('all')
        
# %% main
if __name__=='__main__':
    flag_buildtrack=True
    #ensemble,pos=buildneurons(build_tracking=flag_buildtrack,arena_reset=False)
    pathname=r'C:\Users\Qixin\OneDrive\Lab_stuff\EPHY_Qiushou\Ctx_discrimination_data\192076\20190423'  
    ftype='nex'
    ensemble,pos=buildneurons(pathname,
                              file_type=ftype,
                              build_tracking=flag_buildtrack,
                              arena_reset=True,
                              body_part='Body',
                              bootstrap=True                              
                              )
    # %% inspect each unit here
    plt.close('all')
    myUnit=ensemble[0]
    myUnit.waveform.Plotwf() 
    spkhist_wholetrial(myUnit,tbin=1,sigma=1)
#    ctx=myUnit.split_context(pos,plot_ctx=True,plot_buffer=False) 
    plot_context_summary(myUnit)
    if flag_buildtrack:
        real_binsize=1
        sigma=2
        speed_threshold=3
        #Plot the spike map 
        plot_spikemap(myUnit,speed_threshold=speed_threshold)       
        #occupancy,occupancy_map,xrange,yrange=plot_occupancy_map(myUnit,real_binsize=1,sigma=3,occupancy_threshold=10)
        #FR_map,pcsig = plot_place_field(myUnit,real_binsize=real_binsize,sigma=sigma,speed_threshold=speed_threshold)    
        #is_place_cell,spatial_information=plot_ctxwise_PF(myUnit,real_binsize=real_binsize,sigma=sigma,speed_threshold=speed_threshold)
        
    #%%
#    
#    for myUnit in ensemble:
#        hd_tuning_all_quadrant(myUnit,pathname,speed_threshold=speed_threshold,spkc_threshold=10)
#    #%%
##    cell_identity, ctx_pref=bootstrap_context(myUnit,isfig=True)
##    if flag_buildtrack:
##        spatial_information_observed,spatial_information_shuffle=spatial_information_bootstrap(myUnit,shuffle_num=200,real_binsize=real_binsize,sigma=sigma,speed_threshold=speed_threshold)
    save_all_units(ensemble,pathname)



#    save_pdf=True
#    if save_pdf:
#        save_result_as_pdf(ensemble,pathname,real_binsize=real_binsize,sigma=sigma,speed_threshold=speed_threshold)
#    
#    #%%
#    save_all_units(ensemble,pathname)
        