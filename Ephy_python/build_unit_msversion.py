import nexfile as nex
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import build_trajectory_msversion as bt
from scipy import interpolate
from scipy.ndimage.filters import gaussian_filter1d
from scipy.ndimage import gaussian_filter
import copy
import glob
import pickle
import scipy.io
# %%Class definition
class Marker:
    def __init__(self,record,door,protocol):
        self.record=record
        self.door=door
        self.protocol=protocol
        
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
        plt.plot(x,avgwf)
        plt.fill_between(x,avgwf+sdwf,avgwf-sdwf,alpha=0.3)
        plt.show()
        print('Peak-Trough duration is %.3f ms' %self.peak_trough_dur)

#Calculate the peak trough duration 
def FindPTdur(wfvalue,samplingrate):
    avgwf=np.mean(wfvalue,axis=0)
    avgquad=avgwf.reshape(4,-1)
    return (avgquad.min(axis=0).argmax()-avgquad.min(axis=0).argmin())*(1/samplingrate)*1000 #in ms
    
class Unit:
    def __init__(self,spktrain,marker,waveform,pos,name,date,animal_id,bootstrap): #marker and waveform are class as well 
        self.spktrain=spktrain
        self.marker=marker
        self.waveform=waveform 
        self.ctx=self.split_context(pos)
        self.cdi=[calc_cdi(self.ctx),calc_cdi(self.ctx,method=1)]
        self.max_spkc_ctx=np.max([np.max(self.ctx[0]['spkc']),np.max(self.ctx[1]['spkc'])])
        self.name=name
        self.date=date
        self.animal_id=animal_id
        if bootstrap:
            self.context_selectivity=bootstrap_context(self)       
        if pos:
            self.arena_ratio=pos['arena']['ratio']
            self.pos=pos
            try:
                self.is_place_cell,self.spatial_information=plot_ctxwise_PF(self,isfig=False);
            except:
                self.is_place_cell=False
                self.spatial_information=np.nan
                
                    
            
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
            for n,t in enumerate(context[c]['index']):
                ctxstart=self.marker.door[1][t]  
                context[c]['spkt'].append(self.spktrain[(self.spktrain>=ctxstart) &(self.spktrain<=self.marker.record[1][t])]-ctxstart)
                context[c]['spkc'].append(context[c]['spkt'][n].size)
                context[c]['dur'].append(self.marker.record[1][t]-ctxstart)
                context[c]['fr'].append(context[c]['spkt'][n].size/context[c]['dur'][n])
                context[c]['ctx_start'].append(ctxstart-self.marker.record[0][t])
                context[c]['ctx_end'].append(self.marker.record[1][t]-self.marker.record[0][t])
                if pos:
                    if context[c]['spkc'][n]!=0:
                        spktemp=self.spktrain[(self.spktrain>=self.marker.record[0][t]) &(self.spktrain<=self.marker.record[1][t])]-self.marker.record[0][t] #the spikes from record start to record end 
                        spk_postemp=align_spike_pos(pos['trial'][t],spktemp)
                        context[c]['spkt_raw'].append(spktemp[spktemp>=(ctxstart-self.marker.record[0][t])])
                        context[c]['pos'].append(pos['trial'][t].loc[(pos['trial'][t]['ts']>=(ctxstart-self.marker.record[0][t]))])
                        context[c]['spkpos'].append(spk_postemp[spk_postemp['ts']>=(ctxstart-self.marker.record[0][t])])                   
                    else:
                        context[c]['pos'].append(pos['trial'][t].loc[(pos['trial'][t]['ts']>=(ctxstart-self.marker.record[0][t]))])
                        context[c]['spkpos'].append([])
                        context[c]['spkt_raw'].append([])

        #construct the buffer context 
        context[-1]['name']='buffer'
        context[-1]['index']=np.asarray([int(i) for i in range(len(self.marker.protocol))])
        context[-1]['spkt']=[]
        context[-1]['spkc']=[]
        context[-1]['dur']=[]
        context[-1]['fr']=[]
        for n,t in enumerate(context[-1]['index']):
            context[-1]['spkt'].append(self.spktrain[(self.spktrain>=self.marker.record[0][t])&(self.spktrain<=self.marker.door[0][t])]-self.marker.record[0][t])
            context[-1]['spkc'].append(context[-1]['spkt'][n].size)
            context[-1]['dur'].append(self.marker.door[0][t]-self.marker.record[0][t])
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
        return trial
    

# %%The primary function for builidng a unit 
def buildneurons(pathname=r'C:\Users\Qixin\XuChunLab\nexdata\192043',file_type='nex',build_tracking=False,arena_reset=False,body_part='Body',bootstrap=False):
    #import nex file with a GUI window 
    experiment_date=os.path.split(pathname)[1]
    animal_id=os.path.split(os.path.split(pathname)[0])[1]  
    if build_tracking:
        #build position 
        pos=bt.build_pos(pathname,reset_arena=arena_reset,body_part=body_part)
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
        try:
            suppm=pd.read_excel(glob.glob(os.path.join(pathname,'*csv'))[0])
            protocol=(suppm['order'])
            protocol=protocol[~protocol.isnull()]
            input_protocol=protocol.values
        except:
            input_protocol=[str(x) for x in input('Enter the order of context protocol: ').split() or 'A B A B A B B A'.split()] 
            print(input_protocol)    
        record_marker=[events[0]['Timestamps'],events[1]['Timestamps']]
        door_marker=[]
        for mrker in markers:
            if mrker['Header']['Name']=='KBD1':
                door_marker.insert(0,mrker['Timestamps'])
            elif mrker['Header']['Name']=='KBD3':
                door_marker.insert(1,mrker['Timestamps'])
        door_marker=door_marker[0:2]
        allmarker=Marker(record_marker,door_marker,input_protocol)
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
# %%Some general plot functions are defined here 
#1. raster
def plot_raster(neuralData,linecolor=[0,0,0],linesize=0.5,x='trial#',title_name='Spike raster plot'):
    plt.eventplot(neuralData,color=linecolor,linelengths=linesize)
    plt.title(title_name)
    plt.ylabel(x)

#2. spike histogram
def plot_spkhist(neuralData,ax,dt=5,tlim=(0,300)):
    spkc_hist=np.histogram(neuralData,bins=tlim[1]//dt,range=tlim)
    fr_hist=spkc_hist[0]/dt
    ax.bar(spkc_hist[1][:-1],fr_hist,width=dt)
    plt.xlabel('Time (s)')
    plt.ylabel('Firing rate (Hz)')
    return 

#3. spike map
def plot_spikemap(Unit):
     f, ax = plt.subplots(2,4, sharey=True, sharex=True)
     f.set_size_inches(12,6)
     for c in range(2):    
         for t,(pos,spkpos) in enumerate(zip(Unit.ctx[c]['pos'],Unit.ctx[c]['spkpos'])):
             ax[c,t].set_aspect(1)
             ax[c,t].plot(pos['x'],pos['y'],alpha=0.5)
             #ax[c,t].scatter(spkpos['x'],spkpos['y'],s=5,color='r')
             if len(spkpos)>=5:
                 ax[c,t].scatter(spkpos['x'],spkpos['y'],s=Unit.ctx[c]['spkpos'][t]['speed'],color='r',alpha=0.4)
             
             

# %% Numeric computation functions are defined here ###
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

def bootstrap_context(Unit,eval_method='fr1',shuffle_num=10000,isfig=False):
    init_t=0
    spkt_ob=[]
    tstart=[]
    tend=[]
    dur=[]
    ttemp=0
    #create a long spiketrain containing only spikes from context 
    for i,record_end in enumerate(Unit.marker.record[1]):
        spkctx=Unit.spktrain[(Unit.spktrain>=Unit.marker.door[0][i])&(Unit.spktrain<=record_end)]-Unit.marker.door[0][i]
        spkt_ob.append(spkctx+init_t)
        dur.append((record_end-Unit.marker.door[0][i]))
        init_t+=dur[i]
        tstart.append(ttemp)
        tend.append(ttemp+dur[i])
        ttemp+=dur[i]        
    spkt_observed=np.concatenate(spkt_ob)
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
        thres=2.17 #99% zscore value
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
        return cell_identity
    
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
        return cell_identity, ctx_pref
                    
                
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

def align_spike_pos(postemp,spktemp,isfig=False):
    pos_t=postemp['ts']
    fx=interpolate.interp1d(pos_t,postemp['x'],fill_value="extrapolate")
    fy=interpolate.interp1d(pos_t,postemp['y'],fill_value="extrapolate")
    #note here we filtered the speed and head-direction with a 1D gaussian kernel of sigma around 250ms (7 datapoints)
    fs=interpolate.interp1d(pos_t,gaussian_filter1d(postemp['speed'],7),fill_value="extrapolate")
    fhd=interpolate.interp1d(pos_t,gaussian_filter1d(postemp['hd'],7),fill_value="extrapolate")
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
def refine_arena(Unit):
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
    arena_new={'x':(np.floor(np.min(xmin)),np.ceil(np.max(xmax))),'y':(np.floor(np.min(ymin)),np.ceil(np.max(ymax)))}
    return arena_new

def plot_occupancy_map(Unit,filtered=True,real_binsize=2,sigma=2,occupancy_threshold=3,isfig=True,speed_threshold=5):
    #occupancy_threshold: the animal has to pass the bin more than this many times 
    if isfig:
        f, ax = plt.subplots(2,4, sharey=True, sharex=True)
        f.set_size_inches(12,6)
    arena_new=refine_arena(Unit)
    binsize=real_binsize*np.floor(Unit.arena_ratio) #2cm per bin 
    xbincount=int((np.ceil(arena_new['x'][1]-arena_new['x'][0])/binsize))
    ybincount=int(np.ceil((arena_new['y'][1]-arena_new['y'][0])/binsize))
    xnudge=(binsize*xbincount-(arena_new['x'][1]-arena_new['x'][0]))/2
    ynudge=(binsize*ybincount-(arena_new['y'][1]-arena_new['y'][0]))/2
    x_new=(arena_new['x'][0]-xnudge,arena_new['x'][1]+xnudge)
    y_new=(arena_new['y'][0]-ynudge,arena_new['y'][1]+ynudge)
    yrange=np.arange(y_new[0],y_new[1]+binsize,binsize)
    xrange=np.arange(x_new[0],x_new[1]+binsize,binsize)
    occupancy=[[],[]]
    occupancy_map=[[],[]]
    for c in range(2):
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

def plot_place_field(Unit,filtered=True,real_binsize=2,sigma=2,isfig=True,speed_threshold=5):
    fps=Unit.pos['FPS']
    if isfig:
        f, ax = plt.subplots(2,4, sharey=True, sharex=True)
        f.set_size_inches(12,6)
    occupancy,occupancy_map,xrange,yrange=plot_occupancy_map(Unit,filtered=True,sigma=sigma,real_binsize=real_binsize,isfig=False,speed_threshold=speed_threshold)
    FR_map=[[],[]]
    for c in range(2):
        for t,spkpostemp in enumerate(Unit.ctx[c]['spkpos']):            
            if np.any(spkpostemp):
                x=spkpostemp['x'][spkpostemp['speed']>=speed_threshold]
                y=spkpostemp['y'][spkpostemp['speed']>=speed_threshold]
            else:
                x=[]
                y=[]
            H,xedges,yedges=np.histogram2d(x,y,bins=(xrange,yrange))
            H=H.T
            if filtered:
                H=gaussian_filter(H,sigma=sigma)
            FR_map[c].append(H/(occupancy_map[c][t]*(1/fps)))


                
    if isfig:
        for c in range(2):
            for t,spkpostemp in enumerate(Unit.ctx[c]['spkpos']):
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                im=ax[c,t].imshow(FR_map[c][t],extent=extent, vmin=0,vmax=np.max(FR_map),cmap='jet', origin='lower')
        f.subplots_adjust(right=0.8)
        cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
        f.colorbar(im,cax=cbar_ax)
        plt.show()
    return FR_map

def plot_ctxwise_PF(Unit,filtered=True,real_binsize=2,sigma=2,isfig=True,speed_threshold=5):
    fps=Unit.pos['FPS']
    if isfig:
        f, ax = plt.subplots(2,1, sharey=True, sharex=True)
        f.set_size_inches(12,6)
    occupancy,occupancy_map,xrange,yrange=plot_occupancy_map(Unit,filtered=False,real_binsize=real_binsize,isfig=False,speed_threshold=speed_threshold) #Do not filter just yet 
    occupancy_ctx=[[],[]]
    P_occup=[[],[]]
    dur_ctx=[[],[]] #the total duration in a context excluding the quiet periods
    for c,ctx_om in enumerate(occupancy_map):
        occupancy_ctx[c]=gaussian_filter(np.sum(ctx_om,axis=0),sigma=sigma)
        P_occup[c]=occupancy_ctx[c]/np.sum(occupancy_ctx[c])
        dur_ctx[c]=np.sum(np.sum(ctx_om,axis=0))*(1/fps)

    spk_map=[[],[]]
    spkc_ctx=[0,0] #the total spike count in a context 
    for c in range(2):
        for t,spkpostemp in enumerate(Unit.ctx[c]['spkpos']):
            if np.any(spkpostemp):
                x=spkpostemp['x'][spkpostemp['speed']>=speed_threshold]
                y=spkpostemp['y'][spkpostemp['speed']>=speed_threshold]
            else:
                x=[]
                y=[]
            spkc_ctx[c]+=np.size(x)
            H,xedges,yedges=np.histogram2d(x,y,bins=(xrange,yrange))
            H=H.T
            spk_map[c].append(H)
    FR_ctx=np.divide(spkc_ctx,dur_ctx)
    FR_map_ctx=[[],[]]
    for c,ctx_spkmap in enumerate(spk_map):
        FR_map_ctx[c]=gaussian_filter(((np.sum(ctx_spkmap,axis=0))/((occupancy_ctx[c])*(1/fps))),sigma=sigma)
    if isfig:
        for c,FRmap in enumerate(FR_map_ctx):
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            im=ax[c].imshow(FRmap,extent=extent, vmin=0,vmax=np.max(FR_map_ctx),cmap='jet', origin='lower')
        f.subplots_adjust(right=0.8)
        cbar_ax = f.add_axes([0.6, 0.15, 0.05, 0.7])
        f.colorbar(im,cax=cbar_ax)
        try:
            plt.title('Context '+Unit.context_selectivity[0]+' cell')
        except:
            pass
        plt.show()

    
    spatial_information=[0,0]
    for i,(pbin,FR_bin,meanFR) in enumerate(zip(P_occup,FR_map_ctx,FR_ctx)):
        for pbiny,FR_biny in zip(pbin,FR_bin):
            for pbinyx,FR_binyx in zip(pbiny,FR_biny):
                if pbinyx==0:
                    spatial_information[i]+=0 
                else:
                    spatial_information[i]+=pbinyx*(FR_binyx/meanFR)*(np.log2(FR_binyx/meanFR))
    if np.max(spatial_information)>=0.2:
        is_place_cell=True
    else:
        is_place_cell=False
        
    return is_place_cell,spatial_information

def make_pseudoUnit(Unit):
    pseudoUnit=copy.deepcopy(Unit)
    #We need to change the 'spkpos' in ctx: that means the x, y, and the speed
    #First let's shuffle the spikes within trial 
    for c,ctx in enumerate(Unit.ctx[:-1]):
        for n,(spkpostemp,spktemp) in enumerate(zip(ctx['spkpos'],ctx['spkt'])):
            if ctx['spkc'][n]>=0:
                t=ctx['index'][n]
                ctxstart=Unit.marker.door[1][t]-Unit.marker.record[0][t]
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
                pseudoUnit.ctx[c]['spkpos'][n]=align_spike_pos(postemp,pseudospk+ctxstart) 
    
    return pseudoUnit

def spatial_information_bootstrap(Unit,shuffle_num=100,isfig=True,real_binsize=1,sigma=2,speed_threshold=5):
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

def save_all_units(ensemble,pathname):
    for i,unit in enumerate(ensemble):
        if unit.max_spkc_ctx>=50:
            filename=os.path.join(pathname, unit.date +'_'+ unit.animal_id +'_'+ str(i)+ '.pkl')
            with open(filename,'wb') as output:
                pickle.dump(unit,output,pickle.HIGHEST_PROTOCOL)
# %% main
if __name__=='__main__':
    flag_buildtrack=True
    #ensemble,pos=buildneurons(build_tracking=flag_buildtrack,arena_reset=False)
    pathname=r'D:\EPHY_Qiushou\Ephy_data_organized\M028\20181112'  
    ftype='nex'
    ensemble,pos=buildneurons(pathname,
                              file_type=ftype,
                              build_tracking=flag_buildtrack,
                              arena_reset=False,
                              body_part='Body',
                              bootstrap=False                              
                              )
    # %% inspect each unit here
    plt.close('all')
    myUnit=ensemble[0]
    myUnit.waveform.Plotwf() 
    ctx=myUnit.split_context(pos,plot_ctx=True,plot_buffer=False)      
    if flag_buildtrack:
        real_binsize=0.5
        sigma=3
        speed_threshold=0
        #Plot the spike map 
        plot_spikemap(myUnit)       
        #occupancy,occupancy_map,xrange,yrange=plot_occupancy_map(myUnit,real_binsize=1,sigma=3,occupancy_threshold=10)
#        FR_map=plot_place_field(myUnit,real_binsize=real_binsize,sigma=sigma,speed_threshold=speed_threshold)    
#        is_place_cell,spatial_information=plot_ctxwise_PF(myUnit,real_binsize=real_binsize,sigma=sigma,speed_threshold=speed_threshold)
    #%%
#    cell_identity, ctx_pref=bootstrap_context(myUnit,isfig=True)
#    if flag_buildtrack:
#        spatial_information_observed,spatial_information_shuffle=spatial_information_bootstrap(myUnit,shuffle_num=200,real_binsize=real_binsize,sigma=sigma,speed_threshold=speed_threshold)
    #save_all_units(ensemble,pathname)

    