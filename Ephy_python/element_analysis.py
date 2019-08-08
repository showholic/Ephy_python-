# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 15:25:18 2019

@author: Qixin
"""

from build_unit import *
#%%
flag_buildtrack=True
    #ensemble,pos=buildneurons(build_tracking=flag_buildtrack,arena_reset=False)
pathname=r'D:\EPHY_Qiushou\Ctx_element_data\192086\20190610'  
ftype='nex'
ensemble,pos=buildneurons(pathname,
                          file_type=ftype,
                          build_tracking=True,
                          arena_reset=True,
                          body_part='Body',
                          bootstrap=False                              
                          )
#%%
import matplotlib.backends.backend_pdf
import matplotlib._pylab_helpers
plt.close('all') #make sure everything's closed
for myUnit in ensemble: 
    
    myUnit.waveform.Plotwf() 
    spkhist_wholetrial(myUnit,tbin=1,sigma=1)
    plot_context_summary(myUnit)      

    real_binsize=1
    sigma=2
    speed_threshold=2
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
