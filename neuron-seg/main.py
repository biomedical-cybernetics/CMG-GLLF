import sys
import os
import cremi
import numpy as np
import math
from multiprocessing import Pool
import time
import json
import h5py
import subprocess
from cremi import Annotations, Volume
from cremi.io import CremiFile
sys.path.append('../data/20170312_mala_v2/')
from evaluate import evaluate

def single_seg(aff_fname,gt_fname,output_folder,thresholds,merfs):
    '''
    Segment one single affinity graph using thresholds and merge functions
    '''
    done_marker = os.path.join(output_folder,'done_marker.txt')
    # if os.path.exists(done_marker):
    #     return
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    for mf in merfs:
        time_start = time.time()
        evaluate(
            aff_fname,
            gt_fname,
            thresholds,
            [(output_folder+'/'+mf+'th_0.%f') % t for t in thresholds],
            custom_fragments=True,
            histogram_quantiles=False,
            discrete_queue=True,
            merge_function=mf,
            dilate_mask=1,
            mask_fragments=True,
            # keep_segmentation=False,
            keep_segmentation=True,
        )    
        time_consum = time.time()-time_start
        with open(os.path.join(output_folder,'time.txt'),'a') as f:
            f.write(mf + ' take ' + str(time_consum) + ' seconds to run \n')
    with open(done_marker,'a') as f:
        f.write('The tasks belong to this folder is done!')

def single_seg_for_par(args):
    aff_fname,gt_fname,output_folder,thresholds,merfs = args
    single_seg(aff_fname,gt_fname,output_folder,thresholds,merfs)

if __name__ == "__main__":
    # run for CM-GLLF sample B
    sample_name = 'B'
    mala_folder = '../data/20170312_mala_v2/'
    thresholds = list(np.around(np.arange(0.1,1,0.1),decimals=1))
    merfs = ('median_aff_histograms','85_aff_histograms','median_aff','85_aff')#merge functions    
    Mus = list(np.around(np.arange(0.1,1,0.1),decimals=1))
    Is = list(np.around(np.arange(0.1,1,0.1),decimals=1))
    # get the necessary file names
    gt_fname = os.path.join(mala_folder,'sample_'+sample_name+'.augmented.0.hdf')
    sota_aff_fname = os.path.join(mala_folder,'affs',('sample_'+sample_name+'.augmented.0.hdf'))
    trans_aff_path = '../data/trans_affs/'+sample_name+'/whole'
    seg_path = '../data/segs/'+sample_name+'/whole'
    search=False
    if search:
        # get the SOTA segmentation
        single_seg(sota_aff_fname,
                gt_fname,
                output_folder=os.path.join(seg_path,'sota'),
                thresholds=thresholds,
                merfs=merfs)
        # do affinity graph transformation
        subprocess.check_call(['matlab', '-batch', 'transform_cremi_b'])
        # get the CM-GLLF segmentation in parallel
        for inflec in Is:
            # make the list for parallel computing
            args_list = [
            (os.path.join(trans_aff_path,'mu_'+str(mu)+'_I_'+str(inflec)+'.hdf'),
            gt_fname,
            os.path.join(seg_path,'mu_'+str(mu)+'_I_'+str(inflec)),
            thresholds,
            merfs) for mu in Mus]
            # print(args_list)
            # print('\n')
            time_start = time.time()
            # with Pool() as pool:
            #     pool.map(single_seg_for_par, args_list)
            pool=Pool()
            try:
                pool.map(single_seg_for_par, args_list)
            finally:
                pool.close()  # prevent any more tasks from being added
                pool.join()   # wait for worker processes to exit            
            with open('./time.txt','a') as f:
                f.write('For this single I, it took ' + str(time.time()-time_start) +' seconds to run\n')
    else:
        # first get the CM-GLLF segmentation with mu=0.4 and inflection=0.3
        mu=0.4
        inflec=0.3
        aff_path = os.path.join(trans_aff_path,'mu_'+str(mu)+'_I_'+str(inflec)+'.hdf')
        out_folder = os.path.join(seg_path,'mu_'+str(mu)+'_I_'+str(inflec))
        thresholds = [0.9]
        merfs = ["median_aff_histograms"]
        single_seg(aff_path,
                gt_fname,
                out_folder,
                thresholds,
                merfs)
        # then get the SOTA segmentation
        thresholds = [0.8]
        merfs = ["median_aff"]   
        single_seg(sota_aff_fname,
                gt_fname,
                output_folder=os.path.join(seg_path,'sota'),
                thresholds=thresholds,
                merfs=merfs)
    
        