#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 09:55:11 2023

@author: kevinm
"""



'''
TODO:
    -make a summary file of results
        -each time an analysis is finished, loop over each of the stars 
        -make a new results file for that star if is doesn't exist summarizing the star's measures
        -for stars that have previous analyses (above the threshold time), compare the posterior PM distribution widths
        -save the results that have the smallest posterior PM uncertainties
        -make a global function that collects the results from all the stars and puts them in a big final csv

'''     


import os
import math
import gc
gc.enable()
import argparse
import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

import pandas as pd
import scipy
import scipy.stats as stats
import numpy as np

import emcee
import corner
from tqdm import tqdm
import time
import datetime
    
import astropy
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord
#import astropy.coordinates as coord
import astropy.units as u

#    from schwimmbad import MultiPool
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
# from multiprocessing import Process, Manager
# import multiprocessing
# import itertools    

import process_GaiaHub_outputs as process_GH
import GaiaHub_bayesian_pm_analysis_SINGLE as BPM


os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6

curr_scripts_dir = os.path.dirname(os.path.abspath(__file__))
n_max_threads = BPM.n_max_threads
last_update_time = BPM.last_update_time
cosmos_update_time = BPM.cosmos_update_time
final_file_ext = BPM.final_file_ext

def gaiahub_BPMs(argv):  
    """
    Inputs
    """
       
    examples = '''Examples:
        
    python GaiaHub_BPPPM.py --name "Fornax_dSph"
               
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, usage='%(prog)s [options]', 
                                     description='GaiaHub BPMs computes Bayesian proper motions (PM), parallaxes, and positions by'+\
                                     ' combining HST and Gaia data.', epilog=examples)
   
    # options
    parser.add_argument('--name', type=str, 
                        default = 'Output',
                        help='Name of directory to analyze.')
    parser.add_argument('--path', type=str, 
                        default = f'{os.getcwd()}/', 
                        help='Path to GaiaHub results.')
    parser.add_argument('--overwrite', 
                        action='store_true',
                        default=False,
                        help = 'Overwrite any previous results.')
    parser.add_argument('--overwrite_GH', 
                        action='store_true', 
                        default=False,
                        help = 'Overwrite the GaiaHub summaries used for the Bayesian analysis.')
    parser.add_argument('--individual_fit_only', 
                        action='store_true', 
                        default=False,
                        help = 'When looping over multiple images, only consider each image separately (do not fit together). Default False.')
    parser.add_argument('--repeat_first_fit', 
                        action='store_true', 
                        default=True,
                        help = 'Repeat the first fit. Useful for getting better measures of sources without Gaia priors. Default True.')
    parser.add_argument('--plot_indv_star_pms', 
                        action='store_true', 
                        default=True,
                        help = 'Plot the PM measurments for individual stars. Good for diagnostics.')
    parser.add_argument('--fit_population_pms', 
                        action='store_true', 
                        default=False,
                        help = 'Fit a 2D Gaussian population distribution to the posterior PM measurements. Default False.')
    parser.add_argument('--image_names', type=str, 
                        nargs='+', 
                        default = "y", 
                        help='Specify the HST image names to analyze.'+\
                             ' The default value, "y", will analyze all HST images in a directory (first separately, then combined).'+\
                             ' The user can also specify image names separated by spaces.')
    
    parser.add_argument('--max_iterations', type=int, 
                        default = 3, 
                        help='Maximum number of allowed iterations before convergence. Default 3.')
    parser.add_argument('--max_sources', type=int, 
                        default = 2000, 
                        help='Maximum number of allowed sources per image. Default 2000.')
    parser.add_argument('--max_images', type=int, 
                        default = 10, 
                        help='Maximum number of allowed images to be fit together at the same time. Default 10.')
    
    parser.add_argument('--n_processes', type = int, 
                        default = n_max_threads, 
                        help='The number of jobs to run in parallel. Default uses all the available processors. For single-threaded execution use 1.')

    parser.add_argument('--fit_all_hst', 
                        action='store_true', 
                        default=False,
                        help = 'Find all HST-identified sources (even those without Gaia matches) and cross-matches between HST images.'+\
                        ' THIS IS A BETA FEATURE! THE CROSS-MATCHING WILL LIKELY BE IMPROVED IN THE FUTURE!'+\
                        ' Default False, but only used when one or more HST images share additional sources.')

    if len(argv)==0:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args(argv)
    field = args.name
    path = args.path
    overwrite_previous = args.overwrite
    overwrite_GH_summaries = args.overwrite_GH
    image_names = args.image_names
    n_fit_max = args.max_iterations
    max_stars = args.max_sources
    max_images = args.max_images
    redo_without_outliers = args.repeat_first_fit
    plot_indv_star_pms = args.plot_indv_star_pms
    n_threads = args.n_processes
    fit_population_pms = args.fit_population_pms
    individual_fit_only = args.individual_fit_only
    fit_all_hst = args.fit_all_hst
    
    #probably want to figure out how to ask the user for a thresh_time
    thresh_time = last_update_time
    
#    print('image_names',image_names)
        
    if (image_names == "y") or (image_names == "Y"):
        #then analyse all the images in a field along a path
#        all_image_analysis(field,path,
#                           overwrite_previous=overwrite_previous,
#                           overwrite_GH_summaries=overwrite_GH_summaries,
#                           thresh_time=thresh_time,
#                           n_fit_max=n_fit_max,
#                           max_images=max_images,
#                           redo_without_outliers=redo_without_outliers,
#                           max_stars=max_stars,
#                           plot_indv_star_pms=plot_indv_star_pms)
        linked_image_list = BPM.link_images(field,path,
                                        overwrite_previous=overwrite_previous,
                                        overwrite_GH_summaries=overwrite_GH_summaries,
                                        thresh_time=thresh_time)
        
        if individual_fit_only:
            temp_list = []
            for entry in linked_image_list:
                if len(entry) == 1:
                    temp_list.append(entry)
            linked_image_list = temp_list
        
        for entry_ind,entry in enumerate(linked_image_list):
            print(f'\n\n\nCurrently on list number {entry_ind+1} of {len(linked_image_list)}.')
            
            #check if previous analysis exists
            image_name = '_'.join(entry)
            outpath = f'{path}{field}/Bayesian_PMs/{image_name}/'
                                
            final_file = f'{outpath}{image_name}{final_file_ext}'
            if os.path.isfile(final_file):
                file_time = os.path.getmtime(final_file)
                if (file_time > thresh_time) and (not overwrite_previous):
                    print(f'SKIPPING fit of image {image_name} in {field} because it has recently been analysed.')
                    continue
            
            entry_list = ' '.join(entry)
            overwrite_previous_string = ''
            if overwrite_previous:
                overwrite_previous_string = '--overwrite '
            overwrite_GH_string = ''
            if overwrite_GH_summaries:
                overwrite_GH_string = '--overwrite_GH '
            repeat_string = ''
            if redo_without_outliers:
                repeat_string = '--repeat_first_fit '
            plot_string = ''
            if plot_indv_star_pms:
                plot_string = '--plot_indv_star_pms '
            pop_fit_string = ''
            if fit_population_pms:
                pop_fit_string = '--fit_population_pms '
            fit_all_hst_string = ''
            if fit_all_hst:
                fit_all_hst_string = '--fit_all_hst '
                
            #use os.system call so that each image set analysis is separate 
            #to prevent a creep of memory leak (probably from numpy) that 
            #uses up all the RAM and slows down the calculations significantly
            os.system(f"python {curr_scripts_dir}/GaiaHub_bayesian_pm_analysis_SINGLE.py --name {field} --path {path} --image_list {entry_list} "+\
                      f"--max_iterations {n_fit_max} --max_sources {max_stars} --max_images {max_images} --n_processes {n_threads} "+\
                      f"{overwrite_previous_string}{overwrite_GH_string}{repeat_string}{plot_string}{pop_fit_string}{fit_all_hst_string}")
            
                        
#            BPM.analyse_images(entry,
#                           field,path,
#                           overwrite_previous=overwrite_previous,
#                           overwrite_GH_summaries=overwrite_GH_summaries,
#                           thresh_time=thresh_time,
#                           n_fit_max=n_fit_max,
#                           max_images=max_images,
#                           redo_without_outliers=redo_without_outliers,
#                           max_stars=max_stars,
#                           plot_indv_star_pms=plot_indv_star_pms,
#                            n_threads = n_threads)
            gc.collect()

    else:
        
        BPM.analyse_images(image_names,
                       field,path,
                       overwrite_previous=overwrite_previous,
                       overwrite_GH_summaries=overwrite_GH_summaries,
                       thresh_time=thresh_time,
                       n_fit_max=n_fit_max,
                       max_images=max_images,
                       redo_without_outliers=redo_without_outliers,
                       max_stars=max_stars,
                       plot_indv_star_pms=plot_indv_star_pms,
                       fit_population_pms=fit_population_pms,
                       n_threads=n_threads,
                       fit_all_hst=fit_all_hst)

    return 


if __name__ == '__main__':
    
    testing = False
#    testing = True
    
    if not testing:
        gaiahub_BPMs(sys.argv[1:])
    else:
        
        
        fields = [
        '47Tuc',
        'Arp2',
        'COSMOS_field',
        'DDO_216',
        'Draco_dSph',
        'E3',
        'Fornax_dSph',
        'IC_10',
        'Leo_I',
        'LMC',
        'M31',
        'NGC_205',
        'NGC2419',
        'Pal1',
        'Pal2',
        'Pal4',
        'Pal13',
        'Pal15',
        'Sculptor_dSph',
        'Sextans_dSph',
        'Terzan8',
        ]
        
        fields = [
        'COSMOS_field',
        'Fornax_dSph',
        'Sculptor_dSph',
        'Sextans_dSph',
        'Draco_dSph',
        'E3',
        'Leo_II',
        'Leo_A',
        'NGC6822',
        'Phoenix',
        'SAG_DIG',
        'Leo_T',
        'IC_1613',
        'WLM',
        'Arp2',
        'DDO_216',
        'IC_10',
        'Leo_I',
        'NGC_205',
        'Pal1',
        'Pal4',
        'Pal13',
        'Pal15',
        'M31',
        'Pal2',
        'Terzan8',
        'NGC2419',
        '47Tuc',
        ]
        
        fields = [        
        'COSMOS_field',
        ]
        
#        fields = [
#        'Sculptor_dSph',
#        '47Tuc',
#        ]
        
#        fields = [
#        'FAKE_ROMAN_01',
#        ]
#        
#        fields = [
##        'FAKE_ROMAN_01',
#        'FAKE_FIELD_07',
#        'FAKE_FIELD_01',
#        'FAKE_FIELD_06',
#        'FAKE_FIELD_05',
#        ]
#        
#        fields = [
##        'FAKE_ROMAN_01',
##        'FAKE_ROMAN_02',
#        'FAKE_ROMAN_03',
##        'FAKE_FIELD_07',
##        'FAKE_FIELD_01',
#        ]
#        
        
        overwrite_previous = True
        overwrite_previous = False
        overwrite_GH_summaries = False
        overwrite_GH_summaries = True
        n_fit_max = 3
        n_fit_max = 5
        max_stars = 2000
        max_stars = 500
        max_images = 10
        max_images = 5
        max_images = 2
        redo_without_outliers = True
        plot_indv_star_pms = True
        n_threads = n_max_threads
        
        individual_fit_only = False
#        individual_fit_only = True
        
        fit_all_hst = False
        fit_population_pms = False
        
        #probably want to figure out how to ask the user for a thresh_time, but for now, it is the last time I changed the main code
        thresh_time = last_update_time
        
        image_names = 'y'
        
        for field in fields:
            
            thresh_time = last_update_time
            if field == 'COSMOS_field':
                thresh_time = cosmos_update_time
            
            if 'FAKE' in field:
                path = '/Volumes/Kevin_Astro/Astronomy/HST_Gaia_PMs/FAKE_GaiaHub_results/'
            else:
                path = '/Volumes/Kevin_Astro/Astronomy/HST_Gaia_PMs/GaiaHub_results/'
            
#            process_GH.collect_gaiahub_results(field,path=path,overwrite=True)
#            continue
            max_stars = 500
            max_stars = 1000
            if field in ['COSMOS_field']:
                fit_population_pms = False
                overwrite_GH_summaries = False
                overwrite_previous = False
            elif 'FAKE' in field:
                fit_population_pms = False
                overwrite_GH_summaries = False
                max_stars = 1000*6
            else:
                fit_population_pms = True
                overwrite_GH_summaries = False
            overwrite_GH_summaries = False
            
            linked_image_list = BPM.link_images(field,path,
                                            overwrite_previous=overwrite_previous,
                                            overwrite_GH_summaries=overwrite_GH_summaries,
                                            thresh_time=thresh_time)
            orig_linked_image_list = linked_image_list
            
            outpath = f'{path}{field}/Bayesian_PMs/'
            trans_file_df = pd.read_csv(f'{outpath}gaiahub_image_transformation_summaries.csv')
            trans_file_names = trans_file_df['image_name'].to_numpy()
            trans_file_mjds = trans_file_df['HST_time'].to_numpy()
            indv_list = []
            indv_list_times = []
            for entry in linked_image_list:
                if len(entry) == 1:
                    indv_list.append(entry)
                    indv_list_times.append(trans_file_mjds[np.where(trans_file_names == entry)[0][0]])
            
            indv_list_array = np.array(indv_list)[:,0]
            indv_list_times = np.array(indv_list_times)
            
            nearest_time_neighbour_inds = np.zeros(len(indv_list_times)).astype(int)
            match_time_list = []
            for j,mjd in enumerate(indv_list_times):
                curr_time_diffs = np.abs(indv_list_times-mjd)
                curr_time_diffs[j] = np.inf
                nearest_time_neighbour_inds[j] = np.argmin(curr_time_diffs)
                match_time_list.append([indv_list_array[j],indv_list_array[nearest_time_neighbour_inds[j]]])
                
#            group_times = []
#            time_inds = np.zeros(len(indv_list_times)).astype(int)
#            for j,mjd in enumerate(indv_list_times):
#                if j == 0:
#                    group_times.append(mjd)
#                    time_inds[j] = 0
#                    continue
#                curr_times = np.array(group_times)
#                curr_time_diffs = np.abs(curr_times-mjd)
#                nearest_ind = np.argmin(curr_time_diffs)
#                nearest_diff = curr_time_diffs[nearest_ind]
#                if nearest_diff > 0.5:
#                    #then add the new time to the list
#                    time_inds[j] = len(group_times)
#                    group_times.append(mjd)
#                else:
#                    time_inds[j] = nearest_ind
#            group_times = np.array(group_times)
#            match_time_list = []
#            for time_ind in np.unique(time_inds):
#                match_time_list.append(list(indv_list_array[np.where(time_inds == time_ind)[0]]))
            
            if individual_fit_only:
                linked_image_list = indv_list
            
            if field == 'COSMOS_field':
                #only list of nearby observation times
                
                new_list = []
#                for j,entry in enumerate(tqdm(indv_list_array,total=len(indv_list_array))):
                for j,entry in enumerate(indv_list_array):
                    linked_df = pd.read_csv(f'{outpath}{entry}/{entry}_linked_images.csv')
                    curr_names = linked_df['image_name'].to_numpy()
                    curr_time_offsets = linked_df['time_offset'].to_numpy()
                    oldest_ind = np.argmin(curr_time_offsets)
                    newest_ind = np.argmax(curr_time_offsets)
                    
#                    curr_im_ind = np.argmin(np.abs(curr_time_offsets))
#                    max_offset_ind = np.argmax(np.abs(curr_time_offsets))
#                    if curr_time_offsets[curr_im_ind] < curr_time_offsets[max_offset_ind]:
#                        newest_ind = max_offset_ind
#                        oldest_ind = curr_im_ind
#                    else:
#                        newest_ind = curr_im_ind
#                        oldest_ind = max_offset_ind
                    
                    hst_baseline = curr_time_offsets[newest_ind]-curr_time_offsets[oldest_ind]
                    if hst_baseline < 0.3:
#                    if abs(hst_baseline) < 1:
                        continue
                    new_entry = [curr_names[oldest_ind],curr_names[newest_ind]]
                    if new_entry not in new_list:
                        new_list.append(new_entry)
                        
                linked_image_list.extend(new_list)
                        
#                linked_image_list.extend(match_time_list)
                
#                linked_image_list = new_list
                
#                linked_image_list = match_time_list
#            elif field == 'FAKE_ROMAN_01':
#                #only list of nearby observation times
#                linked_image_list = [
#                        ['im01n180t092s101'],
#                        ['im02n180t092s101'],
#                        ['im03n180t092s101'],
#                        ['im01n180t092s101','im02n180t092s101','im03n180t092s101'],
#                        ]
#            elif 'FAKE_FIELD' in field:
#                #only list of nearby observation times
#                linked_image_list = [
#                        ['im01n180t092s101'],
#                        ['im02n180t092s101'],
#                        ['im03n180t092s101'],
#                        ['im01n180t092s101','im02n180t092s101','im03n180t092s101'],
#                        ]
            
            for entry_ind,entry in enumerate(linked_image_list):
                print(f'\n\n\nCurrently on list number {entry_ind+1} of {len(linked_image_list)}.\n')
                
#                if entry_ind > 0:
#                    overwrite_GH_summaries = False
#                if (field in ['M31']) and (entry_ind == 0):
#                    overwrite_GH_summaries = True

                #check if previous analysis exists
                image_name = '_'.join(entry)
                outpath = f'{path}{field}/Bayesian_PMs/{image_name}/'
                                
#                if 'FAKE_FIELD' in field:
#                    chosen_nstar = 200
#                    chosen_nstar = 10
#                    nstar_string = 'n%03d'%chosen_nstar
#                    chosen_nstar2 = 5
#                    nstar_string2 = 'n%03d'%chosen_nstar2
#                    
#                    chosen_nstar = 200
#                    nstar_string = 'n%03d'%chosen_nstar
#                    chosen_nstar2 = 150
#                    nstar_string2 = 'n%03d'%chosen_nstar2
#                    skip = True
#                    if (nstar_string in image_name):
#                        skip = False
#                    elif (nstar_string2 in image_name):
#                        skip = False
#                    
#                    curr_n_star = int(entry[0].split('n')[1][:3])
#                    curr_dtime = float(entry[0].split('t')[1][:3])/10
#                    curr_seed = int(entry[0].split('s')[1][:3])
##                    if curr_n_star not in [200,150,100]:
##                    if curr_n_star not in [5,10,15,20]:
#                    if curr_n_star not in [10]:
#                        skip = True
#                    else:
#                        skip = False
#                    
##                    if field in ['FAKE_FIELD_07']:
##                        skip = False
#                    
#                    if skip:
#                        print(f'SKIPPING fit of image {image_name} in {field} because it does not have the right number of sources.')
#                        continue
#                    
#                    if curr_dtime not in [15]:
#                        skip = True
#                    else:
#                        skip = False
#                    
#                    if field in ['FAKE_FIELD_07']:
#                        skip = False
#
#                    if skip:
#                        print(f'SKIPPING fit of image {image_name} in {field} because it does not have the right time offset.')
#                        continue
                    
#                    if curr_seed not in np.arange(101,109+1e-10,1).astype(int):
##                    if curr_seed not in [101]:
#                        continue
                
                if field in ['FAKE_FIELD_06','FAKE_FIELD_05']:
                    if ('im01' not in image_name):
                        print(f'SKIPPING fit of image {image_name} in {field} because it is a copy of a previous analysis.')
                        continue
                    
                overwrite_previous = False
                if 'FAKE_ROMAN' in field:
                    pass
#                    chosen_seed = 102
#                    seed_string = 's%03d'%chosen_seed
#                    if seed_string not in image_name:
#                        print(f'SKIPPING fit of image {image_name} in {field} because it does not have {chosen_seed} as a seed.')
#                        continue
#                    
##                    if ('t088' in image_name) and (len(entry) == 1):
##                        overwrite_previous = True
##                    if ('t088' in image_name) and (len(entry) > 1):
##                        overwrite_previous = True
##                    if ('t089' in image_name) and (len(entry) == 1):
##                        overwrite_previous = True
##                    if ('t089' in image_name) and (len(entry) > 1):
##                        overwrite_previous = True
##                    if ('t091' in image_name) and (len(entry) == 1):
##                        overwrite_previous = True
##                    if ('t091' in image_name) and (len(entry) > 1):
##                        overwrite_previous = True
##                    if ('t085' in image_name) and (len(entry) == 1):
##                        overwrite_previous = True
##                    if ('t085' in image_name) and (len(entry) > 1):
##                        overwrite_previous = True
##                    if ('t087' in image_name) and (len(entry) == 1):
##                        overwrite_previous = True
#                    if ('t087' in image_name) and (len(entry) > 1):
#                        overwrite_previous = True
##                    if ('t093' in image_name) and (len(entry) == 1):
##                        overwrite_previous = True
#                    if ('t093' in image_name) and (len(entry) > 1):
#                        overwrite_previous = True
                        
#                if field in ['FAKE_FIELD_07']:
#                    overwrite_previous = True
                    
                        
#                final_file = f'{outpath}{image_name}{final_file_ext}'
#                if os.path.isfile(final_file):
#                    file_time = os.path.getmtime(final_file)
#                    if (file_time > thresh_time) and (not overwrite_previous):
#                        print(f'SKIPPING fit of image {image_name} in {field} because it has recently been analysed.')
#                        continue
                    
                fit_all_hst = False
                if field in ['COSMOS_field']:
                    if len(entry) > 1:
                        fit_all_hst = True
                
                entry_list = ' '.join(entry)
                overwrite_previous_string = ''
                if overwrite_previous:
                    overwrite_previous_string = '--overwrite '
                overwrite_GH_string = ''
                if overwrite_GH_summaries:
                    overwrite_GH_string = '--overwrite_GH '
                repeat_string = ''
                if redo_without_outliers:
                    repeat_string = '--repeat_first_fit '
                plot_string = ''
                if plot_indv_star_pms:
                    plot_string = '--plot_indv_star_pms '
                pop_fit_string = ''
                if fit_population_pms:
                    pop_fit_string = '--fit_population_pms '
                fit_all_hst_string = ''
                if fit_all_hst:
                    fit_all_hst_string = '--fit_all_hst '
                    
                #use os.system call so that each image set analysis is separate 
                #to prevent a creep of memory leak (probably from numpy) that 
                #uses up all the RAM and slows down the calculations significantly
                os.system(f"python {curr_scripts_dir}/GaiaHub_bayesian_pm_analysis_SINGLE.py --name {field} --path {path} --image_list {entry_list} "+\
                          f"--max_iterations {n_fit_max} --max_sources {max_stars} --max_images {max_images} --n_processes {n_threads} "+\
                          f"{overwrite_previous_string}{overwrite_GH_string}{repeat_string}{plot_string}{pop_fit_string}{fit_all_hst_string}")
   
            if field in ['Fornax_dSph','Sculptor_dSph','Draco_dSph']:
                entry = orig_linked_image_list[-1]         
                
                if field == 'Sculptor_dSph':
#                    entry = ['j8hofes6q','j8hofesaq','j8hoffsjq','j8hoffsoq','j8hofnt8q']
                    entry = ['j8hofafuq','j8hofafyq','j8hof5gbq','j8hof5ghq','j8hof5gmq','j8hofigsq']
                elif field == 'Draco_dSph':
                    entry = ['j93404ebq','j93404ekq','j93404f1q','j93404fbq','j9qv04p3q','j9qv04pdq','j9qv04pnq','j9qv04pxq','jc2z01ekq','jc2z01feq']
                    
                max_im_nums = np.arange(2,min(10,len(entry))+1e-10,1).astype(int)
#                max_im_nums[0] = max_im_nums[-1] #do the last first
#                max_im_nums[-1] = 2
                
                for ind,curr_max_ims in enumerate(max_im_nums):
                
                    print(f'\n\n\nCurrently on list number {ind+1} of {len(max_im_nums)} using {curr_max_ims} max number of images.\n')
                    
    #                if entry_ind > 0:
    #                    overwrite_GH_summaries = False
    #                if (field in ['M31']) and (entry_ind == 0):
    #                    overwrite_GH_summaries = True
    
                    #check if previous analysis exists
                    image_name = '_'.join(entry)
                    outpath = f'{path}{field}/Bayesian_PMs/{image_name}/'
                                        
                    final_file = f'{outpath}{image_name}{final_file_ext}'
#                    if os.path.isfile(final_file):
#                        file_time = os.path.getmtime(final_file)
#                        if (file_time > thresh_time) and (not overwrite_previous):
#                            print(f'SKIPPING fit of image {image_name} in {field} because it has recently been analysed.')
#                            continue
                    
                    entry_list = ' '.join(entry)
                    overwrite_previous_string = ''
                    if overwrite_previous:
                        overwrite_previous_string = '--overwrite '
                    overwrite_GH_string = ''
                    if overwrite_GH_summaries:
                        overwrite_GH_string = '--overwrite_GH '
                    repeat_string = ''
                    if redo_without_outliers:
                        repeat_string = '--repeat_first_fit '
                    plot_string = ''
                    if plot_indv_star_pms:
                        plot_string = '--plot_indv_star_pms '
                    pop_fit_string = ''
                    if fit_population_pms:
                        pop_fit_string = '--fit_population_pms '
                    fit_all_hst_string = ''
                    if fit_all_hst:
                        fit_all_hst_string = '--fit_all_hst '
                        
                    #use os.system call so that each image set analysis is separate 
                    #to prevent a creep of memory leak (probably from numpy) that 
                    #uses up all the RAM and slows down the calculations significantly
                    os.system(f"python {curr_scripts_dir}/GaiaHub_bayesian_pm_analysis_SINGLE.py --name {field} --path {path} --image_list {entry_list} "+\
                              f"--max_iterations {n_fit_max} --max_sources {max_stars} --max_images {curr_max_ims} --n_processes {n_threads} "+\
                              f"{overwrite_previous_string}{overwrite_GH_string}{repeat_string}{plot_string}{pop_fit_string}{fit_all_hst_string}")
                
    
        print(f'\n\nDone with field {field}\n\n\n')
