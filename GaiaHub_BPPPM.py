#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 09:55:11 2023

@author: kevinm
"""

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
    individual_fit_only = args.individual_fit_only
    
    #probably want to figure out how to ask the user for a thresh_time
    thresh_time = ((datetime.datetime(2023, 6, 16, 15, 47, 19, 264136)-datetime.datetime.utcfromtimestamp(0)).total_seconds()+7*3600)
    
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
            print(f'\nCurrently on list number {entry_ind+1} of {len(linked_image_list)}.\n')
            
            #check if previous analysis exists
            image_name = '_'.join(entry)
            outpath = f'{path}{field}/Bayesian_PMs/{image_name}/'
                                
            final_fig = f'{outpath}{image_name}_posterior_position_uncertainty.png'
            if os.path.isfile(final_fig):
                file_time = os.path.getmtime(final_fig)
                if (file_time > thresh_time) or (not overwrite_previous):
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
                
            #use os.system call so that each image set analysis is separate 
            #to prevent a creep of memory leak (probably from numpy) that 
            #uses up all the RAM and slows down the calculations significantly
            os.system(f"python {curr_scripts_dir}/GaiaHub_bayesian_pm_analysis_SINGLE.py --name {field} --path {path} --image_list {entry_list} "+\
                      f"--max_iterations {n_fit_max} --max_sources {max_stars} --max_images {max_images} --n_processes {n_threads} "+\
                      f"{overwrite_previous_string}{overwrite_GH_string}{repeat_string}{plot_string}")
            
                        
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
                       n_threads = n_threads)

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
        'Fornax_dSph',
        'Sculptor_dSph',
        'Sextans_dSph',
        'Draco_dSph',
        '47Tuc',
        ]
        
        path = '/Volumes/Kevin_Astro/Astronomy/HST_Gaia_PMs/GaiaHub_results/'
        overwrite_previous = True
        overwrite_GH_summaries = False
        n_fit_max = 3
        max_stars = 2000
        max_images = 10
        redo_without_outliers = True
        plot_indv_star_pms = True
        n_threads = n_max_threads
        
        individual_fit_only = False
        individual_fit_only = True
        
        #probably want to figure out how to ask the user for a thresh_time, but for now, it is the last time I changed the main code
        thresh_time = ((datetime.datetime(2023, 6, 16, 15, 47, 19, 264136)-datetime.datetime.utcfromtimestamp(0)).total_seconds()+7*3600)
        
        image_names = 'y'
        
        for field in fields:
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
                print(f'\n\n\nCurrently on list number {entry_ind+1} of {len(linked_image_list)}.\n')
                
                #check if previous analysis exists
                image_name = '_'.join(entry)
                outpath = f'{path}{field}/Bayesian_PMs/{image_name}/'
                                    
                final_fig = f'{outpath}{image_name}_posterior_position_uncertainty.png'
                if os.path.isfile(final_fig):
                    file_time = os.path.getmtime(final_fig)
                    if (file_time > thresh_time) or (not overwrite_previous):
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
                    
                #use os.system call so that each image set analysis is separate 
                #to prevent a creep of memory leak (probably from numpy) that 
                #uses up all the RAM and slows down the calculations significantly
                os.system(f"python {curr_scripts_dir}/GaiaHub_bayesian_pm_analysis_SINGLE.py --name {field} --path {path} --image_list {entry_list} "+\
                          f"--max_iterations {n_fit_max} --max_sources {max_stars} --max_images {max_images} --n_processes {n_threads} "+\
                          f"{overwrite_previous_string}{overwrite_GH_string}{repeat_string}{plot_string}")
            


