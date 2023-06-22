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

def gaiahub_BPMs(argv):  
    """
    Inputs
    """
       
    examples = '''Examples:
       
    gaiahub_BPMs --name "Fornax_dSph"
        
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, usage='%(prog)s [options]', 
                                     description='GaiaHub BPMs computes Bayesian proper motions (PM), parallaxes, and positions by'+\
                                     ' combining HST and Gaia data.', epilog=examples)
   
    # options
    parser.add_argument('--name', type=str, 
                        default = 'Output',
                        help='Name of directory to analyze.')
    parser.add_argument('--path', type=str, 
                        default = '/Volumes/Kevin_Astro/Astronomy/HST_Gaia_PMs/GaiaHub_results/', 
                        help='Path to GaiaHub results.')
    parser.add_argument('--overwrite', 
                        action='store_true',
                        help = 'Overwrite any previous results.')
    parser.add_argument('--overwrite_GH', 
                        action='store_true', 
                        help = 'Overwrite the GaiaHub summaries used for the Bayesian analysis.')
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
    
    #probably want to figure out how to ask the user for a thresh_time
    thresh_time = ((datetime.datetime(2023,5,22,15,20,38,259741)-datetime.datetime.utcfromtimestamp(0)).total_seconds()+7*3600)
    if field in ['COSMOS_field']:
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
                      f"--max_iterations {n_fit_max} --max_sources {max_stars} --max_images {max_images} "+\
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
#                           plot_indv_star_pms=plot_indv_star_pms)
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
                       plot_indv_star_pms=plot_indv_star_pms)

    return 


if __name__ == '__main__':
    
    testing = False
#    testing = True
    
    if not testing:
        gaiahub_BPMs(sys.argv[1:])


