#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6

def gaiahub_single_BPMs(argv):  
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
    parser.add_argument('--image_list', type=str, 
                        nargs='+', 
                        default = [None], 
                        help='Specify the list of HST image name to analyze together.')
    
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
    image_list = args.image_list
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
        
    print()
        
    analyse_images(image_list,
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




font = {'family' : 'serif',
#        'weight' : 'bold',
        'size'   : 16,}
matplotlib.rc('font', **font)
    
gaia_dr3_date = '2017-05-28'
gaia_dr3_time = Time(gaia_dr3_date)

pixel_scale_ratios = {'ACS':50,'WFC3':40} #mas/pixel

get_matrix_params = process_GH.get_matrix_params
correlation_names = process_GH.correlation_names

def link_images(field,path,
                       overwrite_previous=True,overwrite_GH_summaries=False,thresh_time=0):
    '''
    
    '''
    
    print(f'Reading in GaiaHub summaries for field {field} to determine how HST images are linked together.')
    
    outpath = f'{path}{field}/Bayesian_PMs/'
    if (not os.path.isfile(f'{outpath}gaiahub_image_transformation_summaries.csv')) or (overwrite_GH_summaries):
        process_GH.collect_gaiahub_results(field,path=path,overwrite=overwrite_GH_summaries)
        
    trans_file_df = pd.read_csv(f'{outpath}gaiahub_image_transformation_summaries.csv')
    
    image_names = np.array(trans_file_df['image_name'].to_list())

    linked_image_info = {}
    for image_name in image_names:
        if not os.path.isfile(f'{outpath}/{image_name}/{image_name}_linked_images.csv'):
            process_GH.image_lister(field,path)
        
#        curr_dict = {'image_name':indv_linked_image_lists[curr_image_name],
#                     'time_offset':indv_linked_image_dtimes[curr_image_name],
#                     'n_common_sources':indv_linked_image_matches[curr_image_name]}
        linked_image_info[image_name] = pd.read_csv(f'{outpath}/{image_name}/{image_name}_linked_images.csv')

    full_linked_images = {}
    for image_name in image_names:
        if image_name not in full_linked_images:
            full_linked_images[image_name] = [image_name]
        for other_image_name in linked_image_info[image_name]['image_name']:
            if other_image_name not in full_linked_images:
                full_linked_images[other_image_name] = [image_name]
            if other_image_name not in full_linked_images[image_name]:
                full_linked_images[image_name].append(other_image_name)
    full_linked_image_mjds = {}
    for image_name in image_names:
        full_linked_image_mjds[image_name] = np.zeros(len(full_linked_images[image_name]))
        for j in range(len(full_linked_image_mjds[image_name])):
            match_ind = np.where(image_names == full_linked_images[image_name][j])[0][0]
            full_linked_image_mjds[image_name][j] = trans_file_df['HST_time'][match_ind]
        new_order = np.argsort(full_linked_image_mjds[image_name])
        full_linked_image_mjds[image_name] = list(np.array(full_linked_image_mjds[image_name])[new_order])
        full_linked_images[image_name] = list(np.array(full_linked_images[image_name])[new_order])
        
                
    linked_image_list = []
    for image_ind,image_name in enumerate(image_names):
        if image_ind == 0:
            linked_image_list.append(full_linked_images[image_name])
        else:
            found_match = False
            for item in full_linked_images[image_name]:
                for lind,llist in enumerate(linked_image_list):
                    if item in llist:
                        #then this list should
                        found_match = True
                        break
                if found_match:
                    break
            if found_match:
                for item in full_linked_images[image_name]:
                    if item not in llist:
                        linked_image_list[lind].append(item)
            if not found_match:
                #then make a new entry
                linked_image_list.append(full_linked_images[image_name])
    print(f'Found {len(linked_image_list)} sets of linked HST images.')
                
    for image_ind,image_name in enumerate(image_names):
        if [image_name] not in linked_image_list:
            linked_image_list.insert(image_ind,[image_name])
    print(f'Must perform {len(linked_image_list)} analyses.')
    print()
    
    return linked_image_list


def lnpost_vector(params,
                  n_param_indv,x,x0s,ags,bgs,cgs,dgs,img_nums,xy,xy0s,xy_g,hst_covs,
                  proper_offset_jacs,proper_offset_jac_invs,unique_gaia_offset_inv_covs,use_inds,unique_inds,
                  parallax_offset_vector,delta_times,unique_inv_inds,delta_time_identities,global_pm_inv_cov,
                  unique_gaia_pm_inv_covs,unique_V_theta_i_inv_dot_theta_i,unique_V_mu_i_inv_dot_mu_i,
                  V_mu_global_inv_dot_mu_global,unique_gaia_offsets,unique_gaia_pms,global_pm_mean,
                  unique_gaia_parallax_ivars,global_parallax_ivar,unique_ids,unique_gaia_parallaxes,
                  global_parallax_mean,log_unique_gaia_pm_covs_det,log_global_pm_cov_det,
                  unique_gaia_parallax_vars,log_unique_gaia_offset_covs_det,unique_keep,
                  seed=101,n_pms=1) -> np.ndarray:
#    print(params)
#    np.random.seed(seed)
#    print(seed)
#    seed = (os.getpid() * int(time.time())) % 123456789
    np.random.seed(seed)

    #use the current transformation parameter draws to define draws of PMs and parallaxes
    #then evaluate posterior probabilities from those, and sum over to get trans_params posterior probs
    ags = params[0::n_param_indv]
    bgs = params[1::n_param_indv]
    w0s = params[2::n_param_indv]
    z0s = params[3::n_param_indv]
    cgs = params[4::n_param_indv]
    dgs = params[5::n_param_indv]
    # x0s = param_outputs[:,0]
    # y0s = param_outputs[:,1]

    # if np.any(np.abs(w0s-param_outputs[:,2]) > 1000) or np.any(np.abs(z0s-param_outputs[:,3]) > 1000):
    #     result = np.zeros((len(unique_ids),5+1))
    #     result[:,0] = -np.inf
    #     return result                
    
    star_hst_gaia_pos = np.zeros((len(x),2)) #in gaia pixels
    star_hst_gaia_pos_cov = np.zeros((len(x),2,2)) #in gaia pixels
    star_ratios = np.zeros(len(x))
        
    matrices = np.zeros((len(x),2,2))
    matrices_T = np.zeros((len(x),2,2))
    
    star_ratios[:] = 1
    poss_matrices = np.zeros((len(x0s),2,2))
    poss_matrices[:,0,0] = ags
    poss_matrices[:,0,1] = bgs
    poss_matrices[:,1,0] = cgs
    poss_matrices[:,1,1] = dgs
    poss_matrices_T = np.copy(poss_matrices)
    poss_matrices_T[:,0,1] = cgs
    poss_matrices_T[:,1,0] = bgs
    
    matrices = poss_matrices[img_nums]
    matrices_T = poss_matrices_T[img_nums]
    wz0s = np.array([w0s,z0s]).T[img_nums]
    
    xy_trans = np.einsum('nij,nj->ni',matrices,xy-xy0s)+wz0s
    star_hst_gaia_pos = xy_g-xy_trans
        
    star_hst_gaia_pos_cov = np.einsum('nij,njk->nik',matrices,np.einsum('nij,njk->nik',hst_covs,matrices_T))
    star_ratios = star_ratios[:,None,None]
    star_hst_gaia_pos_inv_cov = np.linalg.inv(star_hst_gaia_pos_cov)    
                    
    jac_V_data_inv_jac = np.einsum('nji,njk->nik',proper_offset_jacs,np.einsum('nij,njk->nik',star_hst_gaia_pos_inv_cov,proper_offset_jacs))
    inv_jac_dot_d_ij = np.einsum('nij,nj->ni',proper_offset_jac_invs,star_hst_gaia_pos)
    summed_jac_V_data_inv_jac = np.add.reduceat(jac_V_data_inv_jac*use_inds[:,None,None],unique_inds)
    Sigma_theta_i_inv = unique_gaia_offset_inv_covs+summed_jac_V_data_inv_jac
    Sigma_theta_i = np.linalg.inv(Sigma_theta_i_inv)
    
    jac_V_data_inv_jac_dot_parallax_vects = np.einsum('nij,nj->ni',jac_V_data_inv_jac,parallax_offset_vector)
    summed_jac_V_data_inv_jac_dot_parallax_vects = np.add.reduceat(jac_V_data_inv_jac_dot_parallax_vects*use_inds[:,None],unique_inds)
    jac_V_data_inv_jac_dot_d_ij = np.einsum('nij,nj->ni',jac_V_data_inv_jac,inv_jac_dot_d_ij)
    summed_jac_V_data_inv_jac_dot_d_ij = np.add.reduceat(jac_V_data_inv_jac_dot_d_ij*use_inds[:,None],unique_inds)
    summed_jac_V_data_inv_jac_times = np.add.reduceat(jac_V_data_inv_jac*delta_times[:,None,None]*use_inds[:,None,None],unique_inds)
    
    A_mu_i = np.einsum('nij,njk->nik',Sigma_theta_i,summed_jac_V_data_inv_jac_times)
    C_mu_ij = delta_time_identities-A_mu_i[unique_inv_inds]
    A_mu_i_inv = np.linalg.inv(A_mu_i)
    C_mu_ij_inv = np.linalg.inv(C_mu_ij)
    
    Sigma_mu_theta_i_inv = np.einsum('nij,njk->nik',np.einsum('nji,njk->nik',A_mu_i,unique_gaia_offset_inv_covs),A_mu_i)
    Sigma_mu_d_ij_inv = np.einsum('nij,njk->nik',np.einsum('nji,njk->nik',C_mu_ij,jac_V_data_inv_jac),C_mu_ij)
    
    Sigma_mu_i_inv = global_pm_inv_cov+unique_gaia_pm_inv_covs+Sigma_mu_theta_i_inv+\
                     np.add.reduceat(Sigma_mu_d_ij_inv*use_inds[:,None,None],unique_inds)
    Sigma_mu_i = np.linalg.inv(Sigma_mu_i_inv)
    
    A_plx_mu_i = np.einsum('nij,nj->ni',Sigma_theta_i,-1*summed_jac_V_data_inv_jac_dot_parallax_vects)
    B_plx_mu_i = np.einsum('nij,nj->ni',Sigma_theta_i,unique_V_theta_i_inv_dot_theta_i\
                                                        -summed_jac_V_data_inv_jac_dot_d_ij)
    
    Sigma_mu_theta_i_inv_dot_A_mu_i_inv = np.einsum('nij,njk->nik',Sigma_mu_theta_i_inv,A_mu_i_inv)
    Sigma_mu_d_ij_inv_dot_C_mu_ij_inv = np.einsum('nij,njk->nik',Sigma_mu_d_ij_inv,C_mu_ij_inv)
    
    C_plx_mu_i = np.einsum('nij,nj->ni',Sigma_mu_i,np.einsum('nij,nj->ni',Sigma_mu_theta_i_inv_dot_A_mu_i_inv,A_plx_mu_i)\
                                                     -np.add.reduceat(np.einsum('nij,nj->ni',Sigma_mu_d_ij_inv_dot_C_mu_ij_inv,parallax_offset_vector+A_plx_mu_i[unique_inv_inds])*use_inds[:,None],unique_inds))
    D_plx_mu_i = -1*np.einsum('nij,nj->ni',Sigma_mu_i,unique_V_mu_i_inv_dot_mu_i+V_mu_global_inv_dot_mu_global+\
                                                        +np.einsum('nij,nj->ni',Sigma_mu_theta_i_inv_dot_A_mu_i_inv,unique_gaia_offsets-B_plx_mu_i)\
                                                        +np.add.reduceat(np.einsum('nij,nj->ni',Sigma_mu_d_ij_inv_dot_C_mu_ij_inv,inv_jac_dot_d_ij+B_plx_mu_i[unique_inv_inds])*use_inds[:,None],unique_inds))
                        
    E_plx_theta_i = np.einsum('nij,nj->ni',A_mu_i,C_plx_mu_i)-A_plx_mu_i
    F_plx_theta_i = np.einsum('nij,nj->ni',A_mu_i,D_plx_mu_i)-B_plx_mu_i
    
    G_plx_d_ij = np.einsum('nij,nj->ni',C_mu_ij,C_plx_mu_i[unique_inv_inds])+A_plx_mu_i[unique_inv_inds]+parallax_offset_vector
    H_plx_d_ij = np.einsum('nij,nj->ni',C_mu_ij,D_plx_mu_i[unique_inv_inds])+B_plx_mu_i[unique_inv_inds]+inv_jac_dot_d_ij

    G_plx_d_ij_T_dot_V_data_inv = np.einsum('nj,nij->ni',G_plx_d_ij,jac_V_data_inv_jac)  
    ivar_plx_d_ij = np.einsum('ni,ni->n',G_plx_d_ij_T_dot_V_data_inv,G_plx_d_ij)
    mu_times_ivar_plx_d_ij = np.einsum('ni,ni->n',G_plx_d_ij_T_dot_V_data_inv,H_plx_d_ij)
#                mu_plx_d_ij = mu_times_ivar_plx_d_ij/ivar_plx_d_ij
    summed_ivar_plx_d_ij = np.add.reduceat(ivar_plx_d_ij*use_inds,unique_inds)
    summed_mu_times_ivar_plx_d_ij = np.add.reduceat(mu_times_ivar_plx_d_ij*use_inds,unique_inds)
    
    C_plx_mu_i_T_dot_V_mu_i_inv = np.einsum('nj,nij->ni',C_plx_mu_i,unique_gaia_pm_inv_covs)  
    ivar_plx_mu_i = np.einsum('ni,ni->n',C_plx_mu_i_T_dot_V_mu_i_inv,C_plx_mu_i)
    mu_times_ivar_plx_mu_i = np.einsum('ni,ni->n',C_plx_mu_i_T_dot_V_mu_i_inv,D_plx_mu_i+unique_gaia_pms)
#                mu_plx_mu_i = mu_times_ivar_plx_mu_i/ivar_plx_mu_i
    
    C_plx_mu_i_T_dot_V_mu_global_inv = np.einsum('nj,ij->ni',C_plx_mu_i,global_pm_inv_cov)  
    ivar_plx_mu_global = np.einsum('ni,ni->n',C_plx_mu_i_T_dot_V_mu_global_inv,C_plx_mu_i)
    mu_times_ivar_plx_mu_global = np.einsum('ni,ni->n',C_plx_mu_i_T_dot_V_mu_global_inv,D_plx_mu_i+global_pm_mean)
#                mu_plx_mu_global = mu_times_ivar_plx_mu_global/ivar_plx_mu_global
    
    E_plx_theta_i_T_dot_V_theta_i_inv = np.einsum('nj,nij->ni',E_plx_theta_i,unique_gaia_offset_inv_covs)  
    ivar_plx_theta_i = np.einsum('ni,ni->n',E_plx_theta_i_T_dot_V_theta_i_inv,E_plx_theta_i)
    mu_times_ivar_plx_theta_i = np.einsum('ni,ni->n',E_plx_theta_i_T_dot_V_theta_i_inv,F_plx_theta_i+unique_gaia_offsets)
#                mu_plx_theta_i = mu_times_ivar_plx_theta_i/ivar_plx_theta_i
    
    
    ivar_plx_i = summed_ivar_plx_d_ij+ivar_plx_mu_i+ivar_plx_mu_global\
                 +ivar_plx_theta_i+unique_gaia_parallax_ivars+global_parallax_ivar
    var_plx_i = 1/ivar_plx_i
    std_plx_i = np.sqrt(var_plx_i)
    # print('plx',summed_ivar_plx_d_ij.min(),ivar_plx_theta_i.min(),ivar_plx_mu_global.min())
    # print('plx',(unique_gaia_parallax_errs/std_plx_i).min())
    mu_plx_i = (summed_mu_times_ivar_plx_d_ij+mu_times_ivar_plx_mu_i+mu_times_ivar_plx_mu_global\
                 +mu_times_ivar_plx_theta_i\
                 +unique_gaia_parallax_ivars*unique_gaia_parallaxes\
                 +global_parallax_ivar*global_parallax_mean)/ivar_plx_i
                             
    parallax_draws = np.random.randn(n_pms,*std_plx_i.shape)*std_plx_i+mu_plx_i
#            single_parallax_draws = np.random.randn(*std_plx_i.shape)*std_plx_i+mu_plx_i
#            parallax_draws[:] = single_parallax_draws
    
    B_mu_i = parallax_draws[:,:,None]*A_plx_mu_i[None,:]-B_plx_mu_i
#                D_mu_ij = inv_jac_dot_d_ij-B_mu_i[:,unique_inv_inds]-parallax_draws[:,unique_inv_inds,None]*parallax_offset_vector
#            mu_mu_theta_i = np.einsum('nij,nj->ni',,)
#            mu_mu_i = np.einsum('nij,njk->nik',Sigma_mu_i,V_mu_global_inv_dot_mu_global+unique_V_mu_i_inv_dot_mu_i\
#                                +np.einsum('nij,njk->nik',Sigma_mu_theta_i_inv,mu_mu_theta_i))
    mu_mu_i = parallax_draws[:,:,None]*C_plx_mu_i[None,:]-D_plx_mu_i

    eig_vals,eig_vects = np.linalg.eig(Sigma_mu_i)
    eig_signs = np.sign(eig_vals)
    eig_vals *= eig_signs
    eig_vects[:,:,0] *= eig_signs[:,0][:,None]
    eig_vects[:,:,1] *= eig_signs[:,1][:,None]
    pm_gauss_draws = np.random.randn(n_pms,len(unique_ids),eig_vals.shape[-1])
#            single_gauss_draws = np.random.randn(len(unique_ids),eig_vals.shape[-1])
#            pm_gauss_draws[:] = single_gauss_draws
    pm_draws = pm_gauss_draws*np.sqrt(eig_vals) #pms in x,y HST
    pm_draws = np.einsum('nij,wnj->wni',eig_vects,pm_draws)+mu_mu_i
    
    mu_theta_i = np.einsum('nij,wnj->wni',A_mu_i,pm_draws)-B_mu_i
                
    eig_vals,eig_vects = np.linalg.eig(Sigma_theta_i)
    eig_signs = np.sign(eig_vals)
    eig_vals *= eig_signs
    eig_vects[:,:,0] *= eig_signs[:,0][:,None]
    eig_vects[:,:,1] *= eig_signs[:,1][:,None]
    offset_gauss_draws = np.random.randn(n_pms,len(unique_ids),eig_vals.shape[-1])
#            single_gauss_draws = np.random.randn(len(unique_ids),eig_vals.shape[-1])
#            offset_gauss_draws[:] = single_gauss_draws
    offset_draws = offset_gauss_draws*np.sqrt(eig_vals) #pms in x,y HST
    offset_draws = np.einsum('nij,wnj->wni',eig_vects,offset_draws)+mu_theta_i
    
#            data_diff_vals = star_hst_gaia_pos-np.einsum('nij,wnj->wni',proper_offset_jacs,delta_times[None,:,None]*pm_draws[:,unique_inv_inds]\
#                                                                                           +parallax_draws[:,unique_inv_inds][:,:,None]*parallax_offset_vector[None,:]\
#                                                                                           -offset_draws[:,unique_inv_inds])
#            ll = -0.5*np.einsum('wni,wni->wn',data_diff_vals,np.einsum('nij,wnj->wni',star_hst_gaia_pos_inv_cov,data_diff_vals))\
#                 -0.5*np.log(np.linalg.det(star_hst_gaia_pos_cov))

    data_diff_vals = inv_jac_dot_d_ij-(delta_times[None,:,None]*pm_draws[:,unique_inv_inds]\
                                       +parallax_draws[:,unique_inv_inds][:,:,None]*parallax_offset_vector[None,:]\
                                       -offset_draws[:,unique_inv_inds])
    
    ll = -0.5*np.einsum('wni,wni->wn',data_diff_vals,np.einsum('nij,wnj->wni',jac_V_data_inv_jac,data_diff_vals))\
         -0.5*np.log(np.linalg.det(np.linalg.inv(jac_V_data_inv_jac)))
    summed_ll = np.add.reduceat(ll*use_inds[None,:],unique_inds,axis=1)
    
    prior_pm_diff = pm_draws-unique_gaia_pms
    prior_parallax_diff = parallax_draws-unique_gaia_parallaxes
    prior_offset_diff = offset_draws-unique_gaia_offsets
    global_pm_diff = pm_draws-global_pm_mean
    global_parallax_diff = parallax_draws-global_parallax_mean
                
    lp = -0.5*np.einsum('wni,wni->wn',prior_pm_diff,np.einsum('nij,wnj->wni',unique_gaia_pm_inv_covs,prior_pm_diff))\
         -0.5*log_unique_gaia_pm_covs_det\
         -0.5*np.einsum('wni,wni->wn',global_pm_diff,np.einsum('ij,wnj->wni',global_pm_inv_cov,global_pm_diff))\
         -0.5*log_global_pm_cov_det\
         -0.5*np.power(prior_parallax_diff,2)*unique_gaia_parallax_ivars-0.5*np.log(unique_gaia_parallax_vars)\
         -0.5*np.power(global_parallax_diff,2)*global_parallax_ivar\
         -0.5*np.einsum('wni,wni->wn',prior_offset_diff,np.einsum('nij,wnj->wni',unique_gaia_offset_inv_covs,prior_offset_diff))\
         -0.5*log_unique_gaia_offset_covs_det
         
    draw_pm_diff = pm_draws-mu_mu_i
    draw_parallax_diff = parallax_draws-mu_plx_i
    draw_offset_diff = offset_draws-mu_theta_i
    
    lnorm = -0.5*np.einsum('wni,wni->wn',draw_pm_diff,np.einsum('nij,wnj->wni',Sigma_mu_i_inv,draw_pm_diff)) \
            -0.5*np.log(np.linalg.det(Sigma_mu_i))\
            -0.5*np.einsum('wni,wni->wn',draw_offset_diff,np.einsum('nij,wnj->wni',Sigma_theta_i_inv,draw_offset_diff)) \
            -0.5*np.log(np.linalg.det(Sigma_theta_i))\
            -0.5*np.power(draw_parallax_diff,2)*ivar_plx_i-0.5*np.log(var_plx_i)
                                
    logprobs = np.sum((summed_ll+lp-lnorm)[:,unique_keep],axis=1)
    
    if n_pms > 1:
        print(logprobs)
        print(logprobs-logprobs.min())
        raise ValueError('alksdjlksjslkjs')
    renorm_fact = np.max(logprobs)
    final_log_post = np.log(np.sum(np.exp(logprobs-renorm_fact)))+renorm_fact
    
    result = np.zeros((len(unique_ids),5+1))
    #give back an example sample of parallaxes and corresponding PMs
    result[:,0] = final_log_post
#                result[:,1:] = vector_draws[0]
    result[:,1:3] = offset_draws[0]
    result[:,3:5] = pm_draws[0]
    result[:,5] = parallax_draws[0]
    
    return result             

def lnpost_new_parallel(walker_inds,thread_ind,new_params,nwalkers_sample,step_ind,
                        n_param_indv,x,x0s,ags,bgs,cgs,dgs,img_nums,xy,xy0s,xy_g,hst_covs,
                        proper_offset_jacs,proper_offset_jac_invs,unique_gaia_offset_inv_covs,use_inds,unique_inds,
                        parallax_offset_vector,delta_times,unique_inv_inds,delta_time_identities,global_pm_inv_cov,
                        unique_gaia_pm_inv_covs,unique_V_theta_i_inv_dot_theta_i,unique_V_mu_i_inv_dot_mu_i,
                        V_mu_global_inv_dot_mu_global,unique_gaia_offsets,unique_gaia_pms,global_pm_mean,
                        unique_gaia_parallax_ivars,global_parallax_ivar,unique_ids,unique_gaia_parallaxes,
                        global_parallax_mean,log_unique_gaia_pm_covs_det,log_global_pm_cov_det,
                        unique_gaia_parallax_vars,log_unique_gaia_offset_covs_det,unique_keep,
                        out_q):
    output = np.zeros((len(walker_inds),len(unique_ids),5+1))
    for count,w_ind in enumerate(walker_inds):
        params = new_params[w_ind]
        output[count] = lnpost_vector(params,
                                      n_param_indv,x,x0s,ags,bgs,cgs,dgs,img_nums,xy,xy0s,xy_g,hst_covs,
                                      proper_offset_jacs,proper_offset_jac_invs,unique_gaia_offset_inv_covs,use_inds,unique_inds,
                                      parallax_offset_vector,delta_times,unique_inv_inds,delta_time_identities,global_pm_inv_cov,
                                      unique_gaia_pm_inv_covs,unique_V_theta_i_inv_dot_theta_i,unique_V_mu_i_inv_dot_mu_i,
                                      V_mu_global_inv_dot_mu_global,unique_gaia_offsets,unique_gaia_pms,global_pm_mean,
                                      unique_gaia_parallax_ivars,global_parallax_ivar,unique_ids,unique_gaia_parallaxes,
                                      global_parallax_mean,log_unique_gaia_pm_covs_det,log_global_pm_cov_det,
                                      unique_gaia_parallax_vars,log_unique_gaia_offset_covs_det,unique_keep,
                                      seed=w_ind+(step_ind+1)*(nwalkers_sample+1))
    out_q.put((output,thread_ind))
    return



def analyse_images(image_list,field,path,
                   overwrite_previous=True,overwrite_GH_summaries=False,thresh_time=0,
                   n_fit_max=3,max_images=10,redo_without_outliers=True,max_stars=2000,
                   plot_indv_star_pms=True):
    '''
    Simultaneously analyzes the images in image_list in field along path
    '''
    
    n_threads = cpu_count()
    print(f'Using {n_threads} CPU(s)')

    outpath = f'{path}{field}/Bayesian_PMs/'
    if (not os.path.isfile(f'{outpath}gaiahub_image_transformation_summaries.csv')) or (overwrite_GH_summaries):
        process_GH.collect_gaiahub_results(field,path=path,overwrite=overwrite_GH_summaries)

    #get the transformation parameters for the images that have been provided
    trans_file_df = pd.read_csv(f'{outpath}gaiahub_image_transformation_summaries.csv')
    allowed_image_names = np.array(trans_file_df['image_name'].to_list())
    keep_im_names = np.zeros(len(allowed_image_names)).astype(bool)
    keep_list_names = np.ones(len(image_list)).astype(bool)
    for name_ind,name in enumerate(image_list):
        if name not in allowed_image_names:
            keep_list_names[name_ind] = False
        else:
            match_ind = np.where(name == allowed_image_names)[0][0]
            keep_im_names[match_ind] = True
    image_list = np.array(image_list)[keep_list_names]
    trans_file_summaries = {field:trans_file_df.loc[np.where(keep_im_names)[0]]}

    #for each of the images, read in the individual source data
    indv_image_source_data = {field:{}}
    for image_ind,image_name in enumerate(image_list):
        if not os.path.isfile(f'{outpath}/{image_name}/{image_name}_linked_images.csv'):
            process_GH.image_lister(field,path)
        
        curr_image_df = pd.read_csv(f'{outpath}/{image_name}/{image_name}_gaiahub_source_summaries.csv')
        for key in curr_image_df.keys():
            if image_ind == 0:
                indv_image_source_data[field][key] = []
            indv_image_source_data[field][key].append(np.array(curr_image_df[key]))
                        
    linked_image_lists = [image_list]
            
    ####Below is code for fitting no terms globally for all images in each mask group
    ####while the all terms are fit for each image individually
    
    #overwrite_previous = False #if there is a previous version with same (nwalkers,nsteps,ndim), just use that
    #number of steps until updating [trans_params,proper_motions] ([1,1] means every update step)
    
    if thresh_time == 0:
        thresh_time = ((datetime.datetime(1900,1,1,0,0,0,0)-datetime.datetime.utcfromtimestamp(0)).total_seconds()+7*3600)
    
    mask_outlier_pms = False #get rid of stars with large Gaia prior PM measures (good for dSph)
    if 'dSph' in field:
        mask_outlier_pms = True #get rid of stars with large Gaia prior PM measures (good for dSph)
    
#    scale_pos = True #True means use the scale_pos_fact to make the starting walkers closer together
#    scale_pos_fact = 1.0
        
    data_combined = {}
    
    poss_fixed_params = []
    
    trans_params = ['Xo','Yo','Wo','Zo','AG','BG','CG','DG','rotate_mult_fact']
    
#    redo_without_outliers = False #use the first set of results to remove obvious outliers, then redo
#    redo_without_outliers = True #use the first set of results to remove obvious outliers, then redo
#    n_fit_max = 5 #don't exceed this number of fitting iterations
#    n_fit_max = 3 #don't exceed this number of fitting iterations
#    
#    max_images = 10 #truncate to this many images if there are more than that to keep runtime/convergence time reasonable
#    max_images = 6 #truncate to this many images if there are more than that to keep runtime/convergence time reasonable
#    max_images = 2 #truncate to this many images if there are more than that to keep runtime/convergence time reasonable
#    max_images = 1 #truncate to this many images if there are more than that to keep runtime/convergence time reasonable
#    max_images = 6 #truncate to this many images if there are more than that to keep runtime/convergence time reasonable
#    # max_images = 4 #truncate to this many images if there are more than that to keep runtime/convergence time reasonable
#    max_images = 10 #truncate to this many images if there are more than that to keep runtime/convergence time reasonable
#    # max_images = 5 #truncate to this many images if there are more than that to keep runtime/convergence time reasonable
    
    gaia_mjd = 57388.5 #J2016.0 in MJD
    
#    hst_pos_err_mult = 1.0 #inflate errors on position in HST pixels
    # hst_pos_err_mult = 2.0 #inflate errors on position in HST pixels
    
#    max_stars = 2000 #maximum number of stars to use
##    max_stars = 100 #maximum number of stars to use
##    max_stars = 50 #maximum number of stars to use
    
    plot_indv_star_pms = True
    if field in ['LMC']:
        plot_indv_star_pms = False
        
    for mask_name in trans_file_summaries:        
        #loop over the different lists in linked_image_lists
            
        for curr_keep_num in range(len(linked_image_lists)):
            skip_fitting = False
            plt.close('all')
            
            temp_image_names = []
            for item in linked_image_lists[curr_keep_num]:
                temp_image_names.append(item.split('_flc')[0])
            
            keep_image_names = np.array(temp_image_names)
                        
            matching_image_names = np.intersect1d(keep_image_names,trans_file_summaries[mask_name]['image_name'])
            if len(matching_image_names) == 0:
                print(f'SKIPPING image list {keep_image_names} because there were no matching names in the output files.')
                continue
            keep_im_nums = np.zeros(len(trans_file_summaries[mask_name]['image_name'])).astype(bool)
            for item_ind,item in enumerate(trans_file_summaries[mask_name]['image_name']):
                if item in matching_image_names:
                    keep_im_nums[item_ind] = True
    
            keep_im_nums = np.where(keep_im_nums)[0]
            #if restricting the list to a smaller amount, then use images seperated by the most time
            #(i.e. oldest image first)
            if max_images < len(keep_im_nums):
                keep_im_mjds = np.array(trans_file_summaries[mask_name]['HST_time'])[keep_im_nums]
                keep_inds = np.arange(0,len(keep_im_nums),int(round(len(keep_im_nums)/max_images)))
                keep_inds = keep_inds[:max_images]
                if len(keep_inds) > 1:
                    keep_inds[-1] = -1
                keep_im_nums = keep_im_nums[np.argsort(keep_im_mjds)[keep_inds]]
                
            keep_im_mjds = np.array(trans_file_summaries[mask_name]['HST_time'])[keep_im_nums]
            keep_image_names = np.array(trans_file_summaries[mask_name]['image_name'])[keep_im_nums]
    #        print(f'Current images have time baselines of {np.round((gaia_mjd-keep_im_mjds)/365,2)} years from Gaia.')
    #        if field == 'COSMOS_field':
    #            years_from_gaia = (gaia_mjd-keep_im_mjds)/365
    #            if years_from_gaia.min() > 7:
    #                continue
            
            curr_hst_image_name = '_'.join(keep_image_names)
    #        if curr_hst_image_name != chosen_hst_image_name:
    #            continue
                    
            previous_analysis_results = {}
            previous_pm_results = {'running_total':{},'average':{}}
            print()
            for name in keep_image_names:
                #check to see if inidividual analysis exists, and use those transform parameters as first guess
                outpath = f'{path}{mask_name}/Bayesian_PMs/{name}/'
                previous_sample_med_file = f'{outpath}{name}_posterior_transformation_6p_medians.npy'
                previous_sample_cov_file = f'{outpath}{name}_posterior_transformation_6p_covs.npy'
                previous_pm_file = f'{outpath}{name}_posterior_PM_parallax_offset_medians.npy'
                if os.path.isfile(previous_sample_med_file) and os.path.isfile(previous_pm_file) and len(keep_im_nums) > 1:
                # if os.path.isfile(previous_sample_med_file) and os.path.isfile(previous_pm_file):
                    print(f'Found previous individual analysis results for image {name} that can be used as a starting guess in the parameter fits.')
                    previous_means = np.load(previous_sample_med_file)
                    previous_covs = np.load(previous_sample_cov_file)
    #                previous_analysis_results[name] = previous_means,previous_covs
                    
                    #transform the means and covs in rot,ratio,on_skew,off_skew to the matrix parameters
                    try:
                        previous_samples = stats.multivariate_normal(mean=previous_means,cov=previous_covs).rvs(10000)
                    except:
                        previous_samples = stats.multivariate_normal(mean=previous_means,cov=np.diag(np.diag(previous_covs))).rvs(10000)
                        
                    trans_samples = np.array(get_matrix_params(previous_samples[:,4],previous_samples[:,5],previous_samples[:,1],previous_samples[:,0])).T
                    new_samples = np.zeros_like(previous_samples)
                    new_samples[:,0] = trans_samples[:,0]
                    new_samples[:,1] = trans_samples[:,1]
                    new_samples[:,2] = previous_samples[:,2] #w0
                    new_samples[:,3] = previous_samples[:,3] #z0
                    new_samples[:,4] = trans_samples[:,2]
                    new_samples[:,5] = trans_samples[:,3]
                    new_means = np.mean(new_samples,axis=0)
                    new_covs = np.cov(new_samples,rowvar=False)
                    previous_analysis_results[name] = new_means,new_covs
                    
                    gaia_ids = np.load(f'{outpath}{name}_Gaia_IDs.npy')      
                    previous_posterior_vectors = np.load(f'{outpath}{name}_posterior_PM_parallax_offset_medians.npy')
                    if len(gaia_ids) != len(previous_posterior_vectors):
                        continue
                    previous_pm_results[name] = {}
                    for star_ind,gaia_id in enumerate(gaia_ids):
                        #change order to be [offset_ra,offset_dec,mu_ra,mu_dec,parallax]
                        previous_pm_results[name][gaia_id] = previous_posterior_vectors[star_ind*5:(star_ind+1)*5][[3,4,0,1,2]]
                        if gaia_id not in previous_pm_results['running_total']:
                            previous_pm_results['running_total'][gaia_id] = [np.zeros(len(previous_pm_results[name][gaia_id])),0]
                        previous_pm_results['running_total'][gaia_id][0] += previous_pm_results[name][gaia_id]
                        previous_pm_results['running_total'][gaia_id][1] += 1
            for star_name in previous_pm_results['running_total']:
                previous_pm_results['average'][star_name] = previous_pm_results['running_total'][gaia_id][0]/previous_pm_results['running_total'][gaia_id][1]
                            
            fit_count = 0 #iteration of the number of fits performed
            total_fit_start = time.time()
            
            while (fit_count < n_fit_max):
                start_time = time.time()
                
                data_combined[mask_name] = {'X':[],'Y':[],'X_G':[],'Y_G':[],
                                            'dX_G':[],'dY_G':[],'g_mag':[],'Gaia_id':[],
                                            'img_num':[],'avg_params':{},'orig_pixel_scale':[],
                                            'delta_times':[],'HST_times':[],'Gaia_times':[],
                                            'pm_x':[],'pm_y':[],'parallax':[],
                                            'X_hst_err':[],'Y_hst_err':[],
                                            'stationary':[],'use_for_fit':[],'q_hst':[],
    #                                        'gaia_pm_x':[],'gaia_pm_y':[],'gaia_pm_x_err':[],'gaia_pm_y_err':[],'gaia_pm_x_y_corr':[],
    #                                        'gaia_ra':[],'gaia_dec':[],'gaia_ra_err':[],'gaia_dec_err':[],'gaia_ra_dec_corr':[],
    #                                        'gaia_parallax':[],'gaia_parallax_err':[],
                                            'gaiahub_pm_x':[],'gaiahub_pm_y':[],'gaiahub_pm_x_err':[],'gaiahub_pm_y_err':[]}
                for label in correlation_names:
                    data_combined[mask_name]['gaia_'+label] = []
                
            #             TRANSFORMATION MATRIX: X_2 = AG*(X_1-Xo)+BG*(Y_1-Yo)+Wo
            #                                    Y_2 = CG*(X_1-Xo)+DG*(Y_1-Yo)+Zo
        
                n_images = len(keep_im_nums)
                n_stars = np.array(trans_file_summaries[mask_name]['n_stars'])[keep_im_nums]
                hst_image_names = np.array(trans_file_summaries[mask_name]['image_name'])[keep_im_nums]
                
                orig_pixel_scales = np.array(trans_file_summaries[mask_name]['orig_pixel_scale'])[keep_im_nums]
                
                param_outputs = np.zeros((n_images,len(trans_params)))
                for ii,param in enumerate(trans_params):
            #         print(param,np.where(~np.isfinite(trans_file_summaries[mask_name][param]))[0])
                    param_outputs[:,ii] = np.array(trans_file_summaries[mask_name][param])[keep_im_nums]
                    
#                ra_centers = trans_file_summaries[mask_name]['ra'][keep_im_nums]
#                dec_centers = trans_file_summaries[mask_name]['dec'][keep_im_nums]
                for image_ind,hst_image in enumerate(hst_image_names):
                    #if there is a previous analysis, use those transformation parameters
                    if hst_image in previous_analysis_results:
                        xo,yo,wo,zo,ag,bg,cg,dg,rot_sign = param_outputs[image_ind]
    #                    rot,ratio,wo,zo,on_skew,off_skew = previous_analysis_results[hst_image][0]
    #                    ag,bg,cg,dg = get_matrix_params(on_skew,off_skew,ratio,rot)
                        ag,bg,wo,zo,cg,dg = previous_analysis_results[hst_image][0]
                        param_outputs[image_ind] = xo,yo,wo,zo,ag,bg,cg,dg,rot_sign
    #                    print(ag,bg,wo,zo,cg,dg)
                                    
                min_n_stars = 5
                ave_param_vals = {}
                for param in poss_fixed_params:
                    vals = np.array(trans_file_summaries[mask_name][param])[keep_im_nums]
                    weights = np.copy(n_stars)
                    weights[weights < min_n_stars] = 0
                    weights = weights/np.sum(weights)
                    ave_val = np.sum(vals*weights)
                    ave_param_vals[param] = ave_val
                data_combined[mask_name]['avg_params'] = ave_param_vals

                for j,orig_ind in enumerate(keep_im_nums):
            #     for j in n_images:
                    x_hst_err = indv_image_source_data[mask_name]['x_hst_err'][orig_ind]
                    y_hst_err = indv_image_source_data[mask_name]['y_hst_err'][orig_ind]
                    gaia_id = indv_image_source_data[mask_name]['Gaia_id'][orig_ind]
                    x_orig = indv_image_source_data[mask_name]['X'][orig_ind]
                    y_orig = indv_image_source_data[mask_name]['Y'][orig_ind]
                    x_gaia = indv_image_source_data[mask_name]['X_G'][orig_ind]
                    y_gaia = indv_image_source_data[mask_name]['Y_G'][orig_ind]
                    dx_orig = indv_image_source_data[mask_name]['dX_G'][orig_ind]
                    dy_orig = indv_image_source_data[mask_name]['dY_G'][orig_ind]
                    use_for_fit = indv_image_source_data[mask_name]['use_for_fit'][orig_ind]
                    g_mag = indv_image_source_data[mask_name]['g_mag'][orig_ind]
                    q_hst = indv_image_source_data[mask_name]['q_hst'][orig_ind]
                    for label in correlation_names:
                        data_combined[mask_name]['gaia_'+label].extend(indv_image_source_data[mask_name][label][orig_ind])
                    hst_times = np.ones(len(gaia_id))*np.array(trans_file_summaries[mask_name]['HST_time'])[orig_ind]
                    gaia_times = indv_image_source_data[mask_name]['Gaia_time'][orig_ind]
                    
                    gaiahub_pm_x = indv_image_source_data[mask_name]['gaiahub_pm_x'][orig_ind]
                    gaiahub_pm_y = indv_image_source_data[mask_name]['gaiahub_pm_y'][orig_ind]
                    gaiahub_pm_x_err = indv_image_source_data[mask_name]['gaiahub_pm_x_err'][orig_ind]
                    gaiahub_pm_y_err = indv_image_source_data[mask_name]['gaiahub_pm_y_err'][orig_ind]
                    
                    curr_stationary = indv_image_source_data[mask_name]['stationary'][orig_ind]
#                    dpix_orig = np.sqrt(np.power(dx_orig,2)+np.power(dy_orig,2))
        
                    xo,yo,wo,zo,ag,bg,cg,dg,rot_sign = param_outputs[j]
                    a,b,c,d = ag,bg,cg,dg
#                    ratio = np.sqrt(a*d-b*c)
#                    rot = np.arctan2(b-c,a+d)*180/np.pi
#                    on_skew = 0.5*(a-d)
#                    off_skew = 0.5*(b+c)
                    
                    #check if the a,b,c,d values from the first guess need to be multiplied
                    #by -1
                    
    #                a,b,c,d = get_matrix_params(on_skew,off_skew,ratio,rot)
                    x_trans = a*(x_orig-xo)+b*(y_orig-yo)+wo
                    y_trans = c*(x_orig-xo)+d*(y_orig-yo)+zo
        
                    dx_trans = x_trans - x_gaia
                    dy_trans = y_trans - y_gaia
                    dpix_trans = np.sqrt(np.power(dx_trans,2)+np.power(dy_trans,2))
                    
                    #check negative version of trans params to see if that is better
                    neg_params = -1*np.array([ag,bg,cg,dg])
                    neg_x_trans = neg_params[0]*(x_orig-xo)+neg_params[1]*(y_orig-yo)+wo
                    neg_y_trans = neg_params[2]*(x_orig-xo)+neg_params[3]*(y_orig-yo)+zo
        
                    neg_dx_trans = neg_x_trans - x_gaia
                    neg_dy_trans = neg_y_trans - y_gaia
                    neg_dpix_trans = np.sqrt(np.power(neg_dx_trans,2)+np.power(neg_dy_trans,2))
                    
                    #change sign if negative version is better
                    if np.median(neg_dpix_trans) < np.median(dpix_trans):
                        xo,yo,wo,zo,ag,bg,cg,dg,rot_sign = param_outputs[j]
                        rot_sign *= -1
                        ag,bg,cg,dg = -1*np.array([ag,bg,cg,dg])
                        param_outputs[j] = xo,yo,wo,zo,ag,bg,cg,dg,rot_sign
                        a,b,c,d = ag,bg,cg,dg
#                        ratio = np.sqrt(a*d-b*c)
#                        rot = np.arctan2(b-c,a+d)*180/np.pi
#                        on_skew = 0.5*(a-d)
#                        off_skew = 0.5*(b+c)
                        x_trans = a*(x_orig-xo)+b*(y_orig-yo)+wo
                        y_trans = c*(x_orig-xo)+d*(y_orig-yo)+zo
            
                        dx_trans = x_trans - x_gaia
                        dy_trans = y_trans - y_gaia
                        dpix_trans = np.sqrt(np.power(dx_trans,2)+np.power(dy_trans,2))
    #                print(j,ag,bg,cg,dg)
    #                if (np.abs(dx_trans).min() > 100) or (np.abs(dy_trans).min() > 100):
    #                    a,b,c,d = -1*np.array([a,b,c,d])
    #                    x_fixed_params = a*(x_orig-xo)+b*(y_orig-yo)+wo
    #                    y_fixed_params = c*(x_orig-xo)+d*(y_orig-yo)+zo
    #                    dx_fixed = x_fixed_params - x_gaia
    #                    dy_fixed = y_fixed_params - y_gaia
    #                    dpix_fixed = np.sqrt(np.power(dx_fixed,2)+np.power(dy_fixed,2))
                    x_trans = a*(x_orig-xo)+b*(y_orig-yo)+wo
                    y_trans = c*(x_orig-xo)+d*(y_orig-yo)+zo
        
                    dx_trans = x_trans - x_gaia
                    dy_trans = y_trans - y_gaia
                    dpix_trans = np.sqrt(np.power(dx_trans,2)+np.power(dy_trans,2))
                    
                    data_combined[mask_name]['Gaia_id'].extend(gaia_id)
                    data_combined[mask_name]['q_hst'].extend(q_hst)
                    data_combined[mask_name]['X_hst_err'].extend(x_hst_err)
                    data_combined[mask_name]['Y_hst_err'].extend(y_hst_err)
                    data_combined[mask_name]['X'].extend(x_orig)
                    data_combined[mask_name]['Y'].extend(y_orig)
                    data_combined[mask_name]['X_G'].extend(x_gaia)
                    data_combined[mask_name]['Y_G'].extend(y_gaia)
                    data_combined[mask_name]['dX_G'].extend(dx_orig)
                    data_combined[mask_name]['dY_G'].extend(dy_orig)
                    data_combined[mask_name]['g_mag'].extend(g_mag)
                    data_combined[mask_name]['img_num'].extend([j]*len(x_orig))
                    data_combined[mask_name]['HST_times'].extend(hst_times)
                    data_combined[mask_name]['Gaia_times'].extend(gaia_times)
    #                data_combined[mask_name]['gaia_pm_x'].extend(gaia_pm_x)
    #                data_combined[mask_name]['gaia_pm_y'].extend(gaia_pm_y)
    #                data_combined[mask_name]['gaia_pm_x_err'].extend(gaia_pm_x_err)
    #                data_combined[mask_name]['gaia_pm_y_err'].extend(gaia_pm_y_err)
    #                data_combined[mask_name]['gaia_pm_x_y_corr'].extend(gaia_pm_x_y_corr)
    #                data_combined[mask_name]['gaia_ra'].extend(gaia_ra)
    #                data_combined[mask_name]['gaia_dec'].extend(gaia_dec)
    #                data_combined[mask_name]['gaia_ra_err'].extend(gaia_ra_err)
    #                data_combined[mask_name]['gaia_dec_err'].extend(gaia_dec_err)
    #                data_combined[mask_name]['gaia_ra_dec_corr'].extend(gaia_ra_dec_corr)
    #                data_combined[mask_name]['gaia_parallax'].extend(gaia_parallax)
    #                data_combined[mask_name]['gaia_parallax_err'].extend(gaia_parallax_err)
                    data_combined[mask_name]['stationary'].extend(curr_stationary)
                    data_combined[mask_name]['gaiahub_pm_x'].extend(gaiahub_pm_x)
                    data_combined[mask_name]['gaiahub_pm_y'].extend(gaiahub_pm_y)
                    data_combined[mask_name]['gaiahub_pm_x_err'].extend(gaiahub_pm_x_err)
                    data_combined[mask_name]['gaiahub_pm_y_err'].extend(gaiahub_pm_y_err)
                    data_combined[mask_name]['use_for_fit'].extend(use_for_fit)
    
                for param in data_combined[mask_name]:
                    data_combined[mask_name][param] = np.array(data_combined[mask_name][param])
                    
                gaia_id = np.copy(data_combined[mask_name]['Gaia_id'])
                sort_gaia_id_inds = np.argsort(gaia_id)
                if len(sort_gaia_id_inds) > max_stars:
                    sort_gaia_id_inds = sort_gaia_id_inds[:max_stars]
                gaia_id = np.copy(data_combined[mask_name]['Gaia_id'])[sort_gaia_id_inds]
                
                x,y = data_combined[mask_name]['X'][sort_gaia_id_inds],data_combined[mask_name]['Y'][sort_gaia_id_inds] #in HST pixels
                x_hst_err,y_hst_err = data_combined[mask_name]['X_hst_err'][sort_gaia_id_inds],data_combined[mask_name]['Y_hst_err'][sort_gaia_id_inds] #in HST pixels
                x_g,y_g = data_combined[mask_name]['X_G'][sort_gaia_id_inds],data_combined[mask_name]['Y_G'][sort_gaia_id_inds] #in Gaia pixels
                g_mag = data_combined[mask_name]['g_mag'][sort_gaia_id_inds]
                use_for_fit = np.array(data_combined[mask_name]['use_for_fit'])[sort_gaia_id_inds]
                q_hst = np.array(data_combined[mask_name]['q_hst'])[sort_gaia_id_inds]
                img_nums = data_combined[mask_name]['img_num'][sort_gaia_id_inds]
                indv_orig_pixel_scales = orig_pixel_scales[img_nums]
                
                hst_time_strings = data_combined[mask_name]['HST_times'][sort_gaia_id_inds]
    #            hst_times = Time(hst_time_strings)
    #            hst_times = Time(hst_time_strings, format='jyear',scale='tcb')
                hst_times = Time(hst_time_strings, format='mjd')
                gaia_time_strings = data_combined[mask_name]['Gaia_times'][sort_gaia_id_inds]
                gaia_times = Time(gaia_time_strings, format='jyear',scale='tcb')
                delta_times = (gaia_times-hst_times).to(u.year).value
                            
                gaia_pm_xs = np.copy(data_combined[mask_name]['gaia_pmra'])[sort_gaia_id_inds] #pm_RA in mas/yr
                gaia_pm_ys = np.copy(data_combined[mask_name]['gaia_pmdec'])[sort_gaia_id_inds] #pm_Dec in mas/yr
                gaia_pm_x_errs = np.copy(data_combined[mask_name]['gaia_pmra_error'])[sort_gaia_id_inds] #pm_RA in mas/yr
                gaia_pm_y_errs = np.copy(data_combined[mask_name]['gaia_pmdec_error'])[sort_gaia_id_inds] #pm_Dec in mas/yr
                gaia_pm_x_y_corrs = np.copy(data_combined[mask_name]['gaia_pmra_pmdec_corr'])[sort_gaia_id_inds] #pm_Dec in mas/yr
                gaia_ras = np.copy(data_combined[mask_name]['gaia_ra'])[sort_gaia_id_inds] #in deg
                gaia_decs = np.copy(data_combined[mask_name]['gaia_dec'])[sort_gaia_id_inds] #in deg
                gaia_ra_errs = np.copy(data_combined[mask_name]['gaia_ra_error'])[sort_gaia_id_inds] #in mas
                gaia_dec_errs = np.copy(data_combined[mask_name]['gaia_dec_error'])[sort_gaia_id_inds] #in mas
                gaia_ra_dec_corrs = np.copy(data_combined[mask_name]['gaia_ra_dec_corr'])[sort_gaia_id_inds] #in mas
                gaia_parallaxes = np.copy(data_combined[mask_name]['gaia_parallax'])[sort_gaia_id_inds] #in mas
                gaia_parallax_errs = np.copy(data_combined[mask_name]['gaia_parallax_error'])[sort_gaia_id_inds] #in mas
                
                if fit_count == 0:
                    unique_ids,unique_inds,unique_inv_inds,unique_counts = np.unique(gaia_id,return_index=True,
                                                                                     return_inverse=True,return_counts=True)
                    
                    proper_offset_jacs = np.zeros((len(x),2,2))
                    proper_offset_jacs[:,0,0] = -1
    #                proper_offset_jacs[:,0,0] = -1/np.cos(gaia_decs*np.pi/180)
    #                proper_offset_jacs[:,0,0] = -1/np.cos(gaia_decs*np.pi/180)
    #                proper_offset_jacs[:,0,0] = 1
                    proper_offset_jacs[:,1,1] = 1
    #                print(proper_offset_jacs[0])
    #                new_proper_offset_jacs = np.zeros((len(x),2,2))
    #                for star_ind in range(len(x)):
    #                    ra,dec = gaia_ras[star_ind]*np.pi/180,gaia_decs[star_ind]*np.pi/180
    #                    img_num = img_nums[star_ind]
    #                    ra0,dec0 = ra_centers[img_num]*np.pi/180,dec_centers[img_num]*np.pi/180
    #                    new_proper_offset_jacs[star_ind] = offset_jac(ra,dec,ra0,dec0)
    #                print(new_proper_offset_jacs[0])
    #                print(np.dot(new_proper_offset_jacs[0],proper_offset_jacs[0]))
    #                laksjd;lksd
                #gaia prior vectors
                gaia_vectors = np.zeros((len(x),5)) 
                gaia_vectors[:,0] = 0 #delta RA
                gaia_vectors[:,1] = 0 #delta Dec
                gaia_vectors[:,2] = gaia_pm_xs #PM_RA
                gaia_vectors[:,3] = gaia_pm_ys #PM_Dec
                gaia_vectors[:,4] = gaia_parallaxes #parallax
                
                #corresponding gaia prior covariance matrices
                gaia_vector_covs = np.ones((len(x),gaia_vectors.shape[1],gaia_vectors.shape[1])) 
                terms_for_correlation = ['ra','dec','pmra','pmdec','parallax'] #change the order to match the gaia_vector
                for i1 in range(len(terms_for_correlation)):
                    label1 = terms_for_correlation[i1]
                    for i2 in range(i1,len(terms_for_correlation)):
                        label2 = terms_for_correlation[i2]
                        if i1 == i2:
                            gaia_vector_covs[:,i1,i2] = np.power(data_combined[mask_name][f'gaia_{label1}_error'],2)[sort_gaia_id_inds]
                        else:
                            if f'gaia_{label1}_{label2}_corr' in data_combined[mask_name]:
                                gaia_vector_covs[:,i1,i2] = (data_combined[mask_name][f'gaia_{label1}_{label2}_corr']*\
                                                            data_combined[mask_name][f'gaia_{label1}_error']*\
                                                            data_combined[mask_name][f'gaia_{label2}_error'])[sort_gaia_id_inds]
                            else:
                                gaia_vector_covs[:,i1,i2] = (data_combined[mask_name][f'gaia_{label2}_{label1}_corr']*\
                                                            data_combined[mask_name][f'gaia_{label1}_error']*\
                                                            data_combined[mask_name][f'gaia_{label2}_error'])[sort_gaia_id_inds]
                                                        
                            gaia_vector_covs[:,i2,i1] = gaia_vector_covs[:,i1,i2]
                            
                stationary = data_combined[mask_name]['stationary'][sort_gaia_id_inds] #boolean array for distant/stationary targets
                not_stationary = ~stationary #gives the stars
                gaiahub_pm_xs = data_combined[mask_name]['gaiahub_pm_x'][sort_gaia_id_inds]
                gaiahub_pm_ys = data_combined[mask_name]['gaiahub_pm_y'][sort_gaia_id_inds]
                gaiahub_pm_x_errs = data_combined[mask_name]['gaiahub_pm_x_err'][sort_gaia_id_inds]
                gaiahub_pm_y_errs = data_combined[mask_name]['gaiahub_pm_y_err'][sort_gaia_id_inds]
                            
                image_name = curr_hst_image_name
    #            image_name = hst_image_names[0]
                outpath = f'{path}{mask_name}/Bayesian_PMs/{image_name}/'
                if not os.path.isdir(outpath):
                    os.makedirs(outpath)
                                    
                star_name = unique_ids[-1]
                indv_star_path = f'{path}{field}/Bayesian_PMs/{image_name}/indv_stars/'
                final_fig = f'{indv_star_path}{image_name}_{star_name}_posterior_PM_comparison.png'
#                final_fig = f'{outpath}{image_name}_posterior_population_PM_offset_analysis_pop_dist.png'
                if os.path.isfile(final_fig):
                    file_time = os.path.getmtime(final_fig)
                    if (file_time > thresh_time) or (not overwrite_previous):
                        print(f'SKIPPING fit of image {image_name} in {mask_name} because it has recently been analysed.')
                        skip_fitting = True
                        break
                
                gaia_pms = np.array([gaia_pm_xs,gaia_pm_ys]).T        
                gaia_pms[stationary] = 0
                gaia_pm_xs[stationary] = 0
                gaia_pm_ys[stationary] = 0
                gaia_pm_stationary_err = 0.01 #mas/yr
                gaia_pm_x_errs[stationary] = gaia_pm_stationary_err
                gaia_pm_y_errs[stationary] = gaia_pm_stationary_err
                gaia_pm_x_y_corrs[stationary] = 0
                
                gaia_err_sizes = np.sqrt(np.power(gaia_pm_x_errs,2)+np.power(gaia_pm_y_errs,2))
                if fit_count == 0:
                    missing_prior_PM = ~np.isfinite(gaia_err_sizes) #also missing parallax
                
                #change covariance matrices for missing gaia priors and stationary sources
                gaia_vector_covs[missing_prior_PM,2:] = 0
                gaia_vector_covs[missing_prior_PM,:,2:] = 0
                gaia_vector_covs[missing_prior_PM,2,2] = 10000**2
                gaia_vector_covs[missing_prior_PM,3,3] = 10000**2
                gaia_vector_covs[missing_prior_PM,4,4] = 1000**2
                gaia_vector_covs[stationary,2:,2:] = 0
                gaia_vector_covs[stationary,2,2] = 0.01**2
                gaia_vector_covs[stationary,3,3] = 0.01**2
                gaia_vector_covs[stationary,4,4] = (1e-6)**2
                
#                gaia_vector_inv_covs = np.linalg.inv(gaia_vector_covs)
                #make the uncertainty in the offset very small for the 
                first_gaia_vector_covs = np.copy(gaia_vector_covs)
    #            first_gaia_vector_covs[missing_prior_PM,:2] = 0
    #            first_gaia_vector_covs[missing_prior_PM,:,:2] = 0
                first_gaia_vector_covs[missing_prior_PM,:2] *= 1e-3
                first_gaia_vector_covs[missing_prior_PM,:,:2] *= 1e-3
    #            first_gaia_vector_covs[missing_prior_PM,:2,:2] = gaia_vector_covs[missing_prior_PM,:2,:2]*(1e-3)**2
                first_gaia_vector_covs = np.copy(gaia_vector_covs)
#                first_gaia_vector_inv_covs = np.linalg.inv(first_gaia_vector_covs)
                
                if fit_count == 0:
                    if np.sum(~missing_prior_PM) > 1:
                        gaia_pm_summary = np.nanpercentile(gaia_pms[~missing_prior_PM],[16,50,84],axis=0).T
                        gaia_pm_summary = np.array([gaia_pm_summary[:,1],\
                                                    gaia_pm_summary[:,1]-gaia_pm_summary[:,0],\
                                                    gaia_pm_summary[:,2]-gaia_pm_summary[:,1]]).T
                        outlier_pms = np.zeros_like(gaia_pms).astype(bool)
                        outlier_pms[~missing_prior_PM] = (gaia_pms[~missing_prior_PM]-(gaia_pm_summary[:,0]-5*gaia_pm_summary[:,1]) < 0) | \
                                                         (gaia_pms[~missing_prior_PM]-(gaia_pm_summary[:,0]+5*gaia_pm_summary[:,2]) > 0)
                        outlier_pms = outlier_pms[:,0] | outlier_pms[:,1]
                        outlier_pms[missing_prior_PM] = False #don't mask missing data
                    else:
                        outlier_pms = np.zeros(len(gaia_pms)).astype(bool)
                    # print(np.sum(outlier_pms),mask_outlier_pms)
                    
                    all_outliers = np.zeros(len(x)).astype(bool)
                    all_outliers = ~use_for_fit
                        
                    if mask_outlier_pms:
                        all_outliers = all_outliers | outlier_pms
                        
                    if n_images == 1:
                        if (np.sum(~all_outliers & ~missing_prior_PM) >= 10):
                            #don't use the missing prior stars in the first fit if there are enough good stars
                            all_outliers = all_outliers | missing_prior_PM
                        
    #            else:
    #                #then mask out the outliers
    #                gaia_id = gaia_id[keep_stars]
    #                x,y = x[keep_stars],y[keep_stars]
    #                x_g,y_g = x_g[keep_stars],y_g[keep_stars]
    #                g_mag = g_mag[keep_stars]
    #                img_nums = img_nums[keep_stars]
    #                delta_times = delta_times[keep_stars]
    #                hst_time_strings = hst_time_strings[keep_stars]
    ##                hst_times = Time(hst_time_strings)
    #                hst_times = Time(hst_time_strings, format='jyear',scale='tcb')
    #                gaia_time_strings = gaia_time_strings[keep_stars]
    #                gaia_times = Time(gaia_time_strings, format='jyear',scale='tcb')
    #                delta_times = (gaia_times-hst_times).to(u.year).value
    #                gaia_pm_xs = gaia_pm_xs[keep_stars]
    #                gaia_pm_ys = gaia_pm_ys[keep_stars]
    #                gaia_pm_x_errs = gaia_pm_x_errs[keep_stars]
    #                gaia_pm_y_errs = gaia_pm_y_errs[keep_stars]
    #                gaia_pm_x_y_corrs = gaia_pm_x_y_corrs[keep_stars]
    #                gaia_ras = gaia_ras[keep_stars]
    #                gaia_decs = gaia_decs[keep_stars]
    #                gaia_ra_errs = gaia_ra_errs[keep_stars]
    #                gaia_dec_errs = gaia_dec_errs[keep_stars]
    #                gaia_ra_dec_corrs = gaia_ra_dec_corrs[keep_stars]
    #                gaia_parallaxes = gaia_parallaxes[keep_stars]
    #                gaia_parallax_errs = gaia_parallax_errs[keep_stars]
    #                stationary = stationary[keep_stars]
    #                not_stationary = ~stationary #gives the stars
    #                gaiahub_pm_xs = gaiahub_pm_xs[keep_stars]
    #                gaiahub_pm_ys = gaiahub_pm_ys[keep_stars]
    #                gaiahub_pm_x_errs = gaiahub_pm_x_errs[keep_stars]
    #                gaiahub_pm_y_errs = gaiahub_pm_y_errs[keep_stars]
                
        #         gaia_pm_xs = np.copy(true_pm_xs)
        #         gaia_pm_ys = np.copy(true_pm_ys)
        #         gaia_parallaxes = np.copy(true_parallaxes)
        
                n_stars = len(x)
                if fit_count == 0:
                    keep_stars = ~all_outliers
#                n_used_stars = np.sum(keep_stars)
                                                
                #if the star has no good images, then use all the images to measure a PM, but don't use in the transform fits
                #if the star has some good images, only use those images to measure PM, and use in the transform fit of those images
                #if a star has all good images, use all images to measure PM, and use in all tranform fits
                
                keep_star_inds = np.where(keep_stars)[0]
                keep_ids = gaia_id[keep_star_inds]
                unique_keep = np.zeros(len(unique_ids)).astype(bool)
                use_inds = np.zeros(len(keep_stars)).astype(bool)
                unique_missing_prior_PM = missing_prior_PM[unique_inds]
                unique_stationary = stationary[unique_inds]
                unique_not_stationary = not_stationary[unique_inds]
                unique_star_mapping = {}
                multiplicity = np.ones(len(gaia_id)) #keep track of the count so that the priors are applied properly
                for star_ind,star_name in enumerate(unique_ids):
                    curr_inds = np.where(gaia_id == star_name)[0]
                    if star_name not in keep_ids:
                        #then use all the images to measure PMs, but don't use for transform fitting
                        unique_keep[star_ind] = False
                    else:
                        # curr_inds = keep_star_inds[np.where(keep_ids == star_name)[0]]
                        unique_keep[star_ind] = True
                    if np.sum(keep_stars[curr_inds]) == 0:
                        #then use all the images to measure PMs, but don't use for transform fitting
                        use_inds[curr_inds] = True
                    else:
                        curr_inds = curr_inds[keep_stars[curr_inds]]
                        use_inds[curr_inds] = keep_stars[curr_inds]
                    unique_star_mapping[star_name] = curr_inds
                    multiplicity[curr_inds] = np.sum(use_inds[curr_inds])
                n_stars_unique = len(unique_ids)
                n_used_stars_unique = np.sum(unique_keep)
                if n_stars_unique == np.sum(unique_missing_prior_PM):
                    print(f"SKIPPING fit of image {image_name} in {mask_name} because no sources have Gaia priors. The results will be the same as GaiaHub's output.")
                    skip_fitting = True
                    break
                
    #            if (np.sum(unique_counts > 1) < 5) and (field in ['COSMOS_field']):
    #                print(f'\nSKIPPING {hst_image_names} because of too few matching stars')
    #                continue
    
                iteration_string = f'   Iteration {fit_count}   '
                print('\n\n'+f'-'*len(iteration_string))
                print(iteration_string)
                print(f'-'*len(iteration_string))
                
                print()
                print(f"Fitting {n_images} images in {mask_name} using image {hst_image_names}")
                print(f'Current images have time baselines of {np.round((gaia_mjd-keep_im_mjds)/365,2)} years from Gaia.')
                print(f'Current images have a total of {n_stars_unique} unique targets.')
                print(f'The unique targets are found in an average (min,max) of {round(n_stars/n_stars_unique,1)} ({unique_counts.min()},{unique_counts.max()}) images.')
                if np.sum(unique_missing_prior_PM) == 1:
                    print(f'There is {np.sum(unique_missing_prior_PM)} target missing priors from Gaia.')
                else:
                    print(f'There are {np.sum(unique_missing_prior_PM)} targets missing priors from Gaia.')
    
                if (fit_count == 0) and (np.sum(missing_prior_PM) > 0):
                    print(f'Using only targets with good Gaia priors ({n_used_stars_unique}/{n_stars_unique} targets) for the first transformation parameter fitting.')
                else:
                    print(f'Using {n_used_stars_unique}/{n_stars_unique} targets in the transformation parameter fitting.')
                
    #            print(f"\nFitting {n_images} images in {mask_name} using image {hst_image_names}")
    #            print(f'Current images have a total of n_stars = {n_stars}. Using {n_used_stars}/{n_stars} stars in the transformation parameter fitting.\n')
                                
                gaiahub_pm_err_sizes = np.sqrt(np.power(gaiahub_pm_x_errs,2)+np.power(gaiahub_pm_y_errs,2))
                gaiahub_pms = np.array([gaiahub_pm_xs,gaiahub_pm_ys]).T
                gaiahub_pm_covs = np.zeros((len(x),2,2))
                gaiahub_pm_covs[:,0,0] = np.power(gaiahub_pm_x_errs,2)
                gaiahub_pm_covs[:,1,1] = np.power(gaiahub_pm_y_errs,2)
        
                gaia_pms = np.array([gaia_pm_xs,gaia_pm_ys]).T        
                gaia_pms[stationary] = 0
                gaia_pm_xs[stationary] = 0
                gaia_pm_ys[stationary] = 0
                gaia_pm_stationary_err = 0.01 #mas/yr
                gaia_pm_x_errs[stationary] = gaia_pm_stationary_err
                gaia_pm_y_errs[stationary] = gaia_pm_stationary_err
                gaia_err_sizes = np.sqrt(np.power(gaia_pm_x_errs,2)+np.power(gaia_pm_y_errs,2))
                if fit_count == 0:
                    missing_prior_PM = ~np.isfinite(gaia_err_sizes) #also missing parallax
                unique_missing_prior_PM = missing_prior_PM[unique_inds]
                                
                #use the initial parameters and priors on parallaxes to give a better estimate of the 
                #prior on the stars that don't have gaia parallaxes
                if fit_count == 0:
                    #use the mean from the other stars
                    if np.sum(unique_not_stationary&~unique_missing_prior_PM) > 3:
                        curr_inds = unique_inds[unique_not_stationary&~unique_missing_prior_PM]
                    else:
                        curr_inds = unique_inds[unique_not_stationary]
                    gaia_pm_x_ivar = np.power(gaia_pm_x_errs[curr_inds],-2)
                    mean_gaia_pm_x = np.nansum(gaia_pm_xs[curr_inds]*gaia_pm_x_ivar)/np.nansum(gaia_pm_x_ivar)
                    gaia_pm_y_ivar = np.power(gaia_pm_y_errs[curr_inds],-2)
                    mean_gaia_pm_y = np.nansum(gaia_pm_ys[curr_inds]*gaia_pm_y_ivar)/np.nansum(gaia_pm_y_ivar)
                    
                    mean_gaia_pm_x = np.nanmedian(gaia_pm_xs[curr_inds])
                    mean_gaia_pm_y = np.nanmedian(gaia_pm_ys[curr_inds])
                    
                    gaia_pms[missing_prior_PM] = np.array([mean_gaia_pm_x,mean_gaia_pm_y])
                    gaia_pm_xs[missing_prior_PM] = mean_gaia_pm_x
                    gaia_pm_ys[missing_prior_PM] = mean_gaia_pm_y
                    
                    gaia_parallax_ivar = np.power(gaia_parallax_errs[curr_inds],-2)
                    mean_gaia_parallax = np.nansum(gaia_parallaxes[curr_inds]*gaia_parallax_ivar)/np.nansum(gaia_parallax_ivar)
                    
                    mean_gaia_parallax = np.nanmedian(gaia_parallaxes[curr_inds])
                    
                    global_vector_mean = np.array([0,0,mean_gaia_pm_x,mean_gaia_pm_y,mean_gaia_parallax])
                    diff = gaia_vectors[curr_inds]-global_vector_mean
                    bad_pm_x = ~np.isfinite(diff[:,2])
                    bad_pm_y = ~np.isfinite(diff[:,3])
                    bad_parallax = ~np.isfinite(diff[:,4])
                    diff[bad_pm_x,2] = 0
                    diff[bad_pm_y,3] = 0
                    diff[bad_parallax,4] = 0
                    global_vector_cov = (np.einsum('ni,nj->ij',diff,diff)/(len(diff)-1))*10**2
                    if np.sum(bad_pm_x) > 0:
                        mult_val = max(1.0**2,10**2/global_vector_cov[2,2])
                        global_vector_cov[2] *= mult_val
                        global_vector_cov[:,2] *= mult_val
                    if np.sum(bad_pm_y) > 0:
                        mult_val = max(1.0**2,10**2/global_vector_cov[3,3])
                        global_vector_cov[3] *= mult_val
                        global_vector_cov[:,3] *= mult_val
                    if np.sum(bad_parallax) > 0:
                        mult_val = max(1.0**2,10**2/global_vector_cov[4,4])
                        global_vector_cov[4] *= mult_val
                        global_vector_cov[:,4] *= mult_val
                    
                else:
                    if np.sum(unique_not_stationary&unique_keep&~unique_missing_prior_PM) > 3:
                        curr_unique_inds = unique_not_stationary&unique_keep&~unique_missing_prior_PM
                    else:
                        curr_unique_inds = unique_not_stationary&unique_keep
                    curr_inds = unique_inds[curr_unique_inds]
    #                curr_inds = unique_inds[unique_not_stationary&unique_keep]
                        
                    new_samples = np.zeros((sample_pms.shape[0],sample_pms.shape[1],5))
                    new_samples[:,:,0] = sample_offsets[:,:,0]
                    new_samples[:,:,1] = sample_offsets[:,:,1]
                    new_samples[:,:,2] = sample_pms[:,:,0]
                    new_samples[:,:,3] = sample_pms[:,:,1]
                    new_samples[:,:,4] = sample_parallaxes
                        
                    all_post_covs = np.zeros((len(unique_ids),5,5))
                    all_post_meds = np.zeros((len(unique_ids),5))
                    for star_ind in range(len(all_post_covs)):
                        all_post_covs[star_ind] = np.cov(new_samples[:,star_ind],rowvar=False)
                        all_post_meds[star_ind] = np.nanmean(new_samples[:,star_ind],axis=0)
                    post_ivars = 1/all_post_covs[curr_unique_inds][:,np.arange(5),np.arange(5)]
                    post_means = np.nansum(all_post_meds[curr_unique_inds]*post_ivars,axis=0)/np.sum(post_ivars,axis=0)
                                                  
                    post_means = np.nanmedian(all_post_meds[curr_unique_inds],axis=0)
                    
                    gaia_pms[missing_prior_PM] = np.array([post_means[2],post_means[3]])
                    gaia_pm_xs[missing_prior_PM] = post_means[2]
                    gaia_pm_ys[missing_prior_PM] = post_means[3]
                    mean_gaia_parallax = post_means[4]
                    
                    global_vector_mean = np.array([0,0,post_means[2],post_means[3],post_means[4]])
                    diff = all_post_meds[curr_unique_inds]-global_vector_mean
                    global_vector_cov = (np.einsum('ni,nj->ij',diff,diff)/(len(diff)-1))*10**2
                if (not np.isfinite(global_vector_cov[2,2])) or (not np.isfinite(global_vector_cov[3,3])):
                    global_vector_cov[2:4,2:4] = np.array([[100**2,0],[0,100**2]])
                global_vector_mean[~np.isfinite(global_vector_mean)] = 0
                if (not np.isfinite(global_vector_cov[2,2])) or (not np.isfinite(global_vector_cov[3,3])):
                    global_vector_cov[2:4,2:4] = np.array([[100**2,0],[0,100**2]])
                min_pm_err = 10.0
                cov_mult_factor = max(1.0,min_pm_err**2/global_vector_cov[2,2],min_pm_err**2/global_vector_cov[3,3])
                global_vector_cov *= cov_mult_factor
                
                global_vector_cov_copy = np.zeros_like(global_vector_cov)
                global_vector_cov_copy[2:4,2:4] = global_vector_cov[2:4,2:4]
                global_vector_cov = global_vector_cov_copy
                global_vector_cov[0,0] = 1000**2
                global_vector_cov[1,1] = 1000**2 #diffuse prior on the offsets
                global_vector_cov[4,4] = 10**2 #diffuse prior on the parallax
    
#                global_vector_inv_cov = np.linalg.inv(global_vector_cov)
    #            global_vector_inv_cov[:] = 0 #turn off the global vector
#                global_vector_inv_cov_dot_mean = np.dot(global_vector_inv_cov,global_vector_mean)
                print('Global mean:',global_vector_mean)
                print('Global std:',np.sqrt(np.diag(global_vector_cov)))
                
                global_parallax_mean = 0.5
                global_parallax_var = 10**2
                global_parallax_ivar = 1/global_parallax_var
                global_pm_mean = global_vector_mean[2:4]
                global_pm_cov = global_vector_cov[2:4,2:4]
                global_pm_inv_cov = np.linalg.inv(global_pm_cov)
                    
                #put a diffuse prior on the missing proper motions
                gaia_pm_x_errs[missing_prior_PM] = 100 #mas/yr
                gaia_pm_y_errs[missing_prior_PM] = 100 #mas/yr
                gaia_pm_x_y_corrs[missing_prior_PM] = 0
                
                gaia_pm_covs = np.zeros((len(x),2,2))
                gaia_pm_covs[:,0,0] = np.power(gaia_pm_x_errs,2)
                gaia_pm_covs[:,1,1] = np.power(gaia_pm_y_errs,2)
                gaia_pm_covs[:,0,1] = gaia_pm_x_errs*gaia_pm_y_errs*gaia_pm_x_y_corrs
                gaia_pm_covs[:,1,0] = gaia_pm_x_errs*gaia_pm_y_errs*gaia_pm_x_y_corrs
                
                gaia_pm_cov_eig_vals,gaia_pm_cov_eig_vects = np.linalg.eig(gaia_pm_covs)
                gaia_pm_cov_eig_vals = np.sqrt(gaia_pm_cov_eig_vals)
                gaia_pm_err_vects = np.array([gaia_pm_cov_eig_vals[:,0][:,None]*gaia_pm_cov_eig_vects[:,:,0],\
                                              gaia_pm_cov_eig_vals[:,1][:,None]*gaia_pm_cov_eig_vects[:,:,1]])
                gaia_err_size = np.sqrt(np.sum(np.power(gaia_pm_err_vects[0],2),axis=1)+np.sum(np.power(gaia_pm_err_vects[1],2),axis=1))
        #        gaia_err_size = np.sqrt(np.power(gaia_pm_x_errs,2)+np.power(gaia_pm_y_errs,2))
                #change this so that it is the 
#                inv_gaia_pm_covs = np.linalg.inv(gaia_pm_covs)
#                inv_gaia_pm_data_covs = np.linalg.inv(gaia_pm_covs*np.array([[1,-1],[-1,1]])) #in x,y Gaia vector instead of RA,Dec
                
                #put diffuse prior on parallaxes that are missing
                gaia_parallax_errs[missing_prior_PM] = 10 #mas
        #         gaia_parallaxes[missing_prior_PM] = 1e-6 #mas
                gaia_parallaxes[missing_prior_PM] = 0.5 #mas
            
                #remember to include the fact that the true parallax cannot be negative 
                #so the pdf values need to be rescaled by a factor of 1/(1-CDF(gaia_vals,x=0))
                #probably want to MCMC sample the parallaxes in log values to get close to parallax=0 and stay positive
                #which means that p(log(parallax)) = parallax*p(parallax)
                
                if fit_count == 0:
                    parallax_offset_vector = np.zeros((len(x),2))
                    print('Finding vectors for motion due to parallax:')
                    for star_ind,_ in enumerate(tqdm(parallax_offset_vector,total=len(parallax_offset_vector))):
                        #need to change these to change the times to be actual Astropy Time objects
                        hst_date = hst_times[star_ind]
                        gaia_date = gaia_times[star_ind]
                        #negative sign to keep the proper motion and offset vectors in same direction
                        #(i.e. final-initial time)
                        parallax_offset_vector[star_ind] = -1*delta_ra_dec_per_parallax(hst_date,gaia_date,
                                                                                gaia_ras[star_ind],gaia_decs[star_ind])
                        #multiply the DeltaRA by cos(Dec) so that all offsets are in the tangent plane (like XY)
                        #because dX = dRA*cos(Dec)
                        parallax_offset_vector[star_ind,0] *= np.cos(gaia_decs[star_ind]*np.pi/180) 
                        
                    #delta_data-useful_matrix*gaia_vectors = 0
                    useful_matrix = np.zeros((len(x),2,gaia_vectors.shape[1]))
                    useful_matrix[:,0,0] = -1
                    useful_matrix[:,1,1] = -1
                    useful_matrix[:,0,2] = delta_times
                    useful_matrix[:,1,3] = delta_times
                    useful_matrix[:,0,4] = parallax_offset_vector[:,0]
                    useful_matrix[:,1,4] = parallax_offset_vector[:,1]
                    #apply the jacobian that goes from changes in RA,Dec to changes in XY_Gaia pixels
                    useful_matrix = np.einsum('nij,njk->nik',proper_offset_jacs,useful_matrix)
        #            useful_matrix[:,0] *= -1 #change so that x=-RA
                                            
                #put strong prior on parallaxes for stationary sources
                gaia_parallax_errs[stationary] = 1e-6 #mas
                gaia_parallax_errs = np.sqrt(gaia_vector_covs[:,-1,-1])
                gaia_parallaxes[stationary] = 1e-6 #mas
#                gaia_prior_parallaxes = np.copy(gaia_parallaxes) #true prior parallax values from Gaia
                #use the maximum of the prior distribution (i.e. 0 if distribution has negative mean)
        #         gaia_parallaxes = np.maximum(1e-6,gaia_parallaxes)
#                gaia_parallax_dist_lognorms = 0#np.log(1-stats.norm(loc=gaia_prior_parallaxes,scale=gaia_parallax_errs).cdf(0))
                
                
                gaia_vectors[:,2] = gaia_pm_xs #PM_RA
                gaia_vectors[:,3] = gaia_pm_ys #PM_Dec
                gaia_vectors[:,4] = gaia_parallaxes #parallax
                
                if fit_count == 0:
                    hst_pix_sigmas = np.zeros((len(x),2))
#                    indv_hst_image_names = keep_image_names[img_nums]
#                    for star_ind,star_name in enumerate(gaia_id):
#                        hst_pix_sigmas[star_ind] = star_hst_pix_offsets[star_name]['final_std_x_hst'][indv_hst_image_names[star_ind]],\
#                                                   star_hst_pix_offsets[star_name]['final_std_y_hst'][indv_hst_image_names[star_ind]]
#                    hst_pix_sigmas *= hst_pos_err_mult #inflate the uncertainties on position
                    hst_pix_sigmas[:] = q_hst[:,None]*0.8
                    median_hst_pix_sigmas = np.median(hst_pix_sigmas,axis=0)
        #            hst_pix_sigmas = 0.5/50 #in hst pixels
        #            hst_pix_sigmas = 1.0/50 #in hst pixels
        #            hst_pix_sigmas = np.array([x_hst_err,y_hst_err]).T
        ##            hst_pix_sigmas = np.ones_like(hst_pix_sigmas)*(5.0/50) #this is for testing
        #            hst_pix_sigmas = np.ones_like(hst_pix_sigmas)*(0.01) #this is for testing
        
                    hst_covs = np.zeros((len(x),2,2))
                    hst_covs[:,0,0] = np.power(hst_pix_sigmas[:,0],2)
                    hst_covs[:,1,1] = np.power(hst_pix_sigmas[:,1],2)
                    
                    hst_inv_covs = np.linalg.inv(hst_covs)
    
                print('Median HST XY position uncertainties (pixels):',np.round(median_hst_pix_sigmas,3))
                
    #            hst_cov = np.zeros((2,2))
    #            hst_cov[0,0] = hst_pix_sigmas**2
    #            hst_cov[1,1] = hst_pix_sigmas**2   
                
                gaia_covs = np.zeros((len(x),2,2)) #in mas, so remember to convert
                gaia_covs[:,0,0] = np.power(gaia_ra_errs,2)
                gaia_covs[:,1,1] = np.power(gaia_dec_errs,2)
                gaia_covs[:,1,0] = gaia_ra_errs*gaia_dec_errs*gaia_ra_dec_corrs
                gaia_covs[:,0,1] = gaia_ra_errs*gaia_dec_errs*gaia_ra_dec_corrs
                
                n_im_show = min(2,n_images)
                n_param_shared = 0
                n_param_indv = 6      
                
                if fit_count == 0:
                    best_pm_parallax_offsets = np.zeros(len(x)*5)
                    #resort so that it goes rot,ratio,w0,z0,on_skew,off_skew
                    best_trans_params = np.zeros(n_images*n_param_indv)
                    for j in range(n_images):
                        x0,y0,w0,z0,ag,bg,cd,dg,rot_sign = param_outputs[j]
                        best_trans_params[j*n_param_indv:(j+1)*n_param_indv] = ag,bg,w0,z0,cd,dg
                    for star_ind in np.where(missing_prior_PM)[0]:
                        star_name = gaia_id[star_ind]
                        if star_name in previous_pm_results['average']:
                            gaia_vectors[star_ind] = previous_pm_results['average'][star_name]
                else:
                    for star_ind in np.where(missing_prior_PM)[0]:
                        curr_pm_x,curr_pm_y,curr_parallax,curr_offset_x,curr_offset_y = best_pm_parallax_offsets[star_ind*5:(star_ind+1)*5]
                        gaia_vectors[star_ind] = curr_offset_x,curr_offset_y,curr_pm_x,curr_pm_y,curr_parallax
                        gaia_pms[star_ind] = curr_pm_x,curr_pm_y
                        gaia_parallaxes[star_ind] = curr_parallax
                recent_trans_params = best_trans_params
                      
                ags = recent_trans_params[0::n_param_indv]
                bgs = recent_trans_params[1::n_param_indv]
                w0s = recent_trans_params[2::n_param_indv]
                z0s = recent_trans_params[3::n_param_indv]
                cgs = recent_trans_params[4::n_param_indv]
                dgs = recent_trans_params[5::n_param_indv]
                
                if fit_count == 0:
                    proper_offset_jacs /= indv_orig_pixel_scales[:,None,None] #scale by the pixel scale
                
                gaia_vectors[:,0:2] = 0 #mean of all offsets it 0,0
                gaia_offsets = np.copy(gaia_vectors[:,0:2])
                gaia_pms = np.copy(gaia_vectors[:,2:4])
                gaia_parallaxes = np.copy(gaia_vectors[:,4])
                gaia_offset_covs = np.copy(gaia_vector_covs[:,0:2,0:2])
                gaia_pm_covs = np.copy(gaia_vector_covs[:,2:4,2:4])
                gaia_parallax_vars = np.copy(gaia_vector_covs[:,4,4])
                
                gaia_offset_inv_covs = np.linalg.inv(gaia_offset_covs)
                gaia_pm_inv_covs = np.linalg.inv(gaia_pm_covs)
                gaia_parallax_ivars = 1/gaia_parallax_vars
                
                identities = np.zeros((len(x),2,2))
                identities[:] = np.eye(2)
                delta_time_identities = delta_times[:,None,None]*identities
                
                unique_gaia_offset_covs = gaia_offset_covs[unique_inds]
                unique_gaia_offset_inv_covs = gaia_offset_inv_covs[unique_inds]
                unique_gaia_offsets = gaia_offsets[unique_inds]
                unique_gaia_pm_covs = gaia_pm_covs[unique_inds]
                unique_gaia_pm_inv_covs = gaia_pm_inv_covs[unique_inds]
                unique_gaia_pms = gaia_pms[unique_inds]
                unique_gaia_parallaxes = gaia_parallaxes[unique_inds]
                unique_gaia_parallax_vars = gaia_parallax_vars[unique_inds]
#                unique_gaia_parallax_errs = np.sqrt(unique_gaia_parallax_vars)
                unique_gaia_parallax_ivars = gaia_parallax_ivars[unique_inds]
                proper_offset_jac_invs = np.linalg.inv(proper_offset_jacs)
                
                            
                unique_V_theta_i_inv_dot_theta_i = np.einsum('nij,nj->ni',unique_gaia_offset_inv_covs,unique_gaia_offsets)
                unique_V_mu_i_inv_dot_mu_i = np.einsum('nij,nj->ni',unique_gaia_pm_inv_covs,unique_gaia_pms)
                V_mu_global_inv_dot_mu_global = np.dot(global_pm_inv_cov,global_pm_mean)
                
                log_global_pm_cov_det = np.log(np.linalg.det(global_pm_cov))
                log_unique_gaia_pm_covs_det = np.log(np.linalg.det(unique_gaia_pm_covs))
                log_unique_gaia_offset_covs_det = np.log(np.linalg.det(unique_gaia_offset_covs))
                
#                gaia_vector_inv_covs_dot_vectors = np.einsum('nij,nj->ni',gaia_vector_inv_covs,gaia_vectors)
#                first_gaia_vector_inv_covs_dot_vectors = np.einsum('nij,nj->ni',first_gaia_vector_inv_covs,gaia_vectors)
                
                #use the first-measured proper motions as prior means on the PMs for stars that don't have Gaia PMs                    
                first_guess_vectors = np.zeros((len(x),5))
                
                star_hst_gaia_pos = np.zeros((len(x),2)) #in gaia pixels
                star_hst_gaia_pos_cov = np.zeros((len(x),2,2)) #in gaia pixels
                star_ratios = np.zeros(len(x))
                for j in range(n_images):
                    curr_img = np.where(img_nums == j)[0]
                    curr_x,curr_y = x[curr_img],y[curr_img]
                    curr_x_g,curr_y_g = x_g[curr_img],y_g[curr_img]
                    
                    x0,y0 = param_outputs[j,:2]
                    a,b,c,d = ags[j],bgs[j],cgs[j],dgs[j]
#                    ratio = np.sqrt(a*d-b*c)
#                    rot = np.arctan2(b-c,a+d)*180/np.pi
#                    on_skew = 0.5*(a-d)
#                    off_skew = 0.5*(b+c)
                    
                    w0 = w0s[j]
                    z0 = z0s[j]
                    
                    x_trans = a*(curr_x-x0)+b*(curr_y-y0)+w0
                    y_trans = c*(curr_x-x0)+d*(curr_y-y0)+z0
    
                    dx_trans = curr_x_g-x_trans
                    dy_trans = curr_y_g-y_trans
    
                    star_hst_gaia_pos[curr_img,0] = dx_trans
                    star_hst_gaia_pos[curr_img,1] = dy_trans
    #                dpix_trans = np.sqrt(np.power(dx_trans,2)+np.power(dy_trans,2))
    #                star_ratios[curr_img] = ratio
                    star_ratios[curr_img] = 1
                    det = a*d-b*c
                    #inverse matrix for de-transforming
                    ai,bi,ci,di = np.array([d,-b,-c,a])/det
    #                matrix = np.array([[a,b],[c,d]])
    #                inv_matrix = np.array([[ai,bi],[ci,di]])
    #
    #                x_final = ai*(curr_x_g-w0)+bi*(curr_y_g-z0)+x0 #HST final position
    #                y_final = ci*(curr_x_g-w0)+di*(curr_y_g-z0)+y0
    
                    matrices = np.zeros((2,2))
                    matrices_T = np.zeros((2,2))
                    inv_matrices = np.zeros((2,2))
                    inv_matrices_T = np.zeros((2,2))
    
                    matrices[0,0] = a
                    matrices[0,1] = b
                    matrices[1,0] = c
                    matrices[1,1] = d
                    matrices_T[0,0] = a
                    matrices_T[0,1] = c
                    matrices_T[1,0] = b
                    matrices_T[1,1] = d
                    inv_matrices[0,0] = ai
                    inv_matrices[0,1] = bi
                    inv_matrices[1,0] = ci
                    inv_matrices[1,1] = di
                    inv_matrices_T[0,0] = ai
                    inv_matrices_T[0,1] = ci
                    inv_matrices_T[1,0] = bi
                    inv_matrices_T[1,1] = di
                    
                    hst_cov_in_gaia = np.einsum('ij,njk->nik',matrices,np.einsum('nij,jk->nik',hst_covs[curr_img],matrices_T))
    #                hst_cov_in_gaia = np.dot(matrices,np.dot(hst_cov,matrices_T))
                    
                    star_hst_gaia_pos_cov[curr_img] = hst_cov_in_gaia
                    
    #            np.random.seed(100)
                
                star_hst_gaia_pos_inv_cov = np.linalg.inv(star_hst_gaia_pos_cov)
                jac_V_data_inv_jac = np.einsum('nji,njk->nik',proper_offset_jacs,np.einsum('nij,njk->nik',star_hst_gaia_pos_inv_cov,proper_offset_jacs))
                inv_jac_dot_d_ij = np.einsum('nij,nj->ni',proper_offset_jac_invs,star_hst_gaia_pos)
                summed_jac_V_data_inv_jac = np.add.reduceat(jac_V_data_inv_jac*use_inds[:,None,None],unique_inds)
                Sigma_theta_i_inv = unique_gaia_offset_inv_covs+summed_jac_V_data_inv_jac
                Sigma_theta_i = np.linalg.inv(Sigma_theta_i_inv)
                
                jac_V_data_inv_jac_dot_parallax_vects = np.einsum('nij,nj->ni',jac_V_data_inv_jac,parallax_offset_vector)
                summed_jac_V_data_inv_jac_dot_parallax_vects = np.add.reduceat(jac_V_data_inv_jac_dot_parallax_vects*use_inds[:,None],unique_inds)
                jac_V_data_inv_jac_dot_d_ij = np.einsum('nij,nj->ni',jac_V_data_inv_jac,inv_jac_dot_d_ij)
                summed_jac_V_data_inv_jac_dot_d_ij = np.add.reduceat(jac_V_data_inv_jac_dot_d_ij*use_inds[:,None],unique_inds)
                summed_jac_V_data_inv_jac_times = np.add.reduceat(jac_V_data_inv_jac*delta_times[:,None,None]*use_inds[:,None,None],unique_inds)
                
                A_mu_i = np.einsum('nij,njk->nik',Sigma_theta_i,summed_jac_V_data_inv_jac_times)
                C_mu_ij = delta_time_identities-A_mu_i[unique_inv_inds]
                A_mu_i_inv = np.linalg.inv(A_mu_i)
                C_mu_ij_inv = np.linalg.inv(C_mu_ij)
                
                Sigma_mu_theta_i_inv = np.einsum('nij,njk->nik',np.einsum('nji,njk->nik',A_mu_i,unique_gaia_offset_inv_covs),A_mu_i)
                Sigma_mu_d_ij_inv = np.einsum('nij,njk->nik',np.einsum('nji,njk->nik',C_mu_ij,jac_V_data_inv_jac),C_mu_ij)
                
                Sigma_mu_i_inv = global_pm_inv_cov+unique_gaia_pm_inv_covs+Sigma_mu_theta_i_inv+\
                                 np.add.reduceat(Sigma_mu_d_ij_inv*use_inds[:,None,None],unique_inds)
                Sigma_mu_i = np.linalg.inv(Sigma_mu_i_inv)
                
                A_plx_mu_i = np.einsum('nij,nj->ni',Sigma_theta_i,-1*summed_jac_V_data_inv_jac_dot_parallax_vects)
                B_plx_mu_i = np.einsum('nij,nj->ni',Sigma_theta_i,unique_V_theta_i_inv_dot_theta_i\
                                                                    -summed_jac_V_data_inv_jac_dot_d_ij)
                
                Sigma_mu_theta_i_inv_dot_A_mu_i_inv = np.einsum('nij,njk->nik',Sigma_mu_theta_i_inv,A_mu_i_inv)
                Sigma_mu_d_ij_inv_dot_C_mu_ij_inv = np.einsum('nij,njk->nik',Sigma_mu_d_ij_inv,C_mu_ij_inv)
                
                C_plx_mu_i = np.einsum('nij,nj->ni',Sigma_mu_i,np.einsum('nij,nj->ni',Sigma_mu_theta_i_inv_dot_A_mu_i_inv,A_plx_mu_i)\
                                                                 -np.add.reduceat(np.einsum('nij,nj->ni',Sigma_mu_d_ij_inv_dot_C_mu_ij_inv,parallax_offset_vector+A_plx_mu_i[unique_inv_inds])*use_inds[:,None],unique_inds))
                D_plx_mu_i = -1*np.einsum('nij,nj->ni',Sigma_mu_i,unique_V_mu_i_inv_dot_mu_i+V_mu_global_inv_dot_mu_global+\
                                                                    +np.einsum('nij,nj->ni',Sigma_mu_theta_i_inv_dot_A_mu_i_inv,unique_gaia_offsets-B_plx_mu_i)\
                                                                    +np.add.reduceat(np.einsum('nij,nj->ni',Sigma_mu_d_ij_inv_dot_C_mu_ij_inv,inv_jac_dot_d_ij+B_plx_mu_i[unique_inv_inds])*use_inds[:,None],unique_inds))
                                    
                E_plx_theta_i = np.einsum('nij,nj->ni',A_mu_i,C_plx_mu_i)-A_plx_mu_i
                F_plx_theta_i = np.einsum('nij,nj->ni',A_mu_i,D_plx_mu_i)-B_plx_mu_i
                
                G_plx_d_ij = np.einsum('nij,nj->ni',C_mu_ij,C_plx_mu_i[unique_inv_inds])+A_plx_mu_i[unique_inv_inds]+parallax_offset_vector
                H_plx_d_ij = np.einsum('nij,nj->ni',C_mu_ij,D_plx_mu_i[unique_inv_inds])+B_plx_mu_i[unique_inv_inds]+inv_jac_dot_d_ij
    
                G_plx_d_ij_T_dot_V_data_inv = np.einsum('nj,nij->ni',G_plx_d_ij,jac_V_data_inv_jac)  
                ivar_plx_d_ij = np.einsum('ni,ni->n',G_plx_d_ij_T_dot_V_data_inv,G_plx_d_ij)
                mu_times_ivar_plx_d_ij = np.einsum('ni,ni->n',G_plx_d_ij_T_dot_V_data_inv,H_plx_d_ij)
#                mu_plx_d_ij = mu_times_ivar_plx_d_ij/ivar_plx_d_ij
                summed_ivar_plx_d_ij = np.add.reduceat(ivar_plx_d_ij*use_inds,unique_inds)
                summed_mu_times_ivar_plx_d_ij = np.add.reduceat(mu_times_ivar_plx_d_ij*use_inds,unique_inds)
                
                C_plx_mu_i_T_dot_V_mu_i_inv = np.einsum('nj,nij->ni',C_plx_mu_i,unique_gaia_pm_inv_covs)  
                ivar_plx_mu_i = np.einsum('ni,ni->n',C_plx_mu_i_T_dot_V_mu_i_inv,C_plx_mu_i)
                mu_times_ivar_plx_mu_i = np.einsum('ni,ni->n',C_plx_mu_i_T_dot_V_mu_i_inv,D_plx_mu_i+unique_gaia_pms)
#                mu_plx_mu_i = mu_times_ivar_plx_mu_i/ivar_plx_mu_i
                
                C_plx_mu_i_T_dot_V_mu_global_inv = np.einsum('nj,ij->ni',C_plx_mu_i,global_pm_inv_cov)  
                ivar_plx_mu_global = np.einsum('ni,ni->n',C_plx_mu_i_T_dot_V_mu_global_inv,C_plx_mu_i)
                mu_times_ivar_plx_mu_global = np.einsum('ni,ni->n',C_plx_mu_i_T_dot_V_mu_global_inv,D_plx_mu_i+global_pm_mean)
#                mu_plx_mu_global = mu_times_ivar_plx_mu_global/ivar_plx_mu_global
                
                E_plx_theta_i_T_dot_V_theta_i_inv = np.einsum('nj,nij->ni',E_plx_theta_i,unique_gaia_offset_inv_covs)  
                ivar_plx_theta_i = np.einsum('ni,ni->n',E_plx_theta_i_T_dot_V_theta_i_inv,E_plx_theta_i)
                mu_times_ivar_plx_theta_i = np.einsum('ni,ni->n',E_plx_theta_i_T_dot_V_theta_i_inv,F_plx_theta_i+unique_gaia_offsets)
#                mu_plx_theta_i = mu_times_ivar_plx_theta_i/ivar_plx_theta_i
                
                ivar_plx_i = summed_ivar_plx_d_ij+ivar_plx_mu_i+ivar_plx_mu_global\
                             +ivar_plx_theta_i+unique_gaia_parallax_ivars+global_parallax_ivar
                var_plx_i = 1/ivar_plx_i
                std_plx_i = np.sqrt(var_plx_i)
                mu_plx_i = (summed_mu_times_ivar_plx_d_ij+mu_times_ivar_plx_mu_i+mu_times_ivar_plx_mu_global\
                             +mu_times_ivar_plx_theta_i\
                             +unique_gaia_parallax_ivars*unique_gaia_parallaxes\
                             +global_parallax_ivar*global_parallax_mean)/ivar_plx_i
                            
                first_guess_parallaxes = mu_plx_i[unique_inv_inds]
                parallax_draws = mu_plx_i
                
                B_mu_i = parallax_draws[:,None]*A_plx_mu_i-B_plx_mu_i
                mu_mu_i = parallax_draws[:,None]*C_plx_mu_i-D_plx_mu_i
                
                first_guess_pms = mu_mu_i[unique_inv_inds]
                pm_draws = mu_mu_i
    
                mu_theta_i = np.einsum('nij,nj->ni',A_mu_i,pm_draws)-B_mu_i
                
                first_guess_offsets = mu_theta_i[unique_inv_inds]
                offset_draws = mu_theta_i
                            
                first_guess_vectors[:,0:2] = first_guess_offsets
                first_guess_vectors[:,2:4] = first_guess_pms
                first_guess_vectors[:,4] = first_guess_parallaxes
    
                gaia_vectors[missing_prior_PM] = first_guess_vectors[missing_prior_PM] 
    
                gaia_vectors[:,0:2] = 0 #mean of all offsets it 0,0
                gaia_offsets = np.copy(gaia_vectors[:,0:2])
                gaia_pms = np.copy(gaia_vectors[:,2:4])
                gaia_parallaxes = np.copy(gaia_vectors[:,4])
                gaia_offset_covs = np.copy(gaia_vector_covs[:,0:2,0:2])
                gaia_pm_covs = np.copy(gaia_vector_covs[:,2:4,2:4])
                gaia_parallax_vars = np.copy(gaia_vector_covs[:,4,4])
                
                gaia_offset_inv_covs = np.linalg.inv(gaia_offset_covs)
                gaia_pm_inv_covs = np.linalg.inv(gaia_pm_covs)
                gaia_parallax_ivars = 1/gaia_parallax_vars
                
                identities = np.zeros((len(x),2,2))
                identities[:] = np.eye(2)
                delta_time_identities = delta_times[:,None,None]*identities
                
#                unique_gaia_offset_covs = gaia_offset_covs[unique_inds]
                unique_gaia_offset_inv_covs = gaia_offset_inv_covs[unique_inds]
                unique_gaia_offsets = gaia_offsets[unique_inds]
#                unique_gaia_pm_covs = gaia_pm_covs[unique_inds]
                unique_gaia_pm_inv_covs = gaia_pm_inv_covs[unique_inds]
                unique_gaia_pms = gaia_pms[unique_inds]
                unique_gaia_parallaxes = gaia_parallaxes[unique_inds]
#                unique_gaia_parallax_vars = gaia_parallax_vars[unique_inds]
                unique_gaia_parallax_ivars = gaia_parallax_ivars[unique_inds]
                
                #first guess
                pos0 = []
                dimLabels = []
                #widths to explore for pos0
                pos0_widths = []
                trans_param_width = 1e-3
                wo_width = 0.01
                zo_width = 0.01
                pos0 = np.copy(recent_trans_params)
                trans_param_mult = 1
                wz_mult = 1e-3
                walker_mults = []
                for j in range(n_images):
    #                xo,yo,wo,zo,rot,ratio,on_skew,off_skew,rot_sign = param_outputs[j]
    #                pos0.extend([rot,ratio,wo,zo,on_skew,off_skew])
                    pos0_widths.extend([trans_param_width,trans_param_width,wo_width,zo_width,trans_param_width,trans_param_width])
                    walker_mults.extend([trans_param_mult,trans_param_mult,wz_mult,wz_mult,trans_param_mult,trans_param_mult])
                    dimLabels.extend(['AG$_{%d}$'%(j+1),'BG$_{%d}$'%(j+1),r'W0$_{%d}$'%(j+1),r'Z0$_{%d}$'%(j+1),'CG$_{%d}$'%(j+1),'DG$_{%d}$'%(j+1),])
                pos0 = np.array(pos0)
#                orig_pos0 = np.copy(pos0)
                pos0_widths = np.array(pos0_widths)
                walker_mults = np.array(walker_mults)
#                ndim_trans = len(pos0)
        
                nwalkers,ndim,nsteps = int(len(pos0)*3),len(pos0),100
                nwalkers,ndim,nsteps = int(len(pos0)*10),len(pos0),500
                nwalkers,ndim,nsteps = max(200,int(len(pos0)*10)),len(pos0),2000
                nwalkers = max(200,10*ndim)
                nwalkers = 40*ndim
        #        nsteps = 10000
                nwalkers = 10*ndim #number to keep 
                nwalkers_sample = 50*ndim #number to use during MCMC (should be large)
                nwalkers_sample = int(round(nwalkers_sample/n_threads))*n_threads #make it nicely divisble by n_threads
    #            nwalkers_sample = 20*ndim #number to use during MCMC (should be large)
                nwalkers = min(100,nwalkers_sample)
                nsteps = 600
                if fit_count == 0:
                    nsteps = 2000
                    nsteps = 1000
                    nsteps = 600+300*len(keep_im_nums)
                else:
                    nsteps = 1000
                    nsteps = 100+200*len(keep_im_nums)
                if n_images == 1:
                    if fit_count == 0:
                        if np.sum(keep_stars) < 20:
                            nsteps = 1000
                        else:
                            nsteps = 800
                    else:
                        nsteps = 600
                # nsteps = 600
                    
                burnin = int(0.7*nsteps)
            #    nwalkers,ndim,nsteps = int(len(pos0)*2),len(pos0),100
    
            #     pos = pos0*(1+1e-3*np.random.randn(nwalkers*ndim).reshape((nwalkers,ndim)))
    #            pos = pos0+np.random.randn(nwalkers*ndim).reshape((nwalkers,ndim))*pos0_widths
                if fit_count == 0:
                    #start using the pos0_widths to define the covariance matrix
                    best_trans_param_covs = np.diag(np.power(pos0_widths,2))
                    for j,hst_image in enumerate(hst_image_names):
                        #if there are previous analysis results, use those covariances instead
                        if hst_image in previous_analysis_results:
                            best_trans_param_covs[j*n_param_indv:(j+1)*n_param_indv,j*n_param_indv:(j+1)*n_param_indv] = previous_analysis_results[hst_image][1]
                    best_trans_param_covs *= (0.5/0.9)**2 #make smaller for first iteration
                samp_mult = 1e5
                scaled_pos0 = pos0*walker_mults*samp_mult
                scaled_cov = (0.9**2)*best_trans_param_covs
                scaled_cov *= samp_mult*walker_mults[:,None]
                scaled_cov *= samp_mult*walker_mults[None,:]
                pos = stats.multivariate_normal(mean=scaled_pos0,cov=scaled_cov,allow_singular=True).rvs(nwalkers_sample)
                pos += np.random.randn(nwalkers_sample,ndim)*np.sqrt(np.diag(scaled_cov))*0.01 #to fight singular matrices
                pos /= walker_mults[None,:]*samp_mult
                
#                unique_gaia_vectors = gaia_vectors[unique_inds]
#                unique_gaia_vector_inv_covs = gaia_vector_inv_covs[unique_inds]
#                unique_gaia_vector_covs = gaia_vector_covs[unique_inds]
#                unique_det_gaia_vector_covs = np.linalg.det(unique_gaia_vector_covs)
#                unique_gaia_vector_inv_covs_dot_vectors = gaia_vector_inv_covs_dot_vectors[unique_inds]
                
                x0s = param_outputs[:,0]
                y0s = param_outputs[:,1]
                xy0s = np.array([x0s,y0s]).T[img_nums]
                xy = np.array([x,y]).T
                xy_g = np.array([x_g,y_g]).T
                
                print()
                print('MCMC Fitting of transformation parameters, proper motions, and parallaxes:')
            
        #         with Pool(n_threads) as pool:
        # #             sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, pool = pool)
        # #             sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost_MAP, pool = pool)
        #             sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost_integrated, pool = pool)
        #             for j, result in enumerate(tqdm(sampler.sample(pos, iterations=nsteps),total=nsteps,smoothing=0.1)):
        #                 pass
                    
                samplerChain = np.zeros(((nwalkers,nsteps,ndim)))
                step_inds = np.arange(nsteps).astype(int)
                walker_inds = np.arange(nwalkers_sample).astype(int)
                
                lnposts = np.zeros((nwalkers,nsteps))
                accept_fracs = np.zeros(nsteps)
                pos_lnpost = np.zeros((nwalkers_sample))
                previous_params = pos
                
                step_ind = -1 #for the first call
                                
                def lnpost_previous(walker_ind) -> np.ndarray:
                    params = previous_params[walker_ind]
                    return lnpost_vector(params,
                                            n_param_indv,x,x0s,ags,bgs,cgs,dgs,img_nums,xy,xy0s,xy_g,hst_covs,
                                            proper_offset_jacs,proper_offset_jac_invs,unique_gaia_offset_inv_covs,use_inds,unique_inds,
                                            parallax_offset_vector,delta_times,unique_inv_inds,delta_time_identities,global_pm_inv_cov,
                                            unique_gaia_pm_inv_covs,unique_V_theta_i_inv_dot_theta_i,unique_V_mu_i_inv_dot_mu_i,
                                            V_mu_global_inv_dot_mu_global,unique_gaia_offsets,unique_gaia_pms,global_pm_mean,
                                            unique_gaia_parallax_ivars,global_parallax_ivar,unique_ids,unique_gaia_parallaxes,
                                            global_parallax_mean,log_unique_gaia_pm_covs_det,log_global_pm_cov_det,
                                            unique_gaia_parallax_vars,log_unique_gaia_offset_covs_det,unique_keep,
                                            seed=walker_ind+(step_ind+1)*(nwalkers_sample+1))                    
                
                def lnpost_new(walker_ind) -> np.ndarray:
                    params = new_params[walker_ind]
                    return lnpost_vector(params,
                                            n_param_indv,x,x0s,ags,bgs,cgs,dgs,img_nums,xy,xy0s,xy_g,hst_covs,
                                            proper_offset_jacs,proper_offset_jac_invs,unique_gaia_offset_inv_covs,use_inds,unique_inds,
                                            parallax_offset_vector,delta_times,unique_inv_inds,delta_time_identities,global_pm_inv_cov,
                                            unique_gaia_pm_inv_covs,unique_V_theta_i_inv_dot_theta_i,unique_V_mu_i_inv_dot_mu_i,
                                            V_mu_global_inv_dot_mu_global,unique_gaia_offsets,unique_gaia_pms,global_pm_mean,
                                            unique_gaia_parallax_ivars,global_parallax_ivar,unique_ids,unique_gaia_parallaxes,
                                            global_parallax_mean,log_unique_gaia_pm_covs_det,log_global_pm_cov_det,
                                            unique_gaia_parallax_vars,log_unique_gaia_offset_covs_det,unique_keep,
                                            seed=walker_ind+(step_ind+1)*(nwalkers_sample+1))                    
                                
                for w_ind in range(nwalkers_sample):
                    vals = lnpost_previous(w_ind)
                    pos_lnpost[w_ind] = vals[0,0]
                    
                    curr_pos_offset_draw = vals[:,[1,2]]
                    curr_pm_draw = vals[:,[3,4]]
                    curr_parallax_draw = vals[:,5]
                    if w_ind == 0:
                        pos_parallaxes = np.zeros((nwalkers_sample,*curr_parallax_draw.shape))
                        pos_pms = np.zeros((nwalkers_sample,*curr_pm_draw.shape))                    
                        pos_offsets = np.zeros((nwalkers_sample,*curr_pos_offset_draw.shape))
                        
                        samplerChain_parallaxes = np.zeros(((nwalkers,nsteps,*curr_parallax_draw.shape)))
                        samplerChain_pms = np.zeros(((nwalkers,nsteps,*curr_pm_draw.shape)))
                        samplerChain_offsets = np.zeros(((nwalkers,nsteps,*curr_pos_offset_draw.shape)))
                        
                    pos_parallaxes[w_ind] = curr_parallax_draw
                    pos_pms[w_ind] = curr_pm_draw
                    pos_offsets[w_ind] = curr_pos_offset_draw
                        
    #            print('lnpost',pos_lnpost.min(),pos_lnpost.max(),np.nanmedian(pos_lnpost))
                
                stretch_val = 1.0
                stretch_a = 1+stretch_val #recommended amount a=2.0
                stretch_a_vals = np.zeros(nsteps)
                update_stretch_size = 50 #update stretch move every this many steps
                update_mvn_size = 10
                
                for step_ind,_ in enumerate(tqdm(step_inds,total=nsteps)):                
                    np.random.seed(step_ind)
                    if step_ind == 0:
                        previous_params = pos
                        previous_lnpost = pos_lnpost
                        previous_parallaxes = pos_parallaxes
                        previous_pms = pos_pms
                        previous_offsets = pos_offsets
                    
                    accepted_params = np.copy(previous_params)
                    accepted_lnposts = np.copy(previous_lnpost)
                    accepted_parallaxes = np.copy(previous_parallaxes)
                    accepted_pms = np.copy(previous_pms)
                    accepted_offsets = np.copy(previous_offsets)
                        
    #                samplerChain[:,step_ind] = np.copy(previous_params[:nwalkers])
    #                lnposts[:,step_ind] = np.copy(previous_lnpost[:nwalkers])
    #                samplerChain_parallaxes[:,step_ind] = np.copy(previous_parallaxes[:nwalkers])
    #                samplerChain_pms[:,step_ind] = np.copy(previous_pms[:nwalkers])
    #                samplerChain_offsets[:,step_ind] = np.copy(previous_offsets[:nwalkers])
        
                    #MCMC to measure all of the transformation parameters
        
                    prop_factors = np.zeros(nwalkers_sample) #nothing to account for in proposal distribution
                    new_params = np.copy(previous_params)
        
                    #use the stretch move formalism (https://arxiv.org/pdf/1202.3665.pdf)
                    #code similar to https://github.com/dfm/emcee/blob/main/src/emcee/moves/stretch.py
        
                    # Get the move-specific proposal.
        #            if step_ind < 200:
        #                stretch_a = 2.0 #recommended amount
        #            else:
        #                stretch_a = 1.2
                    
                    #think about making an adaptive strech move a factor (i.e. change based on the last 10 acceptance fraction values)
        #             stretch_a = 1.2
                    if (step_ind >= 100*n_images) and (step_ind%update_stretch_size == 0):
                        recent_med_accept = np.median(accept_fracs[step_ind-update_stretch_size:step_ind])
                        stretch_val = max(1e-5,min(1.0,stretch_val*(recent_med_accept/0.5))) #change stretch_val to get median closer to 50%
                        stretch_a = 1+stretch_val #recommended amount a=2.0
                    stretch_a_vals[step_ind] = stretch_a
                    
                    if (step_ind >= 100*n_images):# and (fit_count > 0):
                    # if (step_ind >= update_mvn_size):# and (fit_count > 0):
                    # if (step_ind >= 100):# and (fit_count > 0):
                    # if False:# and (fit_count > 0):
                        if (step_ind%update_mvn_size == 0):
                            # recent_samples = previous_params*samp_mult
                            recent_samples = samp_mult*walker_mults[None,:]*samplerChain[:, step_ind-update_mvn_size:step_ind-1, :].reshape((-1, ndim))
                            recent_med_accept = np.median(accept_fracs[step_ind-update_mvn_size:step_ind-1])
                            prop_cov = np.cov(recent_samples,rowvar=False)
                            prop_cov[np.arange(ndim),np.arange(ndim)] = np.maximum((1e-10*samp_mult)**2,prop_cov[np.arange(ndim),np.arange(ndim)])
                            prop_icov = np.linalg.inv(prop_cov)
                            prop_mean = np.mean(recent_samples,axis=0)
                            # prop_cov *= min(1.0**2,(max(recent_med_accept,1/nwalkers_sample)/1.0/3)**2)
                            prop_dist = stats.multivariate_normal(mean=prop_mean,cov=prop_cov,allow_singular=True)
                            # print(recent_med_accept)#,min(1.0**2,(max(recent_med_accept,1/nwalkers_sample)/1.0)**2))
                        new_params = prop_dist.rvs(nwalkers_sample)
                        # prop_factors = prop_dist.logpdf(previous_params)-prop_dist.logpdf(new_params)
                        new_diff = new_params-prop_mean
                        previous_diff = previous_params*samp_mult*walker_mults[None,:]-prop_mean
                        logprop_new = -0.5*np.einsum('wi,wi->w',new_diff,np.einsum('ij,wj->wi',prop_icov,new_diff))
                        logprop_previous = -0.5*np.einsum('wi,wi->w',previous_diff,np.einsum('ij,wj->wi',prop_icov,previous_diff))
                        prop_factors = logprop_previous-logprop_new
                        new_params /= samp_mult*walker_mults[None,:]
                        # print(new_params[0])
                    else:
                        #use stretch move for the first fitting iteration, code from DFM's emcee
                        n_splits = 2
                        randomize_split = True
#                        live_dangerously = False
                        
                        inds = walker_inds % n_splits
                        if randomize_split:
                            np.random.shuffle(inds)
                            
                        for split in range(n_splits):
                            S1 = (inds == split)
                
                            # Get the two halves of the ensemble.
                            sets = [previous_params[inds == j] for j in range(n_splits)]
                            s = sets[split]
                            c = sets[:split] + sets[split + 1 :]
                            
                            c = np.concatenate(c, axis=0)
                            Ns, Nc = len(s), len(c)
                            zz = np.power((stretch_a-1.0)*np.random.rand(Ns)+1,2.0)/stretch_a
                            prop_factors[S1] = (ndim - 1.0) * np.log(zz)
                            rint = np.random.randint(Nc, size=(Ns,))
                            new_params[S1] = c[rint] - (c[rint] - s) * zz[:, None]
                        
                    # #draw a walker, but not the current walker, for each walker
                    # non_matching_ind_draws = (np.random.randint(nwalkers-1,size=nwalkers_sample)+1+walker_inds)%nwalkers_sample
                    # non_matching_params = previous_params[non_matching_ind_draws]
                    # zz = np.power((stretch_a-1.0)*np.random.rand(nwalkers_sample)+1,2)/stretch_a
                    # prop_factors = (ndim-1.0)*np.log(zz)
                    # new_params = non_matching_params-(non_matching_params-previous_params)*zz[:, None]               
                    if n_threads == 1:
                        first = lnpost_new(walker_inds[0])
                        
                        new_vals = np.zeros((len(walker_inds),*first.shape))
                        for j,walker_ind in enumerate(walker_inds):
                            vals = lnpost_new(walker_ind)
                            if j == 0:
                                new_vals = np.zeros((len(walker_inds),*vals.shape))
                            new_vals[j] = vals
                    else:
                        # num_workers = 4
                        
                        # # new_vals = multiprocessing.Queue()
                        # # p = multiprocessing.Process(target=lnpost_new, args=(walker_inds, new_vals))
                        # # p.start()   
                        # # p.join()
                        
                        # manager = Manager()
                        # results = manager.list()
                        # new_vals = manager.Queue(num_workers)  
                        
                        # # start for workers    
                        # pool = []
                        # for i in range(num_workers):
                        #     p = Process(target=lnpost_new, args=(walker_inds, new_vals))
                        #     p.start()
                        #     pool.append(p)                       
                        # for p in pool:
                        #     p.join()
                        
#                        with MultiPool(n_threads) as pool:
#                            new_vals = pool.map(lnpost_new,walker_inds)
##                        with ThreadPoolExecutor(max_workers=n_threads) as executor:
##                            for walker_ind in walker_inds:
##                                vals = executor.submit(lnpost_new, walker_ind).result()
##                                if walker_ind == 0:
##                                    new_vals = np.zeros((len(walker_inds),*vals.shape))
##                                # new_vals = executor.submit(lnpost_new, walker_ind)
##                                new_vals[walker_ind] = vals
                    
                        out_q = mp.Queue()
                        chunksize = int(math.ceil(len(walker_inds) / float(n_threads)))
                        procs = []
                        
                        for thread_ind in range(n_threads):
                            args = (walker_inds[chunksize * thread_ind:chunksize * (thread_ind + 1)],thread_ind,
                                    new_params,nwalkers_sample,step_ind,
                                    n_param_indv,x,x0s,ags,bgs,cgs,dgs,img_nums,xy,xy0s,xy_g,hst_covs,
                                    proper_offset_jacs,proper_offset_jac_invs,unique_gaia_offset_inv_covs,use_inds,unique_inds,
                                    parallax_offset_vector,delta_times,unique_inv_inds,delta_time_identities,global_pm_inv_cov,
                                    unique_gaia_pm_inv_covs,unique_V_theta_i_inv_dot_theta_i,unique_V_mu_i_inv_dot_mu_i,
                                    V_mu_global_inv_dot_mu_global,unique_gaia_offsets,unique_gaia_pms,global_pm_mean,
                                    unique_gaia_parallax_ivars,global_parallax_ivar,unique_ids,unique_gaia_parallaxes,
                                    global_parallax_mean,log_unique_gaia_pm_covs_det,log_global_pm_cov_det,
                                    unique_gaia_parallax_vars,log_unique_gaia_offset_covs_det,unique_keep,
                                    out_q)
                            p = mp.Process(target=lnpost_new_parallel,
                                           args=args)
                            procs.append(p)
                            p.start()
                    
                        new_vals = np.zeros((len(walker_inds),len(unique_ids),5+1))
                        for thread_ind in range(n_threads):
                            vals,curr_thread = out_q.get()
                            new_vals[chunksize * curr_thread:chunksize * (curr_thread + 1)] = vals
                    
                        for p in procs:
                            p.join()
                                        
##                        if n_images > 1:
#                        if False:
#                            with Pool(processes=n_threads) as pool:
##                            with MultiPool(processes=n_threads) as pool:
#                            # with get_context("spawn").Pool() as pool:
#                                new_vals = pool.map(lnpost_new,walker_inds)
#                                # new_vals = pool.map(lnpost_new,walker_inds,chunksize=1)
#                                # new_vals = pool.map(lnpost_new,walker_inds,chunksize=len(walker_inds)//n_threads)
#                        else:
#                            new_vals = np.zeros((len(walker_inds),len(unique_ids),5+1))
#                            with ThreadPoolExecutor(n_threads) as executor:
#                                for walker_ind in walker_inds:
#                                    vals = executor.submit(lnpost_new, walker_ind).result()
#                                    new_vals[walker_ind] = vals
##                                for walker_ind,vals in enumerate(executor.map(lnpost_new, walker_inds)):
##                                    new_vals[walker_ind] = vals
                                                            
                    new_vals = np.array(new_vals)
#                    print(new_vals)
                    new_lnpost = new_vals[:,0,0]
                    new_parallaxes = new_vals[:,:,5]
                    new_pms = new_vals[:,:,[3,4]]
                    new_offsets = new_vals[:,:,[1,2]]
        
                    #accept with MH condition
                    lnpdiff = new_lnpost - previous_lnpost + prop_factors
                    accepted = np.log(np.random.rand(nwalkers_sample)) < lnpdiff
                    
                    accepted_params[accepted] = new_params[accepted]
                    accepted_lnposts[accepted] = new_lnpost[accepted]
                    accepted_parallaxes[accepted] = new_parallaxes[accepted]
                    accepted_pms[accepted] = new_pms[accepted]
                    accepted_offsets[accepted] = new_offsets[accepted]
                    
                    accept_fracs[step_ind] = np.sum(accepted)/len(accepted)
#                    print(accept_fracs[step_ind],lnpdiff.min(),lnpdiff.max(),np.median(lnpdiff))
#                    print(new_vals[:,0,0])
                    
                    samplerChain[:,step_ind] = accepted_params[:nwalkers]
                    lnposts[:,step_ind] = accepted_lnposts[:nwalkers]
                    samplerChain_parallaxes[:,step_ind] = accepted_parallaxes[:nwalkers]
                    samplerChain_pms[:,step_ind] = accepted_pms[:nwalkers]
                    samplerChain_offsets[:,step_ind] = accepted_offsets[:nwalkers]
                    
                    previous_params = accepted_params
                    previous_lnpost = accepted_lnposts
                    previous_parallaxes = accepted_parallaxes
                    previous_pms = accepted_pms
                    previous_offsets = accepted_offsets
    
        #         samplerChain = sampler.chain
                samples = samplerChain[:, burnin:, :].reshape((-1, ndim))
                if len(samples) > 30000:
                    chosen_inds = np.random.choice(len(samples),size=30000,replace=False)
                    samples = samples[chosen_inds]
                else:
                    chosen_inds = np.arange(len(samples)).astype(int)
                sample_lnposts = (lnposts[:,burnin:].reshape((-1,1)))[chosen_inds]
                sample_parallaxes = (samplerChain_parallaxes[:, burnin:].reshape((-1, pos_parallaxes.shape[1])))[chosen_inds]
                sample_pms = (samplerChain_pms[:, burnin:].reshape((-1, pos_pms.shape[1], 2)))[chosen_inds]
                sample_offsets = (samplerChain_offsets[:, burnin:].reshape((-1, pos_offsets.shape[1], 2)))[chosen_inds]
#                median_params = np.percentile(samples,50,axis=0)
#                median_param_errs = np.std(samples,axis=0)
#                orig_offsets = np.sqrt(np.power(data_combined[mask_name]['dX_G'][sort_gaia_id_inds],2)+np.power(data_combined[mask_name]['dY_G'][sort_gaia_id_inds],2))
#                orig_xy_offsets = np.array([data_combined[mask_name]['dX_G'][sort_gaia_id_inds],data_combined[mask_name]['dY_G'][sort_gaia_id_inds]]).T
#                median_offset = np.zeros_like(orig_offsets)
#                median_xy_offset = np.zeros_like(orig_xy_offsets)
#                median_xy_offset_hst = np.zeros_like(orig_xy_offsets)
                
                sample_med = np.median(samples,axis=0)
                sample_cov = np.cov(samples,rowvar=False)
                best_trans_params = sample_med
                best_trans_param_covs = sample_cov
                    
                ags = samples[:,0::n_param_indv]
                bgs = samples[:,1::n_param_indv]
                cgs = samples[:,4::n_param_indv]
                dgs = samples[:,5::n_param_indv]
                
                samples_with_skews = np.zeros_like(samples)
                #rotations
                samples_with_skews[:,0::n_param_indv] = np.arctan2(bgs-cgs,ags+dgs)*180/np.pi
                #ratios
                samples_with_skews[:,1::n_param_indv] = np.sqrt(ags*dgs-bgs*cgs)
                #W0s
                samples_with_skews[:,2::n_param_indv] = samples[:,2::n_param_indv]
                #Z0s
                samples_with_skews[:,3::n_param_indv] = samples[:,3::n_param_indv] 
                #on_skews
                samples_with_skews[:,4::n_param_indv] = 0.5*(ags-dgs)
                #off_skews
                samples_with_skews[:,5::n_param_indv] = 0.5*(bgs+cgs)
                            
                np.save(f'{outpath}{image_name}_posterior_transformation_6p_medians.npy',np.median(samples_with_skews,axis=0))
                np.save(f'{outpath}{image_name}_posterior_transformation_6p_covs.npy',np.cov(samples_with_skews,rowvar=False))
                
                sample_pms_parallax_offsets = np.zeros((sample_pms.shape[0],5*sample_pms.shape[1]))
                sample_pms_parallax_offsets[:,0::5] = sample_pms[:,:,0]
                sample_pms_parallax_offsets[:,1::5] = sample_pms[:,:,1]
                sample_pms_parallax_offsets[:,2::5] = sample_parallaxes
                sample_pms_parallax_offsets[:,3::5] = sample_offsets[:,:,0]
                sample_pms_parallax_offsets[:,4::5] = sample_offsets[:,:,1]
                            
                post_pm_parallax_offset_meds = np.median(sample_pms_parallax_offsets,axis=0)
                post_pm_parallax_offset_covs = np.cov(sample_pms_parallax_offsets,rowvar=False)
                np.save(f'{outpath}{image_name}_posterior_PM_parallax_offset_medians.npy',post_pm_parallax_offset_meds)
                np.save(f'{outpath}{image_name}_posterior_PM_parallax_offset_covs.npy',post_pm_parallax_offset_covs)
                np.save(f'{outpath}{image_name}_posterior_Gaia_ids.npy',unique_ids)
                
                np.save(f'{outpath}{image_name}_Gaia_IDs.npy',gaia_id)            
                np.save(f'{outpath}{image_name}_Gaia_RAs_Decs_Gmags.npy',np.array([gaia_ras,gaia_decs,g_mag]).T)   
                np.save(f'{outpath}{image_name}_used_stars.npy',keep_stars)
                print('Done MCMC Fitting. Plotting results:')
                
                best_pm_parallax_offsets = np.zeros((len(x)*gaia_vectors.shape[1]))
                for ind in range(gaia_vectors.shape[1]):
                    best_pm_parallax_offsets[ind::gaia_vectors.shape[1]] = post_pm_parallax_offset_meds[ind::gaia_vectors.shape[1]][unique_inv_inds]
        
#                median_param_errs = np.std(samples,axis=0)
#                best_samp_ind = np.argmin(np.sum(np.power((samples-median_params)/median_param_errs,2),axis=1))
#                best_sample = samples[best_samp_ind] 
                        
                acpt_fracs = np.sum((np.sum(np.abs(samplerChain[:,:-1]-samplerChain[:,1:]),axis=2)>1e-15),axis=0)/samplerChain.shape[0]
                minKeep = burnin
                stats_vals = (acpt_fracs[minKeep:].min(),np.median(acpt_fracs[minKeep:]),np.mean(acpt_fracs[minKeep:]),acpt_fracs[minKeep:].max())
        
                fig = plt.figure(figsize=[12,6])
                gs = gridspec.GridSpec(1,2,width_ratios=[3,1],wspace=0)
                ax0 = plt.subplot(gs[:, 0])    
                plt.plot(np.arange(len(acpt_fracs)),acpt_fracs,lw=1,alpha=1)
                acc_lim = plt.ylim()
                plt.axvline(minKeep,c='r',label=f'Burnin ({burnin} steps)')
                plt.axhline(stats_vals[1],label='Median: %.3f'%stats_vals[1],c='k',ls='--')
                plt.axhline(stats_vals[2],label='Mean: %.3f'%stats_vals[2],c='k',ls='-')
                plt.axhline(stats_vals[0],label='Min: %.3f\nMax: %.3f'%(stats_vals[0],stats_vals[-1]),c='grey')
                plt.axhline(stats_vals[-1],c='grey')
                plt.legend(loc='best')
                ax0.tick_params(axis='both',direction='inout',length=5,bottom=True,left=True,right=True)
                plt.xlabel('Step Number')
                plt.ylabel('Acceptance Fraction')
                ax1 = plt.subplot(gs[:, 1])
                ax1.axis('off')
                plt.hist(acpt_fracs[minKeep:],bins=min(len(acpt_fracs)-minKeep,100),density=True,cumulative=True,histtype='step',lw=3,orientation='horizontal')
                plt.axhline(stats_vals[1],label='Median: %.3f'%stats_vals[1],c='k',ls='--')
                plt.axhline(stats_vals[2],label='Mean: %.3f'%stats_vals[2],c='k',ls='-')
                plt.axhline(stats_vals[0],label='Min: %.3f\nMax: %.3f'%(stats_vals[0],stats_vals[-1]),c='grey')
                plt.axhline(stats_vals[-1],c='grey')
                #plt.legend(loc='best')
                plt.ylim(acc_lim)
                xlim = np.array(plt.xlim());xlim[-1] *= 1.15
                plt.xlim(xlim)
                plt.tight_layout()
                plt.savefig(f'{outpath}{image_name}_MCMC_acceptance_fraction_it%02d.png'%(fit_count))
                plt.close('all')
                # plt.show()
        
                fig = plt.figure(figsize=[12,6])
                #plt.plot(np.arange(len(acpt_fracs))[minKeep:],acpt_fracs[minKeep:],lw=1,alpha=1)
                plt.plot(np.arange(len(stretch_a_vals)),stretch_a_vals,lw=1,alpha=1)
                ax0.tick_params(axis='both',direction='inout',length=5,bottom=True,left=True,right=True)
                plt.xlabel('Step Number')
                plt.ylabel('Stretch Move $a$ value')
                plt.tight_layout()
                plt.savefig(f'{outpath}{image_name}_MCMC_stretch_move_value_it%02d.png'%(fit_count))
                plt.close('all')
                # plt.show()
            
                n_plot_walkers = min(100,nwalkers)
                chosen_walker_inds = np.random.choice(nwalkers,size=n_plot_walkers,replace=False)
    
                ndim_plot = n_param_shared+n_param_indv*n_im_show
                plt.figure(figsize=[13,9/5*ndim_plot])
                for dim in range(ndim_plot):
                    plt.subplot(ndim_plot,1,dim+1)
                    vals = samplerChain[chosen_walker_inds,:,dim].T
                    val_bounds = np.percentile(np.ravel(vals[burnin:]),[16,50,84])
                    val_bounds = np.array([val_bounds[1],val_bounds[1]-val_bounds[0],val_bounds[2]-val_bounds[1]])
                    ylim = (val_bounds[0]-5*val_bounds[1],val_bounds[0]+5*val_bounds[2])        
                    plt.plot(vals,alpha=0.25)
                    if dim != ndim_plot-1:
                        plt.xticks([])
                    else:
                        plt.xticks(np.arange(0, samplerChain.shape[1]+1, samplerChain.shape[1]/10).astype(int))
                    plt.ylabel(dimLabels[dim])
                    plt.axvline(x=burnin,lw=2,ls='--',c='r')
                    plt.ylim(ylim)        
                plt.xlabel('Step Number')
                plt.savefig(f'{outpath}{image_name}_MCMC_walker_traces_it%02d.png'%(fit_count))
                #plt.tight_layout()
                plt.close('all')
                # plt.show()
        
                corner.corner(samples[:,:ndim_plot], 
                              labels=dimLabels[:ndim_plot], 
                              quantiles=[0.16, 0.5, 0.84], show_titles=True,
                              title_kwargs={"fontsize": 12})
                plt.savefig(f'{outpath}{image_name}_MCMC_corner_plot_it%02d.png'%(fit_count))
                plt.close('all')
                # plt.show()
        
                sample_cov = np.cov(samples[:,:ndim_plot],rowvar=False)
#                sample_median = np.median(samples[:,:ndim_plot],axis=0)
        
                n_sim = min(len(samples),1000)
                samp_inds = np.random.choice(len(samples),n_sim,replace=False).astype(int)
                
                new_offset_samps = np.zeros((n_sim,len(x)))
                new_offset_sigma_samps = np.zeros((n_sim,len(x)))
                new_offset_xy_samps = np.zeros((n_sim,len(x),2))
#                new_pm_xy_hst_samps = np.zeros((n_sim,len(x),2))
#                new_parallax_samps = np.zeros((n_sim,len(x)))
        
                pm_data_measures = np.zeros((n_sim,len(x),2))
        
                x0s = np.copy(param_outputs[:,0])
                y0s = np.copy(param_outputs[:,1])
                for i in range(n_sim):
                    sample = np.copy(samples[samp_inds[i]])
                                        
                    ags = sample[0::n_param_indv]
                    bgs = sample[1::n_param_indv]
                    w0s = sample[2::n_param_indv]
                    z0s = sample[3::n_param_indv]
                    cgs = sample[4::n_param_indv]
                    dgs = sample[5::n_param_indv]
                        
                    star_hst_gaia_pos = np.zeros((len(x),2)) #in gaia pixels
                    star_hst_gaia_pos_cov = np.zeros((len(x),2,2)) #in gaia pixels
                    star_ratios = np.zeros(len(x))
                        
                    matrices = np.zeros((len(x),2,2))
                    matrices_T = np.zeros((len(x),2,2))
                    
                    for j in range(len(x0s)):
                        curr_img = np.where(img_nums == j)[0]
                        curr_x,curr_y = x[curr_img],y[curr_img]
                        curr_x_g,curr_y_g = x_g[curr_img],y_g[curr_img]
                        
                        a,b,c,d = ags[j],bgs[j],cgs[j],dgs[j]
    #                    ratio = np.sqrt(a*d-b*c)
    #                    rot = np.arctan2(b-c,a+d)*180/np.pi
    #                    on_skew = 0.5*(a-d)
    #                    off_skew = 0.5*(b+c)
        
                        x0,y0,w0,z0 = x0s[j],y0s[j],w0s[j],z0s[j]
        
                        x_trans = a*(curr_x-x0)+b*(curr_y-y0)+w0
                        y_trans = c*(curr_x-x0)+d*(curr_y-y0)+z0
        
                        dx_trans = curr_x_g-x_trans
                        dy_trans = curr_y_g-y_trans
            
                        star_hst_gaia_pos[curr_img,0] = dx_trans
                        star_hst_gaia_pos[curr_img,1] = dy_trans
        #                dpix_trans = np.sqrt(np.power(dx_trans,2)+np.power(dy_trans,2))
    #                    star_ratios[curr_img] = ratio
                        star_ratios[curr_img] = 1
            
                        matrices[curr_img,0,0] = a
                        matrices[curr_img,0,1] = b
                        matrices[curr_img,1,0] = c
                        matrices[curr_img,1,1] = d
                        matrices_T[curr_img,0,0] = a
                        matrices_T[curr_img,0,1] = c
                        matrices_T[curr_img,1,0] = b
                        matrices_T[curr_img,1,1] = d
                        
    #                    hst_cov_in_gaia = np.dot(matrices,np.dot(hst_cov,matrices_T))
    #                    star_hst_gaia_pos_cov[curr_img] = hst_cov_in_gaia
                    inv_matrices = np.linalg.inv(matrices)
                    
                    star_hst_gaia_pos_cov = np.einsum('nij,njk->nik',matrices,np.einsum('nij,njk->nik',hst_covs,matrices_T))
                    star_ratios = star_ratios[:,None,None]
                    star_hst_gaia_pos_inv_cov = np.linalg.inv(star_hst_gaia_pos_cov)
                                    
                    jac_V_data_inv_jac = np.einsum('nji,njk->nik',proper_offset_jacs,np.einsum('nij,njk->nik',star_hst_gaia_pos_inv_cov,proper_offset_jacs))
                    inv_jac_dot_d_ij = np.einsum('nij,nj->ni',proper_offset_jac_invs,star_hst_gaia_pos)
                    summed_jac_V_data_inv_jac = np.add.reduceat(jac_V_data_inv_jac*use_inds[:,None,None],unique_inds)
                    Sigma_theta_i_inv = unique_gaia_offset_inv_covs+summed_jac_V_data_inv_jac
                    Sigma_theta_i = np.linalg.inv(Sigma_theta_i_inv)
                    
                    jac_V_data_inv_jac_dot_parallax_vects = np.einsum('nij,nj->ni',jac_V_data_inv_jac,parallax_offset_vector)
                    summed_jac_V_data_inv_jac_dot_parallax_vects = np.add.reduceat(jac_V_data_inv_jac_dot_parallax_vects*use_inds[:,None],unique_inds)
                    jac_V_data_inv_jac_dot_d_ij = np.einsum('nij,nj->ni',jac_V_data_inv_jac,inv_jac_dot_d_ij)
                    summed_jac_V_data_inv_jac_dot_d_ij = np.add.reduceat(jac_V_data_inv_jac_dot_d_ij*use_inds[:,None],unique_inds)
                    summed_jac_V_data_inv_jac_times = np.add.reduceat(jac_V_data_inv_jac*delta_times[:,None,None]*use_inds[:,None,None],unique_inds)
                    
                    A_mu_i = np.einsum('nij,njk->nik',Sigma_theta_i,summed_jac_V_data_inv_jac_times)
                    C_mu_ij = delta_time_identities-A_mu_i[unique_inv_inds]
                    A_mu_i_inv = np.linalg.inv(A_mu_i)
                    C_mu_ij_inv = np.linalg.inv(C_mu_ij)
                    
                    Sigma_mu_theta_i_inv = np.einsum('nij,njk->nik',np.einsum('nji,njk->nik',A_mu_i,unique_gaia_offset_inv_covs),A_mu_i)
                    Sigma_mu_d_ij_inv = np.einsum('nij,njk->nik',np.einsum('nji,njk->nik',C_mu_ij,jac_V_data_inv_jac),C_mu_ij)
                    
                    Sigma_mu_i_inv = global_pm_inv_cov+unique_gaia_pm_inv_covs+Sigma_mu_theta_i_inv+\
                                     np.add.reduceat(Sigma_mu_d_ij_inv*use_inds[:,None,None],unique_inds)
                    Sigma_mu_i = np.linalg.inv(Sigma_mu_i_inv)
                    
                    A_plx_mu_i = np.einsum('nij,nj->ni',Sigma_theta_i,-1*summed_jac_V_data_inv_jac_dot_parallax_vects)
                    B_plx_mu_i = np.einsum('nij,nj->ni',Sigma_theta_i,unique_V_theta_i_inv_dot_theta_i\
                                                                        -summed_jac_V_data_inv_jac_dot_d_ij)
                    
                    Sigma_mu_theta_i_inv_dot_A_mu_i_inv = np.einsum('nij,njk->nik',Sigma_mu_theta_i_inv,A_mu_i_inv)
                    Sigma_mu_d_ij_inv_dot_C_mu_ij_inv = np.einsum('nij,njk->nik',Sigma_mu_d_ij_inv,C_mu_ij_inv)
                    
                    C_plx_mu_i = np.einsum('nij,nj->ni',Sigma_mu_i,np.einsum('nij,nj->ni',Sigma_mu_theta_i_inv_dot_A_mu_i_inv,A_plx_mu_i)\
                                                                     -np.add.reduceat(np.einsum('nij,nj->ni',Sigma_mu_d_ij_inv_dot_C_mu_ij_inv,parallax_offset_vector+A_plx_mu_i[unique_inv_inds])*use_inds[:,None],unique_inds))
                    D_plx_mu_i = -1*np.einsum('nij,nj->ni',Sigma_mu_i,unique_V_mu_i_inv_dot_mu_i+V_mu_global_inv_dot_mu_global+\
                                                                        +np.einsum('nij,nj->ni',Sigma_mu_theta_i_inv_dot_A_mu_i_inv,unique_gaia_offsets-B_plx_mu_i)\
                                                                        +np.add.reduceat(np.einsum('nij,nj->ni',Sigma_mu_d_ij_inv_dot_C_mu_ij_inv,inv_jac_dot_d_ij+B_plx_mu_i[unique_inv_inds])*use_inds[:,None],unique_inds))
                                        
                    E_plx_theta_i = np.einsum('nij,nj->ni',A_mu_i,C_plx_mu_i)-A_plx_mu_i
                    F_plx_theta_i = np.einsum('nij,nj->ni',A_mu_i,D_plx_mu_i)-B_plx_mu_i
                    
                    G_plx_d_ij = np.einsum('nij,nj->ni',C_mu_ij,C_plx_mu_i[unique_inv_inds])+A_plx_mu_i[unique_inv_inds]+parallax_offset_vector
                    H_plx_d_ij = np.einsum('nij,nj->ni',C_mu_ij,D_plx_mu_i[unique_inv_inds])+B_plx_mu_i[unique_inv_inds]+inv_jac_dot_d_ij
        
                    G_plx_d_ij_T_dot_V_data_inv = np.einsum('nj,nij->ni',G_plx_d_ij,jac_V_data_inv_jac)  
                    ivar_plx_d_ij = np.einsum('ni,ni->n',G_plx_d_ij_T_dot_V_data_inv,G_plx_d_ij)
                    mu_times_ivar_plx_d_ij = np.einsum('ni,ni->n',G_plx_d_ij_T_dot_V_data_inv,H_plx_d_ij)
    #                mu_plx_d_ij = mu_times_ivar_plx_d_ij/ivar_plx_d_ij
                    summed_ivar_plx_d_ij = np.add.reduceat(ivar_plx_d_ij*use_inds,unique_inds)
                    summed_mu_times_ivar_plx_d_ij = np.add.reduceat(mu_times_ivar_plx_d_ij*use_inds,unique_inds)
                    
                    C_plx_mu_i_T_dot_V_mu_i_inv = np.einsum('nj,nij->ni',C_plx_mu_i,unique_gaia_pm_inv_covs)  
                    ivar_plx_mu_i = np.einsum('ni,ni->n',C_plx_mu_i_T_dot_V_mu_i_inv,C_plx_mu_i)
                    mu_times_ivar_plx_mu_i = np.einsum('ni,ni->n',C_plx_mu_i_T_dot_V_mu_i_inv,D_plx_mu_i+unique_gaia_pms)
    #                mu_plx_mu_i = mu_times_ivar_plx_mu_i/ivar_plx_mu_i
                    
                    C_plx_mu_i_T_dot_V_mu_global_inv = np.einsum('nj,ij->ni',C_plx_mu_i,global_pm_inv_cov)  
                    ivar_plx_mu_global = np.einsum('ni,ni->n',C_plx_mu_i_T_dot_V_mu_global_inv,C_plx_mu_i)
                    mu_times_ivar_plx_mu_global = np.einsum('ni,ni->n',C_plx_mu_i_T_dot_V_mu_global_inv,D_plx_mu_i+global_pm_mean)
    #                mu_plx_mu_global = mu_times_ivar_plx_mu_global/ivar_plx_mu_global
                    
                    E_plx_theta_i_T_dot_V_theta_i_inv = np.einsum('nj,nij->ni',E_plx_theta_i,unique_gaia_offset_inv_covs)  
                    ivar_plx_theta_i = np.einsum('ni,ni->n',E_plx_theta_i_T_dot_V_theta_i_inv,E_plx_theta_i)
                    mu_times_ivar_plx_theta_i = np.einsum('ni,ni->n',E_plx_theta_i_T_dot_V_theta_i_inv,F_plx_theta_i+unique_gaia_offsets)
    #                mu_plx_theta_i = mu_times_ivar_plx_theta_i/ivar_plx_theta_i
                    
                    ivar_plx_i = summed_ivar_plx_d_ij+ivar_plx_mu_i+ivar_plx_mu_global\
                                 +ivar_plx_theta_i+unique_gaia_parallax_ivars+global_parallax_ivar
                    var_plx_i = 1/ivar_plx_i
                    std_plx_i = np.sqrt(var_plx_i)
                    mu_plx_i = (summed_mu_times_ivar_plx_d_ij+mu_times_ivar_plx_mu_i+mu_times_ivar_plx_mu_global\
                                 +mu_times_ivar_plx_theta_i\
                                 +unique_gaia_parallax_ivars*unique_gaia_parallaxes\
                                 +global_parallax_ivar*global_parallax_mean)/ivar_plx_i
                                             
                    parallax_draws = np.random.randn(*std_plx_i.shape)*std_plx_i+mu_plx_i
        #            single_parallax_draws = np.random.randn(*std_plx_i.shape)*std_plx_i+mu_plx_i
        #            parallax_draws[:] = single_parallax_draws
                    
                    B_mu_i = parallax_draws[:,None]*A_plx_mu_i-B_plx_mu_i
    #                D_mu_ij = inv_jac_dot_d_ij-B_mu_i[:,unique_inv_inds]-parallax_draws[:,unique_inv_inds,None]*parallax_offset_vector
        #            mu_mu_theta_i = np.einsum('nij,nj->ni',,)
        #            mu_mu_i = np.einsum('nij,njk->nik',Sigma_mu_i,V_mu_global_inv_dot_mu_global+unique_V_mu_i_inv_dot_mu_i\
        #                                +np.einsum('nij,njk->nik',Sigma_mu_theta_i_inv,mu_mu_theta_i))
                    mu_mu_i = parallax_draws[:,None]*C_plx_mu_i-D_plx_mu_i
        
                    eig_vals,eig_vects = np.linalg.eig(Sigma_mu_i)
                    eig_signs = np.sign(eig_vals)
                    eig_vals *= eig_signs
                    eig_vects[:,:,0] *= eig_signs[:,0][:,None]
                    eig_vects[:,:,1] *= eig_signs[:,1][:,None]
                    pm_gauss_draws = np.random.randn(len(unique_ids),eig_vals.shape[-1])
        #            single_gauss_draws = np.random.randn(len(unique_ids),eig_vals.shape[-1])
        #            pm_gauss_draws[:] = single_gauss_draws
                    pm_draws = pm_gauss_draws*np.sqrt(eig_vals) #pms in x,y HST
                    pm_draws = np.einsum('nij,nj->ni',eig_vects,pm_draws)+mu_mu_i
                    
                    mu_theta_i = np.einsum('nij,nj->ni',A_mu_i,pm_draws)-B_mu_i
                                
                    eig_vals,eig_vects = np.linalg.eig(Sigma_theta_i)
                    eig_signs = np.sign(eig_vals)
                    eig_vals *= eig_signs
                    eig_vects[:,:,0] *= eig_signs[:,0][:,None]
                    eig_vects[:,:,1] *= eig_signs[:,1][:,None]
                    offset_gauss_draws = np.random.randn(len(unique_ids),eig_vals.shape[-1])
        #            single_gauss_draws = np.random.randn(len(unique_ids),eig_vals.shape[-1])
        #            offset_gauss_draws[:] = single_gauss_draws
                    offset_draws = offset_gauss_draws*np.sqrt(eig_vals) #pms in x,y HST
                    offset_draws = np.einsum('nij,nj->ni',eig_vects,offset_draws)+mu_theta_i
                    
    #                data_vector_means = np.einsum('nij,ni->nj',useful_matrix,star_hst_gaia_pos)/(indv_orig_pixel_scales[:,None]*star_ratios[:,0])
                    data_pm_draws = inv_jac_dot_d_ij-(parallax_draws[unique_inv_inds,None]*parallax_offset_vector\
                                                          -offset_draws[unique_inv_inds])
                    eig_vals,eig_vects = np.linalg.eig(np.linalg.inv(jac_V_data_inv_jac))
                    eig_signs = np.sign(eig_vals)
                    eig_vals *= eig_signs
                    eig_vects[:,:,0] *= eig_signs[:,0][:,None]
                    eig_vects[:,:,1] *= eig_signs[:,1][:,None]
                    data_gauss_draws = np.random.randn(len(data_pm_draws),eig_vals.shape[-1])
                    data_draws = data_gauss_draws*np.sqrt(eig_vals) #pms in x,y HST
                    data_draws = np.einsum('nij,nj->ni',eig_vects,data_draws)+data_pm_draws
                    pm_data_measures[i] = data_draws/delta_times[:,None]
                                    
                    data_diff_vals = star_hst_gaia_pos-np.einsum('nij,nj->ni',proper_offset_jacs,delta_times[:,None]*pm_draws[unique_inv_inds]\
                                                                                                   +parallax_draws[unique_inv_inds,None]*parallax_offset_vector\
                                                                                                   -offset_draws[unique_inv_inds])
                    data_diff_vals = np.einsum('nij,nj->ni',inv_matrices,data_diff_vals)
                    data_diff_vals[:,0] += np.random.randn(len(inv_jac_dot_d_ij))*np.sqrt(hst_covs[:,0,0])
                    data_diff_vals[:,1] += np.random.randn(len(inv_jac_dot_d_ij))*np.sqrt(hst_covs[:,1,1])
    #                dpixels = np.zeros((n_pms,len(x),2))
                    dpixels = data_diff_vals
                    
                    # dpixels_sigma_dists = np.sqrt(np.einsum('ni,ni->n',dpixels,np.einsum('nij,nj->ni',star_hst_gaia_pos_inv_cov,dpixels)))
                    dpixels_sigma_dists = np.sqrt(np.einsum('ni,ni->n',dpixels,np.einsum('nij,nj->ni',hst_inv_covs,dpixels)))
                                    
                    new_offset_xy_samps[i,:,0] = dpixels[:,0]
                    new_offset_xy_samps[i,:,1] = dpixels[:,1]
                    new_offset_samps[i,:] = np.sqrt(np.power(dpixels[:,0],2)+np.power(dpixels[:,1],2))
                    
                    new_offset_sigma_samps[i,:] = dpixels_sigma_dists
                        
            #     new_offset_summary = np.percentile(new_offset_samps,[16,50,84],axis=0)
                new_offset_summary = np.percentile(np.sort(new_offset_samps,axis=1),[16,50,84],axis=0)
                
                pm_data_means = np.zeros((len(x),2))
                pm_data_covs = np.zeros((len(x),2,2))
                for star_ind in range(len(unique_stationary)):
                    star_name = unique_ids[star_ind]
                    curr_inds = unique_star_mapping[star_name]
                    
                    for ind in curr_inds:
                        curr_cov = np.cov(pm_data_measures[:,ind],rowvar=False)
                        curr_med = np.median(pm_data_measures[:,ind],axis=0)
                        
                        pm_data_covs[ind] = curr_cov
                        pm_data_means[ind] = curr_med
                        
                dist_from_prior_means = pm_data_means-gaia_pms
                dist_from_prior_covs = pm_data_covs+gaia_pm_covs
                dist_from_prior_inv_covs = np.linalg.inv(dist_from_prior_covs)
                sigma_dist_from_prior = np.sqrt(np.einsum('ni,ni->n',dist_from_prior_means,np.einsum('nij,nj->ni',dist_from_prior_inv_covs,dist_from_prior_means)))
                sigma_dist_from_prior[missing_prior_PM] = 0
                poss_bad_match = (sigma_dist_from_prior > 3)
                print('Found %d possible bad cross-matches between HST and Gaia.'%(np.sum(poss_bad_match)))
        
                plt.figure(figsize=[10,5])
                plt.title(f"Fixing mask group {mask_name} parameters "+"(N$_{images} = %d$, N$_{fixed} = %d$)"%(len(param_outputs),0))
        #         plt.hist(orig_offsets,bins=10000,density=True,cumulative=True,histtype='step',lw=3,label='Pipeline Output')
                plt.hist(new_offset_summary[1],bins=10000,density=True,cumulative=True,histtype='step',label='New Fit',lw=2)
                plt.hist(new_offset_summary[0],bins=10000,density=True,cumulative=True,histtype='step',color='C1')
                plt.hist(new_offset_summary[2],bins=10000,density=True,cumulative=True,histtype='step',color='C1')
        #         plt.hist(median_offset,bins=10000,density=True,cumulative=True,histtype='step',label='Median Params',lw=2)
                xlim = plt.xlim()
                for i in range(min(n_sim,200)):
                    plt.hist(new_offset_samps[i],bins=10000,density=True,cumulative=True,histtype='step',lw=1,alpha=0.1,color='grey',zorder=-1e10)
                plt.xlim(xlim)
                plt.xlabel('Residual Pixel Offset (HST Pixels)');plt.ylabel('CDF')
                plt.legend(loc=6,bbox_to_anchor=(0.3,0.2))
                plt.savefig(f'{outpath}{image_name}_posterior_pixel_offsets_histogram_it%02d.png'%(fit_count))
                plt.close('all')
                # plt.show()
        
                new_offsets_xy_samps_1d = np.zeros((n_sim*len(x),2))
                new_offsets_xy_samps_1d[:,0] = np.ravel(new_offset_xy_samps[:,:,0])
                new_offsets_xy_samps_1d[:,1] = np.ravel(new_offset_xy_samps[:,:,1])
                curr_x_vals = np.ravel(np.median(new_offset_xy_samps[:,:,0],axis=0))
                curr_y_vals = np.ravel(np.median(new_offset_xy_samps[:,:,1],axis=0))
                
                #else, check to see if there are obvious outliers in the pixel offsets,
                #remove them, and then repeat the fitting
                
                xpixel_summary = np.percentile(curr_x_vals[keep_stars],[16,50,84])
                xpixel_summary = np.array([xpixel_summary[1],
                                           xpixel_summary[1]-xpixel_summary[0],
                                           xpixel_summary[2]-xpixel_summary[1]])
                xpixel_med = xpixel_summary[0]
                xpixel_std = max(0.5*(xpixel_summary[1]+xpixel_summary[2]),0.05)
                
                ypixel_summary = np.percentile(curr_y_vals[keep_stars],[16,50,84])
                ypixel_summary = np.array([ypixel_summary[1],
                                           ypixel_summary[1]-ypixel_summary[0],
                                           ypixel_summary[2]-ypixel_summary[1]])
                ypixel_med = ypixel_summary[0]
                ypixel_std = max(0.5*(ypixel_summary[1]+ypixel_summary[2]),0.05)
                
                n_sigma_keep = 2
                #make n_sigma_keep be the as small as 
                diff_x = (curr_x_vals-xpixel_med)
                diff_y = (curr_y_vals-ypixel_med)
                
                # outlier_x = (np.abs(diff_x/xpixel_std) > n_sigma_keep) & (np.abs(diff_x) > hst_pix_sigmas[:,0]*0.5)
                # outlier_y = (np.abs(diff_y/ypixel_std) > n_sigma_keep) & (np.abs(diff_y) > hst_pix_sigmas[:,1]*0.5)
                outlier_x = (np.abs(diff_x/xpixel_std) > n_sigma_keep) & (np.abs(diff_x) > hst_pix_sigmas[:,0]*2.0)
                outlier_y = (np.abs(diff_y/ypixel_std) > n_sigma_keep) & (np.abs(diff_y) > hst_pix_sigmas[:,1]*2.0)
                offset_dists = np.sqrt(np.power(diff_x/hst_pix_sigmas[:,0],2)+np.power(diff_y/hst_pix_sigmas[:,1],2))
                offset_limit = np.nanpercentile(offset_dists[keep_stars],[50,84])
                offset_limit = offset_limit[0]+2*(offset_limit[1]-offset_limit[0])
                
                # print('offset_limit',offset_limit,np.median(offset_dists))
                
                # x_ind = np.argmax(np.abs(diff_x))
                # print(np.abs(diff_x)[x_ind],xpixel_med,xpixel_std,hst_pix_sigmas[x_ind,0]*0.5,np.abs(diff_x/xpixel_std)[x_ind],outlier_x[x_ind])
                # y_ind = np.argmax(np.abs(diff_y))
                # print(np.abs(diff_y)[y_ind],ypixel_med,ypixel_std,hst_pix_sigmas[y_ind,1]*0.5,np.abs(diff_y/ypixel_std)[y_ind],outlier_y[y_ind])
                
                # outliers = (outlier_x | outlier_y) & (~stationary) #keep the stationary targets
                # if np.sum(~outliers) < 10:
                #     outliers = (offset_dists > offset_limit) & (~stationary) 
                
                
                fig = corner.corner(new_offsets_xy_samps_1d, 
                                      labels=[r'$\Delta X$',r'$\Delta Y$'], 
                                      quantiles=[0.16, 0.5, 0.84], show_titles=True,
                                      title_kwargs={"fontsize": 12},bins=30)
                ax = fig.axes[0]
                xlim = ax.get_xlim()
                new_ax = ax.twinx()
        #         new_ax.hist(orig_xy_offsets[:,0],density=True,alpha=0.75,
        #                 range=xlim,bins=10,histtype='step',lw=2,color='C0')
                new_ax.hist(curr_y_vals,density=True,alpha=0.75,
                        range=xlim,bins=10,histtype='step',lw=2,color='C1')
                new_ax.set_yticks([])
                ax = fig.axes[3]
                xlim = ax.get_xlim()
                new_ax = ax.twinx()
        #         new_ax.hist(orig_xy_offsets[:,1],density=True,alpha=0.75,
        #                 range=xlim,bins=10,histtype='step',lw=2,color='C0')
                new_ax.hist(curr_y_vals,density=True,alpha=0.75,
                        range=xlim,bins=10,histtype='step',lw=2,color='C1')
                new_ax.set_yticks([])
                ax = fig.axes[2]
        #         ax.scatter(orig_xy_offsets[:,0],orig_xy_offsets[:,1],s=5,color='C0')
                ax.scatter(curr_x_vals,curr_y_vals,s=2,color='C1')
                plt.savefig(f'{outpath}{image_name}_posterior_pixel_offsets_corner_plot_it%02d.png'%(fit_count))
                plt.close('all')
                # plt.show()
                
                if np.sum(keep_stars) != len(keep_stars):
                
                    new_offsets_xy_samps_1d = np.zeros((n_sim*np.sum(keep_stars),2))
                    new_offsets_xy_samps_1d[:,0] = np.ravel(new_offset_xy_samps[:,keep_stars,0])
                    new_offsets_xy_samps_1d[:,1] = np.ravel(new_offset_xy_samps[:,keep_stars,1])
        
                    fig = corner.corner(new_offsets_xy_samps_1d, 
                                          labels=[r'$\Delta X$',r'$\Delta Y$'], 
                                          quantiles=[0.16, 0.5, 0.84], show_titles=True,
                                          title_kwargs={"fontsize": 12},bins=30,
                                          range=[[-7*median_hst_pix_sigmas[0],7*median_hst_pix_sigmas[0]],\
                                                 [-7*median_hst_pix_sigmas[1],7*median_hst_pix_sigmas[1]]])
                    ax = fig.axes[0]
                    xlim = ax.get_xlim()
                    new_ax = ax.twinx()
            #         new_ax.hist(orig_xy_offsets[:,0],density=True,alpha=0.75,
            #                 range=xlim,bins=10,histtype='step',lw=2,color='C0')
                    new_ax.hist(curr_y_vals[keep_stars],density=True,alpha=0.75,
                            range=xlim,bins=20,histtype='step',lw=2,color='C1')
                    new_ax.set_yticks([])
                    ax = fig.axes[3]
                    xlim = ax.get_xlim()
                    new_ax = ax.twinx()
            #         new_ax.hist(orig_xy_offsets[:,1],density=True,alpha=0.75,
            #                 range=xlim,bins=10,histtype='step',lw=2,color='C0')
                    new_ax.hist(curr_y_vals[keep_stars],density=True,alpha=0.75,
                            range=xlim,bins=20,histtype='step',lw=2,color='C1')
                    new_ax.set_yticks([])
                    ax = fig.axes[2]
            #         ax.scatter(orig_xy_offsets[:,0],orig_xy_offsets[:,1],s=5,color='C0')
                    ax.scatter(curr_x_vals[keep_stars],curr_y_vals[keep_stars],s=5,color='C1')
                    plt.savefig(f'{outpath}{image_name}_posterior_pixel_GOOD_offsets_corner_plot_it%02d.png'%(fit_count))
                plt.close('all')
                # plt.show()
                
                parallax_samps = sample_parallaxes
                pm_x_samps = sample_pms[:,:,0]
                pm_y_samps = sample_pms[:,:,1]
                offset_x_samps = sample_offsets[:,:,0]
                offset_y_samps = sample_offsets[:,:,1]
                
                posterior_pm_covs = np.zeros((len(pm_x_samps[0]),2,2))
                posterior_pm_meds = np.zeros((len(pm_x_samps[0]),2))
                posterior_offset_covs = np.zeros((len(pm_x_samps[0]),2,2))
                posterior_offset_meds = np.zeros((len(pm_x_samps[0]),2))
                posterior_parallax_errs = np.zeros((len(pm_x_samps[0]),2))
                posterior_parallax_meds = np.zeros((len(pm_x_samps[0])))
                
        #        for star_ind in range(len(unique_ids)):
                for star_ind in range(len(pm_x_samps[0])):
                    pm_vals = np.array([pm_x_samps[:,star_ind],
                                        pm_y_samps[:,star_ind]]).T
                    curr_cov = np.cov(pm_vals,rowvar=False) 
                    curr_diff = np.median(pm_vals,axis=0)
                    
                    posterior_pm_covs[star_ind] = curr_cov
                    posterior_pm_meds[star_ind] = curr_diff
                    
                    offset_vals = np.array([offset_x_samps[:,star_ind],
                                            offset_y_samps[:,star_ind]]).T
                    curr_cov = np.cov(offset_vals,rowvar=False) 
                    curr_diff = np.median(offset_vals,axis=0)
                    
                    posterior_offset_covs[star_ind] = curr_cov
                    posterior_offset_meds[star_ind] = curr_diff
        
                    posterior_parallax_meds[star_ind] = np.median(parallax_samps[:,star_ind])
                    posterior_parallax_errs[star_ind] = np.abs(np.percentile(parallax_samps[:,star_ind],[16,84])-posterior_parallax_meds[star_ind])
    
                err_vectors = np.zeros((len(pm_x_samps[0]),2,2))
                
                plt.figure(figsize=(3*5*(1.3**2),5))
                gs = gridspec.GridSpec(1,3,wspace=0.3)
                ax = plt.subplot(gs[:,0])    
    #            ax = plt.gca()
    #            ax.set_aspect('equal')
                plt.grid(visible=True, which='major', color='#666666', linestyle='-',alpha=0.3)
                plt.minorticks_on()
                plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
                ax.tick_params(axis='both',direction='inout',length=5,bottom=True,left=True,right=True,top=True)
                err_lengths = np.zeros((len(pm_x_samps[0]),2))
                for star_ind in range(len(pm_x_samps[0])):
                    curr_med = posterior_pm_meds[star_ind]      
                    if unique_keep[star_ind]:
                        color = 'C0'
                    else:
                        color = 'C1'
                    zorder = -1e5
                    plt.scatter(curr_med[0],curr_med[1],edgecolor=color,facecolor='None',alpha=0.7,zorder=zorder)
                    
                    if unique_missing_prior_PM[star_ind]:
                        plt.scatter(curr_med[0],curr_med[1],edgecolor='r',facecolor='None',alpha=0.7,zorder=-1e10,s=200)
                    else:
                        prior_med = gaia_pms[unique_inds[star_ind]]
                        plt.scatter(prior_med[0],prior_med[1],color='k',alpha=0.7,zorder=-1e20,s=10)
                        line = [prior_med[0],curr_med[0]],[prior_med[1],curr_med[1]]
                        plt.plot(line[0],line[1],color='grey',ls=':',lw=1,zorder=-1e20)
                        
                xlim,ylim = plt.xlim(),plt.ylim()
                for star_ind in range(len(pm_x_samps[0])):
                    curr_cov = posterior_pm_covs[star_ind]
                    curr_med = posterior_pm_meds[star_ind]
                    
                    curr_vals,curr_vects = np.linalg.eig(curr_cov)
                    curr_vals = np.sqrt(curr_vals)
                
                    err_vects = np.zeros_like(curr_vects)
                    err_vects[0] = curr_vals[0]*curr_vects[:,0]
                    err_vects[1] = curr_vals[1]*curr_vects[:,1]    
                    err_vectors[star_ind] = err_vects
                    
                    err_lengths[star_ind] = np.sqrt(np.sum(np.power(err_vects[0],2))),np.sqrt(np.sum(np.power(err_vects[1],2)))
                    
                    err1_plot = [curr_med[0]-err_vects[0,0],curr_med[0]+err_vects[0,0]],\
                                [curr_med[1]-err_vects[0,1],curr_med[1]+err_vects[0,1]]
                    err2_plot = [curr_med[0]-err_vects[1,0],curr_med[0]+err_vects[1,0]],\
                                [curr_med[1]-err_vects[1,1],curr_med[1]+err_vects[1,1]]
                    
                    if unique_keep[star_ind]:
                        color = 'C0'
                    else:
                        color = 'C1'
                    zorder = -1e5
                    plt.plot(err1_plot[0],err1_plot[1],color=color,lw=1,alpha=0.7,zorder=zorder-2)
                    plt.plot(err2_plot[0],err2_plot[1],color=color,lw=1,alpha=0.7,zorder=zorder-1)
                    if ~unique_missing_prior_PM[star_ind]:
                        curr_cov = gaia_pm_covs[unique_inds[star_ind]]
                        curr_med = gaia_pms[unique_inds[star_ind]]
                        
                        curr_vals,curr_vects = np.linalg.eig(curr_cov)
                        curr_vals = np.sqrt(curr_vals)
                    
                        err_vects = np.zeros_like(curr_vects)
                        err_vects[0] = curr_vals[0]*curr_vects[:,0]
                        err_vects[1] = curr_vals[1]*curr_vects[:,1]    
                                            
                        err1_plot = [curr_med[0]-err_vects[0,0],curr_med[0]+err_vects[0,0]],\
                                    [curr_med[1]-err_vects[0,1],curr_med[1]+err_vects[0,1]]
                        err2_plot = [curr_med[0]-err_vects[1,0],curr_med[0]+err_vects[1,0]],\
                                    [curr_med[1]-err_vects[1,1],curr_med[1]+err_vects[1,1]]
                        color = 'k'
                        plt.plot(err1_plot[0],err1_plot[1],color=color,lw=1,alpha=0.1,zorder=-1e30)
                        plt.plot(err2_plot[0],err2_plot[1],color=color,lw=1,alpha=0.1,zorder=-1e30)
                plt.xlim(xlim);plt.ylim(ylim)
                plt.xlabel(r'$\mu_{\mathrm{RA}}$ (mas/yr)')
                plt.ylabel(r'$\mu_{\mathrm{Dec}}$ (mas/yr)')
                plt.axhline(0,c='k',ls='--',lw=0.5)
                plt.axvline(0,c='k',ls='--',lw=0.5)
    #            plt.show()
                
    #                plt.figure(figsize=(6,6))
    #            ax = plt.gca()
    #            ax.set_aspect('equal')
                ax = plt.subplot(gs[:,1])    
                plt.grid(visible=True, which='major', color='#666666', linestyle='-',alpha=0.3)
                plt.minorticks_on()
                plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
                ax.tick_params(axis='both',direction='inout',length=5,bottom=True,left=True,right=True,top=True)
                y_vals = posterior_pm_meds[~missing_prior_PM[unique_inds],0]
                y_errs = np.sqrt(posterior_pm_covs[~missing_prior_PM[unique_inds],0,0])
                x_vals = gaia_pms[unique_inds][~missing_prior_PM[unique_inds]][:,0]
                x_errs = np.sqrt(gaia_pm_covs[unique_inds][~missing_prior_PM[unique_inds]][:,0,0])
                curr_cond = (~unique_keep)[~missing_prior_PM[unique_inds]]
                if np.sum(curr_cond) > 0:
                    color = 'C1'
                    plt.errorbar(x_vals[curr_cond],y_vals[curr_cond],
                                 xerr=x_errs[curr_cond],yerr=y_errs[curr_cond],fmt='o',color=color,alpha=0.5,ms=1)
                curr_cond = unique_keep[~missing_prior_PM[unique_inds]]
                if np.sum(curr_cond) > 0:
                    color = 'C0'
                    plt.errorbar(x_vals[curr_cond],y_vals[curr_cond],
                                 xerr=x_errs[curr_cond],yerr=y_errs[curr_cond],fmt='o',color=color,alpha=0.5,ms=1)
                xlim,ylim = plt.xlim(),plt.ylim()
                plt.plot(xlim,xlim,color='k',zorder=1e10,lw=1,ls='--')
                plt.xlim(xlim);plt.ylim(ylim)
                plt.xlabel('$\mu_{\mathrm{RA,Gaia}}$ (mas/yr)')
                plt.ylabel('$\mu_{\mathrm{RA,KM}}$ (mas/yr)')
    #            plt.show()
                
    #            plt.figure(figsize=(6,6))
    #            ax = plt.gca()
    #            ax.set_aspect('equal')
                ax = plt.subplot(gs[:,2])    
                plt.grid(visible=True, which='major', color='#666666', linestyle='-',alpha=0.3)
                plt.minorticks_on()
                plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
                ax.tick_params(axis='both',direction='inout',length=5,bottom=True,left=True,right=True,top=True)
                y_vals = posterior_pm_meds[~missing_prior_PM[unique_inds],1]
                y_errs = np.sqrt(posterior_pm_covs[~missing_prior_PM[unique_inds],1,1])
                x_vals = gaia_pms[unique_inds][~missing_prior_PM[unique_inds]][:,1]
                x_errs = np.sqrt(gaia_pm_covs[unique_inds][~missing_prior_PM[unique_inds]][:,1,1])
                curr_cond = (~unique_keep)[~missing_prior_PM[unique_inds]]
                if np.sum(curr_cond) > 0:
                    color = 'C1'
                    plt.errorbar(x_vals[curr_cond],y_vals[curr_cond],
                                 xerr=x_errs[curr_cond],yerr=y_errs[curr_cond],fmt='o',color=color,alpha=0.5,ms=1)
                curr_cond = unique_keep[~missing_prior_PM[unique_inds]]
                if np.sum(curr_cond) > 0:
                    color = 'C0'
                    plt.errorbar(x_vals[curr_cond],y_vals[curr_cond],
                                 xerr=x_errs[curr_cond],yerr=y_errs[curr_cond],fmt='o',color=color,alpha=0.5,ms=1)
                xlim,ylim = plt.xlim(),plt.ylim()
                plt.plot(xlim,xlim,color='k',zorder=1e10,lw=1,ls='--')
                plt.xlim(xlim);plt.ylim(ylim)
                plt.xlabel('$\mu_{\mathrm{Dec,Gaia}}$ (mas/yr)')
                plt.ylabel('$\mu_{\mathrm{Dec,KM}}$ (mas/yr)')
    #            plt.tight_layout()
                plt.savefig(f'{outpath}{image_name}_posterior_VS_prior_PMs_it%02d.png'%(fit_count))
                plt.close('all')
                # plt.show()
                
                if plot_indv_star_pms:
                
                    indv_star_path = f'{path}{field}/Bayesian_PMs/{image_name}/indv_stars/'
                    if not os.path.isdir(indv_star_path):
                        os.makedirs(indv_star_path)
                        
                    print(f'Plotting comparison of data and prior PMs for each star.')
                    
                    for star_ind in range(len(pm_x_samps[0])):
                        star_name = unique_ids[star_ind]
                        curr_inds = unique_star_mapping[star_name]
                        
                        plt.figure(figsize=(6,6))
                        ax = plt.gca()
                        ax.set_aspect('equal')
                        plt.grid(visible=True, which='major', color='#666666', linestyle='-',alpha=0.3)
                        plt.minorticks_on()
                        plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
                        ax.tick_params(axis='both',direction='inout',length=5,bottom=True,left=True,right=True,top=True)
                        plt.title(f'{star_name}\n G = {round(g_mag[unique_inds[star_ind]],1)} mag, Gaia Prior = {~unique_missing_prior_PM[star_ind]}\nUsed in Fit = {unique_keep[star_ind]}\n({field}, '+\
                                  r'$N_{\mathrm{im}}$ = %d)'%(len(curr_inds)))
                        plt.xlabel('$\mu_{\mathrm{RA}}$ (mas/yr)')
                        plt.ylabel('$\mu_{\mathrm{Dec}}$ (mas/yr)')
                        
                        #plot the posterior PM 
                        curr_cov = posterior_pm_covs[star_ind]
                        curr_med = posterior_pm_meds[star_ind]
                        curr_vals,curr_vects = np.linalg.eig(curr_cov)
                        curr_vals = np.sqrt(curr_vals)
                        err_vects = np.zeros_like(curr_vects)
                        err_vects[0] = curr_vals[0]*curr_vects[:,0]
                        err_vects[1] = curr_vals[1]*curr_vects[:,1]    
                        err1_plot = [curr_med[0]-err_vects[0,0],curr_med[0]+err_vects[0,0]],\
                                    [curr_med[1]-err_vects[0,1],curr_med[1]+err_vects[0,1]]
                        err2_plot = [curr_med[0]-err_vects[1,0],curr_med[0]+err_vects[1,0]],\
                                    [curr_med[1]-err_vects[1,1],curr_med[1]+err_vects[1,1]]
                        color = 'r'
                        zorder = 1e10
                        plt.scatter(curr_med[0],curr_med[1],edgecolor=color,facecolor='None',alpha=0.7,zorder=zorder)
                        plt.plot(err1_plot[0],err1_plot[1],color=color,lw=1,alpha=0.7,zorder=zorder-2)
                        plt.plot(err2_plot[0],err2_plot[1],color=color,lw=1,alpha=0.7,zorder=zorder-1)
                                        
                        #plot the Gaia prior PMs if they exist
                        if not unique_missing_prior_PM[star_ind]:
                            curr_cov = gaia_vector_covs[unique_inds[star_ind]][2:4,2:4]
                            curr_med = gaia_vectors[unique_inds[star_ind]][2:4]
                            curr_vals,curr_vects = np.linalg.eig(curr_cov)
                            curr_vals = np.sqrt(curr_vals)
                            err_vects = np.zeros_like(curr_vects)
                            err_vects[0] = curr_vals[0]*curr_vects[:,0]
                            err_vects[1] = curr_vals[1]*curr_vects[:,1]    
                            err1_plot = [curr_med[0]-err_vects[0,0],curr_med[0]+err_vects[0,0]],\
                                        [curr_med[1]-err_vects[0,1],curr_med[1]+err_vects[0,1]]
                            err2_plot = [curr_med[0]-err_vects[1,0],curr_med[0]+err_vects[1,0]],\
                                        [curr_med[1]-err_vects[1,1],curr_med[1]+err_vects[1,1]]
                            color = 'k'
                            zorder = -1e5
                            plt.scatter(curr_med[0],curr_med[1],edgecolor=color,facecolor='None',alpha=0.7,zorder=zorder)
                            plt.plot(err1_plot[0],err1_plot[1],color=color,lw=1,alpha=0.7,zorder=zorder-2)
                            plt.plot(err2_plot[0],err2_plot[1],color=color,lw=1,alpha=0.7,zorder=zorder-1)
                            
                        #plot the data-measured PMs
                        for ind in curr_inds:
                            color = 'C0'
                            zorder = 1e5
                            curr_med = np.median(pm_data_measures[:,ind],axis=0)
                            plt.scatter(curr_med[0],curr_med[1],edgecolor=color,facecolor='None',alpha=0.7,zorder=zorder)
                        
                        #plot the Global PM prior
                        curr_med = global_vector_mean[2:4]
                        color = 'C1'
                        zorder = -1e10
                        plt.scatter(curr_med[0],curr_med[1],edgecolor=color,facecolor='None',alpha=0.7,zorder=zorder)
                            
                        xlim = plt.xlim()
                        ylim = plt.ylim()
                        
                        #plot the data-measured PMs
                        for ind in curr_inds:
                            color = 'C0'
                            zorder = 1e5
                            
                            curr_cov = np.cov(pm_data_measures[:,ind],rowvar=False)
                            curr_med = np.median(pm_data_measures[:,ind],axis=0)
                            curr_vals,curr_vects = np.linalg.eig(curr_cov)
                            curr_vals = np.sqrt(curr_vals)
                            err_vects = np.zeros_like(curr_vects)
                            err_vects[0] = curr_vals[0]*curr_vects[:,0]
                            err_vects[1] = curr_vals[1]*curr_vects[:,1]    
                            err1_plot = [curr_med[0]-err_vects[0,0],curr_med[0]+err_vects[0,0]],\
                                        [curr_med[1]-err_vects[0,1],curr_med[1]+err_vects[0,1]]
                            err2_plot = [curr_med[0]-err_vects[1,0],curr_med[0]+err_vects[1,0]],\
                                        [curr_med[1]-err_vects[1,1],curr_med[1]+err_vects[1,1]]
                            plt.plot(err1_plot[0],err1_plot[1],color=color,lw=1,alpha=0.7,zorder=zorder-2)
                            plt.plot(err2_plot[0],err2_plot[1],color=color,lw=1,alpha=0.7,zorder=zorder-1)
                        
                        #plot the Global PM prior
                        curr_cov = global_vector_cov[2:4,2:4]
                        curr_med = global_vector_mean[2:4]
                        curr_vals,curr_vects = np.linalg.eig(curr_cov)
                        curr_vals = np.sqrt(curr_vals)
                        err_vects = np.zeros_like(curr_vects)
                        err_vects[0] = curr_vals[0]*curr_vects[:,0]
                        err_vects[1] = curr_vals[1]*curr_vects[:,1]    
                        err1_plot = [curr_med[0]-err_vects[0,0],curr_med[0]+err_vects[0,0]],\
                                    [curr_med[1]-err_vects[0,1],curr_med[1]+err_vects[0,1]]
                        err2_plot = [curr_med[0]-err_vects[1,0],curr_med[0]+err_vects[1,0]],\
                                    [curr_med[1]-err_vects[1,1],curr_med[1]+err_vects[1,1]]
                        color = 'C1'
                        zorder = -1e10
                        plt.plot(err1_plot[0],err1_plot[1],color=color,lw=1,alpha=0.7,zorder=zorder-2)
                        plt.plot(err2_plot[0],err2_plot[1],color=color,lw=1,alpha=0.7,zorder=zorder-1)
                        
                        plt.xlim(xlim)
                        plt.ylim(ylim)
                        plt.savefig(f'{indv_star_path}{image_name}_{star_name}_posterior_PM_comparison_it%02d.png'%(fit_count),bbox_inches='tight')
                        # plt.show()
                        plt.close('all')
                        # if star_ind > 25:
                        #     break
                
                outliers = (~use_for_fit) 
                if np.sum(~((poss_bad_match) | outliers)) > 5:
                    outliers = poss_bad_match | outliers
#                if np.sum(~((offset_dists > offset_limit) | outliers)) > 5:
#                    outliers = (offset_dists > offset_limit) | outliers
                if np.sum(~((outlier_x | outlier_y) | outliers)) > 5:
                    outliers = (outlier_x | outlier_y) | outliers
                
                # print(outliers[x_ind])
                if mask_outlier_pms:
                    outliers = outliers | outlier_pms
                outliers = outliers & (~stationary) #keep the stationary targets
                    
    #            if np.sum(~missing_prior_PM) >= 10:
    #                outliers = (outliers | missing_prior_PM)
                outlier_inds = np.where(outliers)[0]
    #            keep_stars = ~outliers
                
                all_outlier_inds = np.where(all_outliers)[0]
                # if fit_count > 0:
                print(f'There are {len(all_outlier_inds)} previously identified outlier indices:',all_outlier_inds)
                print(f'There are {len(outlier_inds)} currently identified outlier indices: ',outlier_inds)                
                
                keep_stars = ~outliers
                
        #        break
                end_time = time.time()
                
                if (not redo_without_outliers):
#                    stop_fitting = True
                    print(f'Iteration {fit_count} took {round(end_time-start_time,2)} seconds.')
                    break
                
                print(f'Iteration {fit_count} took {round(end_time-start_time,2)} seconds.')
                
                force_repeat = False
                if (np.sum(missing_prior_PM) > 0) and (fit_count == 0):
                    #then repeat at least one iteration to have better
                    #prior PMs on the stars that are missing Gaia priors
                    force_repeat = True
                    pass
                # elif (np.sum(missing_prior_PM)/len(missing_prior_PM) > 1.0/3) and (fit_count == 1) and (len(unique_missing_prior_PM) < 20):
                #     #then repeat at least one iteration to have better
                #     #prior PMs on the stars that are missing Gaia priors
                #     force_repeat = True
                #     pass
                elif np.all(outliers == all_outliers) or (len(outlier_inds) == 0):
                    #then no stars (or fewer stars) were removed, so don't repeat the fit
                    print(f'Found no new outliers outside {n_sigma_keep} sigma, so not repeating the fit. Stopped after {fit_count+1} fitting iterations.')
#                    stop_fitting = True
                    break
                elif len(outlier_inds) < len(all_outlier_inds):
                    #common case is that some measures are removed from the outlier list, but
                    #we probably don't need to repeat the fit because it is costly
                    
                    #check if outliers is a subset of all_outliers
                    if np.all(outliers == (outliers & all_outliers)):
                        n_dropped_outliers = len(all_outlier_inds)-len(outlier_inds)
                        #check that the number of removed outliers is small
#                        if (n_dropped_outliers <= 5) and (n_dropped_outliers/np.sum(keep_stars) < 0.1):
                        if (n_dropped_outliers/np.sum(keep_stars) < 0.1):
                            print(f'Current outliers are a smaller subset of the previous outliers (dropped {n_dropped_outliers} measures), so not repeating the fit. Stopped after {fit_count+1} fitting iterations.')
#                            stop_fitting = True
                            break
                    
                fit_count += 1 #increment counter
    #            all_outliers[np.where(~all_outliers)[0][outliers]] = True
    #            keep_stars = ~all_outliers
                all_outliers = np.copy(outliers)
                            
                if fit_count >= n_fit_max:
                    print(f'Stopping fitting iterations because we have reached the maximum of {n_fit_max}, though there are still currently {np.sum(outliers)} outliers of the {len(outliers)} targets outside {n_sigma_keep} sigma.')
#                    stop_fitting = True
                    break
                
                if force_repeat:
                    print(f'Repeating the fit to get better priors for the {np.sum(unique_missing_prior_PM)} target(s) without Gaia priors.')
                else:
                    print(f'There are {np.sum(outliers)} outliers of the {len(outliers)} measures outside {n_sigma_keep} sigma, so repeating the fit.')
                print()
    
    #        break
            if skip_fitting:
                continue
            
            print()
            total_fit_end = time.time()
            print(f'Done fitting. Total process took {round(total_fit_end-total_fit_start,2)} seconds.')
            best_sample_lnprob = sample_lnposts.max()
            print('Best log posterior:',best_sample_lnprob)
            print()

    #        parallax_samps = sample_parallaxes[:,unique_inds]
    #        pm_x_samps = sample_pms[:,unique_inds,0]
    #        pm_y_samps = sample_pms[:,unique_inds,1]
    #        offset_x_samps = sample_offsets[:,unique_inds,0]
    #        offset_y_samps = sample_offsets[:,unique_inds,1]
    #        
    #        posterior_pm_covs = np.zeros((len(unique_ids),2,2))
    #        posterior_pm_meds = np.zeros((len(unique_ids),2))
    #        posterior_offset_covs = np.zeros((len(unique_ids),2,2))
    #        posterior_offset_meds = np.zeros((len(unique_ids),2))
    #        posterior_parallax_errs = np.zeros((len(unique_ids),2))
    #        posterior_parallax_meds = np.zeros((len(unique_ids)))
            
    #        parallax_samps = sample_parallaxes
    #        pm_x_samps = sample_pms[:,:,0]
    #        pm_y_samps = sample_pms[:,:,1]
    #        offset_x_samps = sample_offsets[:,:,0]
    #        offset_y_samps = sample_offsets[:,:,1]
            
            posterior_pm_covs = np.zeros((len(pm_x_samps[0]),2,2))
            posterior_pm_meds = np.zeros((len(pm_x_samps[0]),2))
            posterior_offset_covs = np.zeros((len(pm_x_samps[0]),2,2))
            posterior_offset_meds = np.zeros((len(pm_x_samps[0]),2))
            posterior_parallax_errs = np.zeros((len(pm_x_samps[0]),2))
            posterior_parallax_meds = np.zeros((len(pm_x_samps[0])))
            
    #        for star_ind in range(len(unique_ids)):
            for star_ind in range(len(pm_x_samps[0])):
                pm_vals = np.array([pm_x_samps[:,star_ind],
                                    pm_y_samps[:,star_ind]]).T
                curr_cov = np.cov(pm_vals,rowvar=False) 
                curr_diff = np.median(pm_vals,axis=0)
                
                posterior_pm_covs[star_ind] = curr_cov
                posterior_pm_meds[star_ind] = curr_diff
                
                offset_vals = np.array([offset_x_samps[:,star_ind],
                                        offset_y_samps[:,star_ind]]).T
                curr_cov = np.cov(offset_vals,rowvar=False) 
                curr_diff = np.median(offset_vals,axis=0)
                
                posterior_offset_covs[star_ind] = curr_cov
                posterior_offset_meds[star_ind] = curr_diff
    
                posterior_parallax_meds[star_ind] = np.median(parallax_samps[:,star_ind])
                posterior_parallax_errs[star_ind] = np.abs(np.percentile(parallax_samps[:,star_ind],[16,84])-posterior_parallax_meds[star_ind])
                
            posterior_pm_errs = np.zeros((len(posterior_pm_covs),2))
            posterior_pm_errs[:,0] = np.sqrt(posterior_pm_covs[:,0,0])
            posterior_pm_errs[:,1] = np.sqrt(posterior_pm_covs[:,1,1])
            
            if np.sum((~unique_missing_prior_PM) & unique_not_stationary) > 3:
                curr_keep = unique_inds[(~unique_missing_prior_PM) & unique_not_stationary]
            else:
                curr_keep = unique_inds
            ave_ivars = np.power(np.array([gaia_pm_covs[:,0,0],gaia_pm_covs[:,1,1]]).T,-1)[curr_keep]
            ave_offsets = np.sum(ave_ivars*gaia_pms[curr_keep],axis=0)/np.sum(ave_ivars,axis=0)
            print('Average Gaia PMs:',ave_offsets)
            curr_keep = unique_inds
            ave_offsets = np.nanmean(gaiahub_pms[curr_keep],axis=0)
            ave_ivars = np.power(np.array([gaiahub_pm_x_errs,gaiahub_pm_y_errs]).T[curr_keep],-2)
            ave_offsets = np.nansum(ave_ivars*gaiahub_pms[curr_keep],axis=0)/np.nansum(ave_ivars,axis=0)
            print('Average GaiaHub PMs:',ave_offsets)
            ave_offsets = np.mean(posterior_pm_meds,axis=0)
            ave_ivars = np.power(posterior_pm_errs,-2)
            ave_offsets = np.sum(ave_ivars*posterior_pm_meds,axis=0)/np.sum(ave_ivars,axis=0)
            print('Average posterior PMs:',ave_offsets)
            
#            if np.sum((~unique_missing_prior_PM) & unique_not_stationary) > 3:
#                curr_keep = unique_inds[(~unique_missing_prior_PM) & unique_not_stationary]
#            else:
#                curr_keep = unique_inds[unique_not_stationary]
#            gaia_icovs = np.linalg.inv(gaia_pm_covs[curr_keep])
#            gaia_summed_icov = np.sum(gaia_icovs,axis=0)
#            gaia_ave_pm_cov = np.linalg.inv(gaia_summed_icov)
#            gaia_ave_pm = np.einsum('ij,j->i',gaia_ave_pm_cov,np.sum(np.einsum('nij,nj->ni',gaia_icovs,gaia_pms[curr_keep]),axis=0))
    
#            curr_keep = unique_inds[unique_not_stationary]
#            gaiahub_icovs = np.linalg.inv(gaiahub_pm_covs[curr_keep])
#            gaiahub_summed_icov = np.sum(gaiahub_icovs,axis=0)
#            gaiahub_ave_pm_cov = np.linalg.inv(gaiahub_summed_icov)
#            gaiahub_ave_pm = np.einsum('ij,j->i',gaiahub_ave_pm_cov,np.sum(np.einsum('nij,nj->ni',gaiahub_icovs,gaiahub_pms[curr_keep]),axis=0))
    
#            curr_keep = unique_not_stationary
#            posterior_icovs = np.linalg.inv(posterior_pm_covs[curr_keep])
#            posterior_summed_icov = np.sum(posterior_icovs,axis=0)
#            posterior_ave_pm_cov = np.linalg.inv(posterior_summed_icov)
#            posterior_ave_pm = np.einsum('ij,j->i',posterior_ave_pm_cov,np.sum(np.einsum('nij,nj->ni',posterior_icovs,posterior_pm_meds[curr_keep]),axis=0))
                                    
            err_vectors = np.zeros((len(pm_x_samps[0]),2,2))
            
            plt.figure(figsize=(6,6))
            plt.gca().set_aspect('equal')
            err_lengths = np.zeros((len(pm_x_samps[0]),2))
            for star_ind in range(len(pm_x_samps[0])):
                curr_cov = posterior_pm_covs[star_ind]
                curr_med = posterior_pm_meds[star_ind]
                
                curr_vals,curr_vects = np.linalg.eig(curr_cov)
                curr_vals = np.sqrt(curr_vals)
            
                err_vects = np.zeros_like(curr_vects)
                err_vects[0] = curr_vals[0]*curr_vects[:,0]
                err_vects[1] = curr_vals[1]*curr_vects[:,1]    
                err_vectors[star_ind] = err_vects
                
                err_lengths[star_ind] = np.sqrt(np.sum(np.power(err_vects[0],2))),np.sqrt(np.sum(np.power(err_vects[1],2)))
                
                err1_plot = [curr_med[0]-err_vects[0,0],curr_med[0]+err_vects[0,0]],                [curr_med[1]-err_vects[0,1],curr_med[1]+err_vects[0,1]]
                err2_plot = [curr_med[0]-err_vects[1,0],curr_med[0]+err_vects[1,0]],                [curr_med[1]-err_vects[1,1],curr_med[1]+err_vects[1,1]]
                
                if stationary[star_ind]:
                    color = 'C1'
                    zorder = 1e10
                else:
                    color = 'C0'
                    zorder = -1e5
                plt.scatter(curr_med[0],curr_med[1],edgecolor=color,facecolor='None',alpha=0.7,zorder=zorder)
                plt.plot(err1_plot[0],err1_plot[1],color=color,lw=1,alpha=0.7,zorder=zorder-2)
                plt.plot(err2_plot[0],err2_plot[1],color=color,lw=1,alpha=0.7,zorder=zorder-1)
                
                if unique_missing_prior_PM[star_ind]:
                    plt.scatter(curr_med[0],curr_med[1],edgecolor='r',facecolor='None',alpha=0.7,zorder=-1e10,s=200)
            
            plt.xlabel(r'$\mu_{\mathrm{RA}}$ (mas/yr)')
            plt.ylabel(r'$\mu_{\mathrm{Dec}}$ (mas/yr)')
            plt.axhline(0,c='k',ls='--',lw=0.5)
            plt.axvline(0,c='k',ls='--',lw=0.5)
            plt.close('all')
            # plt.show()
            
            if np.sum(stationary) > 0:
                plt.figure(figsize=(6,6))
                plt.gca().set_aspect('equal')
                for star_ind in range(len(pm_x_samps[0])):
                    if not unique_stationary[star_ind]:
                        continue
                    curr_cov = posterior_pm_covs[star_ind]
                    curr_med = posterior_pm_meds[star_ind]
            
                    curr_vals,curr_vects = np.linalg.eig(curr_cov)
                    curr_vals = np.sqrt(curr_vals)
            
                    err_vects = np.zeros_like(curr_vects)
                    err_vects[0] = curr_vals[0]*curr_vects[:,0]
                    err_vects[1] = curr_vals[1]*curr_vects[:,1]    
                    
            
                    err1_plot = [curr_med[0]-err_vects[0,0],curr_med[0]+err_vects[0,0]],                    [curr_med[1]-err_vects[0,1],curr_med[1]+err_vects[0,1]]
                    err2_plot = [curr_med[0]-err_vects[1,0],curr_med[0]+err_vects[1,0]],                    [curr_med[1]-err_vects[1,1],curr_med[1]+err_vects[1,1]]
            
                    if stationary[star_ind]:
                        color = 'C1'
                        zorder = 1e10
                    else:
                        color = 'C0'
                        zorder = -1e5
                    plt.scatter(curr_med[0],curr_med[1],edgecolor=color,facecolor='None',alpha=0.7,zorder=zorder)
                    plt.plot(err1_plot[0],err1_plot[1],color=color,lw=1,alpha=0.7,zorder=zorder-2)
                    plt.plot(err2_plot[0],err2_plot[1],color=color,lw=1,alpha=0.7,zorder=zorder-1)
            
                plt.xlabel(r'$\mu_{\mathrm{RA}}$ (mas/yr)')
                plt.ylabel(r'$\mu_{\mathrm{Dec}}$ (mas/yr)')
                plt.axhline(0,c='k',ls='--',lw=0.5)
                plt.axvline(0,c='k',ls='--',lw=0.5)
                plt.close('all')
                # plt.show()
                        
            gaia_pm_summary = np.percentile(gaia_pms[unique_inds],[16,50,84],axis=0)
            gaia_pm_summary = np.array([gaia_pm_summary[1],
                                       gaia_pm_summary[1]-gaia_pm_summary[0],
                                       gaia_pm_summary[2]-gaia_pm_summary[1]]).T
            gaiahub_pm_summary = np.percentile(gaiahub_pms[unique_inds],[16,50,84],axis=0)
            gaiahub_pm_summary = np.array([gaiahub_pm_summary[1],
                                           gaiahub_pm_summary[1]-gaiahub_pm_summary[0],
                                           gaiahub_pm_summary[2]-gaiahub_pm_summary[1]]).T
            posterior_pm_summary = np.percentile(posterior_pm_meds,[16,50,84],axis=0)
            posterior_pm_summary = np.array([posterior_pm_summary[1],
                                           posterior_pm_summary[1]-posterior_pm_summary[0],
                                           posterior_pm_summary[2]-posterior_pm_summary[1]]).T
        
            if field not in ['COSMOS_field']:
#            if ('dSph' in field) or ('M31' in field):
                n_sigma_plot = 1
                too_far_x = (gaiahub_pms[unique_inds,0] <= gaiahub_pm_summary[0,0]-n_sigma_plot*gaiahub_pm_summary[0,1]) |\
                            (gaiahub_pms[unique_inds,0] >= gaiahub_pm_summary[0,0]+n_sigma_plot*gaiahub_pm_summary[0,2])
                too_far_y = (gaiahub_pms[unique_inds,1] <= gaiahub_pm_summary[1,0]-n_sigma_plot*gaiahub_pm_summary[1,1]) |\
                            (gaiahub_pms[unique_inds,1] >= gaiahub_pm_summary[1,0]+n_sigma_plot*gaiahub_pm_summary[1,2])
                            
            else:
    #            n_sigma_plot = 3
    #            too_far_x = (gaia_pms[unique_inds,0] <= gaia_pm_summary[0,0]-n_sigma_plot*gaia_pm_summary[0,1]) |\
    #                        (gaia_pms[unique_inds,0] >= gaia_pm_summary[0,0]+n_sigma_plot*gaia_pm_summary[0,2])
    #            too_far_y = (gaia_pms[unique_inds,1] <= gaia_pm_summary[1,0]-n_sigma_plot*gaia_pm_summary[1,1]) |\
    #                        (gaia_pms[unique_inds,1] >= gaia_pm_summary[1,0]+n_sigma_plot*gaia_pm_summary[1,2])
                
                n_sigma_plot = 3
                too_far_x = (posterior_pm_meds[:,0] <= posterior_pm_summary[0,0]-n_sigma_plot*posterior_pm_summary[0,1]) |\
                            (posterior_pm_meds[:,0] >= posterior_pm_summary[0,0]+n_sigma_plot*posterior_pm_summary[0,2])
                too_far_y = (posterior_pm_meds[:,1] <= posterior_pm_summary[1,0]-n_sigma_plot*posterior_pm_summary[1,1]) |\
                            (posterior_pm_meds[:,1] >= posterior_pm_summary[1,0]+n_sigma_plot*posterior_pm_summary[1,2])
                too_far_x = np.zeros_like(too_far_x).astype(bool)
                too_far_y = np.zeros_like(too_far_y).astype(bool)
                
                        
            keep_plot = ~too_far_x & ~too_far_y
            
            
            plt.figure(figsize=(7*3,6*2))
            full_title = plt.suptitle('%d Targets, %d Images, %d Measures\n'%(len(unique_stationary),len(keep_im_nums),len(x))+\
                                      r'($N_{\mathrm{Stars}}=%d$, $N_{\mathrm{Stationary}}=%d$, $N_{\mathrm{No}\, \mathrm{Gaia}\, \mathrm{Prior}}=%d$)'%(np.sum(unique_not_stationary),np.sum(unique_stationary),np.sum(unique_missing_prior_PM)),
                                      fontsize=40)
            gs = gridspec.GridSpec(2,3*2,hspace=0.3,wspace=0.7)
            
            ax = plt.subplot(gs[0, 0:2])
            plt.grid(visible=True, which='major', color='#666666', linestyle='-',alpha=0.3)
            plt.minorticks_on()
            plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
            ax.tick_params(axis='both',direction='inout',length=5,bottom=True,left=True,right=True,top=True)
            for star_ind in range(len(unique_inds)):
                if unique_missing_prior_PM[star_ind] or not(keep_plot[star_ind]):
                    continue
                    
                curr_cov = gaia_pm_covs[unique_inds[star_ind]]
                curr_med = gaia_pms[unique_inds[star_ind]]
                
                curr_vals,curr_vects = np.linalg.eig(curr_cov)
                curr_vals = np.sqrt(curr_vals)
            
                err_vects = np.zeros_like(curr_vects)
                err_vects[0] = curr_vals[0]*curr_vects[:,0]
                err_vects[1] = curr_vals[1]*curr_vects[:,1]    
                
                err1_plot = [curr_med[0]-err_vects[0,0],curr_med[0]+err_vects[0,0]],\
                            [curr_med[1]-err_vects[0,1],curr_med[1]+err_vects[0,1]]
                err2_plot = [curr_med[0]-err_vects[1,0],curr_med[0]+err_vects[1,0]],\
                            [curr_med[1]-err_vects[1,1],curr_med[1]+err_vects[1,1]]
                
                if stationary[star_ind]:
                    color = 'C1'
                    zorder = 1e10
                else:
                    color = 'C0'
                    zorder = -1e5
                    
                color = 'k'
                zorder = -1e5
            
                plt.scatter(curr_med[0],curr_med[1],edgecolor=color,facecolor='None',alpha=0.7,zorder=zorder)
                plt.plot(err1_plot[0],err1_plot[1],color=color,lw=1,alpha=0.7,zorder=zorder-2)
                plt.plot(err2_plot[0],err2_plot[1],color=color,lw=1,alpha=0.7,zorder=zorder-1)
                
            plt.xlabel(r'Gaia $\mu_{\mathrm{RA}}$ (mas/yr)')
            plt.ylabel(r'Gaia $\mu_{\mathrm{Dec}}$ (mas/yr)')
            plt.axhline(0,c='k',ls='--',lw=0.5)
            plt.axvline(0,c='k',ls='--',lw=0.5)
            xlim = plt.xlim()
            ylim = plt.ylim()
            
            ax = plt.subplot(gs[0, 2:4])
            plt.grid(visible=True, which='major', color='#666666', linestyle='-',alpha=0.3)
            plt.minorticks_on()
            plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
            ax.tick_params(axis='both',direction='inout',length=5,bottom=True,left=True,right=True,top=True)
            for star_ind in range(len(unique_inds)):
                if not(keep_plot[star_ind]):
                    continue
            
                curr_cov = gaiahub_pm_covs[unique_inds[star_ind]]
                if not np.all(np.isfinite(curr_cov)):
                    continue
                curr_med = gaiahub_pms[unique_inds[star_ind]]#+true_pms[star_ind]
                
                curr_vals,curr_vects = np.linalg.eig(curr_cov)
                curr_vals = np.sqrt(curr_vals)
            
                err_vects = np.zeros_like(curr_vects)
                err_vects[0] = curr_vals[0]*curr_vects[:,0]
                err_vects[1] = curr_vals[1]*curr_vects[:,1]    
                    
                err1_plot = [curr_med[0]-err_vects[0,0],curr_med[0]+err_vects[0,0]],\
                            [curr_med[1]-err_vects[0,1],curr_med[1]+err_vects[0,1]]
                err2_plot = [curr_med[0]-err_vects[1,0],curr_med[0]+err_vects[1,0]],\
                            [curr_med[1]-err_vects[1,1],curr_med[1]+err_vects[1,1]]
                
                if stationary[star_ind]:
                    color = 'C1'
                    zorder = 1e10
                else:
                    color = 'C0'
                    zorder = -1e5
                color = 'r'
                zorder = -1e5
            
                plt.scatter(curr_med[0],curr_med[1],edgecolor=color,facecolor='None',alpha=0.7,zorder=zorder)
                plt.plot(err1_plot[0],err1_plot[1],color=color,lw=1,alpha=0.7,zorder=zorder-2)
                plt.plot(err2_plot[0],err2_plot[1],color=color,lw=1,alpha=0.7,zorder=zorder-1)
                
                if unique_missing_prior_PM[star_ind]:
                    plt.scatter(curr_med[0],curr_med[1],edgecolor='blue',facecolor='None',alpha=0.7,zorder=1e10,s=200)
            
            plt.xlabel(r'GaiaHub $\mu_{\mathrm{RA}}$ (mas/yr)')
            plt.ylabel(r'GaiaHub $\mu_{\mathrm{Dec}}$ (mas/yr)')
            plt.axhline(0,c='k',ls='--',lw=0.5)
            plt.axvline(0,c='k',ls='--',lw=0.5)
            if field not in ['COSMOS_field']:
                xlim = plt.xlim()
                ylim = plt.ylim()
    #        else:
    #            plt.ylim(ylim);plt.xlim(xlim)
            
            ax = plt.subplot(gs[0, 4:6])
            plt.grid(visible=True, which='major', color='#666666', linestyle='-',alpha=0.3)
            plt.minorticks_on()
            plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
            ax.tick_params(axis='both',direction='inout',length=5,bottom=True,left=True,right=True,top=True)
            for star_ind in range(len(unique_inds)):
                if not(keep_plot[star_ind]):
                    continue
            
                curr_cov = posterior_pm_covs[star_ind]
                curr_med = posterior_pm_meds[star_ind]#+true_pms[star_ind]
                
                curr_vals,curr_vects = np.linalg.eig(curr_cov)
                curr_vals = np.sqrt(curr_vals)
            
                err_vects = np.zeros_like(curr_vects)
                err_vects[0] = curr_vals[0]*curr_vects[:,0]
                err_vects[1] = curr_vals[1]*curr_vects[:,1]    
                    
                err1_plot = [curr_med[0]-err_vects[0,0],curr_med[0]+err_vects[0,0]],\
                            [curr_med[1]-err_vects[0,1],curr_med[1]+err_vects[0,1]]
                err2_plot = [curr_med[0]-err_vects[1,0],curr_med[0]+err_vects[1,0]],\
                            [curr_med[1]-err_vects[1,1],curr_med[1]+err_vects[1,1]]
                
                if stationary[star_ind]:
                    color = 'C1'
                    zorder = 1e10
                else:
                    color = 'C0'
                    zorder = -1e5
                color = 'C0'
                zorder = -1e5
            
                plt.scatter(curr_med[0],curr_med[1],edgecolor=color,facecolor='None',alpha=0.7,zorder=zorder)
                plt.plot(err1_plot[0],err1_plot[1],color=color,lw=1,alpha=0.7,zorder=zorder-2)
                plt.plot(err2_plot[0],err2_plot[1],color=color,lw=1,alpha=0.7,zorder=zorder-1)
                
                if unique_missing_prior_PM[star_ind]:
                    plt.scatter(curr_med[0],curr_med[1],edgecolor='blue',facecolor='None',alpha=0.7,zorder=1e10,s=200)
            
            plt.xlabel(r'KM $\mu_{\mathrm{RA}}$ (mas/yr)')
            plt.ylabel(r'KM $\mu_{\mathrm{Dec}}$ (mas/yr)')
            plt.axhline(0,c='k',ls='--',lw=0.5)
            plt.axvline(0,c='k',ls='--',lw=0.5)
            if field not in ['COSMOS_field']:
                plt.ylim(ylim);plt.xlim(xlim)
            
            plot_gaia_err_size = np.where(~unique_missing_prior_PM)[0]
            
            gmag_order = np.argsort(g_mag[unique_inds])
            keep_gmag_order = np.argsort(g_mag[unique_inds][plot_gaia_err_size])
            
            ax = plt.subplot(gs[1, 0:3])
            plt.grid(visible=True, which='major', color='#666666', linestyle='-',alpha=0.3)
            plt.minorticks_on()
            plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
            plt.plot(g_mag[unique_inds][plot_gaia_err_size][keep_gmag_order],
                     gaia_err_size[unique_inds][plot_gaia_err_size][keep_gmag_order],
                     marker='.',ms=10,color='k',label='Gaia',lw=2)
            plt.plot(g_mag[unique_inds][gmag_order],
                     gaiahub_pm_err_sizes[unique_inds][gmag_order],
                     marker='.',ms=10,color='r',label='GaiaHub',lw=2)
            plt.plot(g_mag[unique_inds][gmag_order],
                        np.sqrt(np.sum(np.power(err_lengths,2),axis=1))[gmag_order],
                        marker='.',ms=10,color='C0',label='KM',lw=2)
            xlim = plt.xlim()
            plt.axvspan(21,xlim[1],color='grey',alpha=0.2,zorder=-1e10)
            plt.xlim(xlim)
            plt.ylabel(r'$||\sigma_{\mu}||$ (mas/yr)')
            plt.xlabel('G (mag)')
            ylim = plt.ylim()
            plt.ylim(0,ylim[1])
            leg = plt.legend(loc=2,markerscale=1)
            for line in leg.get_lines():
                line.set_linewidth(5)
            
            ax = plt.subplot(gs[1, 3:6])
            plt.grid(visible=True, which='major', color='#666666', linestyle='-',alpha=0.3)
            plt.minorticks_on()
            plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
            plt.axhline(1.0,c='k',ls='--',lw=1)
            plt.plot(g_mag[unique_inds][plot_gaia_err_size][keep_gmag_order],
                    (gaia_err_size/gaiahub_pm_err_sizes)[unique_inds][plot_gaia_err_size][keep_gmag_order],
                    marker='.',ms=10,color='r',label='GaiaHub',lw=2)
            plt.plot(g_mag[unique_inds][plot_gaia_err_size][keep_gmag_order],
                    (gaia_err_size[unique_inds]/np.sqrt(np.sum(np.power(err_lengths,2),axis=1)))[plot_gaia_err_size][keep_gmag_order],
                    marker='.',ms=10,color='C0',label='KM',lw=2)
            plt.ylabel(r'PM Error Improvement Factor')
            plt.xlabel('G (mag)')
            ylim = plt.ylim()
            plt.ylim(0,ylim[1])
            leg = plt.legend(loc=2,markerscale=1)
            for line in leg.get_lines():
                line.set_linewidth(5)
            
            plt.savefig(f'{outpath}{image_name}_posterior_PM_comparison.png',
                        bbox_extra_artists=(full_title,leg,), bbox_inches='tight')
            
            # plt.show()
                        
            #compare posterior parallaxes and uncertainties to Gaia 
            plt.figure(figsize=(3*5*(1.3**2),5))
            gs = gridspec.GridSpec(1,3,wspace=0.3)
            
            ax = plt.subplot(gs[:,0])    
#            ax = plt.gca()
#            ax.set_aspect('equal')
            plt.grid(visible=True, which='major', color='#666666', linestyle='-',alpha=0.3)
            plt.minorticks_on()
            plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
            ax.tick_params(axis='both',direction='inout',length=5,bottom=True,left=True,right=True,top=True)
            y_vals = posterior_parallax_meds[~missing_prior_PM[unique_inds]]
            y_errs = posterior_parallax_errs[~missing_prior_PM[unique_inds]].T
            x_vals = gaia_parallaxes[unique_inds][~missing_prior_PM[unique_inds]]
            x_errs = gaia_parallax_errs[unique_inds][~missing_prior_PM[unique_inds]]
            plt.errorbar(x_vals,y_vals,xerr=x_errs,yerr=y_errs,fmt='o',color='C0',alpha=0.5,ms=1)
            xlim,ylim = plt.xlim(),plt.ylim()
            plt.plot(xlim,xlim,color='k',zorder=1e10,lw=1,ls='--')
            plt.xlim(xlim);plt.ylim(ylim)
            plt.xlabel('$\mathrm{plx}_{\mathrm{Gaia}}$ (mas)')
            plt.ylabel('$\mathrm{plx}_{\mathrm{KM}}$ (mas)')
            
            ax = plt.subplot(gs[:,1])    
#            ax = plt.gca()
#            ax.set_aspect('equal')
            plt.grid(visible=True, which='major', color='#666666', linestyle='-',alpha=0.3)
            plt.minorticks_on()
            plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
            ax.tick_params(axis='both',direction='inout',length=5,bottom=True,left=True,right=True,top=True)
            
            plt.plot(g_mag[unique_inds][plot_gaia_err_size][keep_gmag_order],
                     gaia_parallax_errs[unique_inds][plot_gaia_err_size][keep_gmag_order],
                     marker='.',ms=10,color='k',label='Gaia',lw=2)
            ylim = plt.ylim()
            plt.plot(g_mag[unique_inds][gmag_order],0.5*np.sum(posterior_parallax_errs,axis=1)[gmag_order],
                     marker='.',ms=10,color='C0',label='KM',lw=2)
            # plt.ylim(0,ylim[1])
            plt.ylabel(r'$\sigma_{\mathrm{plx}}$ (mas)')
            plt.xlabel('G (mag)')
            leg = plt.legend(loc=2,markerscale=1)
            for line in leg.get_lines():
                line.set_linewidth(5)
                                
            ax = plt.subplot(gs[:,2])    
            plt.grid(visible=True, which='major', color='#666666', linestyle='-',alpha=0.3)
            plt.minorticks_on()
            plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
            ax.tick_params(axis='both',direction='inout',length=5,bottom=True,left=True,right=True,top=True)
            plt.plot(g_mag[unique_inds][plot_gaia_err_size][keep_gmag_order],
                     (gaia_parallax_errs[unique_inds]/(0.5*np.sum(posterior_parallax_errs,axis=1)))[plot_gaia_err_size][keep_gmag_order],
                     marker='.',ms=10,color='C0',label='KM',lw=2)
            plt.xlabel('G (mag)')
            plt.ylabel(r'Parallax Error Improvement Factor')
            plt.axhline(1.0,c='k',ls='--',lw=1)
            ylim = plt.ylim()
            plt.ylim(max(0,ylim[0]),ylim[1])
            
            plt.savefig(f'{outpath}{image_name}_posterior_parallax_uncertainty.png',
                        bbox_extra_artists=(leg,), bbox_inches='tight')
            plt.close('all')
            # plt.show()
            
            #compare posterior offsets to Gaia 
            plt.figure(figsize=(3*5*(1.3**2),5))
            gs = gridspec.GridSpec(1,3,wspace=0.3)
            
            ax = plt.subplot(gs[:,0])    
            plt.grid(visible=True, which='major', color='#666666', linestyle='-',alpha=0.3)
            plt.minorticks_on()
            plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
            ax.tick_params(axis='both',direction='inout',length=5,bottom=True,left=True,right=True,top=True)
            
            for star_ind in range(len(pm_x_samps[0])):
                curr_med = posterior_offset_meds[star_ind]                
                color = 'C0'
                zorder = -1e5
                plt.scatter(curr_med[0],curr_med[1],edgecolor=color,facecolor='None',alpha=0.7,zorder=zorder)
                
                if unique_missing_prior_PM[star_ind]:
                    plt.scatter(curr_med[0],curr_med[1],edgecolor='r',facecolor='None',alpha=0.7,zorder=-1e10,s=200)
                    
            xlim,ylim = plt.xlim(),plt.ylim()
            for star_ind in range(len(pm_x_samps[0])):
                curr_cov = posterior_offset_covs[star_ind]
                curr_med = posterior_offset_meds[star_ind]
                
                curr_vals,curr_vects = np.linalg.eig(curr_cov)
                curr_vals = np.sqrt(curr_vals)
            
                err_vects = np.zeros_like(curr_vects)
                err_vects[0] = curr_vals[0]*curr_vects[:,0]
                err_vects[1] = curr_vals[1]*curr_vects[:,1]    
                                
                err1_plot = [curr_med[0]-err_vects[0,0],curr_med[0]+err_vects[0,0]],\
                            [curr_med[1]-err_vects[0,1],curr_med[1]+err_vects[0,1]]
                err2_plot = [curr_med[0]-err_vects[1,0],curr_med[0]+err_vects[1,0]],\
                            [curr_med[1]-err_vects[1,1],curr_med[1]+err_vects[1,1]]
                
                color = 'C0'
                zorder = -1e5
                plt.plot(err1_plot[0],err1_plot[1],color=color,lw=1,alpha=0.7,zorder=zorder-2)
                plt.plot(err2_plot[0],err2_plot[1],color=color,lw=1,alpha=0.7,zorder=zorder-1)
            plt.xlim(xlim);plt.ylim(ylim)
            plt.xlabel(r'$\Delta\theta_{\mathrm{RA}}$ (mas)')
            plt.ylabel(r'$\Delta\theta_{\mathrm{Dec}}$ (mas)')
            plt.axhline(0,c='k',ls='--',lw=0.5)
            plt.axvline(0,c='k',ls='--',lw=0.5)
#            plt.show()
            
#                plt.figure(figsize=(6,6))
#            ax = plt.gca()
#            ax.set_aspect('equal')
            ax = plt.subplot(gs[:,1])    
            plt.grid(visible=True, which='major', color='#666666', linestyle='-',alpha=0.3)
            plt.minorticks_on()
            plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
            ax.tick_params(axis='both',direction='inout',length=5,bottom=True,left=True,right=True,top=True)
            y_vals = posterior_offset_meds[:,0]
            y_errs = np.sqrt(posterior_offset_covs[:,0,0])
            x_vals = np.zeros_like(y_vals)
            x_errs = np.sqrt(gaia_offset_covs[unique_inds,0,0])
            plt.errorbar(x_vals,y_vals,xerr=x_errs,yerr=y_errs,fmt='o',color='C0',alpha=0.5,ms=1)
            xlim,ylim = plt.xlim(),plt.ylim()
            plt.plot(xlim,xlim,color='k',zorder=1e10,lw=1,ls='--')
            plt.xlim(xlim);plt.ylim(ylim)
            plt.xlabel(r'$\Delta\theta_{\mathrm{RA,Gaia}}$ (mas)')
            plt.ylabel(r'$\Delta\theta_{\mathrm{RA,KM}}$ (mas)')
#            plt.show()
            
#            plt.figure(figsize=(6,6))
#            ax = plt.gca()
#            ax.set_aspect('equal')
            ax = plt.subplot(gs[:,2])    
            plt.grid(visible=True, which='major', color='#666666', linestyle='-',alpha=0.3)
            plt.minorticks_on()
            plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
            ax.tick_params(axis='both',direction='inout',length=5,bottom=True,left=True,right=True,top=True)
            y_vals = posterior_offset_meds[:,1]
            y_errs = np.sqrt(posterior_offset_covs[:,1,1])
            x_vals = np.zeros_like(y_vals)
            x_errs = np.sqrt(gaia_offset_covs[unique_inds,1,1])
            plt.errorbar(x_vals,y_vals,xerr=x_errs,yerr=y_errs,fmt='o',color='C0',alpha=0.5,ms=1)
            xlim,ylim = plt.xlim(),plt.ylim()
            plt.plot(xlim,xlim,color='k',zorder=1e10,lw=1,ls='--')
            plt.xlim(xlim);plt.ylim(ylim)
            plt.xlabel(r'$\Delta\theta_{\mathrm{Dec,Gaia}}$ (mas)')
            plt.ylabel(r'$\Delta\theta_{\mathrm{Dec,KM}}$ (mas)')
#            plt.tight_layout()

            plt.savefig(f'{outpath}{image_name}_posterior_VS_prior_offsets.png')
            plt.close('all')
            
            
            #compare posterior offset uncertainties to Gaia 
            gaia_pos_err_sizes = np.sqrt(np.power(gaia_ra_errs,2)+np.power(gaia_dec_errs,2))
            posterior_pos_err_sizes = np.sqrt(posterior_offset_covs[:,0,0]+posterior_offset_covs[:,1,1])
            plt.figure(figsize=(2*5*(1.3**2),5))
            gs = gridspec.GridSpec(1,2,wspace=0.3)
                        
            ax = plt.subplot(gs[:,0])    
#            ax = plt.gca()
#            ax.set_aspect('equal')
            plt.grid(visible=True, which='major', color='#666666', linestyle='-',alpha=0.3)
            plt.minorticks_on()
            plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
            ax.tick_params(axis='both',direction='inout',length=5,bottom=True,left=True,right=True,top=True)
            
            plt.plot(g_mag[unique_inds][gmag_order],
                     gaia_pos_err_sizes[unique_inds][gmag_order],
                     marker='.',ms=10,color='k',label='Gaia',lw=2)
            ylim = plt.ylim()
            plt.plot(g_mag[unique_inds][gmag_order],posterior_pos_err_sizes[gmag_order],
                     marker='.',ms=10,color='C0',label='KM',lw=2)
            # plt.ylim(0,ylim[1])
            plt.ylabel(r'$||\sigma_{\mathrm{RA,Dec}}||$ (mas)')
            plt.xlabel('G (mag)')
            leg = plt.legend(loc=2,markerscale=1)
            for line in leg.get_lines():
                line.set_linewidth(5)
                                
            ax = plt.subplot(gs[:,1])    
            plt.grid(visible=True, which='major', color='#666666', linestyle='-',alpha=0.3)
            plt.minorticks_on()
            plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
            ax.tick_params(axis='both',direction='inout',length=5,bottom=True,left=True,right=True,top=True)
            plt.plot(g_mag[unique_inds][plot_gaia_err_size][keep_gmag_order],
                     (gaia_pos_err_sizes[unique_inds]/posterior_pos_err_sizes)[plot_gaia_err_size][keep_gmag_order],
                     marker='.',ms=10,color='C0',label='KM',lw=2)
            plt.xlabel('G (mag)')
            plt.ylabel(r'Position Error Improvement Factor')
            plt.axhline(1.0,c='k',ls='--',lw=1)
            ylim = plt.ylim()
            plt.ylim(max(0,ylim[0]),ylim[1])
            
            plt.savefig(f'{outpath}{image_name}_posterior_position_uncertainty.png',
                        bbox_extra_artists=(leg,), bbox_inches='tight')
            plt.close('all')
            # plt.show()
            
            #plots of the V_RA versus distance for the truth and Gaia Priors and Posterior measures
            #Do the same for V_Dec and V_tangential
            #Look at total trend as well as individual measurements
            
            
            indv_star_path = f'{path}{field}/Bayesian_PMs/{image_name}/indv_stars/'
            if not os.path.isdir(indv_star_path):
                os.makedirs(indv_star_path)
                
            print(f'Plotting comparison of data and prior PMs for each star.')
            
            for star_ind in range(len(pm_x_samps[0])):
                star_name = unique_ids[star_ind]
                curr_inds = unique_star_mapping[star_name]
                
                plt.figure(figsize=(6,6))
                ax = plt.gca()
                ax.set_aspect('equal')
                plt.grid(visible=True, which='major', color='#666666', linestyle='-',alpha=0.3)
                plt.minorticks_on()
                plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
                ax.tick_params(axis='both',direction='inout',length=5,bottom=True,left=True,right=True,top=True)
                plt.title(f'{star_name}\n G = {round(g_mag[unique_inds[star_ind]],1)} mag, Gaia Prior = {~unique_missing_prior_PM[star_ind]}\nUsed in Fit = {unique_keep[star_ind]}\n({field}, '+\
                          r'$N_{\mathrm{im}}$ = %d)'%(len(curr_inds)))
                plt.xlabel('$\mu_{\mathrm{RA}}$ (mas/yr)')
                plt.ylabel('$\mu_{\mathrm{Dec}}$ (mas/yr)')
                
                #plot the posterior PM 
                curr_cov = posterior_pm_covs[star_ind]
                curr_med = posterior_pm_meds[star_ind]
                curr_vals,curr_vects = np.linalg.eig(curr_cov)
                curr_vals = np.sqrt(curr_vals)
                err_vects = np.zeros_like(curr_vects)
                err_vects[0] = curr_vals[0]*curr_vects[:,0]
                err_vects[1] = curr_vals[1]*curr_vects[:,1]    
                err1_plot = [curr_med[0]-err_vects[0,0],curr_med[0]+err_vects[0,0]],\
                            [curr_med[1]-err_vects[0,1],curr_med[1]+err_vects[0,1]]
                err2_plot = [curr_med[0]-err_vects[1,0],curr_med[0]+err_vects[1,0]],\
                            [curr_med[1]-err_vects[1,1],curr_med[1]+err_vects[1,1]]
                color = 'r'
                zorder = 1e10
                plt.scatter(curr_med[0],curr_med[1],edgecolor=color,facecolor='None',alpha=0.7,zorder=zorder)
                plt.plot(err1_plot[0],err1_plot[1],color=color,lw=1,alpha=0.7,zorder=zorder-2)
                plt.plot(err2_plot[0],err2_plot[1],color=color,lw=1,alpha=0.7,zorder=zorder-1)
                                
                #plot the Gaia prior PMs if they exist
                if not unique_missing_prior_PM[star_ind]:
                    curr_cov = gaia_vector_covs[unique_inds[star_ind]][2:4,2:4]
                    curr_med = gaia_vectors[unique_inds[star_ind]][2:4]
                    curr_vals,curr_vects = np.linalg.eig(curr_cov)
                    curr_vals = np.sqrt(curr_vals)
                    err_vects = np.zeros_like(curr_vects)
                    err_vects[0] = curr_vals[0]*curr_vects[:,0]
                    err_vects[1] = curr_vals[1]*curr_vects[:,1]    
                    err1_plot = [curr_med[0]-err_vects[0,0],curr_med[0]+err_vects[0,0]],\
                                [curr_med[1]-err_vects[0,1],curr_med[1]+err_vects[0,1]]
                    err2_plot = [curr_med[0]-err_vects[1,0],curr_med[0]+err_vects[1,0]],\
                                [curr_med[1]-err_vects[1,1],curr_med[1]+err_vects[1,1]]
                    color = 'k'
                    zorder = -1e5
                    plt.scatter(curr_med[0],curr_med[1],edgecolor=color,facecolor='None',alpha=0.7,zorder=zorder)
                    plt.plot(err1_plot[0],err1_plot[1],color=color,lw=1,alpha=0.7,zorder=zorder-2)
                    plt.plot(err2_plot[0],err2_plot[1],color=color,lw=1,alpha=0.7,zorder=zorder-1)
                    
                #plot the data-measured PMs
                for ind in curr_inds:
                    color = 'C0'
                    zorder = 1e5
                    curr_med = np.median(pm_data_measures[:,ind],axis=0)
                    plt.scatter(curr_med[0],curr_med[1],edgecolor=color,facecolor='None',alpha=0.7,zorder=zorder)
                
                #plot the Global PM prior
                curr_med = global_vector_mean[2:4]
                color = 'C1'
                zorder = -1e10
                plt.scatter(curr_med[0],curr_med[1],edgecolor=color,facecolor='None',alpha=0.7,zorder=zorder)
                    
                xlim = plt.xlim()
                ylim = plt.ylim()
                
                #plot the data-measured PMs
                for ind in curr_inds:
                    color = 'C0'
                    zorder = 1e5
                    
                    curr_cov = np.cov(pm_data_measures[:,ind],rowvar=False)
                    curr_med = np.median(pm_data_measures[:,ind],axis=0)
                    curr_vals,curr_vects = np.linalg.eig(curr_cov)
                    curr_vals = np.sqrt(curr_vals)
                    err_vects = np.zeros_like(curr_vects)
                    err_vects[0] = curr_vals[0]*curr_vects[:,0]
                    err_vects[1] = curr_vals[1]*curr_vects[:,1]    
                    err1_plot = [curr_med[0]-err_vects[0,0],curr_med[0]+err_vects[0,0]],\
                                [curr_med[1]-err_vects[0,1],curr_med[1]+err_vects[0,1]]
                    err2_plot = [curr_med[0]-err_vects[1,0],curr_med[0]+err_vects[1,0]],\
                                [curr_med[1]-err_vects[1,1],curr_med[1]+err_vects[1,1]]
                    plt.plot(err1_plot[0],err1_plot[1],color=color,lw=1,alpha=0.7,zorder=zorder-2)
                    plt.plot(err2_plot[0],err2_plot[1],color=color,lw=1,alpha=0.7,zorder=zorder-1)
                
                #plot the Global PM prior
                curr_cov = global_vector_cov[2:4,2:4]
                curr_med = global_vector_mean[2:4]
                curr_vals,curr_vects = np.linalg.eig(curr_cov)
                curr_vals = np.sqrt(curr_vals)
                err_vects = np.zeros_like(curr_vects)
                err_vects[0] = curr_vals[0]*curr_vects[:,0]
                err_vects[1] = curr_vals[1]*curr_vects[:,1]    
                err1_plot = [curr_med[0]-err_vects[0,0],curr_med[0]+err_vects[0,0]],\
                            [curr_med[1]-err_vects[0,1],curr_med[1]+err_vects[0,1]]
                err2_plot = [curr_med[0]-err_vects[1,0],curr_med[0]+err_vects[1,0]],\
                            [curr_med[1]-err_vects[1,1],curr_med[1]+err_vects[1,1]]
                color = 'C1'
                zorder = -1e10
                plt.plot(err1_plot[0],err1_plot[1],color=color,lw=1,alpha=0.7,zorder=zorder-2)
                plt.plot(err2_plot[0],err2_plot[1],color=color,lw=1,alpha=0.7,zorder=zorder-1)
                
                plt.xlim(xlim)
                plt.ylim(ylim)
                plt.savefig(f'{indv_star_path}{image_name}_{star_name}_posterior_PM_comparison.png',bbox_inches='tight')
                # plt.show()
                plt.close('all')
                # if star_ind > 25:

                        
            for star_ind in range(len(unique_ids)):
            #     if stationary[star_ind]:
            #         continue
            
                if unique_missing_prior_PM[star_ind]:
                    continue
                    print(star_ind,'\t\t(MISSING GAIA PMs)')
                else:
    #                continue
                    print(star_ind)
                print('%10.4f'%(gaia_pms[unique_inds[star_ind],0]),
                      np.round(np.percentile(pm_x_samps[:,star_ind],[16,50,84]),5))
                print('%10.4f'%(gaia_pms[unique_inds[star_ind],1]),
                      np.round(np.percentile(pm_y_samps[:,star_ind],[16,50,84]),5))
                print('%10.4f'%(gaia_parallaxes[unique_inds[star_ind]]),
                      np.round(np.percentile(parallax_samps[:,star_ind],[16,50,84]),5))
                print('%10.4f'%(0),
                      np.round(np.percentile(offset_x_samps[:,star_ind],[16,50,84]),5))
                print('%10.4f'%(0),
                      np.round(np.percentile(offset_y_samps[:,star_ind],[16,50,84]),5))
    #            print('%10.4f'%(gaia_parallaxes[star_ind]*parallax_offset_vector[star_ind,0]),
    #                  np.round(np.percentile(parallax_samps[:,star_ind]*parallax_offset_vector[star_ind,0],[16,50,84]),5))
    #            print('%10.4f'%(gaia_parallaxes[star_ind]*parallax_offset_vector[star_ind,1]),
    #                  np.round(np.percentile(parallax_samps[:,star_ind]*parallax_offset_vector[star_ind,1],[16,50,84]),5))
                print()
                
#                n_plot_walkers = min(50,nwalkers)
#                chosen_walker_inds = np.random.choice(nwalkers,size=n_plot_walkers,replace=False)
#            
#                curr_samplerChain = samplerChain[chosen_walker_inds]
#            #     curr_samplerChain = 
#                
#                plt.figure(figsize=[13,9/4*5])
#                plt.subplot(5,1,1)
#            #         vals = curr_samplerChain[:,:,ndim_trans+star_ind*3].T-true_pm_xs[star_ind]
#                vals = samplerChain_pms[chosen_walker_inds,:,star_ind,0].T
#                val_bounds = np.percentile(np.ravel(vals[burnin:]),[16,50,84])
#                val_bounds = np.array([val_bounds[1],val_bounds[1]-val_bounds[0],val_bounds[2]-val_bounds[1]])
#                ylim = (val_bounds[0]-5*val_bounds[1],val_bounds[0]+5*val_bounds[2])        
#                plt.ylim(ylim)
#                plt.plot(vals,alpha=0.25)
#                plt.xticks(np.arange(0, curr_samplerChain.shape[1]+1, curr_samplerChain.shape[1]/10).astype(int))
#                plt.gca().set_xticklabels([])
#                plt.ylabel(r'$\mu_{\mathrm{RA}}$ (mas/yr)')
#                plt.axvline(x=burnin,lw=2,ls='--',c='r')
#                plt.axhline(y=gaia_pms[unique_inds[star_ind],0],lw=2,ls='--',c='C0')
#                
#                plt.subplot(5,1,2)
#            #         vals = curr_samplerChain[:,:,ndim_trans+star_ind*3+1].T-true_pm_ys[star_ind]
#                vals = samplerChain_pms[chosen_walker_inds,:,star_ind,1].T
#                val_bounds = np.percentile(np.ravel(vals[burnin:]),[16,50,84])
#                val_bounds = np.array([val_bounds[1],val_bounds[1]-val_bounds[0],val_bounds[2]-val_bounds[1]])
#                ylim = (val_bounds[0]-5*val_bounds[1],val_bounds[0]+5*val_bounds[2])        
#                plt.ylim(ylim)        
#                plt.plot(vals,alpha=0.25)
#                plt.xticks(np.arange(0, curr_samplerChain.shape[1]+1, curr_samplerChain.shape[1]/10).astype(int))
#                plt.ylabel(r'$\mu_{\mathrm{Dec}}$ (mas/yr)')
#                plt.axvline(x=burnin,lw=2,ls='--',c='r')
#                plt.axhline(y=gaia_pms[unique_inds[star_ind],1],lw=2,ls='--',c='C0')
#                
#                plt.subplot(5,1,3)
#            #         vals = np.exp(curr_samplerChain[:,:,ndim_trans+star_ind*3+2].T)-true_parallaxes[star_ind]
#                vals = samplerChain_parallaxes[chosen_walker_inds,:,star_ind].T
#                val_bounds = np.percentile(np.ravel(vals[burnin:]),[16,50,84])
#                val_bounds = np.array([val_bounds[1],val_bounds[1]-val_bounds[0],val_bounds[2]-val_bounds[1]])
#                ylim = (val_bounds[0]-5*val_bounds[1],val_bounds[0]+5*val_bounds[2])   
#                plt.ylim(ylim)        
#                plt.plot(vals,alpha=0.25)
#                plt.xticks(np.arange(0, curr_samplerChain.shape[1]+1, curr_samplerChain.shape[1]/10).astype(int))
#                plt.ylabel(r'plx (mas)')
#                plt.axvline(x=burnin,lw=2,ls='--',c='r')
#                plt.axhline(y=gaia_parallaxes[unique_inds[star_ind]],lw=2,ls='--',c='C0')
#                plt.gca().set_xticklabels([])
#    
#                plt.subplot(5,1,4)
#            #         vals = np.exp(curr_samplerChain[:,:,ndim_trans+star_ind*3+2].T)-true_parallaxes[star_ind]
#                vals = samplerChain_offsets[chosen_walker_inds,:,star_ind,0].T
#                val_bounds = np.percentile(np.ravel(vals[burnin:]),[16,50,84])
#                val_bounds = np.array([val_bounds[1],val_bounds[1]-val_bounds[0],val_bounds[2]-val_bounds[1]])
#                ylim = (val_bounds[0]-5*val_bounds[1],val_bounds[0]+5*val_bounds[2])   
#                plt.ylim(ylim)        
#                plt.plot(vals,alpha=0.25)
#                plt.xticks(np.arange(0, curr_samplerChain.shape[1]+1, curr_samplerChain.shape[1]/10).astype(int))
#                plt.ylabel(r'$\Delta\mathrm{RA}$ (mas)')
#                plt.axvline(x=burnin,lw=2,ls='--',c='r')
#                plt.axhline(y=0,lw=2,ls='--',c='C0')
#                plt.gca().set_xticklabels([])
#    
#                plt.subplot(5,1,5)
#            #         vals = np.exp(curr_samplerChain[:,:,ndim_trans+star_ind*3+2].T)-true_parallaxes[star_ind]
#                vals = samplerChain_offsets[chosen_walker_inds,:,star_ind,1].T
#                val_bounds = np.percentile(np.ravel(vals[burnin:]),[16,50,84])
#                val_bounds = np.array([val_bounds[1],val_bounds[1]-val_bounds[0],val_bounds[2]-val_bounds[1]])
#                ylim = (val_bounds[0]-5*val_bounds[1],val_bounds[0]+5*val_bounds[2])   
#                plt.ylim(ylim)        
#                plt.plot(vals,alpha=0.25)
#                plt.xticks(np.arange(0, curr_samplerChain.shape[1]+1, curr_samplerChain.shape[1]/10).astype(int))
#                plt.ylabel(r'$\Delta\mathrm{Dec}$ (mas)')
#                plt.axvline(x=burnin,lw=2,ls='--',c='r')
#                plt.axhline(y=0,lw=2,ls='--',c='C0')                        
#                plt.xlabel('Step Number')
#                #plt.tight_layout()
#                plt.close('all')
#                # plt.show()
#            
#            #         corner.corner(np.array([pm_x_samps[:,star_ind]-true_pm_xs[star_ind],
#            #                                 pm_y_samps[:,star_ind]-true_pm_ys[star_ind],
#            #                                 np.log10(parallax_samps[:,star_ind])]).T, 
#            #               labels=[r'$\Delta\mu_x$',r'$\Delta\mu_y$',r'$\log_{10}\Delta$plx (mas)'], 
#            #               quantiles=[0.16, 0.5, 0.84], show_titles=True,
#            #               title_kwargs={"fontsize": 12},bins=10,
#            #               truths=[0,0,np.log10(true_parallaxes[star_ind])])
#            #         parallax_offset_x_samps = parallax_samps[:,star_ind]*parallax_offset_vector[star_ind,0]
#            #         parallax_offset_y_samps = parallax_samps[:,star_ind]*parallax_offset_vector[star_ind,1]
#            #         corner.corner(np.array([pm_x_samps[:,star_ind]-true_pm_xs[star_ind],
#            #                                 pm_y_samps[:,star_ind]-true_pm_ys[star_ind],
#            #                                 parallax_offset_x_samps,
#            #                                 parallax_offset_y_samps]).T, 
#            #               labels=[r'$\Delta\mu_x$',r'$\Delta\mu_y$',\
#            #                       r'$\Delta x_{\mathrm{plx}}$ (mas)',r'$\Delta y_{\mathrm{plx}}$ (mas)'], 
#            #               quantiles=[0.16, 0.5, 0.84], show_titles=True,
#            #               title_kwargs={"fontsize": 12},bins=10,
#            #               truths=[0,0,\
#            #                       true_offset_vectors[star_ind,0],\
#            #                       true_offset_vectors[star_ind,1]])
#                
#                corner.corner(np.array([pm_x_samps[:,star_ind],
#                                        pm_y_samps[:,star_ind],
#                                        parallax_samps[:,star_ind],
#                                        offset_x_samps[:,star_ind],
#                                        offset_y_samps[:,star_ind]]).T, 
#                      labels=[r'$\mu_{\mathrm{RA}}$',r'$\mu_{\mathrm{Dec}}$',r'plx',\
#                              r'$\Delta\mathrm{RA}$',r'$\Delta\mathrm{Dec}$'], 
#                      quantiles=[0.16, 0.5, 0.84], show_titles=True,
#                      title_kwargs={"fontsize": 12},bins=10,
#                      truths=[gaia_pm_xs[unique_inds[star_ind]],gaia_pm_ys[unique_inds[star_ind]],\
#                              gaia_parallaxes[unique_inds[star_ind]],0,0])
#            
#                plt.close('all')
#                # plt.show()
                break
                
            #think about showing the change in pixel position (mas) from parallax instead of the parallax amount
            #because the parallax offset might show really small values, and it doesn't matter how large
            #the parallax is (e.g. a factor of 10 too large in parallax doesn't mean much when delta_RA = 0.0001 mas)
            #maybe show this as a function of time instead of gmag
            
#    
#            if field not in ['COSMOS_field']:
#    #            n_sigma_plot = 1
#    #            too_far_x = (gaiahub_pms[unique_inds,0] <= gaiahub_pm_summary[0,0]-n_sigma_plot*gaiahub_pm_summary[0,1]) |\
#    #                        (gaiahub_pms[unique_inds,0] >= gaiahub_pm_summary[0,0]+n_sigma_plot*gaiahub_pm_summary[0,2])
#    #            too_far_y = (gaiahub_pms[unique_inds,1] <= gaiahub_pm_summary[1,0]-n_sigma_plot*gaiahub_pm_summary[1,1]) |\
#    #                        (gaiahub_pms[unique_inds,1] >= gaiahub_pm_summary[1,0]+n_sigma_plot*gaiahub_pm_summary[1,2])
#                            
#                n_sigma_plot = 1
#                too_far_x = (posterior_pm_meds[:,0] <= posterior_pm_summary[0,0]-n_sigma_plot*posterior_pm_summary[0,1]) |\
#                            (posterior_pm_meds[:,0] >= posterior_pm_summary[0,0]+n_sigma_plot*posterior_pm_summary[0,2])
#                too_far_y = (posterior_pm_meds[:,1] <= posterior_pm_summary[1,0]-n_sigma_plot*posterior_pm_summary[1,1]) |\
#                            (posterior_pm_meds[:,1] >= posterior_pm_summary[1,0]+n_sigma_plot*posterior_pm_summary[1,2])
#            else:
#    #            n_sigma_plot = 3
#    #            too_far_x = (gaia_pms[unique_inds,0] <= gaia_pm_summary[0,0]-n_sigma_plot*gaia_pm_summary[0,1]) |\
#    #                        (gaia_pms[unique_inds,0] >= gaia_pm_summary[0,0]+n_sigma_plot*gaia_pm_summary[0,2])
#    #            too_far_y = (gaia_pms[unique_inds,1] <= gaia_pm_summary[1,0]-n_sigma_plot*gaia_pm_summary[1,1]) |\
#    #                        (gaia_pms[unique_inds,1] >= gaia_pm_summary[1,0]+n_sigma_plot*gaia_pm_summary[1,2])
#                
#                n_sigma_plot = 3
#                too_far_x = (posterior_pm_meds[:,0] <= posterior_pm_summary[0,0]-n_sigma_plot*posterior_pm_summary[0,1]) |\
#                            (posterior_pm_meds[:,0] >= posterior_pm_summary[0,0]+n_sigma_plot*posterior_pm_summary[0,2])
#                too_far_y = (posterior_pm_meds[:,1] <= posterior_pm_summary[1,0]-n_sigma_plot*posterior_pm_summary[1,1]) |\
#                            (posterior_pm_meds[:,1] >= posterior_pm_summary[1,0]+n_sigma_plot*posterior_pm_summary[1,2])
#                too_far_x = np.zeros_like(too_far_x).astype(bool)
#                too_far_y = np.zeros_like(too_far_y).astype(bool)
#                
#            keep_plot = ~too_far_x & ~too_far_y
#            
#            #do full 2D covariance fitting of the proper motion offsets to see if we get 
#            #a mean of (0,0), a small covariance, and minimal correlation
#            
#            #2d versions of the fitting
#            #population cov = [[sigma_feh^2,rho*sigm_feh*sigma_alpha],[rho*sigm_feh*sigma_alpha,sigma_alpha^2]]
#            #use MH-MCMC to find the sigma_feh, sigma_alpha, and rho parameters
#            
#            err_scale = 1.0 #no scaling
#            # err_scale = np.sqrt(chi2_scale) #use the result of the 2D distance fitting
#            
#            curr_err_vects = np.copy(err_vectors[keep_plot])*err_scale
#            n_params = 2
#            curr_vals = np.copy(posterior_pm_meds[keep_plot])
#            curr_vals = curr_vals.reshape((*curr_vals.shape,1))
#            curr_covs = np.copy(posterior_pm_covs[keep_plot])*err_scale**2
#            
#            plt.figure(figsize=(7,7))
#            plt.grid(visible=True, which='major', color='#666666', linestyle='-',alpha=0.3)
#            plt.minorticks_on()
#            plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
#            plt.scatter(curr_vals[:,0,0],curr_vals[:,1,0],
#                        marker='o',alpha=0.7,s=50)
#            for j in range(len(curr_vals)):
#                for err_ind in range(n_params):
#                    curr_vect = curr_err_vects[j,err_ind]
#                    x_vals = [curr_vals[j,0,0]-curr_vect[0],curr_vals[j,0,0]+curr_vect[0]]
#                    y_vals = [curr_vals[j,1,0]-curr_vect[1],curr_vals[j,1,0]+curr_vect[1]]
#                    plt.plot(x_vals,y_vals,c='C0',lw=2,alpha=0.2,zorder=-1e10)
#            plt.axvline(0,c='C1',ls='-',zorder=-1e10)
#            plt.axhline(0,c='C1',ls='-',zorder=-1e10)
#            plt.xlabel(r'$\mu_{\mathrm{RA}}$ (mas/yr)')
#            plt.ylabel(r'$\mu_{\mathrm{Dec}}$ (mas/yr)')
#            ax = plt.gca()
#            ax.set_aspect('equal')
#            plt.tight_layout()
#            plt.close('all')
#            # plt.show()
#            
#            curr_inv_covs = np.linalg.inv(curr_covs)
#            curr_inv_cov_dot_vals = np.sum(curr_inv_covs*curr_vals,axis=1)
#            
#            #initial sigma_feh and sigma_alpha are based on total population indvidual dimension fits
#            if field not in ['COSMOS_field']:
#                initial_sigma_x = 0.05
#                initial_sigma_y = 0.05
#            else:
#                initial_sigma_x = 1
#                initial_sigma_y = 1
#            initial_rho = 0
#            
#            nfit = 100*3,600,3
#            new_pos0 = np.zeros((nfit[0],nfit[2]))
#            
#            new_pos0[:,0] = initial_sigma_x+np.random.randn(nfit[0])*initial_sigma_x/10
#            new_pos0[:,1] = initial_sigma_y+np.random.randn(nfit[0])*initial_sigma_y/10
#            new_pos0[:,2] = initial_rho+np.random.randn(nfit[0])*0.05
#            
#            def lnpost_residuals(theta):
#                sigma_feh,sigma_alpha,rho = theta
#                if (sigma_feh <= 0) or (sigma_alpha <= 0) or (np.abs(rho) > 1):
#                    return -np.inf
#            
#                cov_entries = sigma_feh**2,rho*sigma_feh*sigma_alpha,sigma_alpha**2
#                curr_pop_cov = np.array([[cov_entries[0],cov_entries[1]],[cov_entries[1],cov_entries[2]]])
#    #            curr_pop_inv_cov = np.linalg.inv(curr_pop_cov)
#                curr_pop_cov_det = np.linalg.det(curr_pop_cov)
#            
#                summed_covs = curr_pop_cov+curr_covs
#                summed_cov_dets = np.linalg.det(summed_covs)
#                summed_covs_inv = np.linalg.inv(summed_covs)
#                S = np.linalg.inv(np.sum(summed_covs_inv,axis=0))
#                m = np.dot(S,np.sum(np.sum(summed_covs_inv*curr_vals,axis=1),axis=0))
#            
#                #y_vects - mean_vect
#                curr_diff_from_mean = np.copy(curr_vals)
#                curr_diff_from_mean[:,0] -= m[0]
#                curr_diff_from_mean[:,1] -= m[1]
#            
#                exponent_term = -0.5*np.sum(np.sum(summed_covs_inv*curr_diff_from_mean,axis=1)*curr_diff_from_mean[:,:,0],axis=1)
#            
#                curr_pop_cov_lnprob = 0.5*np.log(curr_pop_cov_det)+np.sum(-0.5*np.log(summed_cov_dets)+exponent_term)
#                if not np.isfinite(curr_pop_cov_lnprob):
#                    return -np.inf
#                return curr_pop_cov_lnprob
#            
#            new_dim_labels = [r'$\sigma_{\mu_{\mathrm{RA}}}$',r'$\sigma_{\mu_{\mathrm{Dec}}}$',r'$\rho$']
#            
#            print('Fitting the residual proper motions:')
#            with Pool(n_threads) as pool:
#                new_sampler = emcee.EnsembleSampler(nfit[0],nfit[2],lnpost_residuals,pool=pool)
#            #     new_sampler.run_mcmc(new_pos0,nfit[1])
#                for j, result in enumerate(tqdm(new_sampler.sample(new_pos0,iterations=nfit[1]),total=nfit[1],smoothing=0.1)):
#                    pass
#            
#            new_burnin = int(0.5*nfit[1])
#            new_samplerChain = new_sampler.chain
#            new_samples = new_samplerChain[:, new_burnin:, :].reshape(-1, nfit[2])
#            
#            acpt_fracs = np.sum((np.sum(np.abs(new_samplerChain[:,:-1]-new_samplerChain[:,1:]),axis=2)>1e-15),axis=0)/new_samplerChain.shape[0]
#            minKeep = new_burnin
#            stats_vals = (acpt_fracs[minKeep:].min(),np.median(acpt_fracs[minKeep:]),np.mean(acpt_fracs[minKeep:]),acpt_fracs[minKeep:].max())
#            
#            fig = plt.figure(figsize=[12,6])
#            gs = gridspec.GridSpec(1,2,width_ratios=[3,1],wspace=0)
#            ax0 = plt.subplot(gs[:, 0])    
#            #plt.plot(np.arange(len(acpt_fracs))[minKeep:],acpt_fracs[minKeep:],lw=1,alpha=1)
#            plt.plot(np.arange(len(acpt_fracs)),acpt_fracs,lw=1,alpha=1)
#            acc_lim = plt.ylim()
#            plt.axvline(minKeep,c='r',label=f'Burnin ({burnin} steps)')
#            plt.axhline(stats_vals[1],label='Median: %.3f'%stats_vals[1],c='k',ls='--')
#            plt.axhline(stats_vals[2],label='Mean: %.3f'%stats_vals[2],c='k',ls='-')
#            plt.axhline(stats_vals[0],label='Min: %.3f\nMax: %.3f'%(stats_vals[0],stats_vals[-1]),c='grey')
#            plt.axhline(stats_vals[-1],c='grey')
#            plt.legend(loc='best')
#            ax0.tick_params(axis='both',direction='inout',length=5,bottom=True,left=True,right=True)
#            plt.xlabel('Step Number')
#            plt.ylabel('Acceptance Fraction')
#            ax1 = plt.subplot(gs[:, 1])
#            ax1.axis('off')
#            plt.hist(acpt_fracs[minKeep:],bins=min(len(acpt_fracs)-minKeep,100),density=True,cumulative=True,histtype='step',lw=3,orientation='horizontal')
#            plt.axhline(stats_vals[1],label='Median: %.3f'%stats_vals[1],c='k',ls='--')
#            plt.axhline(stats_vals[2],label='Mean: %.3f'%stats_vals[2],c='k',ls='-')
#            plt.axhline(stats_vals[0],label='Min: %.3f\nMax: %.3f'%(stats_vals[0],stats_vals[-1]),c='grey')
#            plt.axhline(stats_vals[-1],c='grey')
#            #plt.legend(loc='best')
#            plt.ylim(acc_lim)
#            xlim = np.array(plt.xlim());xlim[-1] *= 1.15
#            plt.xlim(xlim)
#            plt.tight_layout()
#            plt.close('all')
#            # plt.show()
#                    
#            plt.figure(figsize=[13,9/5*nfit[2]])
#            for dim in range(nfit[2]):
#                plt.subplot(nfit[2],1,dim+1)
#                plt.plot(new_samplerChain[:,:,dim].T,alpha=0.25)
#                if dim != nfit[2]-1:
#                    plt.xticks([])
#                else:
#                    plt.xticks(np.arange(0, new_samplerChain.shape[1]+1, new_samplerChain.shape[1]/10).astype(int))
#                plt.ylabel(new_dim_labels[dim])
#                plt.axvline(x=new_burnin,lw=2,ls='--',c='r')
#            plt.xlabel('Step Number')
#            plt.tight_layout()
#            plt.close('all')
#            # plt.show()
#            
#            def mu_vect_samples(theta):
#                #use the sigma_feh,sigma_alpha,rho posterior samples to get samples of the mu vector
#                #as well as theta_i and data_i samples
#                sigma_feh,sigma_alpha,rho = theta
#            
#                cov_entries = sigma_feh**2,rho*sigma_feh*sigma_alpha,sigma_alpha**2
#                curr_pop_cov = np.array([[cov_entries[0],cov_entries[1]],[cov_entries[1],cov_entries[2]]])
#                curr_pop_inv_cov = np.linalg.inv(curr_pop_cov)
#    #            curr_pop_cov_det = np.linalg.det(curr_pop_cov)
#            
#                summed_covs = curr_pop_cov+curr_covs
#    #            summed_cov_dets = np.linalg.det(summed_covs)
#                summed_covs_inv = np.linalg.inv(summed_covs)
#                S = np.linalg.inv(np.sum(summed_covs_inv,axis=0))
#                m = np.dot(S,np.sum(np.sum(summed_covs_inv*curr_vals,axis=1),axis=0))
#            
#                mu_sample = stats.multivariate_normal(m,S,allow_singular=True).rvs()
#            
#                S_i_vals = np.linalg.inv(curr_pop_inv_cov+curr_inv_covs)
#                m_i_temp_vals = curr_inv_cov_dot_vals+np.dot(curr_pop_inv_cov,mu_sample)
#                m_i_vals = np.sum(S_i_vals*m_i_temp_vals.reshape((*m_i_temp_vals.shape,1)),axis=1)
#                theta_samples = np.zeros((len(curr_vals),len(m)))
#                data_samples = np.zeros_like(theta_samples)
#                for j in range(len(theta_samples)):
#                    theta_samples[j] = stats.multivariate_normal(m_i_vals[j],S_i_vals[j],allow_singular=True).rvs()
#                    data_samples[j] = stats.multivariate_normal(theta_samples[j],curr_covs[j],allow_singular=True).rvs()
#                return mu_sample,theta_samples,data_samples
#            
#            n_keep_samples = 1000
#            keep_samples = np.random.choice(len(new_samples),size=n_keep_samples,replace=False)
#            new_samples = new_samples[keep_samples]
#            mu_samples = np.zeros((len(new_samples),nfit[2]-1))
#            theta_samples = np.zeros((len(new_samples),len(curr_vals),nfit[2]-1))
#            data_samples = np.zeros_like(theta_samples)
#            
#            print('Drawing proper motion example samples:')
#            for j,sample in enumerate(tqdm(new_samples,total=len(new_samples))):
#                mu_samples[j],theta_samples[j],data_samples[j] = mu_vect_samples(sample)
#            new_samples = np.hstack((new_samples,mu_samples))
#            
#            new_dim_labels.extend([r'$\mu_{\mu_{\mathrm{RA}}}$',r'$\mu_{\mu_{\mathrm{Dec}}}$'])
#            
#            corner.corner(new_samples, labels=new_dim_labels,              fontsize=10,quantiles=[0.16, 0.5, 0.84],show_titles=True)
#            plt.savefig(f'{outpath}{image_name}_posterior_population_PM_analysis_corner_plot.png')
#            plt.close('all')
#            # plt.show()
#            
#            np.save(f'{outpath}{image_name}_posterior_population_PM_analysis_samples.npy',new_samples)
#            
#            sigma_mu_x_2 = np.power(new_samples[:,0],2)
#            sigma_mu_y_2 = np.power(new_samples[:,1],2)
#            sigma_mu_x_y_rho = new_samples[:,0]*new_samples[:,1]*new_samples[:,2]
#            
#            median_mu_x_2 = np.median(sigma_mu_x_2)
#            median_mu_y_2 = np.median(sigma_mu_y_2)
#            median_mu_x_y_rho = np.median(sigma_mu_x_y_rho)
#            median_cov = np.array([[median_mu_x_2,median_mu_x_y_rho],[median_mu_x_y_rho,median_mu_y_2]])
#            
#            median_mu = np.median(mu_samples,axis=0)
##            offsets = np.copy(median_mu)
#            
#            alpha = 0.3
#            
#            plt.figure(figsize=(10,6))
#            len_data_samps = data_samples.shape[0]*data_samples.shape[1]
#            plt.hist2d(np.ravel(data_samples[:,:,0]),np.ravel(data_samples[:,:,1]),
#                       norm=mcolors.PowerNorm(0.3),bins=50,
#                       weights=np.ones(len_data_samps)/len_data_samps)
#            plt.xlabel(r'$\mu_{\mathrm{RA}}$ (mas/yr)')
#            plt.ylabel(r'$\mu_{\mathrm{Dec}}$ (mas/yr)')
#            ax = plt.gca()
#            ax.set_aspect('equal')
#            plt.colorbar(label='Bin Probability')
#            for data_ind in range(len(data_samples[0])):
#                curr_x,curr_y = curr_vals[data_ind,0,0],curr_vals[data_ind,1,0]
#                plt.scatter(curr_x,curr_y,
#                            marker='o',alpha=alpha,s=50,color='r')
#                for err_ind in range(n_params):
#                    curr_vect = curr_err_vects[data_ind,err_ind]
#                    x_vals = [curr_x-curr_vect[0],curr_x+curr_vect[0]]
#                    y_vals = [curr_y-curr_vect[1],curr_y+curr_vect[1]]
#                    plt.plot(x_vals,y_vals,color='r',lw=2,alpha=alpha)
#            # plt.xlim(feh_lim)
#            # plt.ylim(alpha_lim)
#            plt.close('all')
#            # plt.show()
#            
#            pop_eig_vals,pop_eig_vects = np.linalg.eig(median_cov)
#            
#            ellipse_center = median_mu
#            ellipse_axis_lengths = np.sqrt(pop_eig_vals)
#            
#            angles = np.linspace(0,2*np.pi,1000)
#            ellipse_x = ellipse_axis_lengths[0]*np.cos(angles)
#            ellipse_y = ellipse_axis_lengths[1]*np.sin(angles)
#            
#            semi_major_axis_vect = pop_eig_vects[:,0]
#            rotate_angle = np.arctan2(semi_major_axis_vect[1],semi_major_axis_vect[0])
#            
#            #rotate the ellipse
#            new_ellipse_x = ellipse_x*np.cos(rotate_angle)-ellipse_y*np.sin(rotate_angle)
#            new_ellipse_y = ellipse_x*np.sin(rotate_angle)+ellipse_y*np.cos(rotate_angle)
#            
#            ellipse_x = new_ellipse_x+ellipse_center[0]
#            ellipse_y = new_ellipse_y+ellipse_center[1]
#            
#            pop_err_vects = np.zeros((2,2))
#            pop_err_vects[0] = np.sqrt(pop_eig_vals[0])*pop_eig_vects[:,0]
#            pop_err_vects[1] = np.sqrt(pop_eig_vals[1])*pop_eig_vects[:,1]
#            
#            plt.figure(figsize=(10,6))
#            len_data_samps = data_samples.shape[0]*data_samples.shape[1]
#            plt.xlabel(r'$\mu_{\mathrm{RA}}$ (mas/yr)')
#            plt.ylabel(r'$\mu_{\mathrm{Dec}}$ (mas/yr)')
#            ax = plt.gca()
#            ax.set_aspect('equal')
#            for data_ind in range(len(data_samples[0])):
#                curr_x,curr_y = curr_vals[data_ind,0,0],curr_vals[data_ind,1,0]
#                plt.scatter(curr_x,curr_y,
#                            marker='o',alpha=alpha,s=50,color='r')
#                for err_ind in range(n_params):
#                    curr_vect = curr_err_vects[data_ind,err_ind]
#                    x_vals = [curr_x-curr_vect[0],curr_x+curr_vect[0]]
#                    y_vals = [curr_y-curr_vect[1],curr_y+curr_vect[1]]
#                    plt.plot(x_vals,y_vals,color='r',lw=2,alpha=alpha)
#            plt.plot(ellipse_x,ellipse_y)
#            for err_ind in range(len(pop_err_vects)):
#                plt.plot([ellipse_center[0],ellipse_center[0]+pop_err_vects[err_ind,0]],
#                         [ellipse_center[1],ellipse_center[1]+pop_err_vects[err_ind,1]])
#            plt.axhline(0,c='k',lw=0.5,ls='--',alpha=0.7)
#            plt.axvline(0,c='k',lw=0.5,ls='--',alpha=0.7)
#            plt.savefig(f'{outpath}{image_name}_posterior_population_PM_offset_analysis_pop_dist.png')
#            plt.close('all')
#            # plt.show()
#            
#            plt.figure(figsize=(10,6))
#            pop_samps = stats.multivariate_normal(median_mu,median_cov).rvs(10000)
#            len_data_samps = len(pop_samps)
#            plt.hist2d(pop_samps[:,0],pop_samps[:,1],
#                       norm=mcolors.PowerNorm(0.3),bins=50,
#                       weights=np.ones(len_data_samps)/len_data_samps)
#            plt.xlabel(r'$\mu_{\mathrm{RA}}$ (mas/yr)')
#            plt.ylabel(r'$\mu_{\mathrm{Dec}}$ (mas/yr)')
#            ax = plt.gca()
#            ax.set_aspect('equal')
#            plt.colorbar(label='Bin Probability')
#            for data_ind in range(len(data_samples[0])):
#                curr_x,curr_y = curr_vals[data_ind,0,0],curr_vals[data_ind,1,0]
#                plt.scatter(curr_x,curr_y,
#                            marker='o',alpha=alpha,s=50,color='r')
#                for err_ind in range(n_params):
#                    curr_vect = curr_err_vects[data_ind,err_ind]
#                    x_vals = [curr_x-curr_vect[0],curr_x+curr_vect[0]]
#                    y_vals = [curr_y-curr_vect[1],curr_y+curr_vect[1]]
#                    plt.plot(x_vals,y_vals,color='r',lw=2,alpha=alpha)
#            plt.plot(ellipse_x,ellipse_y)
#            for err_ind in range(len(pop_err_vects)):
#                plt.plot([ellipse_center[0],ellipse_center[0]+pop_err_vects[err_ind,0]],
#                         [ellipse_center[1],ellipse_center[1]+pop_err_vects[err_ind,1]])
#            plt.close('all')
#            # plt.show()
            
    if not skip_fitting:
        del samplerChain
        del samples
        del lnposts
        del accept_fracs
        
        del sample_lnposts
        del sample_parallaxes
        del sample_pms
        del sample_offsets
        del sample_med
        del sample_cov
        del best_trans_params
        del best_trans_param_covs
        del ags
        del bgs
        del cgs
        del dgs
        del samples_with_skews
        del sample_pms_parallax_offsets
        del data_combined
        del indv_image_source_data
        
        del gaia_id
        del sort_gaia_id_inds
        del x
        del y
        del x_hst_err
        del y_hst_err
        del x_g
        del y_g
        del use_for_fit
        del q_hst
        del img_nums
        del indv_orig_pixel_scales
        del hst_time_strings
        del hst_times
        del gaia_time_strings
        del gaia_times
        del delta_times
        del gaia_pm_xs
        del gaia_pm_ys
        del gaia_pm_x_errs
        del gaia_pm_y_errs
        del gaia_pm_x_y_corrs
        del gaia_ras
        del gaia_decs
        del gaia_ra_errs
        del gaia_dec_errs
        del gaia_ra_dec_corrs
        del gaia_parallaxes
        del gaia_parallax_errs
        del gaia_vectors            
        del gaia_vector_covs    
    
        del star_hst_gaia_pos_cov
        del hst_covs
        del star_ratios
        del matrices
        del matrices_T
        del star_hst_gaia_pos_inv_cov
        del proper_offset_jacs
        del star_hst_gaia_pos
        del inv_jac_dot_d_ij
        del summed_jac_V_data_inv_jac
        del Sigma_theta_i_inv
        del Sigma_theta_i
        del parallax_offset_vector
        del jac_V_data_inv_jac_dot_parallax_vects
        del summed_jac_V_data_inv_jac_dot_parallax_vects
        del jac_V_data_inv_jac_dot_d_ij
        del summed_jac_V_data_inv_jac_dot_d_ij
        del summed_jac_V_data_inv_jac_times
        del A_mu_i
        del C_mu_ij
        del A_mu_i_inv
        del C_mu_ij_inv
        del unique_gaia_offset_inv_covs
        del unique_gaia_pm_inv_covs
                        
        del Sigma_mu_theta_i_inv
        del Sigma_mu_d_ij_inv
        del Sigma_mu_i_inv 
        del Sigma_mu_i
        del A_plx_mu_i
        del B_plx_mu_i
        del Sigma_mu_theta_i_inv_dot_A_mu_i_inv
        del Sigma_mu_d_ij_inv_dot_C_mu_ij_inv
        del C_plx_mu_i
        del D_plx_mu_i
        del E_plx_theta_i
        del F_plx_theta_i
        del G_plx_d_ij
        del H_plx_d_ij
        del G_plx_d_ij_T_dot_V_data_inv
        del ivar_plx_d_ij
        del mu_times_ivar_plx_d_ij
        del summed_ivar_plx_d_ij
        del summed_mu_times_ivar_plx_d_ij
        del C_plx_mu_i_T_dot_V_mu_i_inv
        del ivar_plx_mu_i
        del mu_times_ivar_plx_mu_i
        del C_plx_mu_i_T_dot_V_mu_global_inv
        del ivar_plx_mu_global
        del mu_times_ivar_plx_mu_global
        del E_plx_theta_i_T_dot_V_theta_i_inv
        del ivar_plx_theta_i
        del mu_times_ivar_plx_theta_i
        del ivar_plx_i
        del var_plx_i
        del std_plx_i
        del mu_plx_i
        del parallax_draws
        del B_mu_i
        del mu_mu_i 
        del pm_gauss_draws
        del pm_draws
        del mu_theta_i
        del offset_gauss_draws
        del offset_draws
        del data_pm_draws
        del eig_signs
        del eig_vals
        del eig_vects
        del data_gauss_draws
        del data_draws
        del pm_data_measures
        del data_diff_vals
        del dpixels
        del dpixels_sigma_dists
        del new_offset_xy_samps
        del new_offset_samps
        del new_offset_sigma_samps
        del new_offset_summary
        
    gc.collect()
            
    return

                


# In[]:


def delta_ra_dec_per_parallax(hst_time,gaia_time,ra,dec):
    #THIS CODE IS REPURPOSED FROM CODE FROM MELODIE KAO
    
    #choose any parallax because we will scale by it later
    parallax = 1.0*u.mas
    distance = (1/parallax.value)*u.kpc
    delta_time = (gaia_time-hst_time).to(u.year).value
    dates = hst_time+np.array([0,delta_time])*u.year
    
    sun_loc = astropy.coordinates.get_sun(dates)
    
    sun_skycoord = SkyCoord(frame='gcrs', obstime=dates,
                            ra = sun_loc.ra, dec = sun_loc.dec)
    sun_eclon = sun_skycoord.geocentrictrueecliptic.lon
#    sun_eclat = sun_skycoord.geocentrictrueecliptic.lat

#     coord_gaia = SkyCoord( ra          = ra*u.deg,
#                            dec         = dec*u.deg,
#                            distance    = distance*u.kpc,
#                            obstime     = gaia_ref_epoch)

    coord_gaia = SkyCoord( ra          = ra*u.deg,
                           dec         = dec*u.deg,
                           distance    = distance,
#                            pm_ra_cosdec= c.icrs.pm_ra_cosdec,
#                            pm_dec      = c.icrs.pm_dec,
                           obstime     = gaia_time)

    # Get geocentric ecliptic coordinates of star after correcting for pm
    star_eclon = coord_gaia.geocentrictrueecliptic.lon
    star_eclat = coord_gaia.geocentrictrueecliptic.lat

    plx_delta_eclon = -parallax * np.sin(star_eclon - sun_eclon) / np.cos(star_eclat)
    plx_delta_eclat = -parallax * np.cos(star_eclon - sun_eclon) * np.sin(star_eclat)

#    offset_lon = plx_delta_eclon - plx_delta_eclon[1] #the second date is Gaia
#    offset_lat = plx_delta_eclat - plx_delta_eclat[1]

#    offset_total = np.sqrt(np.power(offset_lon,2)+np.power(offset_lat,2))

    #### COMPUTE 	parallax and proper motion offsets in equatorial coordinates
    # Ecliptic coordinates: pm-corrected location of star + the parallax offset at each date
    coord_plx = astropy.coordinates.GeocentricTrueEcliptic(
                    lon = star_eclon + plx_delta_eclon,
                    lat = star_eclat + plx_delta_eclat)

    # Transform the pm-corrected location of the star + parallax offset at each date to ICRS coordinates (ra/dec)
    coord_plx_icrs = coord_plx.transform_to(astropy.coordinates.ICRS)
    # FIX THIS AT SOME POINT
    # WARNING: AstropyDeprecationWarning: Transforming a frame instance to a frame class 
    # (as opposed to another frame instance) will not be supported in the future.  
    # Either explicitly instantiate the target frame, or first convert the 
    # source frame instance to a `astropy.coordinates.SkyCoord` and use its
    # `transform_to()` method. [astropy.coordinates.baseframe]

    plx_delta_ra_dec = coord_gaia.spherical_offsets_to(coord_plx_icrs)
    plx_delta_ra   = plx_delta_ra_dec[0]
    plx_delta_dec  = plx_delta_ra_dec[1]

    offset_ra = plx_delta_ra - plx_delta_ra[1] #the second date is Gaia
    offset_dec = plx_delta_dec - plx_delta_dec[1]
#    offset_ra_dec_total = np.sqrt(np.power(offset_ra,2)+np.power(offset_dec,2))

    offset_ra_div_para = offset_ra.to(u.mas)/parallax
    offset_dec_div_para = offset_dec.to(u.mas)/parallax
#    offset_ra_dec_total_div_para = np.sqrt(np.power(offset_ra_div_para,2)+np.power(offset_dec_div_para,2))    
    
    return np.array([offset_ra_div_para.value[0],offset_dec_div_para.value[0]])

#transformation from RA,Dec to XY plane (both XY and RA,Dec in radians)
def offset_jac(ra,dec,ra0,dec0):
    #jacobian to transform deltaRA,deltaDec to deltaX,deltaY (both in radians)
    denom = np.cos(dec0)*np.cos(dec)*np.cos(ra-ra0)+np.sin(dec)*np.sin(dec0)
    if np.isinf(denom):
        denom = np.sign(denom)*1e-100
    ddenom_dra = (-1*denom**-2*(-1*np.cos(dec0)*np.cos(dec)*np.sin(ra-ra0)))
    ddenom_ddec = (-1*denom**-2*(-1*np.cos(dec0)*np.sin(dec)*np.cos(ra-ra0)+np.cos(dec)*np.sin(dec0)))
    dxdra = (-1*np.cos(dec)*np.cos(ra-ra0))/denom+\
            (-1*np.cos(dec)*np.sin(ra-ra0))*ddenom_dra
    dxddec = (-1*-1*np.sin(dec)*np.sin(ra-ra0))/denom+\
             (-1*np.cos(dec)*np.sin(ra-ra0))*ddenom_ddec
    dydra = (-1*np.sin(dec0)*np.cos(dec)*np.sin(ra-ra0))/denom+\
            (np.sin(dec0)*np.cos(dec)*np.cos(ra-ra0)-np.cos(dec0)*np.sin(dec))*ddenom_dra
    dyddec = (-1*np.sin(dec0)*np.sin(dec)*np.cos(ra-ra0)-np.cos(dec0)*np.cos(dec))/denom+\
             (np.sin(dec0)*np.cos(dec)*np.cos(ra-ra0)-np.cos(dec0)*np.sin(dec))*ddenom_ddec
    jac = np.array([[dxdra,dxddec],[dydra,dyddec]])
    return jac


    # In[2]:

if __name__ == '__main__':
    gaiahub_single_BPMs(sys.argv[1:])
    pass
    
#    testing = False
##    testing = True
#    
#    if not testing:
#        gaiahub_BPMs(sys.argv[1:])
#    else:
#        overwrite = True
#        overwrite_GH_summaries = False
#        path = '/Volumes/Kevin_Astro/Astronomy/HST_Gaia_PMs/GaiaHub_results/'
#        
#    #    field = 'Fornax_dSph'
#        field = 'COSMOS_field'
#    
#        thresh_time = ((datetime.datetime(2023,5,22,15,20,38,259741)-datetime.datetime.utcfromtimestamp(0)).total_seconds()+7*3600)
#        if field in ['COSMOS_field']:
#            thresh_time = ((datetime.datetime(2023, 6, 16, 15, 47, 19, 264136)-datetime.datetime.utcfromtimestamp(0)).total_seconds()+7*3600)
#    
#    #    analyse_images(['j8pu0bswq'],
#    #                   field,path,
#    #                   overwrite_previous=True,overwrite_GH_summaries=False,thresh_time=thresh_time)
##        analyse_images(['j8pu0bswq','j8pu0bsyq'],
##                       field,path,
##                       overwrite_previous=True,overwrite_GH_summaries=False,thresh_time=thresh_time)
#    #    analyse_images(['j8pu0bswq','j8pu0bsyq','j8pu0bt1q','j8pu0bt5q'],
#    #                   field,path,
#    #                   overwrite_previous=True,overwrite_GH_summaries=False,thresh_time=thresh_time)
#        
#        
#        
    



            