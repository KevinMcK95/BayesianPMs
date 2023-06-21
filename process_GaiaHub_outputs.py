#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np

import os

import matplotlib
#import matplotlib.pyplot as plt
#import matplotlib.colors as mcolors
#import matplotlib.gridspec as gridspec

import pandas as pd
import scipy
import scipy.stats as stats

#import emcee
#import corner
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
#from multiprocessing import Pool, cpu_count
#from concurrent.futures import ThreadPoolExecutor
# from multiprocessing import Process, Manager
# import multiprocessing
# import itertools    


# In[]:


font = {'family' : 'serif',
#        'weight' : 'bold',
        'size'   : 16,}
matplotlib.rc('font', **font)
    
gaia_dr3_date = '2017-05-28'
gaia_dr3_time = Time(gaia_dr3_date)

pixel_scale_ratios = {'ACS':50,'WFC3':40} #mas/pixel

def get_matrix_params(on_skew,off_skew,ratio,rot):
    '''
    Uses skew, rotation, and ratio terms to measure transformation matrix parameters
    '''
    
    #on_skew = (a-d)/2
    #off_skew = (b+c)/2
    #ratio^2 = a*d-b*c
    #tan(rot) = (b-c)/(a+d)
    
    tan_theta = np.tan(rot/180*np.pi)
    sqrt_term = np.sqrt((tan_theta**2+1)*(ratio**2+on_skew**2+off_skew**2))
    sign = np.sign(rot)
    
    a = sign*((sqrt_term+tan_theta**2*on_skew+on_skew)/(tan_theta**2+1))
    b = sign*((tan_theta*sqrt_term+tan_theta**2*off_skew+off_skew)/(tan_theta**2+1))
    c = sign*(-(tan_theta*sqrt_term)/(tan_theta**2+1)+off_skew)
    d = sign*((sqrt_term)/(tan_theta**2+1)-on_skew)
    
    return a,b,c,d


gaia_labels = ['source_id','ref_epoch','gmag','gmag_error',\
               'HST_image','hst_gaia_pmra_wmean','hst_gaia_pmdec_wmean',\
               'hst_gaia_pmra_wmean_error','hst_gaia_pmdec_wmean_error',\
               'xc_hst_mean_error','yc_hst_mean_error']

#get all the following values, their errors, and their pairwise correlations
terms_for_correlation = ['ra','dec','parallax','pmra','pmdec']
correlation_names = []
for i in range(len(terms_for_correlation)):
    correlation_names.append(terms_for_correlation[i])
    correlation_names.append(f'{terms_for_correlation[i]}_error')
    for j in range(i+1,len(terms_for_correlation)):
        correlation_names.append(f'{terms_for_correlation[i]}_{terms_for_correlation[j]}_corr')
gaia_labels.extend(correlation_names)

def collect_gaiahub_results(field,
                            path='/Volumes/Kevin_Astro/Astronomy/HST_Gaia_PMs/GaiaHub_results/',
#                            overwrite=False):
                            overwrite=True):
    '''
    Finds all GaiaHub output files in field along path and collects them together for Bayesian PM analysis
    '''    
    
    outpath = f'{path}{field}/Bayesian_PMs/'
    
    if not os.path.isdir(outpath):
        os.makedirs(outpath)
    
    if os.path.isfile(f'{outpath}gaiahub_image_transformation_summaries.csv') and (not overwrite):
        print(f'SKIPPING: Found previous GaiaHub transformation summary file for field {field}.')
        return
    
    datapath = f'{path}{field}/HST/mastDownload/HST/'
    all_poss_image_names = os.listdir(datapath)
    hst_image_paths = []
    print(f'Found {len(all_poss_image_names)} possible HST images in field {field}.')

    #find HST images in this field that were able to be analyzed by GaiaHub
    for image_name in all_poss_image_names:
        trans_file = f'{datapath}{image_name}/{image_name}_flc_6p_transformation.txt'
        lnk_file = f'{datapath}{image_name}/{image_name}_flc.LNK'
        if not os.path.isfile(trans_file) or not os.path.isfile(lnk_file):
            continue
        hst_image_paths.append(trans_file)
        
    print(f'{len(hst_image_paths)}/{len(all_poss_image_names)} are possibly useful HST images.')

    #get the HST observation times of all possible images in this field
    #hst_image_times = {}
    hst_image_obsids = {}
    csvpath = f'{path}{field}/HST/'
    for csv_name in os.listdir(csvpath):
        if (field not in csv_name) or ('_data_products.csv' not in csv_name):
            continue
        if 'OLD_' == csv_name[:4]:
            continue

        with open(f'{csvpath}{csv_name}','r') as f:
            lines = f.readlines()
        #find the index where the obs_time is stored:
        variable_names = np.array(lines[0][:-1].split(','))
    #    obs_time_ind = np.where(variable_names == 'obs_time')[0][0]-len(variable_names)
        obs_id_ind = np.where(variable_names == 'obs_id')[0][0]
        parent_obs_id_ind = np.where(variable_names == 'parent_obsid')[0][0]
        
        for j in range(1,len(lines)):
            split_vals = lines[j][:-1].split(',')
            curr_obs_id = split_vals[obs_id_ind]
    #        curr_obs_time = split_vals[obs_time_ind]
            curr_parent_obs_id = split_vals[parent_obs_id_ind]
            
            if curr_obs_id not in hst_image_obsids:
                hst_image_obsids[curr_obs_id] = curr_parent_obs_id
    #        if curr_obs_id not in hst_image_times:
    #            hst_image_times[curr_obs_id] = curr_obs_time
    
    #had to add "ref_epoch" to GaiaHub's call to the Gaia servers        
    gaia_measures = {}
    #gaia_labels = ['source_id','ra','dec','parallax','pmra','pmdec','ref_epoch',\
    #               'ra_error','dec_error','ra_dec_corr','parallax_error',\
    #               'pmra_error','pmdec_error','pmra_pmdec_corr',\
    #               'gmag','gmag_error','ra_parallax_corr','dec_parallax_corr',\
    #               'HST_image','hst_gaia_pmra_wmean','hst_gaia_pmdec_wmean',\
    #               'hst_gaia_pmra_wmean_error','hst_gaia_pmdec_wmean_error']
    
    for label in gaia_labels:
        gaia_measures[label] = []
    
    resultpath = f'{path}{field}/'
    for gaia_name in os.listdir(resultpath):
        if (field not in gaia_name) or ('.csv' not in gaia_name) or ('._' == gaia_name[:2]) or ('_used_HST_images.csv' in gaia_name):
            continue
        if 'OLD_' == gaia_name[:4]:
            continue
        gaia_data = pd.read_csv(f'{resultpath}{gaia_name}')
#        print(f'{resultpath}{gaia_name}')
        for j in range(len(gaia_data)):
#            if gaia_data['source_id'][j] not in gaia_measures['source_id']:
            if gaia_data['source_id'][j] not in gaia_measures['source_id']:
                for label in gaia_labels:
                    if label == 'ref_epoch':
                        if label not in gaia_data:
                            gaia_measures[label].append(2016.0)
                        else:
                            gaia_measures[label].append(gaia_data[label][j])
                    else:
                        gaia_measures[label].append(gaia_data[label][j])
    
    for label in gaia_labels:
        gaia_measures[label] = np.array(gaia_measures[label])
    
    hst_image_tbaselines = {}
    for gaia_name in os.listdir(resultpath):
        if (field not in gaia_name) or ('.csv' not in gaia_name) or ('._' == gaia_name[:2]) or ('_used_HST_images.csv' not in gaia_name):
            continue
        if 'OLD_' == gaia_name[:4]:
            continue
        hst_image_data = pd.read_csv(f'{resultpath}{gaia_name}')
        for j in range(len(hst_image_data)):
            if (hst_image_data['obs_id'][j] not in hst_image_tbaselines):
                hst_image_tbaselines[hst_image_data['obs_id'][j]] = hst_image_data['t_baseline'][j]
    
        
#    print(len(hst_image_paths),hst_image_paths[0])
    unique_images = []
    for name in hst_image_paths:
        unique_images.append(name.split('/')[-2])
    
    hst_image_mjds = {}
    hst_image_exp_info = {}
    
    for j,name in enumerate(hst_image_paths):
        curr_path = '/'.join(name.split('/')[:-1])
        curr_image = unique_images[j]
        hdul = fits.open(f'{curr_path}/{curr_image}_flc.fits')
        exp_start = hdul[0].header['EXPSTART'] #mjd
        exp_end = hdul[0].header['EXPEND'] #mjd
        ave_exp_mjd = 0.5*(exp_start+exp_end)
        exp_time = hdul[0].header['EXPTIME']
        ra_targ = hdul[0].header['RA_TARG']
        dec_targ = hdul[0].header['DEC_TARG']
        hdul.close()
            
        hst_image_mjds[curr_image] = ave_exp_mjd
        hst_image_exp_info[curr_image] = [ra_targ,dec_targ,exp_start,exp_end,exp_time]
        
#    for curr_image in all_poss_image_names:
#        if curr_image in hst_image_mjds:
#            continue
#        if 'hst_' == curr_image[:4]:
#            continue
#        if not os.path.isfile(f'{datapath}{curr_image}/{curr_image}_flc.fits'):
#            continue
#        
#        hdul = fits.open(f'{datapath}{curr_image}/{curr_image}_flc.fits')
#        exp_start = hdul[0].header['EXPSTART'] #mjd
#        exp_end = hdul[0].header['EXPEND'] #mjd
#        ave_exp_mjd = 0.5*(exp_start+exp_end)
#        exp_time = hdul[0].header['EXPTIME']
#        ra_targ = hdul[0].header['RA_TARG']
#        dec_targ = hdul[0].header['DEC_TARG']
#        hdul.close()
#            
#        hst_image_mjds[curr_image] = ave_exp_mjd
#        hst_image_exp_info[curr_image] = [ra_targ,dec_targ,exp_start,exp_end,exp_time]
                
    unique_images = np.unique(unique_images)
            
    
    #####EXAMPLE OF MATRIX TRANSFORMATION SECTION##########
    #             TRANSFORMATION MATRIX: X_2 = AG*(X_1-Xo)+BG*(Y_1-Yo)+Wo
    #                                    Y_2 = CG*(X_1-Xo)+DG*(Y_1-Yo)+Zo
    #
    #                                AG:  0.13445487019
    #                                BG:  0.98645672253
    #                                CG: -0.98680188642
    #                                DG:  0.13332047009
    #                                Xo:      2139.0912
    #                                Yo:      1624.4170
    #                                Wo:      4658.3364
    #                                Zo:      4915.2915
    #
    #                    ROTATION (deg):  82.27
    #                 PIXEL-SCALE RATIO:  0.99567211
    #    REAL IMG PIXEL SCALE (mas/pix): 49.78360526
    #
    #                      MAGNITUDE ZP:  33.13
    ####From Jay Anderson's paper:
    #   rotation = arctan(B-C,A+D)
    #   pixel_scale_ratio = sqrt(A*D-B*C)
    #   on_axis_skew = 0.5*(A-D)
    #   off_axis_skew = 0.5*(B+C)
    
    trans_file_summaries = {}
    
    gaiahub_output_info = {}
    
    print(f'Reading in data for {len(hst_image_paths)} images in field {field}.')
        
    good_count = 0
    for _,trans_file in enumerate(tqdm(hst_image_paths,total=len(hst_image_paths))):
        mask_name = field
        image_name = trans_file.split('/')[-2]
#        mask_name = image_name
        if mask_name not in trans_file_summaries:
            trans_file_summaries[mask_name] = {'image_name':[],
                                               'ra':[],'dec':[],'n_stars':[],
                                               'AG':[],'BG':[],'CG':[],'DG':[],
                                               'Xo':[],'Yo':[],'Wo':[],'Zo':[],
                                               'rot':[],'pix_scale_ratio':[],'real_img_pix_scale':[],
                                               'mag_zp':[], 
                                               'on_axis_skew':[],'off_axis_skew':[],'rotate_mult_fact':[],
                                               'orig_rot':[],'orig_pixel_scale':[],'obsid':[],'exptime':[],
                                               'HST_time':[]}
#            trans_file_summaries[mask_name] = {}
    
            #X in Gaia pixels of cross matched star
            #Y in Gaia pixels of cross matched star
            #X in HST pixels of original star
            #Y in HST pixels of original star
            #Gaia G magnitude
            #HST magnitude
            #difference in Gaia X and HST transformed to Gaia X
            #difference in Gaia Y and HST transformed to Gaia Y
            #HST X transformed to Gaia X
            #HST Y transformed to Gaia Y
            gaiahub_output_info[mask_name] = {'Gaia_id':[],'x_hst_err':[],'y_hst_err':[],
                                              'X_G':[],'Y_G':[],'X':[],
                                              'Y':[],'g_mag':[],'mag':[],'dX_G':[],
                                              'dY_G':[],'X_2':[],'Y_2':[],'use_for_fit':[],'q_hst':[],
                                              'gaiahub_pm_x':[],'gaiahub_pm_y':[],
                                              'gaiahub_pm_x_err':[],'gaiahub_pm_y_err':[],
                                              'stationary':[],'hst_images':[],'Gaia_time':[]
                                              }
            for label in correlation_names:
                gaiahub_output_info[mask_name][label] = []
            
        trans_path = '/'.join(trans_file.split('/')[:-1])
        
        ra,dec = np.nan,np.nan
        ag,bg,cg,dg,xo,yo,wo,zo,rot,pix_scale_rat,pix_scale,mag_zp = np.zeros(12)*np.nan
#        mjd = hst_image_mjds[image_name]
        
        #read in Andres' transformation parameter file, looking for the lines that
        #contain the RA, Dec of the image as well as the transformation parameters
        bad_file = False
        with open(trans_file,'r') as f:
            try:
                lines = f.readlines()
            except:
                bad_file = True
        if bad_file:
            continue
        
    #     print(trans_file)
        in_position_info = False
        in_trans_info = False
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            elif 'THE FOLLOWING LIMITS FOR HST' in line:
                saturation_limit = float(line.split('<')[1].split(',')[0])
                q_hst_limit = float(line.split(',')[1].split('<')[1].split(')')[0])
#            elif 'Delta_Time' in line:
#                delta_time = float(line.split(':')[1])
            elif 'MASTER FRAME INFO' in line:
                in_position_info = True #start region for position of (RA,Dec) of image
            elif 'GAIA INFO' in line:
                in_position_info = False #end region for position of (RA,Dec) of image
            elif 'NEW TRANSFORMATIONS AFTER CLIP' in line:
                in_trans_info = True #start region for transformation parameters
            elif 'SAVING MAT FILE' in line:
                in_trans_info = False #end region for transformation parameters
                break #stop looking through lines
            elif in_position_info:
                if 'RA_CENTER' in line:
                    ra = float(line.split(':')[1])
                elif 'DEC_CENTER' in line:
                    dec = float(line.split(':')[1])   
                elif 'SCALE (mas/pix)' in line:
                    orig_pixel_scale = float(line.split(':')[1])   
            elif in_trans_info:
                if 'STARS FOUND IN COMMON:' in line:
                    n_stars = int(line.split(':')[1])
                elif 'AG:' in line:
                    ag = float(line.split(':')[1])
                elif 'BG:' in line:
                    bg = float(line.split(':')[1])
                elif 'CG:' in line:
                    cg = float(line.split(':')[1])
                elif 'DG:' in line:
                    dg = float(line.split(':')[1])
                elif 'Xo:' in line:
                    xo = float(line.split(':')[1])
                elif 'Yo:' in line:
                    yo = float(line.split(':')[1])
                elif 'Wo:' in line:
                    wo = float(line.split(':')[1])
                elif 'Zo:' in line:
                    zo = float(line.split(':')[1])
                elif 'ROTATION' in line:
                    rot = float(line.split(':')[1])
                elif 'PIXEL-SCALE RATIO' in line:
                    pix_scale_rat = float(line.split(':')[1])
                elif 'REAL IMG PIXEL SCALE' in line:
                    test_val = line.split(':')[1]
                    if '*' in test_val:
                        pix_scale = np.nan
                        print(trans_file,line)
                    else:
                        pix_scale = float(test_val)
                elif 'MAGNITUDE ZP' in line:
                    mag_zp = float(line.split(':')[1])
            else:
                continue        
        
        if not np.isfinite(ag): #remove no-match (i.e. between HST and Gaia) images
            continue
    
        position_file = f"{trans_path}/{image_name}_flc.LNK"
        with open(position_file,'r') as f:
            lines = f.readlines()
        temp_params = []
        for line in lines:
            if len(line) == 0:
                continue
            elif line[0] == '#':
                continue
            else:
                line = np.array(line.split()).astype(float)
                temp_params.append(line)
        temp_params = np.array(temp_params)
    #    X_G,Y_G,X,Y,m_g,m_hst,dX_G,dY_G,X_2,Y_2 = temp_params.T
        X_G,Y_G,X,Y,m_g,m_hst,X_2,Y_2 = temp_params.T[[8,9,0,1,10,2,6,7]]
        q_hst = temp_params[:,3]
        keep_stars = (X > 0) | (Y > 0) #mask out bad matches
    #    plt.scatter(m_hst[keep_stars],q_hst[keep_stars])
    #    plt.gca().invert_xaxis()
    #    plt.xlabel(r'$m_{\mathrm{HST}}$ (mag)')
    #    plt.ylabel(r'$q_{\mathrm{HST}}$')
    #    plt.savefig('/Users/kevinm/Downloads/q_vs_mag.png')
    #    plt.show()
        
    #    if field == 'Sculptor_dSph':
    #        saturation_limit = 18.5
    #    elif field == 'Fornax_dSph':
    #        saturation_limit = 18.75
    #    elif field == 'Draco_dSph':
    #        saturation_limit = 19
    #    elif field == 'Sextans_dSph':
    #        saturation_limit = 18.75
    #    else:
    #        saturation_limit = 18
    #    keep_stars = keep_stars & (m_g > saturation_limit)
        keep_stars = keep_stars & (q_hst > 0)
    #    keep_stars = keep_stars & (m_g > 17)
        temp_params = temp_params[keep_stars]
        
            
        X_G,Y_G,X,Y,m_g,m_hst,X_2,Y_2 = temp_params.T[[8,9,0,1,10,2,6,7]]
        dX_G = X_G-X_2
        dY_G = Y_G-Y_2
        q_hst = temp_params[:,3]
        #stars that shouldn't be used to measure the transformation
        use_for_fit = (q_hst < q_hst_limit) & (m_hst < saturation_limit)
        
        ra_gaia = temp_params[:,15]
        dec_gaia = temp_params[:,16]
        source_id_inds = np.zeros(len(X)).astype(int)
        found_good = np.ones(len(X)).astype(bool)
        for j in range(len(ra_gaia)):
            #cross match to find the Gaia source ids of the targets in the LNK files
            ang_dist_2 = np.power(gaia_measures['ra']-ra_gaia[j],2)+np.power(gaia_measures['dec']-dec_gaia[j],2)
            if ang_dist_2.min() > (1e-5)**2:
                found_good[j] = False
            source_id_inds[j] = np.argmin(ang_dist_2)
            
        if not np.all(found_good):
            temp_params = temp_params[found_good]
            X_G,Y_G,X,Y,m_g,m_hst,X_2,Y_2 = np.array([X_G,Y_G,X,Y,m_g,m_hst,X_2,Y_2]).T[found_good].T
            dX_G = X_G-X_2
            dY_G = Y_G-Y_2
    
            ra_gaia = ra_gaia[found_good]
            dec_gaia = dec_gaia[found_good]
            source_id_inds = source_id_inds[found_good]
            found_good = found_good[found_good]
            
        if len(X_G) < 3:
            print(f'SKIPPING: Not enough stars in image {position_file} (N={len(X_G)})')
            continue
            
    #    if not np.all(found_good):
    #        print(f'SKIPPING: Could not find matches between catalogues for image {position_file}')
    #        continue
        
        if len(np.unique(source_id_inds)) != len(source_id_inds):
            print(f'SKIPPING: Found confusing matches between catalogues for image {position_file}')
            continue
        
        if image_name not in hst_image_obsids:
            print(f'SKIPPING: Could not find image {image_name} in GaiaHub output files.')
            continue
                
        gaiahub_output_info[mask_name]['X_G'].append(X_G)
        gaiahub_output_info[mask_name]['Y_G'].append(Y_G)
        gaiahub_output_info[mask_name]['X'].append(X)
        gaiahub_output_info[mask_name]['Y'].append(Y)
        gaiahub_output_info[mask_name]['g_mag'].append(m_g)
        gaiahub_output_info[mask_name]['mag'].append(m_hst)
        gaiahub_output_info[mask_name]['dX_G'].append(dX_G)
        gaiahub_output_info[mask_name]['dY_G'].append(dY_G)
        gaiahub_output_info[mask_name]['X_2'].append(X_2)
        gaiahub_output_info[mask_name]['Y_2'].append(Y_2)
        gaiahub_output_info[mask_name]['q_hst'].append(q_hst)
        gaiahub_output_info[mask_name]['use_for_fit'].append(use_for_fit)
        
        star_gaia_times = gaia_measures['ref_epoch'][source_id_inds]
    #    star_gaia_times = np.ones_like(star_gaia_times)*gaia_dr3_time.jyear #WANT TO CHECK THIS!
        
        star_hst_images = gaia_measures['HST_image'][source_id_inds]
#        star_hst_times = []
#        
#        for j in range(len(star_gaia_times)):
#            curr_hst_images = star_hst_images[j].split()
#            curr_hst_image_times = {}
#    
#            for hst_image_name in curr_hst_images:
#                real_name = hst_image_name.split('_flc')[0]
#                    
#                if real_name in hst_image_mjds:
#                    curr_hst_image_times[real_name] = hst_image_mjds[real_name]
#    #            elif real_name in hst_image_tbaselines:
#    #                curr_hst_image_times[real_name] = star_gaia_times[j]-hst_image_tbaselines[real_name]
#                else:
#                    curr_hst_image_times[real_name] = np.nan
#            star_hst_times.append(curr_hst_image_times)
            
                    
    #    #MIGHT WANT TO CHECK ON THIS PART!!!
    #    bad_file = False
    #    star_hst_times = np.copy(star_gaia_times).astype(str)
    #    for j in range(len(star_gaia_times)):
    #        curr_hst_images = star_hst_images[j].split()
    #        for hst_image_name in curr_hst_images:
    #            real_name = hst_image_name.split('_flc')[0]
    #            if real_name in hst_image_times:
    #                star_hst_times[j] = hst_image_times[real_name]  
        
        
        gaiahub_output_info[mask_name]['Gaia_time'].append(star_gaia_times)
        
        for label in correlation_names:
            gaiahub_output_info[mask_name][label].append(gaia_measures[label][source_id_inds])
                            
        star_names = gaia_measures['source_id'][source_id_inds]
            
        gaiahub_output_info[mask_name]['Gaia_id'].append(star_names)
        gaiahub_output_info[mask_name]['x_hst_err'].append(gaia_measures['xc_hst_mean_error'][source_id_inds])  
        gaiahub_output_info[mask_name]['y_hst_err'].append(gaia_measures['yc_hst_mean_error'][source_id_inds])
    
        gaiahub_output_info[mask_name]['hst_images'].append(star_hst_images)
        gaiahub_output_info[mask_name]['gaiahub_pm_x'].append(gaia_measures['hst_gaia_pmra_wmean'][source_id_inds])  
        gaiahub_output_info[mask_name]['gaiahub_pm_y'].append(gaia_measures['hst_gaia_pmdec_wmean'][source_id_inds])  
        gaiahub_output_info[mask_name]['gaiahub_pm_x_err'].append(gaia_measures['hst_gaia_pmra_wmean_error'][source_id_inds])  
        gaiahub_output_info[mask_name]['gaiahub_pm_y_err'].append(gaia_measures['hst_gaia_pmdec_wmean_error'][source_id_inds])  
            
        #check if there is file labelling points as stationary, but if it doesn't exist, assume none are stationary
        if os.path.isfile(f"{trans_path}/{image_name}_stationary_points.npy"):
            stationary = np.load(f"{trans_path}/{image_name}_stationary_points.npy")
        else:
            stationary = np.zeros(len(X)).astype(bool)
        gaiahub_output_info[mask_name]['stationary'].append(stationary)
    
        n_stars = len(X) #use the true number instead of the ones used in to calculate the transformation by GaiaHub
    
        trans_file_summaries[mask_name]['orig_rot'].append(rot)
    
        rotation_sign = 1
    #     if rot < 0:
    #         rot = rot%180
    #         rotation_sign = -1
    #    rotation_sign = -1
        
        new_ag = ag*rotation_sign
        new_bg = bg*rotation_sign
        new_cg = cg*rotation_sign
        new_dg = dg*rotation_sign
        on_axis_skew = 0.5*(new_ag-new_dg)
        off_axis_skew = 0.5*(new_bg+new_cg)
    #    on_axis_skew = 0.5*(ag-dg)
    #    off_axis_skew = 0.5*(bg+cg)
    
        trans_file_summaries[mask_name]['ra'].append(ra)
        trans_file_summaries[mask_name]['dec'].append(dec)
        trans_file_summaries[mask_name]['AG'].append(new_ag)
        trans_file_summaries[mask_name]['BG'].append(new_bg)    
        trans_file_summaries[mask_name]['CG'].append(new_cg)
        trans_file_summaries[mask_name]['DG'].append(new_dg)    
        trans_file_summaries[mask_name]['rotate_mult_fact'].append(rotation_sign)
    #     trans_file_summaries[mask_name]['Xo'].append(new_xo)
    #     trans_file_summaries[mask_name]['Yo'].append(new_yo)    
        trans_file_summaries[mask_name]['Xo'].append(xo)
        trans_file_summaries[mask_name]['Yo'].append(yo)    
        trans_file_summaries[mask_name]['Wo'].append(wo)
        trans_file_summaries[mask_name]['Zo'].append(zo)    
        trans_file_summaries[mask_name]['rot'].append(rot)
        trans_file_summaries[mask_name]['pix_scale_ratio'].append(pix_scale_rat)    
        trans_file_summaries[mask_name]['real_img_pix_scale'].append(pix_scale)
        trans_file_summaries[mask_name]['mag_zp'].append(mag_zp)   
#        trans_file_summaries[mask_name]['mjd'].append(mjd)
        trans_file_summaries[mask_name]['image_name'].append(image_name)
        trans_file_summaries[mask_name]['on_axis_skew'].append(on_axis_skew)
        trans_file_summaries[mask_name]['off_axis_skew'].append(off_axis_skew)
        trans_file_summaries[mask_name]['n_stars'].append(n_stars)
        trans_file_summaries[mask_name]['orig_pixel_scale'].append(orig_pixel_scale) #changes based on instrument
        trans_file_summaries[mask_name]['obsid'].append(hst_image_obsids[image_name])
        trans_file_summaries[mask_name]['exptime'].append(hst_image_exp_info[image_name][-1])
        trans_file_summaries[mask_name]['HST_time'].append(hst_image_mjds[image_name])
        
    
#        matrix = np.array([[new_ag,new_bg],[new_cg,new_dg]])
#        inv_matrix = np.linalg.inv(matrix)
#        dXY_H = np.einsum('ij,nj->ni',inv_matrix,np.array((dX_G,dY_G)).T)
#        
#        for star_ind in range(len(X)):
#            star_name = star_names[star_ind]
#            if star_name not in star_hst_pix_offsets:
#                star_hst_pix_offsets[star_name] = {'m_G':[],'m_H':[],'dXY_H':[],'n_im':0,'image_num':[],'mjd':[]}
#            star_hst_pix_offsets[star_name]['n_im'] += 1
#            star_hst_pix_offsets[star_name]['m_G'].append(m_g[star_ind])
#            star_hst_pix_offsets[star_name]['m_H'].append(m_hst[star_ind])
#            star_hst_pix_offsets[star_name]['dXY_H'].append(dXY_H[star_ind])
#            star_hst_pix_offsets[star_name]['mjd'].append(mjd)
#            star_hst_pix_offsets[star_name]['image_num'].append(good_count)
            
        good_count += 1
        
    print(f'\nFound {good_count} useful images in field {field}.')
    
    print(f'Summarizing GaiaHub output files for {good_count} useful images in field {field}.')
    
    for mask in trans_file_summaries:
        for param in gaiahub_output_info[mask]:
            for image_ind in range(len(gaiahub_output_info[mask]['Gaia_id'])):
                gaiahub_output_info[mask][param][image_ind] = np.array(gaiahub_output_info[mask][param][image_ind])
            gaiahub_output_info[mask][param] = np.array(gaiahub_output_info[mask][param])
        for param in trans_file_summaries[mask]:
            trans_file_summaries[mask][param] = np.array(trans_file_summaries[mask][param])
                    
        for image_ind,image_name in enumerate(trans_file_summaries[mask]['image_name']):
            curr_dict = {}
            keys = gaiahub_output_info[mask].keys()
            for key in keys:
                curr_dict[key] = gaiahub_output_info[mask][key][image_ind]
            curr_df = pd.DataFrame.from_dict(curr_dict)
            curr_df.to_csv(f'{outpath}/{image_name}/{image_name}_gaiahub_source_summaries.csv',index=False)

        trans_file_df = pd.DataFrame.from_dict(trans_file_summaries[mask])
        trans_file_df.to_csv(f'{outpath}gaiahub_image_transformation_summaries.csv',index=False)
        
            
#    for star_name in star_hst_pix_offsets:
#        for param in star_hst_pix_offsets[star_name]:
#            star_hst_pix_offsets[star_name][param] = np.array(star_hst_pix_offsets[star_name][param])
                
    return

    # In[]:

def image_lister(field,path):
    '''
    For all useful HST images in field along path, determine which other images share sources, and save the list of common images.
    '''    
    
    outpath = f'{path}{field}/Bayesian_PMs/'
    
    if not os.path.isfile(f'{outpath}gaiahub_image_transformation_summaries.csv'):
        collect_gaiahub_results(field,path=path)
        
    trans_file_df = pd.read_csv(f'{outpath}gaiahub_image_transformation_summaries.csv')
    
    image_names = np.array(trans_file_df['image_name'].to_list())
    image_star_info = {}
    for image_name in image_names:
        image_star_info[image_name] = pd.read_csv(f'{outpath}/{image_name}/{image_name}_gaiahub_source_summaries.csv')
        
#    summary_file_time = os.path.getmtime(f'{outpath}gaiahub_image_transformation_summaries.csv')
            
    #find images that have some overlapping stars, so they should be fit together
    linked_image_lists = []
    star_names_in_images = {}
    
    print(f'Finding overlapping source lists for {len(image_names)} HST images in field {field}.')
    
#    image_order_inds = np.argsort(mjds)

#    for image_name in image_names:
    for _,image_name in enumerate(tqdm(image_names,total=len(image_names))):
        for star_ind,star_name in enumerate(image_star_info[image_name]['Gaia_id']):
            curr_images = image_star_info[image_name]['hst_images'][star_ind].split()
            temp_image_names = []
            temp_image_mjds = []
            for entry in curr_images:
                curr_image_name = entry.split('_flc')[0]
                if curr_image_name not in image_names:
                    continue
                temp_image_names.append(curr_image_name)
                temp_image_mjds.append(trans_file_df['HST_time'][np.where(trans_file_df['image_name'] == curr_image_name)[0][0]])
                if curr_image_name not in star_names_in_images:
                    star_names_in_images[curr_image_name] = []
                if star_name not in star_names_in_images[curr_image_name]:
                    star_names_in_images[curr_image_name].append(star_name)
            curr_images = np.array(temp_image_names)
            curr_image_mjds = np.array(temp_image_mjds)
            keep_images = np.zeros(len(curr_images)).astype(bool)
            for curr_image_ind,curr_image_name in enumerate(curr_images):
                if curr_image_name in image_names:
                    keep_images[curr_image_ind] = True
            curr_images = curr_images[keep_images]
            curr_image_mjds = curr_image_mjds[keep_images]
            ind_order = np.argsort(curr_image_mjds)
            curr_images = list(curr_images[ind_order])
            curr_image_mjds = list(curr_image_mjds[ind_order])
                        
        #    print(star_name,len(curr_images))
            if len(linked_image_lists) == 0:
                linked_image_lists.append(curr_images)
            else:
                #loop over the current image names to see if they are already in a list
                found_match = False
                for curr_image_name in curr_images:
                    for im_list_ind,image_list in enumerate(linked_image_lists):
                        if curr_image_name in image_list:
                            found_match = True
                            break
                    if found_match:
                        break
                if found_match:
                    #then make sure all the images in curr_images are in the corresponding list
                    for curr_image_name in curr_images:
                        if curr_image_name not in image_list:
                            linked_image_lists[im_list_ind].append(curr_image_name)
                else:
                    #no matches, so start a new list
                    linked_image_lists.append(curr_images)
    for image_name in star_names_in_images:
        star_names_in_images[image_name] = np.array(star_names_in_images[image_name])
        
        
    print(f'Found {len(linked_image_lists)} combination(s) of images in field {field}.')
    
    #for each image, look at the stars in that image to see which other
    #images also have those same stars (but don't check for all overlaps)
    indv_linked_image_lists = {}
    indv_linked_image_matches = {}
    indv_linked_image_dtimes = {}
    print(f'Saving linked image lists for each of the {len(image_names)} HST image(s).')

    for _,curr_image_name in enumerate(tqdm(image_names,total=len(image_names))):
        if curr_image_name not in star_names_in_images:
            continue
        curr_star_names = star_names_in_images[curr_image_name]
        curr_image_mjd = trans_file_df['HST_time'][np.where(trans_file_df['image_name'] == curr_image_name)[0][0]]
        
        indv_linked_image_lists[curr_image_name] = [curr_image_name]
        indv_linked_image_matches[curr_image_name] = [len(curr_star_names)]
        indv_linked_image_dtimes[curr_image_name] = [0.0]
        
        for other_image_name in image_names:
            if other_image_name == curr_image_name:
                continue
            elif other_image_name not in star_names_in_images:
                continue
            elif other_image_name in indv_linked_image_lists[curr_image_name]:
                #then it's already in the list, so don't worry about it
                continue
            
            other_star_names = star_names_in_images[other_image_name]
            shared_names = np.intersect1d(other_star_names,curr_star_names)
            if len(shared_names) == 0:
                #then no overlap
                continue
            
            other_image_mjd = trans_file_df['HST_time'][np.where(trans_file_df['image_name'] == other_image_name)[0][0]]
            
            indv_linked_image_lists[curr_image_name].append(other_image_name)
            indv_linked_image_matches[curr_image_name].append(len(shared_names))
            indv_linked_image_dtimes[curr_image_name].append((other_image_mjd-curr_image_mjd)/365.25)
            
        indv_linked_image_lists[curr_image_name] = np.array(indv_linked_image_lists[curr_image_name])
        indv_linked_image_matches[curr_image_name] = np.array(indv_linked_image_matches[curr_image_name])
        indv_linked_image_dtimes[curr_image_name] = np.array(indv_linked_image_dtimes[curr_image_name])
        
        sort_inds = np.argsort(indv_linked_image_dtimes[curr_image_name])
        indv_linked_image_lists[curr_image_name] = indv_linked_image_lists[curr_image_name][sort_inds]
        indv_linked_image_matches[curr_image_name] = indv_linked_image_matches[curr_image_name][sort_inds]
        indv_linked_image_dtimes[curr_image_name] = indv_linked_image_dtimes[curr_image_name][sort_inds]
        
        curr_dict = {'image_name':indv_linked_image_lists[curr_image_name],
                     'time_offset':indv_linked_image_dtimes[curr_image_name],
                     'n_common_sources':indv_linked_image_matches[curr_image_name]}
        curr_df = pd.DataFrame.from_dict(curr_dict)
        curr_df.to_csv(f'{outpath}/{curr_image_name}/{curr_image_name}_linked_images.csv',index=False)
                        
    
#    linked_image_list_matches = {} #pairwise number of shared targets between linked images
#    linked_image_list_dtimes = {} #pairwise times between linked images
#    sort_params = []
#    best_inds = []
#    for j in range(len(linked_image_lists)):
#        linked_image_list_matches[j] = np.zeros((len(linked_image_lists[j]),len(linked_image_lists[j]))).astype(int)
#        linked_image_list_dtimes[j] = np.zeros((len(linked_image_lists[j]),len(linked_image_lists[j])))
#        
#        for k in range(len(linked_image_lists[j])):
#            curr_image_name = linked_image_lists[j][k]
#            curr_image_mjd = trans_file_df['HST_time'][np.where(trans_file_df['image_name'] == curr_image_name)[0][0]]
#            linked_image_list_dtimes[j][k,k] = 0
#            if curr_image_name in star_names_in_images:
#                curr_image_star_names = star_names_in_images[curr_image_name]
#            else:
#                curr_image_star_names = np.array([])
#            linked_image_list_matches[j][k,k] = len(curr_image_star_names)
#    
#            for l in range(k+1,len(linked_image_lists[j])):
#                pair_image_name = linked_image_lists[j][l]
#                pair_image_mjd = trans_file_df['HST_time'][np.where(trans_file_df['image_name'] == pair_image_name)[0][0]]
#                linked_image_list_dtimes[j][k,l] = (curr_image_mjd-pair_image_mjd)/365.25
#                linked_image_list_dtimes[j][l,k] = -1*linked_image_list_dtimes[j][k,l]
#                if pair_image_name in star_names_in_images:
#                    pair_image_star_names = star_names_in_images[pair_image_name]
#                else:
#                    pair_image_star_names = np.array([])
#                shared_names = np.intersect1d(pair_image_star_names,curr_image_star_names)
#                linked_image_list_matches[j][k,l] = len(shared_names)
#                linked_image_list_matches[j][l,k] = linked_image_list_matches[j][k,l]
#                
#        
#        best_ind1 = np.argmax(np.max(linked_image_list_dtimes[j],axis=1))
#        best_ind2 = np.argmax(linked_image_list_dtimes[j][best_ind1])
#        sort_params.append(linked_image_list_dtimes[j][best_ind1,best_ind2])
#        best_inds.append([best_ind1,best_ind2])
#            
#    for j in range(len(linked_image_lists)):    
#        best_ind1,best_ind2 = best_inds[j]
#        print()
#        print(j,'Best image names:  ',linked_image_lists[j][best_ind1],linked_image_lists[j][best_ind2])
#        print(j,'Numbers of stars:  ',linked_image_list_matches[j][best_ind1,best_ind1],linked_image_list_matches[j][best_ind2,best_ind2])
#        print(j,'Max matching stars:',linked_image_list_matches[j][best_ind1,best_ind2])
#        print(j,'Max years offset:  ',linked_image_list_dtimes[j][best_ind1,best_ind2])
#        print(j,'Number of images:  ',len(linked_image_list_dtimes[j]))
#    print()            

#    return linked_image_lists
    return
    
    # In[3]:
    
if __name__ == '__main__':
    overwrite = True
    path = '/Volumes/Kevin_Astro/Astronomy/HST_Gaia_PMs/GaiaHub_results/'
    
    field = 'Fornax_dSph'
    field = 'COSMOS_field'
    
    collect_gaiahub_results(field,path=path,overwrite=overwrite)
    image_lister(field,path)
            
    
            
    # In[]:
    













        