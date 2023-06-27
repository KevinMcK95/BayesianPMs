#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 12:11:06 2023

@author: kevinm
"""

import numpy as np
import os
import pandas as pd
from tqdm import tqdm
#import matplotlib.pyplot as plt
import process_GaiaHub_outputs as process_GH

correlation_names = process_GH.correlation_names


def source_matcher(field,path):
    '''
    Using the outputs from the Bayesian analysis (or GaiaHub if it doesn't exist),
    loop over all the HST images in a field to find all the remaining 
    HST-identified sources (found in the .XYmqxyrd GaiaHub files) and transforms
    them to Gaia pseudo-pixels
    
    Remove sources that already have matches with Gaia.
    
    Cross-match between all the HST images that have any overlap with one another.
    
    '''
    
    outpath = f'{path}{field}/Bayesian_PMs/'
    
    trans_file_df = pd.read_csv(f'{outpath}gaiahub_image_transformation_summaries.csv')
    
    image_names = np.array(trans_file_df['image_name'].to_list())
    not_in_gaia_sources = {}
    for image_ind,image_name in enumerate(tqdm(image_names,total=len(image_names))):
    #    print(f'Looking at all HST-identified sources for {image_name} in field {field}.')
        
        ra_center,dec_center = trans_file_df['ra'][image_ind],trans_file_df['dec'][image_ind]
        x0,y0 = trans_file_df['Xo'][image_ind],trans_file_df['Yo'][image_ind]
        
        datapath = f'{path}{field}/HST/mastDownload/HST/{image_name}/'
        trans_file = f'{datapath}{image_name}_flc_6p_transformation.txt'
        with open(trans_file,'r') as f:
            lines = f.readlines()
        
        in_position_info = False
        in_trans_info = False
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            elif 'THE FOLLOWING LIMITS FOR HST' in line:
                mag_limit = float(line.split('<')[1].split(',')[0])
                q_hst_limit = float(line.split(',')[1].split('<')[1].split(')')[0])
    #            elif 'Delta_Time' in line:
    #                delta_time = float(line.split(':')[1])
            elif 'MASTER FRAME INFO' in line:
                in_position_info = True #start region for position of (RA,Dec) of image
            elif 'GAIA INFO' in line:
                in_position_info = False #end region for position of (RA,Dec) of image
                break
    #        elif 'NEW TRANSFORMATIONS AFTER CLIP' in line:
    #            in_trans_info = True #start region for transformation parameters
    #        elif 'SAVING MAT FILE' in line:
    #            in_trans_info = False #end region for transformation parameters
    #            break #stop looking through lines
            elif in_position_info:
                if 'RA_CENTER' in line:
                    ra_gaia_center = float(line.split(':')[1])
                elif 'DEC_CENTER' in line:
                    dec_gaia_center = float(line.split(':')[1])   
                elif 'X_CENTER' in line:
                    x_gaia_center = float(line.split(':')[1])
                elif 'Y_CENTER' in line:
                    y_gaia_center = float(line.split(':')[1])   
                elif 'SCALE (mas/pix)' in line:
                    gaia_pixel_scale = float(line.split(':')[1])   
            else:
                continue        
        
        outpath = f'{path}{field}/Bayesian_PMs/{image_name}/'
    
        if os.path.isfile(f'{outpath}{image_name}_posterior_transformation_6p_matrix_params_medians.npy'):
    #        print(f'Using Bayesian transformation parameter results for image {image_name}')
            
            trans_sample_med = np.load(f'{outpath}{image_name}_posterior_transformation_6p_matrix_params_medians.npy')
    #        trans_sample_cov = np.load(f'{outpath}{image_name}_posterior_transformation_6p_matrix_params_covs.npy')
        else:
    #        print(f'Using GaiaHub transformation parameter results for image {image_name}')
            trans_sample_med = np.array([trans_file_df['AG'][image_ind],
                                         trans_file_df['BG'][image_ind],
                                         trans_file_df['Wo'][image_ind],
                                         trans_file_df['Zo'][image_ind],
                                         trans_file_df['CG'][image_ind],
                                         trans_file_df['DG'][image_ind]])
            
        a,b,w0,z0,c,d = trans_sample_med
        det = a*d-b*c
        #inverse matrix for de-transforming
        ai,bi,ci,di = np.array([d,-b,-c,a])/det
        matrix = np.array([[a,b],[c,d]])
        inv_matrix = np.array([[ai,bi],[ci,di]])
        
        gaiahub_summary = pd.read_csv(f'{outpath}{image_name}_gaiahub_source_summaries.csv')
        x_hst,y_hst,m_hst,q_hst = gaiahub_summary['X'],gaiahub_summary['Y'],gaiahub_summary['mag'],gaiahub_summary['q_hst']
        x_hst = x_hst.to_numpy()
        y_hst = y_hst.to_numpy()
        m_hst = m_hst.to_numpy()
        q_hst = q_hst.to_numpy()
        ra_gaia = gaiahub_summary['ra'].to_numpy()
        dec_gaia = gaiahub_summary['dec'].to_numpy()
        
        median_mag_offset = np.median(gaiahub_summary['g_mag']-m_hst)
        
        all_hst_sources = np.genfromtxt(f'{datapath}{image_name}_flc.XYmqxyrd',names=('X','Y','m','q','x','y','r','d'))
        found_in_gaia = np.zeros(len(all_hst_sources)).astype(bool)
        source_inds = np.zeros(len(x_hst)).astype(int)
        
    #    print('Finding where Gaia-identified sources occur in HST source list.')
    #    for j,_ in enumerate(tqdm(x_hst,total=len(x_hst))):
        for j,_ in enumerate(x_hst):
            source_inds[j] = np.where(
                    (np.abs(x_hst[j]-all_hst_sources['X']) < 1e-2) &\
                    (np.abs(y_hst[j]-all_hst_sources['Y']) < 1e-2) &\
                    (np.abs(m_hst[j]-all_hst_sources['m']) < 1e-2)
                    )[0][0]
        found_in_gaia[source_inds] = True
        not_in_gaia = ~found_in_gaia
        not_in_gaia_inds = np.where(not_in_gaia)[0]
        
        orig_xy_hst_gaia = np.zeros((len(x_hst),2))
        orig_xy_hst_gaia[:,0] = a*(x_hst-x0)+b*(y_hst-y0)+w0
        orig_xy_hst_gaia[:,1] = c*(x_hst-x0)+d*(y_hst-y0)+z0
        
        xymqrd_hst_gaia = np.zeros((len(not_in_gaia_inds),9))
        xymqrd_hst_gaia[:,0] = a*(all_hst_sources['X'][not_in_gaia_inds]-x0)+b*(all_hst_sources['Y'][not_in_gaia_inds]-y0)+w0
        xymqrd_hst_gaia[:,1] = c*(all_hst_sources['X'][not_in_gaia_inds]-x0)+d*(all_hst_sources['Y'][not_in_gaia_inds]-y0)+z0
        xymqrd_hst_gaia[:,2] = all_hst_sources['m'][not_in_gaia_inds]
        xymqrd_hst_gaia[:,3] = all_hst_sources['q'][not_in_gaia_inds]
        
        #translate the XY_gaia into RA,Dec
        #multiply the DeltaRA by cos(Dec) so that all offsets are in the tangent plane (like XY)
        #because dX = dRA*cos(Dec)
        dX = xymqrd_hst_gaia[:,0]-x_gaia_center
        dY = xymqrd_hst_gaia[:,1]-y_gaia_center
        dRA = (-1*dX/np.cos(dec_gaia_center))*gaia_pixel_scale/(1000*3600)
        dDec = dY*gaia_pixel_scale/(1000*3600)
        RA = ra_gaia_center-dRA
        Dec = dec_gaia_center+dDec
        
        dX_orig = orig_xy_hst_gaia[:,0]-x_gaia_center
        dY_orig = orig_xy_hst_gaia[:,1]-y_gaia_center
        dRA_orig = (-1*dX_orig/np.cos(dec_gaia_center))*gaia_pixel_scale/(1000*3600)
        dDec_orig = dY_orig*gaia_pixel_scale/(1000*3600)
        RA_orig = ra_gaia_center-dRA_orig
        Dec_orig = dec_gaia_center+dDec_orig
        
    #    xymqrd_hst_gaia[:,4] = RA
    #    xymqrd_hst_gaia[:,5] = Dec
        xymqrd_hst_gaia[:,4] = all_hst_sources['r'][not_in_gaia_inds]
        xymqrd_hst_gaia[:,5] = all_hst_sources['d'][not_in_gaia_inds]
        xymqrd_hst_gaia[:,6] = xymqrd_hst_gaia[:,2]+median_mag_offset #estimate of gaia mag
        xymqrd_hst_gaia[:,7] = all_hst_sources['X'][not_in_gaia_inds]
        xymqrd_hst_gaia[:,8] = all_hst_sources['Y'][not_in_gaia_inds]
        
        not_in_gaia_sources[image_name] = xymqrd_hst_gaia
        
    max_pm = 100 #mas/yr
    first_min_dist_thresh = 100 #mas
    min_dist_thresh = 10 #mas
    
    gaia_mjd = 57388.5 #J2016.0 in MJD
    
    
    #now loop over the images to determine cross matches
    #for each image, find the closest neighbour to each star
    
    print(f'Finding cross-matches with sources of in field {field}.')
    
    match_dict = {}
    star_name_dict = {}
    poss_star_pairs = []
    
    star_counter = 0
    
    for image_ind,image_name in enumerate(tqdm(image_names,total=len(image_names))):
    #    print(f'Finding cross-matches with sources of image {image_name} in field {field}.')
        if image_name not in match_dict:
            match_dict[image_name] = {}
        if image_name not in star_name_dict:
            star_name_dict[image_name] = {}
            
        outpath = f'{path}{field}/Bayesian_PMs/{image_name}/'
        linked_image_info = pd.read_csv(f'{outpath}{image_name}_linked_images.csv')
        neighbour_images = linked_image_info['image_name'].to_numpy()
        neighbour_image_dts = linked_image_info['time_offset'].to_numpy()
        match_ind = np.where(neighbour_images == image_name)[0][0]
        keep_images = np.ones(len(neighbour_images)).astype(bool)
        keep_images[match_ind] = False
        neighbour_images = neighbour_images[keep_images]
        neighbour_image_dts = neighbour_image_dts[keep_images]
        
        curr_data = not_in_gaia_sources[image_name]
        
        match_counter = np.zeros(len(neighbour_images))
    #    for other_ind,other_image in enumerate(tqdm(neighbour_images,total=len(neighbour_images))):
        for other_ind,other_image in enumerate(neighbour_images):
            if other_image not in match_dict:
                match_dict[other_image] = {}
            if other_image in match_dict[image_name]:
                continue
            
            if other_image not in match_dict[image_name]:
                match_dict[image_name][other_image] = {}
            if image_name not in match_dict[other_image]:
                match_dict[other_image][image_name] = {}
            if other_image not in star_name_dict:
                star_name_dict[other_image] = {}
                
            other_data = not_in_gaia_sources[other_image]
            other_dt = np.abs(neighbour_image_dts[other_ind])
            other_ra_decs = other_data[:,[4,5]]
            max_distance = max_pm*other_dt+first_min_dist_thresh #in mas
            max_distance = max_distance/1000/3600 #in degrees
            max_dist2 = max_distance**2
            
            curr_match_counter = 0
            
            ave_offsets = np.zeros((len(curr_data),2))*np.nan
            
            for star_ind in range(len(curr_data)):
                curr_ra_dec = curr_data[star_ind,[4,5]]
                ang_dist2 = np.sum(np.power(other_ra_decs-curr_ra_dec,2),axis=1)
                best_match_ind = np.argmin(ang_dist2)
                closest_dist = ang_dist2[best_match_ind]**0.5
                if closest_dist > max_distance:
                    #not a close enough match
                    continue
                dra_dec = other_ra_decs[best_match_ind]-curr_ra_dec
    #            print(star_ind,np.round([closest_dist*1000*3600,
    #                                     curr_data[star_ind,2],
    #                                     other_data[best_match_ind,2],
    #                                     dra_dec[0]*1000*3600,
    #                                     dra_dec[1]*1000*3600],1))
                ave_offsets[star_ind] = dra_dec
                curr_match_counter += 1
                
            #determine if there is a constant offset in RA and Dec between the two lists,
            #and then redo the searching to apply it if it is significant
            good_offsets = ave_offsets[np.isfinite(ave_offsets[:,0])]
            median_offset = np.median(good_offsets,axis=0)
            median_offset[~np.isfinite(median_offset)] = 0
            median_offset_vect_size = (median_offset[0]**2+median_offset[1]**2)**0.5
            
            #repeat the matching but subtract off a possible constant offset in RA,Dec alignment
            
    #       print(f'Repeating matching criteria because a significant constant offset of {round(median_offset_vect_size*1000*3600,1)} mas was found')
            curr_match_counter = 0
            max_distance = max_pm*other_dt+min_dist_thresh #in mas
            max_distance = max_distance/1000/3600 #in degrees
            max_dist2 = max_distance**2
            
            for star_ind in range(len(curr_data)):
                curr_ra_dec = curr_data[star_ind,[4,5]]
                ang_dist2 = np.sum(np.power(other_ra_decs-(curr_ra_dec+median_offset),2),axis=1)
                best_match_ind = np.argmin(ang_dist2)
                closest_dist = ang_dist2[best_match_ind]**0.5
                if closest_dist > max_distance:
                    #not a close enough match
                    continue
                dra_dec = other_ra_decs[best_match_ind]-(curr_ra_dec+median_offset)
    #            print(star_ind,np.round([closest_dist*1000*3600,
    #                                     curr_data[star_ind,2],
    #                                     other_data[best_match_ind,2],
    #                                     dra_dec[0]*1000*3600,
    #                                     dra_dec[1]*1000*3600],1))
                
                ave_offsets[star_ind] = dra_dec
                curr_match_counter += 1
                
                #closest distance in mas, 
                #hst_mag image 1
                #hst_mag image 2
                #gaia_mag image 1
                #gaia_mag image 2
                #ra image 1
                #dec image 1
                #ra image 2
                #dec image 2
                #dRA (image2-image1), in mas
                #dDec (image2-image1), in mas
                #index in image 1
                #index in image 2
    #            print(star_ind,closest_dist)
                overwrite_previous = True
                
                if star_ind in match_dict[image_name][other_image]:
                    #compare the previous entry's distance 
                    old_dist = match_dict[image_name][other_image][star_ind][0]
                    if old_dist < closest_dist*1000*3600:
                        overwrite_previous = False
    #                else:
    #                    print(f'Overwriting previous distance, {old_dist} mas, for star {star_ind} in image {image_name} with new one, {closest_dist*1000*3600} mas')
                if best_match_ind in match_dict[other_image][image_name]:
                    old_dist = match_dict[other_image][image_name][best_match_ind][0]
                    if old_dist < closest_dist*1000*3600:
                        overwrite_previous = False
    #                else:
    #                    print(f'Overwriting previous distance, {old_dist} mas, for star {star_ind} in image {image_name} with new one, {closest_dist*1000*3600} mas')
                if overwrite_previous:
                    match_dict[image_name][other_image][star_ind] = np.array([closest_dist*1000*3600,
                                                                               curr_data[star_ind,2],other_data[best_match_ind,2],
                                                                               curr_data[star_ind,6],other_data[best_match_ind,6],
                                                                               curr_ra_dec[0],curr_ra_dec[1],
                                                                               other_ra_decs[best_match_ind][0],other_ra_decs[best_match_ind][1],                                                                       
                                                                               dra_dec[0]*1000*3600,
                                                                               dra_dec[1]*1000*3600,
                                                                               star_ind,best_match_ind])
                
                    match_dict[other_image][image_name][best_match_ind] = np.array([closest_dist*1000*3600,
                                                                               other_data[best_match_ind,2],curr_data[star_ind,2],
                                                                               other_data[best_match_ind,6],curr_data[star_ind,6],
                                                                               other_ra_decs[best_match_ind][0],other_ra_decs[best_match_ind][1],                                                                       
                                                                               curr_ra_dec[0],curr_ra_dec[1],
                                                                               -1*dra_dec[0]*1000*3600,
                                                                               -1*dra_dec[1]*1000*3600,
                                                                               best_match_ind,star_ind])
            
                
                
            good_offsets = ave_offsets[np.isfinite(ave_offsets[:,0])]
            median_offset = np.median(good_offsets,axis=0)
            median_offset_vect_size = (median_offset[0]**2+median_offset[1]**2)**0.5
            match_counter[other_ind] = curr_match_counter
            
    #        print(f'New constant offset of {round(median_offset_vect_size*1000*3600,1)} mas was found')
            
    #    print(f'Found {match_counter} possible matches for {len(curr_data)} sources of image {image_name} in field {field}.')
        
    #    break
        
            
    match_completed_dict = {}
    star_name_dict = {}
    star_rename_dict = {}
    star_counter = 0
    for image_ind,image_name in enumerate(tqdm(image_names,total=len(image_names))):
        for other_ind,other_image in enumerate(match_dict[image_name]):
            for star_ind in match_dict[image_name][other_image]:
                first_entry = match_dict[image_name][other_image][star_ind]
                best_match_ind = int(first_entry[-1])
                best_dist = first_entry[0]
                if best_match_ind not in match_dict[other_image][image_name]:
                    continue
                
                second_entry = match_dict[other_image][image_name][best_match_ind]
                if (first_entry[-2] != second_entry[-1]) or (first_entry[-1] != second_entry[-2]):
                    continue
                            
                if image_name not in match_completed_dict:
                    match_completed_dict[image_name] = {}
                if other_image not in match_completed_dict:
                    match_completed_dict[other_image] = {}
                    
                if (star_ind in match_completed_dict[image_name]) and (best_match_ind in match_completed_dict[other_image]):
                    star_name1 = match_completed_dict[image_name][star_ind]
                    star_name2 = match_completed_dict[other_image][best_match_ind]
                    if star_name1 == star_name2:
                        star_name = star_name1
                    else:
                        #choose the smaller of the two numbers
                        star_num1 = int(star_name1[3:])
                        star_num2 = int(star_name2[3:])
                        smaller_num_ind = np.argmin([star_num1,star_num2])
                        star_name = 'HST%016d'%(min(star_num1,star_num2))
                        
                        if smaller_num_ind == 0: 
                            star_rename_dict[star_name2] = star_name
                            for key in star_name_dict[star_name2]:
                                entry = star_name_dict[star_name2][key]
                                star_name_dict[star_name1][key] = entry
                                match_completed_dict[key][entry] = star_name
    #                        del star_name_dict[star_name2]
    #                        print('deteled',star_name2)
                        else: 
                            star_rename_dict[star_name1] = star_name
                            for key in star_name_dict[star_name1]:
                                entry = star_name_dict[star_name1][key]
                                star_name_dict[star_name2][key] = entry
                                match_completed_dict[key][entry] = star_name
    #                        del star_name_dict[star_name1]
    #                        print('deteled',star_name1)
    #                    print('AHHHH!',image_name,star_name1,other_image,star_name2)
    #                    asd;lfkjasdl;fjasdkl
                elif star_ind in match_completed_dict[image_name]:
                    star_name = match_completed_dict[image_name][star_ind]
                elif best_match_ind in match_completed_dict[other_image]:
                    star_name = match_completed_dict[other_image][best_match_ind]
                else:
                    #new star
                    star_name = 'HST%016d'%star_counter
                    star_counter += 1
                
                match_completed_dict[image_name][star_ind] = star_name
                match_completed_dict[other_image][best_match_ind] = star_name
                
                if star_name not in star_name_dict:
                    star_name_dict[star_name] = {}
                    
                star_name_dict[star_name][image_name] = star_ind
                star_name_dict[star_name][other_image] = best_match_ind
                
    star_properties_dict = {}
    image_new_stars = {}
    for star_name in star_name_dict:
        ave_properties = np.zeros((len(star_name_dict[star_name]),10))
        for image_ind,image_name in enumerate(star_name_dict[star_name]):
            if image_name not in image_new_stars:
                image_new_stars[image_name] = []
            image_new_stars[image_name].append(star_name)
            star_ind = star_name_dict[star_name][image_name]
            
            curr_mjd = trans_file_df['HST_time'][np.where(image_names == image_name)[0][0]]
            curr_dt = (gaia_mjd-curr_mjd)/365.25
            
            ave_properties[image_ind,:-1] = not_in_gaia_sources[image_name][star_ind] #X_G,Y_G,m_hst,q_fit,ra,dec,m_g,X,Y
            ave_properties[image_ind,-1] = curr_dt
            
        dxs = ave_properties[:,0]-ave_properties[0,0]
        dys = ave_properties[:,1]-ave_properties[0,1]
        dts = ave_properties[:,-1]-ave_properties[0,-1]
        dts[0] = 1
            
        pm_ras = dxs/dts
        pm_decs = dys/dts
        pm_weights = np.power(ave_properties[:,3],2)+np.power(ave_properties[0,3],2)
        if np.sum(pm_weights) == 0:
            pm_weights[:] = 1
        pm_weights /= np.sum(pm_weights)
        ave_pm_ra = np.sum(pm_weights*pm_ras)
        ave_pm_dec = np.sum(pm_weights*pm_decs)
        
        gaia_xs = ave_properties[:,0]+ave_pm_ra*curr_dt
        gaia_ys = ave_properties[:,1]+ave_pm_dec*curr_dt
        
        weights = ave_properties[:,3]
        if np.sum(weights) == 0:
            weights[:] = 1
        weights /= np.sum(weights)
        star_properties_dict[star_name] = {}
        star_properties_dict[star_name]['Gaia_id'] = star_name
        star_properties_dict[star_name]['hst_images'] = ' '.join(list(star_name_dict[star_name].keys()))
        star_properties_dict[star_name]['ra_G'] = np.sum(ave_properties[:,4]*weights)
        star_properties_dict[star_name]['dec_G'] = np.sum(ave_properties[:,5]*weights)
        
    #    star_properties_dict[star_name]['X_G'] = np.sum(ave_properties[:,0]*weights)
    #    star_properties_dict[star_name]['Y_G'] = np.sum(ave_properties[:,1]*weights)
        star_properties_dict[star_name]['X_G'] = np.sum(gaia_xs*weights)
        star_properties_dict[star_name]['Y_G'] = np.sum(gaia_ys*weights)
        
        star_properties_dict[star_name]['g_mag'] = np.sum(ave_properties[:,6]*weights)    
    
    for image_name in image_new_stars:
        outpath = f'{path}{field}/Bayesian_PMs/{image_name}/'
        gaiahub_output_info = {'Gaia_id':[],'x_hst_err':[],'y_hst_err':[],
                                          'X_G':[],'Y_G':[],'X':[],
                                          'Y':[],'g_mag':[],'mag':[],'dX_G':[],
                                          'dY_G':[],'X_2':[],'Y_2':[],'use_for_fit':[],'q_hst':[],
                                          'gaiahub_pm_x':[],'gaiahub_pm_y':[],
                                          'gaiahub_pm_x_err':[],'gaiahub_pm_y_err':[],
                                          'stationary':[],'hst_images':[],'Gaia_time':[]
                                          }
        for label in correlation_names:
            gaiahub_output_info[label] = []
            
        star_names = image_new_stars[image_name]
        star_hst_images = []
        X_G = np.zeros(len(star_names))
        Y_G = np.zeros(len(star_names))
        ra_G = np.zeros(len(star_names))
        dec_G = np.zeros(len(star_names))
        m_g = np.zeros(len(star_names))
        X = np.zeros(len(star_names))
        Y = np.zeros(len(star_names))
        q_hst = np.zeros(len(star_names))
        m_hst = np.zeros(len(star_names))
        X_2 = np.zeros(len(star_names))
        Y_2 = np.zeros(len(star_names))
        for star_ind,star_name in enumerate(star_names):
            star_hst_images.append(star_properties_dict[star_name]['hst_images'])
    #        X_G[star_ind] = star_properties_dict[star_name]['X_G']
    #        Y_G[star_ind] = star_properties_dict[star_name]['Y_G']
            ra_G[star_ind] = star_properties_dict[star_name]['ra_G']
            dec_G[star_ind] = star_properties_dict[star_name]['dec_G']
            m_g[star_ind] = star_properties_dict[star_name]['g_mag']
            
            curr_ind = star_name_dict[star_name][image_name]
            star_properties = not_in_gaia_sources[image_name][curr_ind] #X_G,Y_G,m_hst,q_fit,ra,dec,m_g,X,Y
            
            X[star_ind] = star_properties[7]
            Y[star_ind] = star_properties[8]
            q_hst[star_ind] = star_properties[3]
            m_hst[star_ind] = star_properties[2]
            X_2[star_ind] = star_properties[0]
            Y_2[star_ind] = star_properties[1]
            
            X_G[star_ind] = star_properties[0]
            Y_G[star_ind] = star_properties[1]
        dX_G = X_G-X_2
        dY_G = Y_G-Y_2
            
        nan_array = np.nan*np.zeros(len(X_G))
        zeros_array = np.zeros(len(X_G))
        ones_array = np.zeros(len(X_G))
                
        gaiahub_output_info['X_G'].extend(X_G)
        gaiahub_output_info['Y_G'].extend(Y_G)
        gaiahub_output_info['X'].extend(X)
        gaiahub_output_info['Y'].extend(Y)
        gaiahub_output_info['g_mag'].extend(m_g)
        gaiahub_output_info['mag'].extend(m_hst)
        gaiahub_output_info['dX_G'].extend(dX_G)
        gaiahub_output_info['dY_G'].extend(dY_G)
        gaiahub_output_info['X_2'].extend(X_2)
        gaiahub_output_info['Y_2'].extend(Y_2)
        gaiahub_output_info['q_hst'].extend(q_hst)
        gaiahub_output_info['use_for_fit'].extend(zeros_array.astype(bool))
                
        gaiahub_output_info['Gaia_time'].extend(2016.0*ones_array)
        
        for label in correlation_names:
            gaiahub_output_info[label].extend(nan_array)
        gaiahub_output_info['ra'] = ra_G
        gaiahub_output_info['dec'] = dec_G
        gaiahub_output_info['ra_error'] = 51200.0 #0.5*2048*50
        gaiahub_output_info['dec_error'] = 51200.0
        gaiahub_output_info['ra_dec_corr'] = zeros_array
                                    
        gaiahub_output_info['Gaia_id'].extend(star_names)
    
        gaiahub_output_info['x_hst_err'].extend(nan_array)  
        gaiahub_output_info['y_hst_err'].extend(nan_array)
    
        gaiahub_output_info['hst_images'].extend(star_hst_images)
        gaiahub_output_info['gaiahub_pm_x'].extend(nan_array)  
        gaiahub_output_info['gaiahub_pm_y'].extend(nan_array)  
        gaiahub_output_info['gaiahub_pm_x_err'].extend(nan_array)  
        gaiahub_output_info['gaiahub_pm_y_err'].extend(nan_array)  
            
        #check if there is file labelling points as stationary, but if it doesn't exist, assume none are stationary
        gaiahub_output_info['stationary'].extend(zeros_array.astype(bool))
    
        output_df = pd.DataFrame.from_dict(gaiahub_output_info)
        output_df.to_csv(f'{outpath}/{image_name}_gaiahub_source_summaries_ALL_HST.csv',index=False)
        
    return
                    
    
if __name__ == '__main__':
    pass

    path = '/Volumes/Kevin_Astro/Astronomy/HST_Gaia_PMs/GaiaHub_results/'
    
    field = 'Fornax_dSph'
    field = 'COSMOS_field'

    source_matcher(field,path)






