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
from decimal import getcontext,Decimal
from math import atan2,asin,sqrt,pow


#import matplotlib.pyplot as plt
import process_GaiaHub_outputs as process_GH

correlation_names = process_GH.correlation_names

def n_combs(n,k):
    #return n choose k combinations
    return int(round(np.exp(np.sum(np.log(np.arange(n)+1))-np.sum(np.log(np.arange(k)+1))-np.sum(np.log(np.arange(n-k)+1)))))

getcontext().prec = 28

def XY_from_RADec(ra,dec,ra0,dec0,X0,Y0,pixel_scale):
    '''
    ra,dec,ra0,dec0 in degrees
    pixel_scale in mas/pixel
    X0,Y0 in pixels
    X,Y in pixels
    
    equations come from xym2pm_GH.F, after line 2047
    '''
    
    dec0_rad = dec0*np.pi/180
    dec_rad = dec*np.pi/180
    ra0_rad = ra0*np.pi/180
    ra_rad = ra*np.pi/180
    
    cosra = np.cos(ra_rad-ra0_rad)
    sinra = np.sin(ra_rad-ra0_rad)
    cosde = np.cos(dec_rad)
    sinde = np.sin(dec_rad)
    cosd0 = np.cos(dec0_rad)
    sind0 = np.sin(dec0_rad)
    
    rrrr = sind0*sinde + cosd0*cosde*cosra
    #offsets in radians
    dY = (cosd0*sinde-sind0*cosde*cosra)/rrrr
    dX = cosde*sinra/rrrr
    
    x  = cosde*np.cos(ra_rad)
    y  = cosde*np.sin(ra_rad)
    z  = sinde
    xx = cosd0*np.cos(ra0_rad)
    yy = cosd0*np.sin(ra0_rad)
    zz = sind0
    bad_cond = (x*xx + y*yy + z*zz < 0)
    dY[bad_cond] = np.pi/2
    dX[bad_cond] = np.pi/2
            
    X = X0-dX*180/np.pi*3600*1000/pixel_scale
    Y = Y0+dY*180/np.pi*3600*1000/pixel_scale
    
    return X,Y


def RADec_from_XY(X,Y,ra0,dec0,X0,Y0,pixel_scale):
    '''
    ra,dec,ra0,dec0 in degrees
    pixel_scale in mas/pixel
    X0,Y0 in pixels
    X,Y in pixels    
    
    equations from https://www.researchgate.net/publication/333841450_Astrometry_The_Foundation_for_Observational_Astronomy,
    though our dY is their Y*-1
    '''
    
    dec0_rad = dec0*np.pi/180
    ra0_rad = ra0*np.pi/180
    
    cosd0 = np.cos(dec0_rad)
    sind0 = np.sin(dec0_rad)
    
#    decimal_cosd0 = Decimal(cosd0)
#    decimal_sind0 = Decimal(sind0)
    
    dX = (X0-X)/(180/np.pi*3600*1000/pixel_scale)
    dY = (Y-Y0)/(180/np.pi*3600*1000/pixel_scale)
    
    ra_rad = np.zeros_like(X)
    dec_rad = np.zeros_like(Y)
    
    atan_args = dX,cosd0-dY*sind0
    asin_args = (sind0+dY*cosd0)/np.sqrt(1+np.power(dX,2)+np.power(dY,2))
    
    for j in range(len(X)):
#        decimal_dx = Decimal(dX[j])
#        decimal_dy = Decimal(dY[j])
#        ra_rad[j] = ra0_rad + atan2(decimal_dx,decimal_cosd0-decimal_dy*decimal_sind0)
        
        ra_rad[j] = ra0_rad + atan2(Decimal(atan_args[0][j]),Decimal(atan_args[1][j]))
        dec_rad[j] = asin(Decimal(asin_args[j]))
    
    ra = ra_rad*180/np.pi
    dec = dec_rad*180/np.pi
                
    return ra,dec

def RADec_and_Jac_from_XY(X,Y,ra0,dec0,X0,Y0,pixel_scale):
    '''
    ra,dec,ra0,dec0 in degrees
    pixel_scale in mas/pixel
    X0,Y0 in pixels
    X,Y in pixels    
    
    equations from https://www.researchgate.net/publication/333841450_Astrometry_The_Foundation_for_Observational_Astronomy,
    though our dY is their Y*-1
    '''
        
    deg_to_mas = 3600*1000 #mas/deg
    mas_to_pix = 1/pixel_scale #pix/mas
    rad_to_deg = 180/np.pi #deg/rad
    
    rad_to_mas = rad_to_deg*deg_to_mas
    rad_to_pix = rad_to_mas*mas_to_pix
    
    dec0_rad = dec0/rad_to_deg
    ra0_rad = ra0/rad_to_deg
    
    cosd0 = np.cos(dec0_rad)
    sind0 = np.sin(dec0_rad)
    
    dX = (X0-X)/rad_to_pix
    dY = (Y-Y0)/rad_to_pix
    
    ra_rad = np.zeros_like(X)
    dec_rad = np.zeros_like(Y)
    
    atan_args = dX,cosd0-dY*sind0
    summed_dx_dy = 1+np.power(dX,2)+np.power(dY,2)
    sqrt_summed_dx_dy = np.sqrt(summed_dx_dy)
    asin_args = (sind0+dY*cosd0)/sqrt_summed_dx_dy
    
    for j in range(len(X)):
#        decimal_dx = Decimal(dX[j])
#        decimal_dy = Decimal(dY[j])
#        ra_rad[j] = ra0_rad + atan2(decimal_dx,decimal_cosd0-decimal_dy*decimal_sind0)
        
        ra_rad[j] = atan2(Decimal(atan_args[0][j]),Decimal(atan_args[1][j]))
        dec_rad[j] = asin(Decimal(asin_args[j]))
        
    ra_rad += ra0_rad
    ra = ra_rad*rad_to_deg
    dec = dec_rad*rad_to_deg
    
    atan_div = atan_args[0]/atan_args[1]
    datan = 1/(np.power(atan_div,2)+1)
    dasin = 1/np.sqrt(1-np.power(asin_args,2))
    ddeltaX_dX = -1/rad_to_pix
    ddeltaY_dY = 1/rad_to_pix
    
    dra_dX = rad_to_deg*datan*(1/atan_args[1])*ddeltaX_dX
    dra_dY = rad_to_deg*datan*(-1*atan_div/atan_args[1]*(-1*sind0))*ddeltaY_dY
    dracosdec_dX = dra_dX*np.cos(dec)*deg_to_mas
    dracosdec_dY = dra_dY*np.cos(dec)*deg_to_mas
    ddec_dX = rad_to_mas*dasin*(-1*asin_args/summed_dx_dy)*dX*ddeltaX_dX
    ddec_dY = rad_to_mas*dasin*(-1*asin_args/summed_dx_dy*dY+cosd0/sqrt_summed_dx_dy)*ddeltaY_dY
    
    jac = np.zeros((len(X),2,2))
    jac[:,0,0] = dracosdec_dX
    jac[:,0,1] = dracosdec_dY
    jac[:,1,0] = ddec_dX
    jac[:,1,1] = ddec_dY
                
    return np.array([ra,dec]).T,jac


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
    gaia_trans_params = {}
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
        
        #change the Gaia pseudo-pixels in Gaia RA,Dec
        ra_gaia_new,dec_gaia_new = RADec_from_XY(xymqrd_hst_gaia[:,0],
                                                 xymqrd_hst_gaia[:,1],
                                                 ra_gaia_center,dec_gaia_center,x_gaia_center,y_gaia_center,gaia_pixel_scale)
        xymqrd_hst_gaia[:,4] = ra_gaia_new
        xymqrd_hst_gaia[:,5] = dec_gaia_new
        
        not_in_gaia_sources[image_name] = xymqrd_hst_gaia
        gaia_trans_params[image_name] = np.array([ra_gaia_center,dec_gaia_center,x_gaia_center,y_gaia_center,gaia_pixel_scale])
        
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
                
    max_allowed_pm_size = 300 #mas/yr
                
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
            
        #use all pairwise combinations of positions to estimate the PM, which is used to predict a Gaia RA,Dec
        #which is then turned into an X_G,Y_G for each image
        curr_n_velo_combs = min(100,n_combs(len(ave_properties),2))
                
        poss_gaia_ras = np.zeros((curr_n_velo_combs,2))
        poss_gaia_ra_errs = np.zeros((curr_n_velo_combs,2))
        poss_gaia_decs = np.zeros((curr_n_velo_combs,2))
        poss_gaia_dec_errs = np.zeros((curr_n_velo_combs,2))
        
        counter = 0
        for ind1 in range(len(ave_properties)-1):
            for ind2 in range(ind1+1,len(ave_properties)):
                dx = ave_properties[ind2,4]-ave_properties[ind1,4]
                dy = ave_properties[ind2,5]-ave_properties[ind1,5]
                dt = ave_properties[ind2,-1]-ave_properties[ind1,-1]
                diff_err = np.sqrt(np.power(ave_properties[ind2,3],2)+np.power(ave_properties[ind1,3],2))
                pm_err = diff_err/np.abs(dt)
                
                pm_ra = dx/dt
                pm_dec = dy/dt
                pm_size_mas = ((pm_ra**2+pm_dec**2)**0.5)*3600*1000
                
                if pm_size_mas > max_allowed_pm_size:
                    pm_scale_fact = max_allowed_pm_size/pm_size_mas
                else:
                    pm_scale_fact = 1
                    
                pm_ra *= pm_scale_fact
                pm_dec *= pm_scale_fact
                pm_err *= pm_scale_fact                
                
                curr_gaia_ras = ave_properties[ind1,4]+ave_properties[ind1,-1]*pm_ra,\
                                ave_properties[ind2,4]+ave_properties[ind2,-1]*pm_ra
                curr_gaia_ra_errs = np.sqrt(np.power(ave_properties[ind1,3],2)+np.power(pm_err*ave_properties[ind1,-1],2)),\
                                    np.sqrt(np.power(ave_properties[ind2,3],2)+np.power(pm_err*ave_properties[ind2,-1],2))
                curr_gaia_decs = ave_properties[ind1,5]+ave_properties[ind1,-1]*pm_dec,\
                                 ave_properties[ind2,5]+ave_properties[ind2,-1]*pm_dec
                curr_gaia_dec_errs = np.sqrt(np.power(ave_properties[ind1,3],2)+np.power(pm_err*ave_properties[ind1,-1],2)),\
                                     np.sqrt(np.power(ave_properties[ind2,3],2)+np.power(pm_err*ave_properties[ind2,-1],2))
                
                poss_gaia_ras[counter] = curr_gaia_ras
                poss_gaia_ra_errs[counter] = curr_gaia_ra_errs
                poss_gaia_decs[counter] = curr_gaia_decs
                poss_gaia_dec_errs[counter] = curr_gaia_dec_errs
                
                counter += 1
                
                if counter >= curr_n_velo_combs:
                    break
            if counter >= curr_n_velo_combs:
                break
            
        poss_gaia_ra_errs[poss_gaia_ra_errs == 0] = np.inf
        poss_gaia_dec_errs[poss_gaia_dec_errs == 0] = np.inf
        
        poss_gaia_ra_weights = np.power(poss_gaia_ra_errs,-2)
        poss_gaia_ra_weights[~np.isfinite(poss_gaia_ra_weights)] = 0
        if np.sum(poss_gaia_ra_weights) == 0:
            poss_gaia_ra_weights[:] = 1
        poss_gaia_ra_weights /= np.sum(poss_gaia_ra_weights)
        poss_gaia_dec_weights = np.power(poss_gaia_dec_errs,-2)
        poss_gaia_dec_weights[~np.isfinite(poss_gaia_dec_weights)] = 0
        if np.sum(poss_gaia_dec_weights) == 0:
            poss_gaia_dec_weights[:] = 1
        poss_gaia_dec_weights /= np.sum(poss_gaia_dec_weights)
        
        best_gaia_ra = np.sum(poss_gaia_ra_weights*poss_gaia_ras)
        best_gaia_dec = np.sum(poss_gaia_dec_weights*poss_gaia_decs)
                        
        errs = np.copy(ave_properties[:,3])
        errs[errs == 0] = 1000
        weights = np.power(errs,-2)
        if np.sum(weights) == 0:
            weights[:] = 1
        weights /= np.sum(weights)
        star_properties_dict[star_name] = {}
        star_properties_dict[star_name]['Gaia_id'] = star_name
        star_properties_dict[star_name]['hst_images'] = ' '.join(list(star_name_dict[star_name].keys()))
        star_properties_dict[star_name]['ra_G'] = best_gaia_ra
        star_properties_dict[star_name]['dec_G'] = best_gaia_dec
                
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
            
        ra_gaia_center,dec_gaia_center,x_gaia_center,y_gaia_center,gaia_pixel_scale = gaia_trans_params[image_name]
            
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
            
#            X_G[star_ind] = star_properties[0]
#            Y_G[star_ind] = star_properties[1]
            
        X_G,Y_G = XY_from_RADec(ra_G,dec_G,
                                ra_gaia_center,dec_gaia_center,x_gaia_center,y_gaia_center,gaia_pixel_scale)
        
        dX_G = X_G-X_2
        dY_G = Y_G-Y_2
        
#        print(np.all(np.isfinite(dX_G)),np.all(np.isfinite(dY_G)))
            
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
#    field = 'COSMOS_field'
    field = 'Draco_dSph'

    source_matcher(field,path)






