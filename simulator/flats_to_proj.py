#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniil Kazantsev
"""

from scipy.ndimage import shift
import random
import numpy as np
from tomophantom.supp.artifacts import noise


def add_flats_proj(projData3D_clean, flatfield_poisson, flats_combined3D, blurred_speckles_map, source_intensity, variations_number, detectors_miscallibration, jitter_projections):
    """
    Adds flats to the projection data
    
    variations_number - the number of functions to control stripe type (1 - linear, 2, - sinusoidal, 3 - exponential)
    """
    [DetectorsDimV, projectionsNo, DetectorsDimH] = np.shape(projData3D_clean)
    projData3D_raw = np.zeros(np.shape(projData3D_clean),dtype='float32')
    
    sinusoidal_response = np.sin(np.linspace(0,1.5*np.pi,projectionsNo)) + np.random.random(projectionsNo) * 0.1
    sinusoidal_response /= np.max(sinusoidal_response)
    exponential_response = np.exp(np.linspace(0,np.pi,projectionsNo)) + np.random.random(projectionsNo) * 0.1
    exponential_response /= np.max(exponential_response)
  
    # convert synthetic projections to raw-data like projection ready for normalisation
    for i in range(0,projectionsNo):
        proj_exp = np.exp(-projData3D_clean[:,i,:])*source_intensity*flatfield_poisson # raw projection gets multiplied with a flat field (background)
        for j in range(0,variations_number):
            if j == 0:
                # adding a consistent offset for certain detectors
                proj_exp -= blurred_speckles_map[:,:,j]*detectors_miscallibration*source_intensity
            if j == 1:
                # adding a sinusoidal-like response offset for certain detectors
                proj_exp += sinusoidal_response[i]*blurred_speckles_map[:,:,j]*detectors_miscallibration*source_intensity
            if j == 2:
                # adding an exponential response offset for certain detectors
                proj_exp += exponential_response[i]*blurred_speckles_map[:,:,j]*detectors_miscallibration*source_intensity
                
        projection_poisson = noise(proj_exp, source_intensity, noisetype='Poisson')

        # apply jitter to projections
        if jitter_projections != 0.0:
            horiz_shift = random.uniform(-jitter_projections,jitter_projections)  #generate random directional shift
            vert_shift = random.uniform(-jitter_projections,jitter_projections)  #generate random directional shift
            projection_poisson = shift(projection_poisson.copy(),[vert_shift,horiz_shift], mode='reflect')
        projData3D_raw[:,i,:] = projection_poisson

    projData3D_raw /= np.max(projData3D_raw)
    return np.uint16(projData3D_raw*65535)