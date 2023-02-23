#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniil Kazantsev
"""

from scipy.ndimage import shift
import random
import numpy as np
from tomophantom.supp.artifacts import noise


def add_flats_proj(projData3D_clean, flatfield_poisson, flats_combined3D,
                   blurred_speckles_map, source_intensity, variations_number,
                   detectors_miscallibration, jitter_projections):
    """
    Adds flats to the projection data.
    Parameters:
        projData3D_clean : np.ndarray
            3D projection data to add flats to.
        flatfield_poisson : np.ndarray
            2D Flat field with poission noise added
        flats_combined3D : np.ndarray
            3D flat field data
        blurred_speckles_map : np.ndarray
            3D array representing miscalibrated detector pixels.
        source_intensity : int
            Source intensity which affects the amount of the Poisson noise
            added to data.
        variations_number : int
            The number of functions to control stripe type.
            1 - linear, 2 - sinusoidal, 3 - exponential.
        detectors_miscallibration : float
            Multiplier for defective detectors.
        jitter_projections : float
            A random jitter to the projections in pixels.
    Returns:
        np.ndarray
            3D array of projection with flat fields added in.
            Same shape as `projData3D_clean`.
    """
    [DetectorsDimV, projectionsNo, DetectorsDimH] = np.shape(projData3D_clean)
    projData3D_raw = np.zeros(np.shape(projData3D_clean), dtype='float32')
    
    sinusoidal_response = np.sin(np.linspace(0, 1.5*np.pi, projectionsNo)
                                 ) + np.random.random(projectionsNo) * 0.1
    sinusoidal_response /= np.max(sinusoidal_response)
    exponential_response = np.exp(np.linspace(0, np.pi, projectionsNo)
                                  ) + np.random.random(projectionsNo) * 0.1
    exponential_response /= np.max(exponential_response)
  
    # convert synthetic projections to raw-data like projection ready for
    # normalisation
    for i in range(0, projectionsNo):
        # raw projection gets multiplied with a flat field (background)
        proj_exp = np.exp(-projData3D_clean[:, i, :]) * \
                   source_intensity * flatfield_poisson
        for j in range(0, variations_number):
            if j == 0:
                # adding a consistent offset for certain detectors
                proj_exp -= blurred_speckles_map[:, :, j] * \
                            detectors_miscallibration * source_intensity
            if j == 1:
                # add sinusoidal-like response offset for certain detectors
                proj_exp += sinusoidal_response[i] * \
                            blurred_speckles_map[:,:,j] * \
                            detectors_miscallibration * source_intensity
            if j == 2:
                # add exponential response offset for certain detectors
                proj_exp += exponential_response[i] * \
                            blurred_speckles_map[:,:,j] * \
                            detectors_miscallibration * source_intensity

        # Potential problem: in the below function, tomophantom offers no way
        # of changing the seed
        # It is either always fixed at 1, or completely random.
        # If fixed, the network could overfit on this particular distribution
        # of noise
        # If random, the network may struggle to translate between different
        # distributions of noise in inputs and targets
        projection_poisson = noise(proj_exp, source_intensity,
                                   noisetype='Poisson')

        # apply jitter to projections
        if jitter_projections != 0.0:
            # generate random horizontal shift
            horiz_shift = random.uniform(-jitter_projections,
                                         jitter_projections)
            # generate random vertical shift
            vert_shift = random.uniform(-jitter_projections,
                                        jitter_projections)
            projection_poisson = shift(projection_poisson.copy(),
                                       [vert_shift, horiz_shift],
                                       mode='reflect')
        projData3D_raw[:, i, :] = projection_poisson

    projData3D_raw /= np.max(projData3D_raw)
    return np.uint16(projData3D_raw*65535)
