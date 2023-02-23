#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniil Kazantsev
"""

from scipy.special import spherical_yn
from scipy.ndimage import gaussian_filter
from scipy.ndimage import shift
import random
import numpy as np
from tomophantom.supp.artifacts import noise
from tomophantom.supp.speckle_routines import simulate_speckles_with_shot_noise


def synth_flats_mod(projData3D_clean, source_intensity, variations_number=3,
                    arguments_Bessel=(1, 25), specklesize=2, kbar=2,
                    sigmasmooth=2, jitter_projections=0.0, flatsnum=20):
    """
    A function to generate synthetic flat field images and raw data for
    projection data normalisation.
    This is a way to more realistic modelling of stripes leading to ring
    artifacts.
    the format of the input (clean) data is:
        [detectorsX, Projections, detectorsY]
    Parameters:
        projData3D_clean : np.ndarray
            3D array containing projection data without any artifacts or noise.
        source_intensity : int
            Source intensity which affects the amount of the Poisson noise
            added to data.
        variations_number : int
            Number of miscalibrated detectors to simulate. Default is 3.
        arguments_Bessel : Tuple[int, int]
            Tuple of 2 Arguments for 2 Bessel functions to control background
            variations. Default is (1, 25).
        specklesize : int
            Speckle size in pixel units for background simulation.
            Default is 2.
        kbar : int
            Mean photon density (photons per pixel) for background simulation.
            Default is 2.
        sigmasmooth : float
            Gaussian smoothing parameter to blur the speckled backround
            (1,3,5,7...). Default is 2.0
        jitter_projections : float
            A random jitter to the projections in pixels. Default is 0.0
        flatsnum : int
            Number of flat fields to generate.
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            First argument is 2D flatfield with noise.
            Second argument is 3D flatfield data.
            Third argument is 3D array representing defective detector pixels.
    """
    DetectorsDimV, projectionsNo, DetectorsDimH = np.shape(projData3D_clean)
    
    # output datasets
    flats_combined3D = np.zeros(
        (DetectorsDimV, flatsnum, DetectorsDimH), dtype='uint16')

    # normalise the data
    projData3D_clean /= np.max(projData3D_clean)
    
    # using spherical Bessel functions to emulate the background (scintillator)
    # variations
    func = spherical_yn(1, np.linspace(arguments_Bessel[0],
                                       arguments_Bessel[1], DetectorsDimV,
                                       dtype='float32'))
    func += abs(np.min(func))
    
    flatfield = np.zeros((DetectorsDimV, DetectorsDimH))
    for i in range(0, DetectorsDimH):
        flatfield[:, i] = func
    for i in range(0, DetectorsDimH):
        flatfield[:, i] += np.flipud(func)
    
    if specklesize != 0.0:
        # using speckle generator routines to create a photon count texture in
        # the background
        speckle_background = simulate_speckles_with_shot_noise(
            [DetectorsDimV, DetectorsDimH], 1, specklesize, kbar)
    else:
        speckle_background = np.ones((DetectorsDimV, DetectorsDimH))

    # model miscallibrated detectors (possible path to generate ring artifacts)
    blurred_speckles_map = np.zeros(
        (DetectorsDimV, DetectorsDimH, variations_number))
    for i in range(0, variations_number):
        speckles = simulate_speckles_with_shot_noise(
            [DetectorsDimV, DetectorsDimH], 1, 10, 0.03)
        # blur the speckled background
        blurred_speckles = gaussian_filter(speckles.copy(), sigma=sigmasmooth)
        # threshold the result
        blurred_speckles[blurred_speckles < 0.6*np.max(blurred_speckles)] = 0
        blurred_speckles_map[:, :, i] = blurred_speckles
    blurred_speckles_map /= np.max(blurred_speckles_map)
    
    # prepeare flat fields
    for i in range(0, flatsnum):
        # add speckled background to initial image with the Bessel background
        flatfield_combined = flatfield.copy() + 0.5*(
                speckle_background/np.max(speckle_background))
        flatfield_combined /= np.max(flatfield_combined)
        
        # adding Poisson noise to flat fields
        flatfield_poisson = noise(flatfield_combined*source_intensity,
                                  source_intensity, noisetype='Poisson')
        flatfield_poisson /= np.max(flatfield_poisson)
        
        flats_combined3D[:, i, :] = np.uint16(flatfield_poisson*65535)

    return flatfield_poisson, np.uint16(flats_combined3D), blurred_speckles_map
