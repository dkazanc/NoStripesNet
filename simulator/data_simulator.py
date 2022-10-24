#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 10:52:20 2022

@author: Daniil Kazantsev
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from tomophantom import TomoP3D
from tomophantom.randphant.generator import foam3D
from tomobar.supp.suppTools import normaliser
from flatgen_mod import synth_flats_mod
from flats_to_proj import add_flats_proj
from data_io import *


def generateSample(N_size, tot_objects, output_path=None, sampleNo=None, verbose=False, visual=False):
    # define ranges for parameters
    x0min = -0.9
    x0max = 0.9
    y0min = -0.9
    y0max = 0.9
    z0min = -0.8
    z0max = 0.8
    c0min = 0.01
    c0max = 1.0
    ab_min = 0.01
    ab_max = 0.25

    if verbose:
        print("Generating 3D random phantom")
    (Objfoam3D, myObjects) = foam3D(x0min, x0max, y0min, y0max, z0min, z0max, c0min, c0max, ab_min, ab_max, N_size,
                                    tot_objects, object_type='ellipsoid')

    if visual:
        plt.figure()
        sliceSel = int(0.5 * N_size)
        plt.subplot(131)
        plt.imshow(Objfoam3D[sliceSel, :, :], vmin=0, vmax=0.5)
        plt.title('3D Object, axial view')

        plt.subplot(132)
        plt.imshow(Objfoam3D[:, sliceSel, :], vmin=0, vmax=0.5)
        plt.title('3D Object, coronal view')

        plt.subplot(133)
        plt.imshow(Objfoam3D[:, :, sliceSel], vmin=0, vmax=0.5)
        plt.title('3D Object, sagittal view')
        plt.show()
    # %%

    if verbose:
        print("Generating exact 3D projection data")
    Horiz_det = int(np.sqrt(2) * N_size)  # detector column count (horizontal)
    Vert_det = N_size  # detector row count (vertical) (no reason for it to be > N)
    angles_num = int(0.5 * np.pi * N_size)  # angles number
    angles = np.linspace(0.0, 179.9, angles_num, dtype='float32')  # in degrees

    if verbose:
        print("Building 3D analytical projection data with TomoPhantom")
    ProjData3D = TomoP3D.ObjectSino(N_size, Horiz_det, Vert_det, angles, myObjects)

    if visual:
        plt.figure()
        plt.subplot(131)
        plt.imshow(ProjData3D[:, sliceSel, :])
        plt.title('2D Projection (analytical)')
        plt.subplot(132)
        plt.imshow(ProjData3D[sliceSel, :, :])
        plt.title('Sinogram view')
        plt.subplot(133)
        plt.imshow(ProjData3D[:, :, sliceSel])
        plt.title('Tangentogram view')
        plt.show()

    # Save Clean Sinogram
    if output_path:
        if sampleNo is None:
            raise RuntimeError("If supplying an output path, a sample number must also be supplied.")
        filename = os.path.join(output_path, str(sampleNo).zfill(4) + '_clean')
        save3DTiff(ProjData3D, filename)

    return ProjData3D


# %%
def simulateFlats(ProjData3D, N_size, I0=40000, flatsnum=20, shifted_positions_no=5, shift_step=2,
                  output_path=None, sampleNo=None, verbose=False, visual=False):
    if verbose:
        print("Simulate synthetic flat fields ")
    # Moving the flat field data verically to emulate the movement of the phantom (sample)

    Horiz_det = int(np.sqrt(2) * N_size)  # detector column count (horizontal)
    Vert_det = N_size  # detector row count (vertical) (no reason for it to be > N)
    angles_num = int(0.5 * np.pi * N_size)  # angles number
    intens_max_clean = np.max(ProjData3D)

    off_center_index = -int(shifted_positions_no / 2) * int(shift_step) + int(shift_step)

    projData3D_norm = np.zeros((Vert_det, angles_num, Horiz_det, shifted_positions_no))

    for i in range(0, shifted_positions_no):
        if verbose:
            print("Offset vertical index {}".format(off_center_index))
        [flats_no_noise, flats_combined3D, blurred_speckles_map] = synth_flats_mod(ProjData3D,
                                                                                   source_intensity=I0,
                                                                                   variations_number=3,
                                                                                   arguments_Bessel=(1, 25),
                                                                                   specklesize=3,
                                                                                   kbar=2,
                                                                                   jitter_projections=0.0,
                                                                                   sigmasmooth=2,
                                                                                   flatsnum=flatsnum)
        if visual:
            plt.figure()
            plt.imshow(flats_no_noise)
            plt.title('2D flat field without noise')
            plt.show()

        flats_no_noise = np.roll(flats_no_noise, off_center_index, axis=0)
        flats_combined3D = np.roll(flats_combined3D, off_center_index, axis=0)
        off_center_index += shift_step

        # adding flats to clean projections
        projData3D_raw = add_flats_proj(ProjData3D,
                                        flats_no_noise,
                                        flats_combined3D,
                                        blurred_speckles_map,
                                        source_intensity=I0,
                                        variations_number=3,
                                        detectors_miscallibration=0.07,
                                        jitter_projections=0.0)

        if visual:
            plt.figure()
            plt.imshow(projData3D_raw[:, 0, :])
            plt.title('2D Projection (before normalisation) with a shifted flat field')
            plt.show()

        print("Normalise projections with flats")
        projData3D_norm[:, :, :, i] = normaliser(projData3D_raw, flats_combined3D, darks=None, log='true', method='mean')
        projData3D_norm[:, :, :, i] *= intens_max_clean
        # note that I'm saving the result in 4D data array for demonstration purposes ONLY

        # Save normalised projection (i.e. sinogram with artifacts)
        if output_path:
            if sampleNo is None:
                raise RuntimeError("If supplying an output path, a sample number must also be supplied.")
            shiftDir = os.path.join(output_path, 'shift'+str(i).zfill(2))
            filename = os.path.join(shiftDir, str(sampleNo)+'_shift'+str(i).zfill(2))
            save3DTiff(projData3D_norm[:, :, :, i], filename)

    return projData3D_norm

# %%%


if __name__ == '__main__':
    N_size = 256  # define the grid size (cubic)
    tot_objects = 300  # the total number of objects to generate
    I0 = 40000  # full-beam photon flux intensity
    flatsnum = 20  # the number of the flat fields generated
    shifted_positions_no = 5  # the total number of the shifted vertical positions of a sample
    shift_step = 2  # the shift step of a sample in pixels
    projData3D = generateSample(N_size, tot_objects, verbose=True, visual=True)
    projData3D_norm = simulateFlats(projData3D, N_size, I0=I0, flatsnum=flatsnum,
                                    shifted_positions_no=shifted_positions_no, shift_step=shift_step, verbose=True)

    """
     note that when off_center_index = 0 this is your centered (normal) sample to which you need to compare
     all the shifted ones. So let me demonstrate the idea here
    """
    intens_max_clean = np.max(projData3D)
    slice_sel = 140
    # this is a sinogram of the centered array
    plt.figure()
    plt.imshow(projData3D_norm[slice_sel, :, :, 2], vmin=0, vmax=intens_max_clean)
    plt.title('Cenetred array sinogram view')

    plt.figure()
    plt.imshow(projData3D_norm[slice_sel, :, :, 4], vmin=0, vmax=intens_max_clean)
    plt.title('Shifted array sinogram view')

# so the idea here that this is the same slice of the data (sinogram) but it has got different noise and artifact distribution!
# %%
