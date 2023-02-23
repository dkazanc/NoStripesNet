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
from tomophantom.supp.artifacts import _Artifacts_
from tomobar.supp.suppTools import normaliser
from .flatgen_mod import synth_flats_mod
from .flats_to_proj import add_flats_proj
from utils.data_io import rescale, save3DTiff


def generateSample(N_size, tot_objects, output_path=None, sampleNo=None,
                   verbose=False, visual=False):
    """Generate a 3D sample and simulate its forward projection.
    Samples consist of a series of spheres of various sizes and intensities.
    Parameters:
            N_size : int
                Cubic size of the sample generated.
            tot_objects : int
                Total number of spheres to generate.
            output_path : str
                 The output path to save the projection to.
                 If passed, `sampleNo` must also be passed. Default is None.
            sampleNo : int
                The number of the sample. This will be appended to
                `output_path` when the projection is saved.
            verbose : bool
                Print out some extra information when running.
                Default is False.
            visual : bool
                Plot slices of the sample. Default is False.
    Returns:
        np.ndarray
            The simulated projection of the generated sample.
    """
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
    (Objfoam3D, myObjects) = foam3D(x0min, x0max, y0min, y0max, z0min, z0max,
                                    c0min, c0max, ab_min, ab_max, N_size,
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
    # detector column count (horizontal)
    Horiz_det = int(np.sqrt(2) * N_size)
    # detector row count (vertical) (no reason for it to be > N)
    Vert_det = N_size
    angles_num = int(0.5 * np.pi * N_size)  # angles number
    angles = np.linspace(0.0, 179.9, angles_num, dtype='float32')  # in degrees

    if verbose:
        print("Building 3D analytical projection data with TomoPhantom")
    ProjData3D = TomoP3D.ObjectSino(N_size, Horiz_det, Vert_det, angles,
                                    myObjects)

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
            raise RuntimeError("If supplying an output path, a sample number "
                               "must also be supplied.")
        filename = os.path.join(output_path, str(sampleNo).zfill(4) + '_clean')
        save3DTiff(ProjData3D, filename)

    return ProjData3D


# %%
def simulateFlats(ProjData3D, N_size, I0=40000, flatsnum=20,
                  shifted_positions_no=5, shift_step=2, output_path=None,
                  sampleNo=None, verbose=False, visual=False):
    """Simulate flat fields to add realistic synthetic noise and artifacts to
    a projection.
    Also simulates vertically shifting the sample, as is done in real-life
    data to create a sinogram with no artifacts.
    Parameters:
        ProjData3D : np.ndarray
            3D projection to add noise & artifacts to.
        N_size : int
            Cubic size of the sample.
        I0 : int
            Source intensity which affects the amount of the Poisson noise
            added to data.
        flatsnum : int
            Number of flat fields to generate.
        shifted_positions_no : int
            Number of vertical shifts to simulate.
        shift_step : int
            Pixel step between shifts.
        output_path : str
             The output path to save the projection to.
             If passed, `sampleNo` must also be passed. Default is None.
        sampleNo : int
            The number of the sample. This will be appended to
            `output_path` when the projection is saved.
        verbose : bool
            Print out some extra information when running.
            Default is False.
        visual : bool
            Plot slices of the sample. Default is False.
    Returns:
        np.ndarray
            A 4D array where the axis 3 indicates the shift, and axes 0-2
            contain the projection with flat field noise & artifacts.
    """
    if verbose:
        print("Simulate synthetic flat fields ")

    # detector column count (horizontal)
    Horiz_det = int(np.sqrt(2) * N_size)
    # detector row count (vertical) (no reason for it to be > N)
    Vert_det = N_size
    angles_num = int(0.5 * np.pi * N_size)  # angles number
    intens_max_clean = np.max(ProjData3D)

    off_center_index = -int(shifted_positions_no / 2) * int(shift_step)

    projData3D_norm = np.zeros(
        (Vert_det, angles_num, Horiz_det, shifted_positions_no))

    if visual:
        plt.figure()
        plt.suptitle("2D Projection with a shifted flat field")

    # Simulate flat fields
    flats_no_noise, flats_combined3D, blurred_speckles_map = synth_flats_mod(
        ProjData3D,
        source_intensity=I0,
        variations_number=3,
        arguments_Bessel=(1, 25),
        specklesize=3,
        kbar=2,
        jitter_projections=0.0,
        sigmasmooth=2,
        flatsnum=flatsnum)
    flats_unshifted = flats_no_noise.copy()
    flats_3D_unshifted = flats_combined3D.copy()
    speckles_unshifted = blurred_speckles_map.copy()

    # Make "clean" projection; i.e. with flat field noise but no stripes
    # Hence why variations_number = -1 and detectors_miscallibration = 0
    projData3D_clean = add_flats_proj(ProjData3D,
                                      flats_no_noise,
                                      flats_combined3D,
                                      blurred_speckles_map,
                                      source_intensity=I0,
                                      variations_number=-1,
                                      detectors_miscallibration=0.0,
                                      jitter_projections=0.0)
    projData3D_clean = normaliser(projData3D_clean, flats_combined3D,
                                  darks=None, log='true', method='mean')
    projData3D_clean = rescale(projData3D_clean, b=intens_max_clean)
    # Save "clean" projection; i.e. with flat field noise but no stripes
    if output_path:
        if sampleNo is None:
            raise RuntimeError("If supplying an output path, a sample number "
                               "must also be supplied.")
        cleanPath = os.path.join(output_path, 'clean')
        filename = os.path.join(cleanPath, str(sampleNo).zfill(4) + '_clean')
        save3DTiff(projData3D_clean, filename)

    # Loop through shifts and add flats + speckles (causing stripes) to each
    for i in range(0, shifted_positions_no):
        if verbose:
            print("Offset vertical index {}".format(off_center_index))

        # Moving the flat field data vertically to emulate the movement of
        # the phantom (sample)
        flats_no_noise = np.roll(flats_unshifted, off_center_index,
                                 axis=0)
        flats_combined3D = np.roll(flats_3D_unshifted, off_center_index,
                                   axis=0)
        blurred_speckles_map = np.roll(speckles_unshifted, off_center_index,
                                       axis=0)
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


        if verbose:
            print("Normalise projections with flats")
        projData3D_norm[:, :, :, i] = normaliser(projData3D_raw,
                                                 flats_combined3D, darks=None,
                                                 log='true', method='mean')
        projData3D_norm[:, :, :, i] = rescale(projData3D_norm[:, :, :, i],
                                              b=intens_max_clean)
        # note that I'm saving the result in 4D data array for demonstration
        # purposes ONLY

        if visual:
            images = [flats_unshifted, flats_no_noise, ProjData3D[:, 0, :],
                      projData3D_clean[:, 0, :], projData3D_norm[:, 0, :]]
            titles = ['2D flat field unshifted',
                      f'2D flat field shifted by '
                      f'{off_center_index - shift_step}',
                      'Input', 'With Noise', 'With Stripes']
            idx = i * len(images)
            for j in range(len(images)):
                plt.subplot(shifted_positions_no, 5, idx + j+1)
                plt.imshow(images[j], cmap='gray')
                plt.title(titles[j])

        # Save normalised projection (i.e. sinogram with artifacts)
        if output_path:
            if sampleNo is None:
                raise RuntimeError("If supplying an output path, a sample "
                                   "number must also be supplied.")
            shiftDir = os.path.join(output_path, f'shift{i:02}')
            filename = os.path.join(shiftDir, f'{sampleNo:04}_shift{i:02}')
            save3DTiff(projData3D_norm[:, :, :, i], filename)

    if visual:
        plt.show()
    return projData3D_clean, projData3D_norm

# %%%


def simulateStripes(ProjData3D, percentage=1.2, max_thickness=3.0,
                    intensity=0.25, kind='mix', variability=0,
                    output_path=None, sampleNo=None, verbose=False,
                    visual=False):
    """Add synthetic stripes to a projection. The simulation is not as
    realistic as `simulateFlats`; no flat fields are simulated, so there is no
    noise; stripes are simply added in to sinograms.
    Parameters:
        ProjData3D : np.ndarray
            3D projection to add stripes to.#
        percentage : float
            Percentage of entire projection size to add stripes to.
            Must be in range (0, 100]. Default is 1.2
        max_thickness : float
            Maximum thickness of stripes.
            Must be in range [0, 10]. Default is 3.0
        intensity : float
            Stripe intensity multiplier. Stripe intensities are calculated
            like so:
                max intesity of projection * random float [-1.0, 0.5)
                * `intensity`
            Default is 0.25
        kind : str
            Type of stripes to generate. Must be either 'partial', 'full',
            or 'mix' (to get both partial and full stripes).
        variability : float
            Variability multiplier to incorporate change of intensity in the
            stripe.
        output_path : str
             The output path to save the projection to.
             If passed, `sampleNo` must also be passed. Default is None.
        sampleNo : int
            The number of the sample. This will be appended to
            `output_path` when the projection is saved.
        verbose : bool
            Print out some extra information when running.
            Default is False.
        visual : bool
            Plot slices of the sample. Default is False.
    Returns:
        np.ndarray
            The projection with stripes added in. Same shape as input.
    """
    if verbose:
        print("Simulating stripes on projection data...")
    _stripes_ = {'stripes_percentage': percentage,
                 'stripes_maxthickness': max_thickness,
                 'stripes_intensity': intensity,
                 'stripes_type': kind,
                 'stripes_variability': variability}
    # normalize data in range [0, 1]
    ProjData3D = rescale(ProjData3D, a=0, b=1)
    # add stripes
    projData3D_stripes = _Artifacts_(ProjData3D, **_stripes_)
    # stripes add a bit of extra intensity (i.e. > 1) so must be clipped back
    # to range [0, 1]
    projData3D_stripes = np.clip(np.abs(projData3D_stripes), 0, 1)

    if visual:
        titles = ["Raw Sinogram", "Sinogram with Stripes"]
        for i, img in enumerate([ProjData3D, projData3D_stripes]):
            plt.subplot(1, 2, i+1)
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            plt.title(titles[i])
        plt.show()

    if output_path:
        if sampleNo is None:
            raise RuntimeError("If supplying an output path, a sample number"
                               "must also be supplied.")
        shiftDir = os.path.join(output_path, f'shift00')
        filename = os.path.join(shiftDir, f'{sampleNo:04}_shift00')
        save3DTiff(projData3D_stripes, filename)

    return projData3D_stripes


if __name__ == '__main__':
    N_size = 256  # define the grid size (cubic)
    tot_objects = 300  # the total number of objects to generate
    I0 = 40000  # full-beam photon flux intensity
    flatsnum = 20  # the number of the flat fields generated
    shifted_positions_no = 5  # number of shifted vertical positions
    shift_step = 2  # the shift step of a sample in pixels
    projData3D = generateSample(N_size, tot_objects, verbose=True, visual=True)
    projData3D_norm = simulateFlats(projData3D, N_size, I0=I0,
                                    flatsnum=flatsnum,
                                    shifted_positions_no=shifted_positions_no,
                                    shift_step=shift_step, verbose=True)

    """
    note that when off_center_index = 0 this is your centered (normal) sample 
    to which you need to compare all the shifted ones. 
    So let me demonstrate the idea here
    """
    intens_max_clean = np.max(projData3D)
    slice_sel = 140
    # this is a sinogram of the centered array
    plt.figure()
    plt.imshow(projData3D_norm[slice_sel, :, :, 2], vmin=0,
               vmax=intens_max_clean)
    plt.title('Cenetred array sinogram view')

    plt.figure()
    plt.imshow(projData3D_norm[slice_sel, :, :, 4], vmin=0,
               vmax=intens_max_clean)
    plt.title('Shifted array sinogram view')

# so the idea here that this is the same slice of the data (sinogram) but it
# has got different noise and artifact distribution!
# %%
