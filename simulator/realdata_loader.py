import os
import numpy as np
from skimage.transform import resize
from tomophantom.supp.artifacts import stripes as add_stripes
from utils.data_io import saveTiff, loadTiff
from utils.misc import rescale
from utils.stripe_detection import detect_stripe_larix
from utils.tomography import TomoH5


def resize_chunk(chunk, new_shape):
    return resize(chunk, new_shape, anti_aliasing=True)


def get_raw_data(tomogram):
    """Given a 3D tomogram (or subset), return the raw sinograms in
    it.
    Parameters:
        tomogram : np.ndarray
            3D tomographic data. Can be a subset of the full tomogram.
            Must have shape (detector Y, angles, detector X).
    Returns:
        np.ndarray
            The raw sinograms in the given tomogram.
    """
    return resize_chunk(tomogram, (tomogram.shape[0], 402, 362))


def get_dynamic_data(tomogram, frame_angles=900):
    """Given a 3D tomogram (or subset), return the frames of the dynamic scan.
    Parameters:
        tomogram : np.ndarray
            3D tomographic data. Can be a subset of the full tomogram.
            Must have shape (detector Y, angles, detector X).
        frame_angles : int
            Number of angles per frame of dynamic scan.
    Returns:
        np.ndarray
            4D array of shape (chunk idx, no. frames, sino Y, sino X).
    """
    num_frames = tomogram.shape[1] // (frame_angles * 2)
    out = np.ndarray((tomogram.shape[0], num_frames, 402, 362))
    for s in range(tomogram.shape[0]):
        for f in range(0, num_frames * 2, 2):
            frame = tomogram[s, f * frame_angles:(f + 1) * frame_angles, :]
            frame = resize_chunk(frame, (402, 362))
            out[s, f//2] = frame
    return out


def get_paired_data(tomogram, mask=None):
    """Given a 3D tomogram (or subset), return the pair of a clean sinogram
    and the same sinogram with stripes.
    Parameters:
        tomogram : np.ndarray
            3D tomographic data. Can be a subset of the full tomogram.
            Must have shape (detector Y, angles, detector X).
        mask : np.ndarray
            Mask indicating locations of stripes in the tomogram. If not
            provided, a mask will be generated (this can take multiple hours).
            Must have same shape as `tomogram`. Default is None.
    Returns:
        np.ndarray
            The clean/stripe pairs in the given tomogram.
    """
    # Resize
    tomogram = resize_chunk(tomogram, (tomogram.shape[0], 402, 362))
    out = np.ndarray(tomogram.shape[0], dtype=[('real_artifact', '?'),
                                               ('stripe', 'f', (402, 362)),
                                               ('clean', 'f', (402, 362))])
    # Generate mask if not given
    if mask is None:
        print("Mask not given so generating one...")
        mask = detect_stripe_larix(tomogram, threshold=0.63)
        print(f"Done.")
    for s in range(tomogram.shape[0]):
        if s % 100 == 0:
            print(f"Processing sinogram {s}/{tomogram.shape[0]}...", end=' ')
        # if sinogram doesn't contain stripes, add stripes synthetically
        if np.sum(mask[s]) == 0:
            clean = tomogram[s]
            stripe_type = np.random.choice(['partial', 'full'])
            stripe = add_stripes(clean, percentage=1,
                                 maxthickness=2, intensity_thresh=0.2,
                                 stripe_type=stripe_type,
                                 variability=0.005)
            real_artifact = False
        else:
            # Otherwise, stripe is current sinogram
            clean = None  # clean does not exist
            stripe = tomogram[s]
            real_artifact = True
        out[s] = real_artifact, stripe, clean
        if s % 100 == 0:
            print("Done.")
    return out


def create_patches(data, patch_size):
    """Split a 2D array into a number of smaller 2D patches.
    Parameters:
        data : np.ndarray
            Data to be split into patches. If data does not evenly fit into
            the size of patch specified, it will be cropped.
        patch_size : Tuple[int, int]
            Size of patches. Must have form:
                (patch_height, patch_width)
    Returns:
        np.ndarray
            Array containing patches. Has shape:
                (num_patches, patch_height, patch_width)
    """
    # Check that data can be evenly split into patches of size `patch_size`
    remainder = np.mod(data.shape, patch_size)
    if remainder[0] != 0:
        # If patch height doesn't evenly go into image height, crop image
        if remainder[0] % 2 == 0:
            # If remainder is even, crop evenly on bottom & top
            data = data[remainder[0]//2:-(remainder[0]//2)]
        else:
            # Otherwise, crop one more from top
            data = data[remainder[0]//2:-(remainder[0]//2) - 1]
    if remainder[1] != 0:
        # If patch width doesn't evenly go into image width, crop image
        if remainder[1] % 2 == 0:
            # If remainder is even, crop evenly on left & right
            data = data[:, remainder[1]//2:-(remainder[1]//2)]
        else:
            # Otherwise, crop one more from right
            data = data[:, remainder[1]//2:-(remainder[1]//2) - 1]
    # First, split into patches by width
    num_patches_w = data.shape[1] // patch_size[1]
    patches_w = np.split(data, num_patches_w, axis=1)
    # Then, split into patches by height
    num_patches_h = data.shape[0] // patch_size[0]
    patches_h = [np.split(p, num_patches_h, axis=0) for p in patches_w]
    # Finally, combine into one array
    num_patches = num_patches_h * num_patches_w
    patches = np.asarray(patches_h).reshape(num_patches, *patch_size)
    return patches


def get_patch_paired_data(tomogram, mask=None, patch_size=(1801, 256)):
    """Given a 3D tomogram (or subset), split each sinogram into patches and
     return pairs of a clean patch and the same patch with stripes.
    Parameters:
        tomogram : np.ndarray
            3D tomographic data. Can be a subset of the full tomogram.
            Must have shape (detector Y, angles, detector X).
        mask : np.ndarray
            Mask indicating locations of stripes in the tomogram. If not
            provided, a mask will be generated (this can take multiple hours).
            Must have same shape as `tomogram`. Default is None.
        patch_size : Tuple[int, int]
            Size of patches to split each sinogram into. If a sinogram does not
            evenly go into the size given, it will be cropped.
    Returns:
        np.ndarray
            The clean/stripe pairs in the given tomogram.
    """
    out = []
    for s in range(tomogram.shape[0]):
        if s % 100 == 0:
            print(f"Processing sinogram {s}/{tomogram.shape[0]}...", end=' ')
        sino_patches = create_patches(tomogram[s], patch_size)
        mask_patches = create_patches(mask[s], patch_size)
        tmp_out = np.ndarray(len(sino_patches),
                             dtype=[('real_artifact', '?'),
                                    ('stripe', 'f', patch_size),
                                    ('clean', 'f', patch_size)])
        # Loop through each patch
        for p in range(len(sino_patches)):
            # if sinogram doesn't contain stripes, add stripes synthetically
            if mask_patches[p].sum() == 0:
                clean = sino_patches[p]
                # Add synthetic stripes to clean sinogram patch
                # currently done with TomoPhantom, other methods could be used
                stripe = add_stripes(clean, percentage=0.4, maxthickness=6,
                                     intensity_thresh=0.2, stripe_type='full',
                                     variability=0)
                # Clip back to original range
                stripe = np.clip(stripe, clean.min(), clean.max())
                real_artifact = False
            else:
                # Otherwise, stripe is current patch and clean is None
                clean = None
                stripe = sino_patches[p]
                real_artifact = True
            tmp_out[p] = real_artifact, stripe, clean
        out.append(tmp_out)
        if s % 100 == 0:
            print("Done.")
    return np.asarray(out)


def get_data(mode, data, chunk_size, chunk_num, hdf_idx, **kwargs):
    """Process the data according to the given mode.
    Parameters:
        mode : str
            Mode to process the data in. Must be one of 'raw', 'paired',
            'dynamic', 'patch'.
        data : np.ndarray
            The data to process.
        chunk_size : int
            Size of chunks that full dataset was split into.
        chunk_num : int
            Index of chunk that `data` belongs to.
        hdf_idx : int
            Index of HDF file that data was loaded from. Only applies when
            mode in [`paired`, 'patch'] and number of shifts is greater than 1.
        kwargs:
            Keyword arguments for subsequent processing functions.
            If mode == 'paired', kwarg 'mask' must be given.
            If mode == 'dynamic', kwarg 'frame_angles' must be given.
            If mode == 'patch', kwargs 'mask' and 'patch_size' must be given.
    Returns:
         np.ndarray
            Array of processed data to be saved to disk.
    """
    if mode == 'raw':
        return get_raw_data(data)
    elif mode in ['paired', 'patch']:
        if 'mask' not in kwargs:
            raise ValueError("A mask should be given.")
        # Get correct mask for current shift
        mask = kwargs['mask'][f'm{hdf_idx}']
        # Crop mask to correct size
        mask_idx = np.s_[:, chunk_num * chunk_size:(chunk_num+1) * chunk_size]
        mask = np.swapaxes(mask[mask_idx], 0, 1)
        if mode == 'paired':
            return get_paired_data(data, mask=mask)
        if 'patch_size' not in kwargs:
            raise ValueError("Patch size should be given.")
        patch_size = kwargs['patch_size']
        return get_patch_paired_data(data, mask=mask, patch_size=patch_size)
    elif mode == 'dynamic':
        if 'frame_angles' not in kwargs:
            raise ValueError("Angles per frame should be given.")
        return get_dynamic_data(data, frame_angles=kwargs['frame_angles'])
    else:
        raise ValueError(f"Mode must be one of ['raw', 'paired', 'dynamic', "
                         f"'patch']. Instead got mode = '{mode}'.")


def save_rescaled_sino(sino, imin, imax, path):
    """Helper function to rescale a sinogram and then save it.
    Also returns the min & max of the sinogram before rescaling.
    Parameters:
        sino : np.ndarray
            2D sinogram to save.
        imin : float
            Min value to rescale sinogram w.r.t.
        imax : float
            Max value to rescale sinogram w.r.t.
        path : str
            Path to save sinogram to.
    Returns:
        Tuple[float, float]
            Min and Max of sinogram before it was rescaled.
    """
    # Store sino min & max before rescaling
    smin, smax = sino.min(), sino.max()
    # Rescale sinogram w.r.t chunk min & max
    np.clip(sino, imin, imax, out=sino)
    sino = rescale(sino, b=65535, imin=imin, imax=imax)
    sino = sino.astype(np.uint16, copy=False)
    # Save sinogram to disk
    saveTiff(sino, path, normalise=False)
    return smin, smax


def save_chunk(chunk, root, mode, start=0, sample_no=0, shift_no=0):
    """Save a chunk of sinograms to disk, given a mode.
    Parameters:
        chunk: np.ndarray
            3D chunk containing sinograms to save.
            Must have shape (detector Y, angles, detector X).
        root : str
            Path to save sinograms to.
        mode : str
            Mode to save sinograms in.
            Must be one of ['raw', 'paired', 'dynamic', 'patch'].
        start : int
            Index of slice to start counting at.
        sample_no : int
            Number of sample. Used in filename when saving sinograms.
        shift_no : int
            Number of current shift. Used in filename when saving sinograms.
    Returns:
        Dict[str, Tuple[float, float]]
            Dictionary where key is path to tiff, and value is pair of min &
            max for each sinogram.
    """
    # Dictionary to store path & min/max of each sinogram
    minmax = {}
    # Store min & max of chunk for rescaling
    if mode == 'raw':
        chunk_min, chunk_max = chunk[20:-20].min(), chunk[20:-20].max()
        filepath = os.path.join(root, f'shift{shift_no:02}')
        for s in range(chunk.shape[0]):
            filename = f'{sample_no:04}_shift{shift_no:02}_{start + s:04}'
            savepath = os.path.join(filepath, filename)
            sino = chunk[s]
            # Store path & min/max in dictionary
            minmax[savepath] = save_rescaled_sino(sino, chunk_min, chunk_max,
                                                  savepath)
    elif mode == 'dynamic':
        chunk_min, chunk_max = chunk[20:-20].min(), chunk[20:-20].max()
        for s in range(chunk.shape[0]):
            for f in range(chunk.shape[1]):
                frame = chunk[s, f]
                filename = f'{sample_no:04}_frame{f:02}_{start + s:04}'
                savepath = os.path.join(root, filename)
                minmax[savepath] = save_rescaled_sino(frame, chunk_min,
                                                      chunk_max, savepath)
    elif mode == 'paired':
        chunk_min = np.nanmin(chunk[20:-20]['stripe'])
        chunk_max = np.nanmax(chunk[20:-20]['stripe'])
        for s in range(chunk.shape[0]):
            if s % 100 == 0:
                print(f"Saving sinogram {s}/{chunk.shape[0]}...", end=' ')
            real_artifact, stripe, clean = chunk[s]
            # Save to disk in correct directory based on whether artifact is
            # real or fake
            if real_artifact:
                filepath = os.path.join(root, 'real_artifacts')
            else:
                filepath = os.path.join(root, 'fake_artifacts')
            filename = f'{sample_no:04}_shift{shift_no:02}_{start + s:04}'
            stripe_path = os.path.join(filepath, 'stripe', filename)
            clean_path = os.path.join(filepath, 'clean', filename)
            # Save sino as tiff and store path & min/max in dictionary
            minmax[stripe_path] = save_rescaled_sino(stripe, chunk_min,
                                                     chunk_max, stripe_path)
            # clean may be all nan as it doesn't exist for real artifacts
            if not np.isnan(clean).all():
                minmax[clean_path] = save_rescaled_sino(clean, chunk_min,
                                                        chunk_max, clean_path)
            if s % 100 == 0:
                print("Done.")
    elif mode == 'patch':
        basename = f'{sample_no:04}_shift{shift_no:02}'
        for s in range(chunk.shape[0]):
            if s % 100 == 0:
                print(f"Saving sinogram {s}/{chunk.shape[0]}...", end=' ')
            # Rescale each patch w.r.t sinogram, rather than chunk
            sino_min = np.nanmin(chunk[s]['stripe'])
            sino_max = np.nanmax(chunk[s]['stripe'])
            for p in range(chunk.shape[1]):
                real_artifact, stripe, clean = chunk[s, p]
                if real_artifact:
                    filepath = os.path.join(root, 'real_artifacts')
                else:
                    filepath = os.path.join(root, 'fake_artifacts')
                filename = f'{basename}_{start + s:04}_w{p:02}'
                stripe_path = os.path.join(filepath, 'stripe', filename)
                clean_path = os.path.join(filepath, 'clean', filename)
                # Save sino as tiff and store path & min/max in dictionary
                minmax[stripe_path] = save_rescaled_sino(stripe, sino_min,
                                                         sino_max, stripe_path)
                # clean may be all nan
                if not np.isnan(clean).all():
                    minmax[clean_path] = save_rescaled_sino(clean, sino_min,
                                                            sino_max,
                                                            clean_path)
            if s % 100 == 0:
                print("Done.")
    return minmax


def chunk_generator(hdf_file, chunk_size, flat_file=None):
    """Generator that returns chunks of an HDF5 dataset.
    Parameters:
        hdf_file : str
            Path to HDF5 file containing data.
        chunk_size : int
            Size of chunks to load data in.
        flat_file : str
            Path to HDF5 file containing flat & dark fields. Default is None.
            If not given, flats & darks will be loaded from `hdf_file`.
            If no flats or darks exist in `hdf_file`, an error will be raised.
    Returns:
        Generator
            Generator representing each chunk in the dataset.
    """
    flats, darks = None, None
    if flat_file is not None:
        flat_h5 = TomoH5(flat_file)
        flats, darks = flat_h5.get_flats(), flat_h5.get_darks()
    tomo = TomoH5(hdf_file)
    num_sinos = tomo.shape[1]
    num_chunks = int(np.ceil(num_sinos / chunk_size))
    for c in range(num_chunks):
        print(f"Loading chunk {c+1}/{num_chunks}...", end=' ', flush=True)
        chunk_slice = np.s_[:, c*chunk_size:(c+1)*chunk_size, :]
        chunk = tomo.get_normalized(chunk_slice, flats, darks)
        # Swap axes so sinograms are in dimension 0
        # i.e. (detector Y, angles, detector X)
        chunk = np.swapaxes(chunk, 0, 1)
        print(f"Done.")
        yield chunk


def reload_save(shape, minmax):
    """Reload a whole dataset from disk, normalize each item w.r.t the whole
    dataset, then save each item to disk again.
    Parameters:
        shape : Tuple[int, int]
            Shape of items saved. Should be two-dimensional.
        minmax : Dict[str, Tuple[float, float]]
            Dictionary containing paths to each item & the item's min and max
            before it was saved. Used to rescale each item back to its original
            range.
    """
    full_tomo = np.ndarray((len(minmax), *shape))
    for idx, (path, (lo, hi)) in enumerate(minmax.items()):
        sino = loadTiff(path, normalise=True)
        sino = rescale(sino, a=lo, b=hi)
        full_tomo[idx] = sino
    # Clip full tomo so it is not skewed by outliers
    full_tomo = np.clip(full_tomo,
                        full_tomo[20:-20].min(), full_tomo[20:-20].max())
    # Normalize again w.r.t. whole 3D tomogram
    full_tomo = rescale(full_tomo, a=0, b=65535)
    # Convert to uint16
    full_tomo = full_tomo.astype(np.uint16, copy=False)
    # Save each sinogram again
    for idx, path in enumerate(minmax.keys()):
        saveTiff(full_tomo[idx], path, normalise=False)


def generate_real_data(root, hdf_file, mode, chunk_size, sample_no, no_shifts,
                       flat_file, **kwargs):
    """General function to generate a dataset from an HDF/nxs file.
    Splits the data into chunks, processes each chunk, then saves each chunk
    to disk. If there is more than 1 chunk, the whole dataset will be re-loaded
    and normalized again w.r.t. the entire 3D dataset.
    Parameters:
        root : str
            Directory to save data to.
        hdf_file : str
            HDF5/Nexus file containing real-life data.
        mode : str
            Type of data to generate. Must be one of 'raw', 'paired',
            'dynamic', or 'patch'.
        chunk_size : int
            Size of chunks to split data into. Data is chunked in axis 1.
        sample_no : int
            Number of sample being generated. Only used in filenames.
        no_shifts : int
            Number of shifts for the dataset.
        flat_file : str
            Path to HDF5/Nexus file containing flat & dark fields for
            normalization.
        kwargs:
            Keyword arguments for processing functions.
            If mode == 'paired', kwarg 'mask' must be given.
            If mode == 'dynamic', kwarg 'frame_angles' must be given.
            If mode == 'patch', kwargs 'mask' and 'patch_size' must be given.
    """
    file_num = int(os.path.basename(hdf_file).split('.')[0])
    for shift_no in range(no_shifts):
        print(f"Shift {shift_no+1}/{no_shifts}")
        hdf_idx = file_num + shift_no
        current_hdf = os.path.join(os.path.dirname(hdf_file), f'{hdf_idx}.nxs')
        rescale_dict = {}
        chunks = chunk_generator(current_hdf, chunk_size, flat_file)
        num_chunks = 0
        for chunk in chunks:
            data = get_data(mode, chunk, chunk_size, num_chunks, hdf_idx,
                            **kwargs)
            chunk_idx = chunk_size * num_chunks
            chunk_dict = save_chunk(data, root, mode, start=chunk_idx,
                                    sample_no=sample_no, shift_no=shift_no)
            rescale_dict.update(chunk_dict)
            num_chunks += 1
        if num_chunks > 1:
            print(f"Re-loading & normalizing data w.r.t entire 3D sample...")
            if mode == 'patch':
                reload_save(kwargs['patch_size'], rescale_dict)
            else:
                reload_save((402, 362), rescale_dict)
