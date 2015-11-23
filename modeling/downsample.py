import numpy as np

def downsample_indices(original_word_code):
    n = min(np.bincount(original_word_code))
    n_even = n/2
    indices = []

    for code in np.arange(max(original_word_code)+1):
        mask = original_word_code == code
        idx = np.sort(np.where(mask)[0])
        # Only sample from the even indices so the downsampled dataset
        # still consists of pairs of positive and negative examples.
        even_idx = idx[idx % 2 == 0]
        sampled_even_idx = np.sort(np.random.choice(even_idx, size=n_even, replace=False))
        # Add the odd-numbered examples of errors.
        sampled_idx = np.concatenate([sampled_even_idx, sampled_even_idx+1])
        sampled_idx = np.sort(sampled_idx)
        indices.extend(sampled_idx)

    return np.sort(indices)

def downsample_hdf5_file(hdf5_file, idx):
    for key in hdf5_file.keys():
        value = hdf5_file[key].value
        del hdf5_file[key]
        hdf5_file.create_dataset(key, data=value[idx], dtype=value.dtype)
