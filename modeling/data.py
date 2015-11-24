import os
import h5py
import numpy as np

def split_data(hdf5_path, split_size, output_dir=None):
    """
    Split the datasets in an HDF5 file into smaller sets and save them
    to new files.  By default the files are put into a subdirectory of
    the directory containing `hdf5_path`.  The subdirectory is created
    if it does not exist; the name of the directory is `hdf5_path` with
    the file suffix removed.  To write to a different directory, provide
    the path to the existing directory in `output_dir`.

    Parameters
    -------
    hdf5_path : str
        The path to the HDF5 file.
    split_size : int
        The size of the 
    """
    f = h5py.File(hdf5_path)
    n = 0
    # Find the largest n.
    for k,v in f.iteritems():
        n = max(n, v.value.shape[0])

    if output_dir is None:
        output_dir = os.path.splitext(hdf5_path)[0]
        os.mkdir(output_dir)

    # Copy subsequences of the data to smaller files.
    width = int(np.ceil(np.log10(n / split_size)))
    for i,j in enumerate(range(0, n, split_size)):
        outfile = '{dir}/{num:{fill}{width}}.h5'.format(
                dir=output_dir, num=i, fill='0', width=width)
        print(outfile)
        fout = h5py.File(outfile, 'w')
        for k,v in f.iteritems():
            subset = v[j:j+split_size]
            fout.create_dataset(k, data=subset, dtype=v.dtype)
        fout.close()

def balance_classes(target):
    """
    Get a subset of the indices in the target variable of an imbalanced dataset
    such that each class has the same number of occurrences.  This is to be used
    in conjunction with `balance_datasets` to create a balanced dataset.

    Parameters
    ---------
    target : array-like of int
        The target variable from which to sample.
    """
    n = min(np.bincount(target))
    n_even = n/2
    indices = []

    for code in np.arange(max(target)+1):
        mask = target == code
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

def balance_datasets(hdf5_file, key='original_word_code'):
    """
    Balance the datasets in an HDF5 file.  A balanced sample of
    the dataset denoted by `key` is taken.  The corresponding 
    examples from all other datasets are sampled, too.

    Parameters
    -----------
    hdf5_file : h5py.File
        An open HDF5 file.
    key : str
        The key of the target variable in `hdf5_file` to balance.  
    """
    idx = balance_classes(hdf5_file[key].value)
    for key in hdf5_file.keys():
        value = hdf5_file[key].value
        del hdf5_file[key]
        hdf5_file.create_dataset(key, data=value[idx], dtype=value.dtype)

def mask_zero_for_rnn(hdf5_fh, n_vocab):
    """
    Given an HDF5 data set with inputs `X` (the entire sentence),
    `Xwindow` (the window of words around e.g. a preposition), and
    `XwindowNULL` (the window of words as in `Xwindow` with the center
    word replaced by a nonce), transform the inputs as follows:

        a) Change 0 in every position before the end of the sentence to
           vocab_size + 1.
        b) Change 0 in every position after the beginning of the sentence
           to vocab_size + 1.

    Unmodified, the inputs `X`, etc., use 0 to indicate both that the
    word is unknown and that the sentence has ended (i.e. for padding
    a variable-length input like a sentence to fill all of the columns
    of a matrix).  The reasons to change this is that (1) some models,
    like recurrent neural networks, pay attention to every detail of
    their input and (2) some frameworks, like Keras, allow you do mask
    out 0's, so the model gets less confused.

    The `len` key has the offset at which the sentence ends in `X`.

    The `window_position` key in the data set has the offset at which
    the preposition occurs in `X`.

    Parameters
    ------------
    hdf5_fh : 
        A open, writable HDF5 file.
    n_vocab : int
        The number of words in the model's vocabulary.
    """
    XRNN = renumber_unknowns_in_sentence(
                hdf5_fh['X'].value,
                hdf5_fh['len'].value,
                n_vocab)
    hdf5_fh.create_dataset('XRNN', data=XRNN, dtype=XRNN.dtype)

    XwindowRNN = renumber_unknowns_in_window(
                hdf5_fh['Xwindow'].value,
                hdf5_fh['window_position'].value,
                n_vocab)
    hdf5_fh.create_dataset('XwindowRNN', data=XwindowRNN, dtype=XwindowRNN.dtype)

    XwindowNULLRNN = renumber_unknowns_in_window(
                hdf5_fh['XwindowNULL'].value,
                hdf5_fh['window_position'].value,
                n_vocab)
    hdf5_fh.create_dataset('XwindowNULLRNN', data=XwindowNULLRNN, dtype=XwindowNULLRNN.dtype)

    return hdf5_fh

def renumber_unknowns_in_sentence(X, lengths, n_vocab):
    """
    So, to transform `X` as described in item (a) above,

        * Find every occurrence of a 0 before the end of a sentence,
          using `len` to determine where the sentence ends.
        * Replace those occurences with `n_vocab`.
    """

    X = X.copy()
    for i,length in enumerate(lengths):
        sent = X[i]
        zeros_in_sent = [False] * X.shape[1]
        # Add 2 for leading '<s>' and trailing '</s>'.
        zeros_in_sent[:length+2] = sent[:length+2] == 0
        if np.any(zeros_in_sent):
            X[i, zeros_in_sent] = n_vocab
    return X

def renumber_unknowns_in_window(Xwindow, window_positions, n_vocab):
    """
    And to transform `Xwindow` and `XwindowNULL` for item (b),

        * Find every occurrence of a 0 after the beginning of a sentence
          using `window_position` to determine where in the window the
          sentence begins.  If `window_position` is 0, the first two
          positions in the window will be 0, because the preposition in
          that case is the first word in the sentence and it appears at
          the center of the window (index 2, with windows of length 5).
          Those first two words must remain 0, as they indicate the
          absence of words.  If `window_position` is 1, only the first
          word must remain 0; the word in the second position of the
          window could be 0 because it is out of vocabulary.  And if
          `window_position` is 2, then the first two words, if 0, are
          0 because they're out of vocabulary.  Thus, the indices in the
          window that should be checked for the "zero because out of
          vocabulary" case start at max(0, 2-`window_position`).  (NB:
          I didn't find any occurrences of `window_position` > `len`,
          just some occurrences of `window_position` == `len` - 2,
          which with sentence-terminating punctuation and the </s>
          padding character at the end of each sentence just means
          that there are several sentences that end with a preposition.
          So we only need to deal with the beginning of the window.)
        * Replace those occurrences with `n_vocab`.
    """
    Xwindow = Xwindow.copy()
    for i,window_position in enumerate(window_positions):
        window = Xwindow[i]
        start = max(0, 2 - window_position)
        zeros_in_window = window == 0
        zeros_in_window[0:start] = False
        if np.any(zeros_in_window):
            Xwindow[i, zeros_in_window] = n_vocab
    return Xwindow

def create_window(sentence, position, size=7, nonce=None):
    """
    Create a fixed-width window onto a sentence centered at some position.
    The sentence is assumed not to contain sentence-initial and -terminating
    markup (i.e. no '<s>' element immediately before the start of the
    sentence and no '</s>' immediately after its end).  (If they were included
    in `sentence`, we would exclude them for backward compatibility with other
    preprocesing code.)  It is also assumed not to be padded with trailing zeros.

    Parameters
    ---------
    sentence : np.ndarray
        An array of integers that represents a sentence.  The integers
        are indices in a model's vocabulary.
    position : int
        The 0-based index of the word in the sentence on which the window
        should be centered.
    size : int
        The size of the window.  Must be odd.
    nonce : int or None
        The index in the vocabulary of the nonce word to put at the
        center of the window, replacing the index of the existing word.
        When None, this does not occur.
    """
    if position < 0 or position >= len(sentence):
        raise ValueError("`position` (%d) must lie within sentence (len=%d)" % 
                (position, len(sentence)))

    # Get exactly the positions in `sentence` to copy to `window`.
    window_start = position - size/2
    window_end = position + size/2
    sent_range = np.arange(window_start, window_end+1)
    sent_mask = (sent_range >= 0) & (sent_range < len(sentence))
    sent_indices = sent_range[sent_mask]

    window_range = np.arange(0, size)
    window_indices = window_range[sent_mask]

    window = np.zeros(size)
    window[window_indices]
    sentence[sent_indices]
    window[window_indices] = sentence[sent_indices]

    if nonce is not None:
        window[size/2] = nonce

    return window

def create_windows(sentences, lengths, positions, window_size, nonce=None):
    windows = np.zeros((len(sentences), window_size))
    for i, sentence in enumerate(sentences):
        length = lengths[i]
        position = positions[i]
        sentence_without_zero_padding = sentence[0:length+2]
        sentence_without_markup = sentence_without_zero_padding[1:-1]
        windows[i] = modeling.data.create_window(
                sentence_without_markup, 
                position=position,
                window_size=window_size,
                nonce=nonce)
    return windows

def add_window_dataset(hdf5_file, name, window_size, nonce=None, sentences_name='X'):
    sentences = hdf5_file[sentences_name].value
    lengths = hdf5_file['len'].value
    positions = hdf5_file['window_position'].value

    windows = create_windows(sentences, lengths, positions, window_size, nonce)
    hdf5_file.create_dataset(name, data=windows, dtype=np.int32)
