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
    hdf5_fh.create_dataset('XRNN', data=X, dtype=X.dtype)

    XwindowRNN = renumber_unknowns_in_window(
                hdf5_fh['Xwindow'].value,
                hdf5_fh['window_position'].value,
                n_vocab)
    hdf5_fh.create_dataset('XwindowRNN', data=XwindowRNN, dtype=X.dtype)

    XwindowNULLRNN = renumber_unknowns_in_window(
                hdf5_fh['XwindowNULL'].value,
                hdf5_fh['window_position'].value,
                n_vocab)
    hdf5_fh.create_dataset('XwindowNULLRNN', data=XwindowNULLRNN, dtype=X.dtype)

def renumber_unknowns_in_sentence(X, lengths, n_vocab):
    """
    So, to transform `X` as described in item (a) above,

        * Find every occurrence of a 0 before the end of a sentence,
          using `len` to determine where the sentence ends.
        * Replace those occurences with `n_vocab`.
    """

    X = X.copy()
    for i,length in enumerate(lengths):
        sentence = X[i]
        # Add 2 for leading '<s>' and trailing '</s>'.
        zeros_in_sent = sentence[:length+2] == 0
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
