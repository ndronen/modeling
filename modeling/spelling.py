from __future__ import absolute_import
from .utils import load_model

import string
import codecs
import six
import re
import itertools
import random
import cPickle
from collections import defaultdict
from operator import itemgetter

import progressbar
import json
import h5py

import Levenshtein
import enchant
import nltk

import numpy as np
import pandas as pd
from numpy.random import RandomState
import sklearn.cross_validation

MODEL_DIR = 'models/spelling/convnet/8d1c58f0737b11e5921d22000aec9897/'
INDEX_FILE = 'models/spelling/data/wikipedia-index.pkl'

def runspell(model_dir=MODEL_DIR, index_file=INDEX_FILE, k=5):
    index = cPickle.load(open(index_file))
    spell = load_model.load_model(model_dir)
    probs = spell.model.predict_proba(spell.data)
    return index, spell, probs
    #ranks = rank.compute_ranks(probs, spell.target)
    #top_k = rank.compute_top_k(probs, index['term'], index, k=k)
    #return index, spell, probs, ranks, top_k

def compute_ranks(probs, target):
    '''
    Compute the rank in a model's softmax output of each example.

    Parameters
    -----------
    probs : np.ndarray
        A 2-d array of softmax outputs with one example per row and one
        column per class.
    target : np.ndarray
        A 1-d array of target values.

    Return
    -----------
    ranks : np.ndarray
        A 1-d array of ranks, with 0 being the highest.
    '''
    ranks = np.zeros_like(target)
    for i in np.arange(len(probs)):
        idx = np.where(np.argsort(probs[i]) == target[i])[0]
        ranks[i] = probs.shape[1] - 1 - idx
    return ranks

def compute_top_k(probs, target, term_index, k=5):
    '''
    Compute the K most probable terms for each example in a model's
    softmax output.

    Parameters
    -----------
    probs : np.ndarray
        A 2-d array of softmax outputs with one example per row and one
        column per class.
    target : np.ndarray
        A 1-d array of target values.
    term_index : dict
        A mapping from vocabulary indices to vocabulary terms.

    Returns
    ----------
    top_k : dict of list of (string, list)
        A mapping from vocabulary terms to a list of lists of the K
        most probable terms and their probabilities.  The inner list has
        one entry for each row in `probs` (equivalently, each entry in
        `target`).  The outer list has one entry per row in `probs`.
    '''
    top_k = defaultdict(list)
    for i in np.arange(len(probs)):
        top_k_tokens = []
        for idx in np.argsort(probs[i])[-k:]:
            try:
                top_k_tokens.append((term_index[idx], probs[i][idx]))
            except KeyError as e:
                print('Unknown key at prediction {i} using prediction index {idx}'.format(
                    i=i, idx=idx))
        top_k_tokens = sorted(top_k_tokens, key=itemgetter(1), reverse=True)
        top_k[term_index[target[i]]].append(top_k_tokens)
    return top_k


# Data needs to be converted to input and targets.  An input is a window
# of k characters around a (possibly corrupted) token t.  A target is a
# one-hot vector representing w.  In English, the average word length
# is a little more than 5 and there are almost no words longer than 20
# characters [7].  Initially we will use a window of 100 characters.
# Since words vary in length, the token t will not be centered in the
# input.  Instead it will start at the 40th position of the window.
# This will (?) make the task easier to learn.
#
# [1] http://www.ravi.io/language-word-lengths

def load_data(path):
    with codecs.open(path, encoding='utf-8') as f:
        return f.read().replace('\n', ' ')

stanford_jar_path = '/work/stanford-corenlp-full-2015-04-20/stanford-corenlp-3.5.2.jar'

def build_tokenizer(tokenizer='stanford'):
    if tokenizer == 'stanford':
        return nltk.tokenize.StanfordTokenizer(
                path_to_jar=stanford_jar_path)
    else:
        return nltk.tokenize.TreebankWordTokenizer()

def is_word(token):
    return re.match(r'[\w.-]{2,}$', token)

def insert_characters(token, index_to_char, n=1, char_pool=string.ascii_lowercase, seed=17):
    if isinstance(seed, RandomState):
        rng = seed
    else:
        rng = RandomState(seed)

    new_token = token
    for i in six.moves.range(n):
        idx = rng.randint(len(new_token))
        #ch = index_to_char[rng.randint(len(index_to_char))]
        ch = rng.choice(list(char_pool))
        new_token = unicode(new_token[0:idx] + ch + new_token[idx:])
    return new_token

def delete_characters(token, index_to_char, n=1, char_pool=None, seed=17):
    if isinstance(seed, RandomState):
        rng = seed
    else:
        rng = RandomState(seed)

    new_token = token
    if n > len(new_token):
        n = len(new_token) - 1
    for i in six.moves.range(n):
        try:
            idx = max(1, rng.randint(len(new_token)))
            new_token = unicode(new_token[0:idx-1] + new_token[idx:])
        except ValueError, e:
            print('new_token', new_token, len(new_token))
            raise e
    return new_token

def replace_characters(token, index_to_char, n=1, char_pool=string.ascii_lowercase, seed=17):
    if isinstance(seed, RandomState):
        rng = seed
    else:
        rng = RandomState(seed)

    new_token = token
    for i in six.moves.range(n):
        idx = max(1, rng.randint(len(new_token)))
        #ch = index_to_char[rng.randint(len(index_to_char))]
        ch = rng.choice(list(char_pool))
        new_token = unicode(new_token[0:idx-1] + ch + new_token[idx:])
    return new_token

def transpose_characters(token, index_to_char, n=1, char_pool=None, seed=17):
    if isinstance(seed, RandomState):
        rng = seed
    else:
        rng = RandomState(seed)

    chars = set(token)
    if len(chars) == 1:
        return token

    new_token = token
    for i in six.moves.range(n):
        idx = max(1, rng.randint(len(new_token)))
        neighbor = 0
        if idx == 0:
            neighbor == 1
        elif idx == len(new_token) - 1:
            neighbor = len(new_token) - 2
        else:
            if rng.uniform() > 0.5:
                neighbor = idx + 1
            else:
                neighbor = idx - 1
        left = min(idx, neighbor) 
        right = max(idx, neighbor)
        new_token = unicode(new_token[0:left] + new_token[right] + new_token[left] + new_token[right+1:])
    return new_token

def tokenize(data, tokenizer=None):
    """
    Tokenize a string using a given tokenizer.

    Parameters 
    -----------
    data : str or unicode
        The string to be tokenized.
    tokenizer : str
        The name of the tokenizer.  Uses Stanford Core NLP tokenizer if
        'stanford'; otherwise, uses the Penn Treebank tokenizer.

    Returns
    ---------
    tokens : list
        An on-order list of the tokens.
    """
    toker = build_tokenizer(tokenizer=tokenizer)
    return [t.lower() for t in toker.tokenize(data)]


def build_index(token_seq, min_freq=100, max_features=1000, downsample=0):
    """
    Builds character and term indexes from a sequence of tokens.

    Parameters
    -----------
    token_seq : list
        A list of tokens from some text.
    min_freq : int
        The minimum number of occurrences a term must have to be included
        in the index.
    max_features : int
        The maximum number of terms to include in the term index.
    downsample : int
        The maximum number of occurrences to allow for any term.  Only
        used if > 0.
    """
    passes = 4
    if downsample > 0:
        passes += 1

    term_vocab = set()
    char_vocab = set([u' '])
    term_freqs = defaultdict(int)
    token_seq_index = defaultdict(list)
    below_min_freq = set()

    # Include most of the characters we care about.  We add any remaining
    # characters after this loop.
    for charlist in [string.ascii_letters, string.punctuation, range(10)]:
        for ch in charlist:
            char_vocab.add(unicode(str(ch)))

    pass_num = 1
    print('pass {pass_num} of {passes}: scanning tokens'.format(
        pass_num=pass_num, passes=passes))
    pass_num += 1
    pbar = progressbar.ProgressBar(term_width=40,
        widgets=[' ', progressbar.Percentage(),
        ' ', progressbar.ETA()],
        maxval=len(token_seq)).start()

    for i, token in enumerate(token_seq):
        pbar.update(i+1)

        if not is_word(token) or len(token) == 1:
            continue

        if term_freqs[token] == 0:
            below_min_freq.add(token)
        elif term_freqs[token] > min_freq:
            try:
                below_min_freq.remove(token)
            except KeyError:
                pass

        term_freqs[token] += 1
        token_seq_index[token].append(i)

        for ch in token:
            char_vocab.add(unicode(ch))

    pbar.finish()

    print('# of terms: ' + str(len(term_freqs)))

    print('pass {pass_num} of {passes}: removing infrequent terms'.format(
        pass_num=pass_num, passes=passes))
    pass_num += 1
    pbar = progressbar.ProgressBar(term_width=40,
        widgets=[' ', progressbar.Percentage(),
        ' ', progressbar.ETA()],
        maxval=len(term_freqs)).start()

    for i, term in enumerate(below_min_freq):
        pbar.update(i+1)
        del term_freqs[term]
        del token_seq_index[term]

    pbar.finish()
    print('# of terms: ' + str(len(term_freqs)))

    print('pass {pass_num} of {passes}: sorting terms by frequency'.format(
        pass_num=pass_num, passes=passes))
    pass_num += 1
    most_to_least_freq = sorted(term_freqs.iteritems(),
            key=itemgetter(1), reverse=True)
    print('')

    term_i = 0
    term_to_index = {}
    index_to_term = {}

    print('pass {pass_num} of {passes}: building term index'.format(
        pass_num=pass_num, passes=passes))
    pass_num += 1
    pbar = progressbar.ProgressBar(term_width=40,
        widgets=[' ', progressbar.Percentage(),
        ' ', progressbar.ETA()],
        maxval=len(most_to_least_freq)).start()

    for i, (term, freq) in enumerate(most_to_least_freq):
        pbar.update(i+1)

        if len(term_vocab) == max_features:
            del token_seq_index[term]
            continue

        term_vocab.add(term)
        term_to_index[term] = term_i
        index_to_term[term_i] = term 
        term_i += 1

    pbar.finish()

    if downsample > 0:
        print('pass {pass_num} of {passes}: downsampling'.format(
            pass_num=pass_num, passes=passes))
        pass_num += 1
        pbar = progressbar.ProgressBar(term_width=40,
            widgets=[' ', progressbar.Percentage(),
            ' ', progressbar.ETA()],
            maxval=len(token_seq_index)).start()
        rng = random.Random(17)
        for i, token in enumerate(token_seq_index.keys()):
            pbar.update(i+1)
            indices = token_seq_index[token]
            token_seq_index[token] = random.sample(indices, downsample)
        pbar.finish()

    char_to_index = dict((c,i+1) for i,c in enumerate(char_vocab))
    char_to_index['NONCE'] = 0

    return token_seq_index, char_to_index, term_to_index


def min_dictionary_edit_distance(terms, dictionary, progress=False):
    """
    Find the edit distance from each of a list of terms to the nearest
    word in the dictionary (where the dictionary presumably defines
    nearness as edit distance).

    Parameters
    -----------
    terms : list
        A list of terms.
    dictionary : enchant.Dict
        A dictionary.

    Returns 
    ----------
    distances : dict
        A dictionary with terms as keys and (distance,term,rank,rejected
        suggestions) as values.
    """
    pbar = None
    if progress:
        pbar = progressbar.ProgressBar(term_width=40,
            widgets=[' ', progressbar.Percentage(),
            ' ', progressbar.ETA()],
            maxval=len(terms)).start()

    distances = {}
    for i, t in enumerate(terms):
        if progress:
            pbar.update(i+1)
        suggestions = [s.lower() for s in dictionary.suggest(t)]
        # If the token itself is the top suggestion, then compute
        # the edit distance to the next suggestion.  We should not
        # return an edit distance of 0 for any token.
        orig_suggestions = list(suggestions)
        rejected_suggestions = []
        rank = 0
        accepted_suggestion = None
        accepted_suggestion_rank = 0
        distance = np.inf

        for suggestion in suggestions:
            rank += 1
            d = Levenshtein.distance(t, suggestion)
            if suggestion == t:
                rejected_suggestions.append((suggestion, d, rank))
            elif ' ' in suggestion or '-' in suggestion or "'" in suggestion:
                # For this study, we don't want to accept suggestions
                # that split a word (e.g. 'antelope' -> 'ant elope'.
                rejected_suggestions.append((suggestion, d, rank))
            elif suggestion == t + 's' or suggestion + 's' == t:
                # Exclude singular-plural variants.
                rejected_suggestions.append((suggestion, d, rank))
            else:
                if d < distance:
                    if accepted_suggestion is not None:
                        rejected_suggestions.append((accepted_suggestion, distance, rank))
                    accepted_suggestion = suggestion
                    accepted_suggestion_rank = rank
                    distance = d
                else:
                    rejected_suggestions.append((suggestion, d, rank))

        distances[t] = {
                'accepted': accepted_suggestion,
                'distance': distance,
                'rank': accepted_suggestion_rank,
                'rejected': rejected_suggestions
                }
            
    if progress:
        pbar.finish()

    return distances

def build_dataset(token_seq, token_seq_index, term_to_index, char_to_index, max_token_length=15, leading_context_size=10, trailing_context_size=10, leading_separator='{', trailing_separator='}', n_examples_per_context=10, n_errors_per_token=[1], n_errors_per_context=[0], dictionary=enchant.Dict('en_US'), seed=17):
    """
    Build a dataset of examples of spelling erors and corresponding
    corrections for training a supervised spelling correction model.
    Each example consists of a window around a token in `token_seq`.  

    The error types are (random) insertion, (random) deletion, (random)
    replacement, and transposition.  (In the future other error types may
    be added, such as sampling characters from the token itself instead
    of randomly (for insertion or replacement) and creating errors that
    are plausible given the layout of a QWERTY keyboard.

    Parameters
    ------------
    token_seq : list 
        A sequence of tokens.
    token_seq_index : dict of list
        A mapping from token to indices of occurrences of the token.
    term_to_index : dict
        A mapping from token to the index of the token in the vocabulary.
    char_to_index : dict
        A mapping from character to the index of the character in the vocabulary.
    max_token_length : int
        The maximum allowed token length for an (example, correction)
        pair.  Tokens that exceed this limit are ignored and no examples
        of spelling errors of this token are created.
    leading_context_size : int
        The size of the window for each token; includes the length of the token itself.
    trailing_context_size : int
        The position at which the token should be placed in the window.
    leading_separator : str
        A string that will be placed immediately before the spelling
        error to demarcate it from the leading context.
    trailing_separator : str
        A string that will be placed immediately after the spelling
        error to demarcate it from the trailing context.
    n_examples_per_context : int
        The number of examples of errors that will be generated per context.
    n_errors_per_token : list of int
        The possible number of errors injected into a word for a training
        example.  The number of errors injected into a word is sampled from
        this list for each training example.
    n_errors_per_context : list of int
        The possible number of errors injected into the leading and trailing
        context for a training example.  The number of errors injected into
        both the leading and trailing context is sampled from this list for
        each training example.
    dictionary : enchant.Dict
        Dictionary for computing edit distance to nearest term.
    seed : int or np.random.RandomState
        The initialization for the random number generator.

    Returns (FIXME: now returns data frame)
    ---------
    spelling_errors : np.ndarray
        A matrix of examples of a deliberately-injected spelling error.
    corrections : np.ndarray
        An array consisting of the indices in the token vocabulary of
        the correct word for each example.
    error_types : list
        A list of the type of error injected into the token.
    """
    if isinstance(seed, RandomState):
        rng = seed
    else:
        rng = RandomState(seed=seed)

    n_contexts = sum([len(token_seq_index[t]) for t in token_seq_index.keys() if len(t) <= max_token_length])
    n_examples = n_examples_per_context * n_contexts
    max_inserts_per_token = max(n_errors_per_token)
    max_inserts_per_context = 2*max(n_errors_per_context)
    max_chars_in_window = leading_context_size + max_token_length + trailing_context_size + len(leading_separator) + len(trailing_separator) + max_inserts_per_token + max_inserts_per_context

    error_examples = np.zeros((n_examples, max_chars_in_window), dtype=np.int32)
    error_examples.fill(char_to_index[' '])

    leading_contexts = np.zeros((n_examples, leading_context_size + max_inserts_per_context))
    leading_contexts.fill(char_to_index[' '])
    spelling_errors = np.zeros((n_examples, max_token_length + max_inserts_per_token))
    spelling_errors.fill(char_to_index[' '])
    trailing_contexts = np.zeros((n_examples, trailing_context_size + max_inserts_per_context))
    trailing_contexts.fill(char_to_index[' '])

    correct_tokens = []
    corrupt_tokens = []
    corrections = np.zeros(n_examples, dtype=np.int32)
    error_types = []
    context_ids = []
    example_ids = []
    term_lens = []
    edit_distance_to_nearest_term = []
    nearest_term = []

    print('starting to construct {n_examples} examples'.format(
            n_examples=n_examples))

    pbar = progressbar.ProgressBar(term_width=40,
        widgets=[' ', progressbar.Percentage(),
        ' ', progressbar.ETA()],
        maxval=n_examples).start()
    n = 0

    index_to_char = dict((i,c) for c,i in char_to_index.iteritems())
    if 63 not in index_to_char.keys():
        # A workaround for now.  Delete ASAP.
        index_to_char[63] = '^'

    def add_chars_to_array(n, array, chars, default_char=' '):
        for k, ch in enumerate(chars):
            try:
                array[n, k] = char_to_index[ch]
            except KeyError, e:
                array[n, k] = char_to_index[default_char]

    for token in token_seq_index.keys():
        if len(token) > max_token_length:
            continue

        term_len = len(token)
        distances = min_dictionary_edit_distance([token], dictionary)
        distance = distances[token]['distance']
        term = distances[token]['accepted']

        for i in token_seq_index[token]:
            for j in six.moves.range(n_examples_per_context):
                pbar.update(n+1)
                corruptor = rng.choice([insert_characters, delete_characters,
                        replace_characters, transpose_characters])
                n_token_errors = rng.choice(n_errors_per_token)
                corrupt_token = corruptor(token, index_to_char, n=n_token_errors, seed=rng)

                n_context_errors = rng.choice(n_errors_per_context)
                leading = leading_context(token_seq, i, leading_context_size)
                leading = corruptor(leading, index_to_char, n=n_context_errors, seed=rng)
                trailing = trailing_context(token_seq, i, trailing_context_size)
                trailing = corruptor(trailing, index_to_char, n=n_context_errors, seed=rng)

                window = ''.join([leading, leading_separator,
                        corrupt_token, trailing_separator, trailing])

                add_chars_to_array(n, error_examples, window)
                add_chars_to_array(n, leading_contexts, leading)
                add_chars_to_array(n, spelling_errors, corrupt_token)
                add_chars_to_array(n, trailing_contexts, trailing)

                correct_tokens.append(token)
                corrections[n] = term_to_index[token]
                corrupt_tokens.append(corrupt_token)
                error_types.append(corruptor.__name__)
                context_ids.append(i)
                example_ids.append(n)
                edit_distance_to_nearest_term.append(distance)
                nearest_term.append(term)
                term_lens.append(term_len)

                n += 1

    pbar.finish()

    df = pd.DataFrame({
            'correction': corrections,
            'error_type': error_types,
            'context_id': context_ids,
            'example_id': example_ids,
            'edit_distance_to_nearest_term': edit_distance_to_nearest_term,
            'nearest': nearest_term,
            'term_length': term_lens,
            'correct_token': correct_tokens,
            'corrupt_token': corrupt_tokens
            })

    def add_columns(df, array, colname_prefix):
        colwidth = int(np.log10(array.shape[1]))+1
        colfmt = colname_prefix + '{col:0' + str(colwidth) + 'd}'
        for col in np.arange(array.shape[1]):
            colname = colfmt.format(col=col)
            df[colname] = array[:, col]

    add_columns(df, error_examples, 'full_error')
    add_columns(df, leading_contexts, 'leading_context')
    add_columns(df, spelling_errors, 'spelling_error')
    add_columns(df, trailing_contexts, 'trailing_context')

    return df

def context_is_complete(tokens, n):
    context = ' '.join(tokens)
    return len(context) >= n

def trim_leading_context(tokens, n):
    context = ' '.join(tokens)
    start = len(context) - n
    return context[start:]

def trim_trailing_context(tokens, n):
    context = ' '.join(tokens)
    return context[:n]

def leading_context(tokens, i, n):
    end = max(0, i-1)
    j = end
    context = []
    while j > 0:
        context.insert(0, tokens[j])
        if context_is_complete(context, n):
            break
        j -= 1
    if not context_is_complete(context, n):
        context.insert(0, ' ' * n)
    return trim_leading_context(context, n)

def trailing_context(tokens, i, n):
    start = min(len(tokens), i+1)
    j = start
    context = []
    while j < len(tokens):
        context.append(tokens[j])
        j += 1
        if context_is_complete(context, n):
            break
    if not context_is_complete(context, n):
        context.append(' ' * n)
    return trim_trailing_context(context, n)

def build_fixed_width_window(tokens, i, token, window_size=100, token_pos=40):
    leading = leading_context(tokens, i, n=token_pos)
    # Subtract 2 for spaces between leading, token, and trailing.
    n = window_size - token_pos - len(token) - 2
    trailing = trailing_context(tokens, i, n)
    return leading + ' ' + token + ' ' + trailing

def context_id_to_idx(ids, groups):
    '''
    '''
    idx = []
    for i in ids:
        idx.extend(groups[i])
    return idx

def build_splits(dataset, train_size, valid_size, by=['context_id'], seed=17):
    """
    Split a dataset created by `build_dataset` into train, validation, and test.
    """
    if isinstance(seed, RandomState):
        rng = seed
    else:
        rng = RandomState(seed)

    groups = dataset.groupby(by).groups
    context_ids = groups.keys()

    train_ids, other_ids = sklearn.cross_validation.train_test_split(
            context_ids, train_size=train_size, random_state=rng)
    valid_ids, test_ids = sklearn.cross_validation.train_test_split(
            other_ids, train_size=valid_size, random_state=rng)

    train_idx = context_id_to_idx(train_ids, groups)
    valid_idx = context_id_to_idx(valid_ids, groups)
    test_idx = context_id_to_idx(test_ids, groups)

    return dataset.ix[train_idx, :], dataset.ix[valid_idx, :], dataset.ix[test_idx, :]

def dataset_to_hdf5(df, filename, what = {'example_id': 'example_id', 'context_id': 'context_id', 'correction': 'correction', 'edit_distance_to_nearest_term': 'edit_distance_to_nearest_term', 'leading_context': r'leading_context\d{2}', 'spelling_error': r'spelling_error\d{2}', 'full_error': r'full_error\d{2}', 'trailing_context': r'trailing_context\d{2}'}):

    f = h5py.File(filename, 'w')
    for name, pattern in what.iteritems():
        if name in df.columns:
            data = df[name].values
        else:
            mask = [True if re.match(pattern, s) else False for s in df.columns]
            data = df.ix[:, np.where(mask)[0]].values
        f.create_dataset(name, data=data, dtype=int)
    f.close()

def save_index(term_to_index, char_to_index, output_prefix):
    indices = {}

    if term_to_index is not None:
        index_to_term = dict((i,t) for t,i in term_to_index.iteritems())
        indices['term'] = index_to_term

    if char_to_index is not None:
        index_to_char = dict((i,t) for t,i in char_to_index.iteritems())
        indices['char'] = index_to_char

    if len(indices):
        cPickle.dump(indices, open(output_prefix + 'index.pkl', 'w'))

def save_target_data(term_to_index, output_prefix): 
    """
    Save vocabulary and weights (right now just 1's) to JSON for use 
    by training scripts.
    """
    index_to_term = dict((i,t) for t,i in term_to_index.iteritems())
    names = sorted(index_to_term.iteritems(), key=itemgetter(0))
    names = [n[1] for n in names]
    target_data = {}
    target_data['y'] = {}
    target_data['y']['names'] = names
    target_data['y']['weights'] = dict(zip(range(len(names)), [1] * len(names)))
    json.dump(target_data, open(output_prefix + 'target-data.json', 'w'))
