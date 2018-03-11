# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains code to read tokenized data from file,
truncate, pad and process it into batches ready for training"""


import random
import time
import re
import nltk

import numpy as np
from six.moves import xrange
from vocab import PAD_ID, UNK_ID
from bilm import Batcher
from copy import deepcopy


class Batch(object):
    """A class to hold the information needed for a training batch"""

    def __init__(self, context_ids, context_mask, context_tokens, qn_ids, qn_mask, qn_tokens, ans_span, ans_tokens, context_char, qn_char, context_pos_ids, qn_pos_ids, context_ne_ids, qn_ne_ids, context_em, uuids=None):
        """
        Inputs:
          {context/qn}_ids: Numpy arrays.
            Shape (batch_size, {context_len/question_len}). Contains padding.
          {context/qn}_mask: Numpy arrays, same shape as _ids.
            Contains 1s where there is real data, 0s where there is padding.
          {context/qn/ans}_tokens: Lists length batch_size, containing lists (unpadded) of tokens (strings)
          ans_span: numpy array, shape (batch_size, 2)
          uuid: a list (length batch_size) of strings.
            Not needed for training. Used by official_eval mode.
        """
        self.context_ids = context_ids
        self.context_mask = context_mask
        self.context_tokens = context_tokens
        self.context_char = context_char

        self.qn_ids = qn_ids
        self.qn_mask = qn_mask
        self.qn_tokens = qn_tokens
        self.qn_char = qn_char
        self.context_pos_ids = context_pos_ids
        self.qn_pos_ids = qn_pos_ids
        self.context_ne_ids = context_ne_ids
        self.qn_ne_ids = qn_ne_ids
        self.context_em = context_em

        self.ans_span = ans_span
        self.ans_tokens = ans_tokens

        self.uuids = uuids

        self.batch_size = len(self.context_tokens)


def split_by_whitespace(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(" ", space_separated_fragment))
    return [w for w in words if w]


def intstr_to_intlist(string):
    """Given a string e.g. '311 9 1334 635 6192 56 639', returns as a list of integers"""
    return [int(s) for s in string.split()]


def sentence_to_token_ids(sentence, word2id):
    """Turns an already-tokenized sentence string into word indices
    e.g. "i do n't know" -> [9, 32, 16, 96]
    Note any token that isn't in the word2id mapping gets mapped to the id for UNK
    """
    tokens = split_by_whitespace(sentence) # list of strings
    ids = [word2id.get(w, UNK_ID) for w in tokens]
    return tokens, ids

# NOTE: CHANGE
def token_to_pos_ne_id(tokens, pos_tag_id_map, ne_tag_id_map):
    """Turns an already-tokenized sentence string into word indices
    e.g. "i do n't know" -> [9, 32, 16, 96]
    Note any token that isn't in the word2id mapping gets mapped to the id for UNK
    """
    pos_tags = nltk.tree2conlltags(nltk.ne_chunk(nltk.pos_tag(tokens), binary=False))
    pos_id = []
    ne_id = []
    for tag_p in pos_tags:
        pos_tag = tag_p[1]
        ne_tag = tag_p[2]
        if pos_tag in pos_tag_id_map:
            pos_id.append(pos_tag_id_map[pos_tag])
        else:
            print ('pos tag mis match')
            pos_id.append(0)
        if ne_tag in ne_tag_id_map:
            ne_id.append(ne_tag_id_map[ne_tag])
        else:
            print ('ne tag mis match')
            ne_id.append(0)
    return pos_id, ne_id

def get_em(context_token, qn_token):
    result = [0] * len(context_token)
    qn_token_set = set(qn_token)
    for i in range(len(context_token)):
        if context_token[i] in qn_token:
            result[i] = 1
    return result

def get_pos_ne_id(line):
    str_result = line.split()
    return list(map(int, str_result))

def padded(token_batch, batch_pad=0):
    """
    Inputs:
      token_batch: List (length batch size) of lists of ints.
      batch_pad: Int. Length to pad to. If 0, pad to maximum length sequence in token_batch.
    Returns:
      List (length batch_size) of padded of lists of ints.
        All are same length - batch_pad if batch_pad!=0, otherwise the maximum length in token_batch
    """
    maxlen = max([len(x) for x in token_batch]) if batch_pad == 0 else batch_pad
    return [token_list + [PAD_ID] * (maxlen - len(token_list)) for token_list in token_batch]


def refill_batches(batches, word2id, context_file, qn_file, ans_file, context_pos_file, qn_pos_file, context_ne_file, qn_ne_file, batch_size, context_len, question_len, discard_long, batcher):
    """
    Adds more batches into the "batches" list.

    Inputs:
      batches: list to add batches to
      word2id: dictionary mapping word (string) to word id (int)
      context_file, qn_file, ans_file: paths to {train/dev}.{context/question/answer} data files
      batch_size: int. how big to make the batches
      context_len, question_len: max length of context and question respectively
      discard_long: If True, discard any examples that are longer than context_len or question_len.
        If False, truncate those exmaples instead.
    """
    print ("Refilling batches...")
    tic = time.time()
    examples = [] # list of (qn_ids, context_ids, ans_span, ans_tokens) triples
    context_line, qn_line, ans_line = context_file.readline(), qn_file.readline(), ans_file.readline() # read the next line from each
    context_pos_line, qn_pos_line, context_ne_line, qn_ne_line = context_pos_file.readline(), qn_pos_file.readline(), context_ne_file.readline(), qn_ne_file.readline()

    while context_line and qn_line and ans_line: # while you haven't reached the end

        # Convert tokens to word ids
        context_tokens, context_ids = sentence_to_token_ids(context_line, word2id)
        qn_tokens, qn_ids = sentence_to_token_ids(qn_line, word2id)
        ans_span = intstr_to_intlist(ans_line)
        context_tokens_char = deepcopy(context_tokens)
        qn_tokens_char = deepcopy(qn_tokens)


        # read the next line from each file
        context_line, qn_line, ans_line = context_file.readline(), qn_file.readline(), ans_file.readline()

        # get ans_tokens from ans_span
        assert len(ans_span) == 2
        if ans_span[1] < ans_span[0]:
            print ("Found an ill-formed gold span: start=%i end=%i" % (ans_span[0], ans_span[1]))
            continue
        ans_tokens = context_tokens[ans_span[0] : ans_span[1]+1] # list of strings

        # discard or truncate too-long questions
        if len(qn_ids) > question_len:
            if discard_long:
                continue
            else: # truncate
                qn_ids = qn_ids[:question_len]
                qn_tokens_char = qn_tokens_char[:question_len]

        # discard or truncate too-long contexts
        if len(context_ids) > context_len:
            if discard_long:
                continue
            else: # truncate
                context_ids = context_ids[:context_len]
                context_tokens_char = context_tokens_char[:context_len]

        # add to examples
        # NOTE: Change
        context_pos_id = get_pos_ne_id(context_pos_line)
        context_ne_id = get_pos_ne_id(context_ne_line)
        qn_pos_id = get_pos_ne_id(qn_pos_line)
        qn_ne_id = get_pos_ne_id(qn_ne_line)
        context_em = get_em(context_tokens_char, qn_tokens_char)
        examples.append((context_ids, context_tokens, qn_ids, qn_tokens, ans_span, ans_tokens, context_tokens_char, qn_tokens_char, context_pos_id, qn_pos_id, context_ne_id, qn_ne_id, context_em))

        # stop refilling if you have 160 batches
        if len(examples) == batch_size * 160:
            break

    # Once you've either got 160 batches or you've reached end of file:

    # Sort by question length
    # Note: if you sort by context length, then you'll have batches which contain the same context many times (because each context appears several times, with different questions)
    examples = sorted(examples, key=lambda e: len(e[2]))

    # Make into batches and append to the list batches
    for batch_start in range(0, len(examples), batch_size):

        # Note: each of these is a list length batch_size of lists of ints (except on last iter when it might be less than batch_size)
        context_ids_batch, context_tokens_batch, qn_ids_batch, qn_tokens_batch, ans_span_batch, ans_tokens_batch, context_tokens_char_batch, qn_tokens_char_batch, context_pos_id_batch, qn_pos_id_batch, context_ne_id_batch, qn_ne_id_batch, context_em_batch= list(zip(*examples[batch_start:batch_start+batch_size]))

        # NOTE: Change
        context_char_batch = batcher.batch_sentences(context_tokens_char_batch, context_len) # already padded
        qn_char_batch = batcher.batch_sentences(qn_tokens_char_batch, question_len) # already padded
        batches.append((context_ids_batch, context_tokens_batch, qn_ids_batch, qn_tokens_batch, ans_span_batch, ans_tokens_batch, context_char_batch, qn_char_batch,context_pos_id_batch, qn_pos_id_batch, context_ne_id_batch, qn_ne_id_batch, context_em_batch))

    # shuffle the batches
    random.shuffle(batches)

    toc = time.time()
    print ("Refilling batches took %.2f seconds" % (toc-tic))
    return


def get_batch_generator(word2id, context_path, qn_path, ans_path, context_pos_path, qn_pos_path, context_ne_path, qn_ne_path, batch_size, context_len, question_len, discard_long, batcher):
    """
    This function returns a generator object that yields batches.
    The last batch in the dataset will be a partial batch.
    Read this to understand generators and the yield keyword in Python: https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do

    Inputs:
      word2id: dictionary mapping word (string) to word id (int)
      context_file, qn_file, ans_file: paths to {train/dev}.{context/question/answer} data files
      batch_size: int. how big to make the batches
      context_len, question_len: max length of context and question respectively
      discard_long: If True, discard any examples that are longer than context_len or question_len.
        If False, truncate those exmaples instead.
    """
    context_file, qn_file, ans_file = open(context_path), open(qn_path), open(ans_path)
    context_pos_file, qn_pos_file, context_ne_file, qn_ne_file = open(context_pos_path), open(qn_pos_path), open(context_ne_path), open(qn_ne_path)
    batches = []

    while True:
        if len(batches) == 0: # add more batches
            refill_batches(batches, word2id, context_file, qn_file, ans_file, context_pos_file, qn_pos_file, context_ne_file, qn_ne_file, batch_size, context_len, question_len, discard_long, batcher)
        if len(batches) == 0:
            break

        # NOTE: CHANGE
        # Get next batch. These are all lists length batch_size
        (context_ids, context_tokens, qn_ids, qn_tokens, ans_span, ans_tokens, context_char, qn_char, context_pos_ids, qn_pos_ids, context_ne_ids, qn_ne_ids, context_em) = batches.pop(0)

        # Pad context_ids and qn_ids
        qn_ids = padded(qn_ids, question_len) # pad questions to length question_len
        context_ids = padded(context_ids, context_len) # pad contexts to length context_len
        context_pos_ids = padded(context_pos_ids, context_len)
        context_ne_ids = padded(context_ne_ids, context_len)
        qn_pos_ids = padded(qn_pos_ids, question_len)
        qn_ne_ids = padded(qn_ne_ids, question_len)
        context_em = padded(qn_ne_ids, context_len)

        # Make qn_ids into a np array and create qn_mask
        qn_ids = np.array(qn_ids) # shape (question_len, batch_size)
        qn_mask = (qn_ids != PAD_ID).astype(np.int32) # shape (question_len, batch_size)
        qn_pos_ids = np.array(qn_pos_ids)
        qn_ne_ids = np.array(qn_ne_ids)

        # Make context_ids into a np array and create context_mask
        context_ids = np.array(context_ids) # shape (context_len, batch_size)
        context_mask = (context_ids != PAD_ID).astype(np.int32) # shape (context_len, batch_size)
        context_pos_ids = np.array(context_pos_ids)
        context_ne_ids = np.array(context_ne_ids)
        context_em = np.array(context_em)

        # Make ans_span into a np array
        ans_span = np.array(ans_span) # shape (batch_size, 2)

        # Make into a Batch object
        batch = Batch(context_ids, context_mask, context_tokens, qn_ids, qn_mask, qn_tokens, ans_span, ans_tokens, context_char, qn_char, context_pos_ids, qn_pos_ids, context_ne_ids, qn_ne_ids, context_em)

        yield batch

    return
