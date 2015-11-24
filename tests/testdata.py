# -*- coding: utf-8 -*-
from __future__ import absolute_import

#import six
import sys
import os
import numpy as np

import unittest
import modeling.data

class TestData(unittest.TestCase):
    def test_create_window_position_at_beginning(self):
        sentence = np.arange(1, 12)
        position = 0
        expected_window = [0, 0, 0, 1, 2, 3, 4]
        window = modeling.data.create_window(sentence, position,
                size=7)

        self.assertEqual(7, len(window))
        self.assertTrue(np.all(window == expected_window))

    def test_create_window_position_at_end_nonce(self):
        sentence = np.arange(1, 12)
        position = len(sentence) - 1
        nonce = 99
        expected_window = [8, 9, 10, nonce, 0, 0, 0]
        window = modeling.data.create_window(sentence, position,
                size=7, nonce=nonce)

        self.assertEqual(7, len(window))
        self.assertTrue(np.all(window == expected_window))

    def test_create_window_position_before_sentence(self):
        sentence = np.arange(1, 12)
        position = -1
        self.assertRaises(
                ValueError,
                modeling.data.create_window,
                sentence, position)

    def test_create_window_position_after_sentence(self):
        sentence = np.arange(1, 12)
        position = 12
        self.assertRaises(
                ValueError,
                modeling.data.create_window,
                sentence, position)

