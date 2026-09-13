#!/usr/bin/env python3

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import sklearn

from src.word_classification import (
    contains_valid_chars,
    get_features,
    get_features_and_labels,
    word_classification,
)


class TestWordClassification(unittest.TestCase):

    def test_get_features(self):
        a = np.array(["abc", "zaka"])
        f = get_features(a)
        self.assertEqual(
            f.shape[0], 2,
            msg="Feature array returned by get_features had incorrect "
                "shape! Expected 2 rows for the 2 input words.")
        self.assertEqual(
            f.shape[1], 29,
            msg="Feature array returned by get_features had incorrect "
                "shape! Expected 29 columns, one per letter of the "
                "alphabet.")
        self.assertEqual(
            f[0, 0], 1,
            msg="Feature array returned by get_features had incorrect "
                "content in pos [0,0]! 'abc' contains exactly one 'a'.")
        self.assertEqual(
            f[0, 1], 1,
            msg="Feature array returned by get_features had incorrect "
                "content in pos [0,1]! 'abc' contains exactly one 'b'.")
        self.assertEqual(
            f[0, 2], 1,
            msg="Feature array returned by get_features had incorrect "
                "content in pos [0,2]! 'abc' contains exactly one 'c'.")

        self.assertEqual(
            f[1, 0], 2,
            msg="Feature array returned by get_features had incorrect "
                "content in pos [1,0]! 'zaka' contains two 'a's.")
        self.assertEqual(
            f[1, 25], 1,
            msg="Feature array returned by get_features had incorrect "
                "content in pos [1,25]! 'zaka' contains one 'z'.")
        self.assertEqual(
            f[1, 10], 1,
            msg="Feature array returned by get_features had incorrect "
                "content in pos [1,10]! 'zaka' contains one 'k'.")

    def test_contains_valid_chars(self):
        alphabet = "abcdefghijklmnopqrstuvwxyzäö-"
        inputs = [alphabet, alphabet + "#", alphabet[1:], "", "ä"]
        expected = [True, False, True, True, True]
        for s, e in zip(inputs, expected):
            self.assertEqual(
                contains_valid_chars(s), e,
                msg="Incorrect result from contains_valid_chars(%r)! "
                    "Expected %r." % (s, e))

    def test_get_features_and_labels(self):
        X, y = get_features_and_labels()
        self.assertEqual(
            len(X.shape), 2,
            msg="Incorrect dimension of feature matrix X returned by "
                "get_features_and_labels! Expected a 2D array.")
        self.assertEqual(
            X.shape, (y.shape[0], 29),
            msg="Incorrect shape of feature matrix X returned by "
                "get_features_and_labels! Expected (%d, 29)."
                % (y.shape[0],))
        self.assertEqual(
            y.shape[0], 157006,
            msg="Incorrect shape of target vector y returned by "
                "get_features_and_labels! Expected 157006 labels total.")
        self.assertEqual(
            sum(y), 63260,
            msg="Incorrect content in target vector y returned by "
                "get_features_and_labels! Expected 63260 positive "
                "(Finnish) labels.")

    def test_word_classification(self):
        v = word_classification()
        self.assertEqual(
            len(v), 5,
            msg="Expected that function word_classification returns 5 "
                "accuracy scores (one per cross-validation fold)! "
                "Got %d." % (len(v),))

        good = True
        try:
            correct = [0.89370104, 0.89678673, 0.89758288, 0.89685042,
                       0.89643642]
            for a, b in zip(correct, v):
                self.assertAlmostEqual(
                    a, b, places=3,
                    msg="Incorrect accuracy score returned by "
                        "word_classification!")
        except AssertionError:
            good = False

        if not good:
            # Result of non-shuffled cross validation
            correct = [0.86833706, 0.96897443, 0.842957, 0.87366338,
                       0.88320352]
            for a, b in zip(correct, v):
                self.assertAlmostEqual(
                    a, b, places=3,
                    msg="Incorrect accuracy score returned by "
                        "word_classification!")

    def test_word_classification_calls(self):
        with patch("src.word_classification.cross_val_score",
                   wraps=sklearn.model_selection.cross_val_score) as cvs, \
             patch("src.word_classification.model_selection.KFold",
                   wraps=sklearn.model_selection.KFold) as kf:
            word_classification()
            cvs.assert_called()
            kf.assert_called()
            args, kwargs = kf.call_args
            self.assertIn(
                "random_state", kwargs,
                msg="You did not specify the random_state argument to "
                    "KFold!")
            self.assertEqual(
                kwargs["random_state"], 0,
                msg="Incorrect random_state argument to KFold! Expected 0.")
            self.assertIn(
                "shuffle", kwargs,
                msg="You did not specify the shuffle argument to KFold!")
            self.assertEqual(
                kwargs["shuffle"], True,
                msg="Incorrect shuffle argument to KFold! Expected True.")


if __name__ == '__main__':
    unittest.main()
