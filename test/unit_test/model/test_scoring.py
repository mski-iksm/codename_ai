import unittest

import numpy as np
import pandas as pd

from codename_ai.model.scoring import ScoringWithRedAndBlue


class TestScoringWithRedAndBlue(unittest.TestCase):

    def test_total_scores(self):
        my_target_words = ['a', 'b']
        opponent_target_words = ['c']
        distance_data_dict = {
            'a': np.array([0, 0, 0.7, 1]),
            'b': np.array([0, 0.1, 0.7, 1]),
            'c': np.array([0, 100, 0.7, 1]),
        }
        candidate_words = ['AAA', 'BBB', 'CCC', 'DDD']
        resulted = ScoringWithRedAndBlue.calculate_scores(my_target_words=my_target_words,
                                                          opponent_target_words=opponent_target_words,
                                                          distance_data_dict=distance_data_dict,
                                                          candidate_words=candidate_words)
        expected_total_scores = pd.DataFrame([
            [0., 100., 0., 0.],
            [0., 99.9, 0., 0.],
            [1., 100.9, 0.3, 0.],
        ],
                                             index=[('a', ), ('b', ), ('a', 'b')],
                                             columns=['AAA', 'BBB', 'CCC', 'DDD'])
        expected = ScoringWithRedAndBlue(total_scores=expected_total_scores)

        pd.testing.assert_frame_equal(resulted._total_scores, expected._total_scores)

    def test_get_best_word_and_count(self):
        expected_total_scores = pd.DataFrame([
            [0., 100., 0., 0.],
            [0., 99.9, 0., 0.],
            [1., 100.9, 0.3, 0.],
        ],
                                             index=[('a', ), ('b', ), ('a', 'b')],
                                             columns=['AAA', 'BBB', 'CCC', 'DDD'])
        scoing_model = ScoringWithRedAndBlue(total_scores=expected_total_scores)
        resulted_best_candidate_word, resulted_expect_count, resulted_expect_words = scoing_model.get_best_word_and_count()

        expected_best_candidate_word = 'BBB'
        expected_expect_count = 2
        expected_expect_words = ('a', 'b')

        self.assertEqual(resulted_best_candidate_word, expected_best_candidate_word)
        self.assertEqual(resulted_expect_count, expected_expect_count)
        self.assertTupleEqual(resulted_expect_words, expected_expect_words)
