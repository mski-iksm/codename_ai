import unittest

import numpy as np
import pandas as pd

from codename_ai.model.scoring import ScoringWithRedAndBlue


class TestScoringWithRedAndBlue(unittest.TestCase):

    def test_total_scores(self):
        my_target_words = ['a', 'b']
        opponent_target_words = ['c']
        distance_data_dict = {
            # 大きい程遠い
            'a': {
                'AAA': 0.,
                'BBB': -1.,
                'CCC': 0.7,
                'DDD': -0.1
            },
            'b': {
                'AAA': 0.,
                'BBB': -1.,
                'CCC': 0.7,
                'DDD': 0.
            },
            'c': {
                'AAA': 1.0,
                'BBB': -1.,
                'CCC': 1.0,
                'DDD': 1.0
            }
        }
        resulted = ScoringWithRedAndBlue.calculate_scores(my_target_words=my_target_words,
                                                          opponent_target_words=opponent_target_words,
                                                          distance_data_dict=distance_data_dict)
        expected_total_scores = pd.DataFrame([
            [0.98, 2, ['a', 'b'], 1.96],
            [None, 0, [], None],
            [0.28, 2, ['a', 'b'], 0.56],
            [1.03, 2, ['a', 'b'], 2.06],
        ],
                                             index=['AAA', 'BBB', 'CCC', 'DDD'],
                                             columns=['score', 'count', 'expecting_my_target_word', 'total_score'])
        expected = ScoringWithRedAndBlue(candidates_table=expected_total_scores)

        pd.testing.assert_frame_equal(resulted._candidates_table, expected._candidates_table)

    def test_get_best_word_and_count(self):
        expected_total_scores = pd.DataFrame([
            [0.98, 2, ['a', 'b'], 1.96],
            [None, 0, [], None],
            [0.28, 2, ['a', 'b'], 0.56],
            [1.03, 2, ['a', 'b'], 2.06],
        ],
                                             index=['AAA', 'BBB', 'CCC', 'DDD'],
                                             columns=['score', 'count', 'expecting_my_target_word', 'total_score'])
        scoing_model = ScoringWithRedAndBlue(candidates_table=expected_total_scores)
        resulted_best_candidate_word, resulted_expect_count, resulted_expect_words = scoing_model.get_best_word_and_count()

        expected_best_candidate_word = 'DDD'
        expected_expect_count = 2
        expected_expect_words = ['a', 'b']

        self.assertEqual(resulted_best_candidate_word, expected_best_candidate_word)
        self.assertEqual(resulted_expect_count, expected_expect_count)
        self.assertListEqual(resulted_expect_words, expected_expect_words)

    def test_calculate_total_score_with_missing_opponent_target_words(self):
        my_target_words = ['a', 'b']
        opponent_target_words = ['c']
        distance_data_dict = {
            # 大きい程遠い
            'a': {
                'BBB': 1.,
            },
            'b': {
                'AAA': 1.,
                'BBB': 1.,
            },
            'c': {
                'AAA': 2.,
            }
        }
        resulted = ScoringWithRedAndBlue.calculate_scores(my_target_words=my_target_words,
                                                          opponent_target_words=opponent_target_words,
                                                          distance_data_dict=distance_data_dict,
                                                          my_target_score_offset=0.,
                                                          fillna_distance_for_me=5.,
                                                          fillna_distance_for_opponent=5.)
        expected_total_scores = pd.DataFrame([
            [1., 1, ['b'], 1.],
            [4., 2, ['a', 'b'], 8.],
        ],
                                             index=['AAA', 'BBB'],
                                             columns=['score', 'count', 'expecting_my_target_word', 'total_score'])
        expected = ScoringWithRedAndBlue(candidates_table=expected_total_scores)

        pd.testing.assert_frame_equal(resulted._candidates_table, expected._candidates_table)
