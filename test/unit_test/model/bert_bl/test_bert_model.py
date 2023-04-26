import unittest

import numpy as np
import torch

from codename_ai.model.bert_bl.bert_model import (
    CalculateWordDistanceWithBERT, GetMultipleSentencesBertVector)

# class TestCalculateWordDistanceWithBERT(unittest.TestCase):

#     def test_run(self):
#         target_word_vector = torch.Tensor([1, 2, 3, 4, 5])
#         candidate_word_vectors = {
#             'word1': torch.Tensor([1, 2, 3, 4, 6]),
#             'word2': torch.Tensor([1, 2, 3, 4, 5]),
#         }
#         resulted = CalculateWordDistanceWithBERT._run(target_word_vector=target_word_vector, candidate_word_vectors=candidate_word_vectors)
#         expected = np.array([0.99585927, 1.0000001])
#         np.testing.assert_allclose(resulted, expected, atol=1e-6)

# class TestGetMultipleSentencesBertVector(unittest.TestCase):

#     def test_sentence_to_vector(self):
#         resulted = GetMultipleSentencesBertVector._sentences_to_vector(input_sentences=['牧場'])
#         self.assertEqual(resulted.shape[0], 768)
