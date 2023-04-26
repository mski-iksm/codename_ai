import unittest

from codename_ai.model.wordnet_model.wordnet_model import (
    CalculateSynsetDistanceWithWordNetBySynset,
    CalculateWordDistanceWithWordNet, _use_closest_distance)


class TestCalculateWordDistanceWithWordNet(unittest.TestCase):

    def test_use_closest_distance(self):
        synset_distances_by_synset = [{'a': 1., 'b': 2.}, {'a': 0.5}]
        resulted = _use_closest_distance(distance_dicts_list=synset_distances_by_synset)
        expected = {'a': 0.5, 'b': 2.}
        self.assertDictEqual(resulted, expected)

    def test_get_distance_by_word(self):
        synset_distances_by_synset = [{'10641755-n': 1., '02121620-n': 2.}, {'10641755-n': 0.5}]
        # '10641755-n': 犬とか
        # '02121620-n': 猫とか
        resulted = CalculateWordDistanceWithWordNet._get_distance_by_word(synset_distances_by_synset=synset_distances_by_synset)
        expected = {
            'いぬ': 0.5,
            'にゃんにゃん': 2.0,
            'ねんねこ': 2.0,
            'まわし者': 0.5,
            'キャット': 2.0,
            'スパイ': 0.5,
            'ネコ': 2.0,
            '回し者': 0.5,
            '回者': 0.5,
            '密偵': 0.5,
            '工作員': 0.5,
            '廻し者': 0.5,
            '廻者': 0.5,
            '探': 0.5,
            '探り': 0.5,
            '犬': 0.5,
            '猫': 2.0,
            '秘密捜査員': 0.5,
            '諜報員': 0.5,
            '諜者': 0.5,
            '間者': 0.5,
            '間諜': 0.5,
            '隠密': 0.5
        }
        self.assertDictEqual(resulted, expected)


class TestCalculateSynsetDistanceWithWordNetBySynset(unittest.TestCase):

    def test_run(self):
        target_synset = 'a'
        connected_synsets = [
            {
                'b': 1.,
                'c': 2.
            },
            {
                'c': 0.5
            },
        ]
        resulted = CalculateSynsetDistanceWithWordNetBySynset._run(target_synset=target_synset, connected_synsets=connected_synsets)
        expected = {'a': 0., 'b': 2., 'c': 1.5}
        self.assertDictEqual(resulted, expected)
