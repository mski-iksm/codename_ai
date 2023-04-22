import os
from typing import Dict, List, Tuple

import gokart
import luigi
import pandas as pd

wordnet_sense_word = pd.read_csv('data/wordnet_sense_word.csv')
wordnet_synlink = pd.read_csv('data/wordnet_synlink.csv')


def word2synsets(word: str) -> List[str]:
    return wordnet_sense_word.loc[wordnet_sense_word['lemma'] == word, 'synset'].tolist()


def synset2words(synset: str) -> List[str]:
    return wordnet_sense_word.loc[wordnet_sense_word['synset'] == synset, 'lemma'].tolist()


def _use_closest_distance(distance_dicts_list: List[Dict[str, float]]) -> Dict[str, float]:
    return {synset_keys: min(d.get(synset_keys, 9999) for d in distance_dicts_list) for synset_keys in set().union(*distance_dicts_list)}


def _get_connected_synsets(synset: str) -> Tuple[List[str], List[str]]:
    match_df = wordnet_synlink.loc[wordnet_synlink['synset1'] == synset]
    hypernym_synsets = list(set(match_df.loc[match_df['link'] == 'hype', 'synset2'].tolist()))
    other_synsets = list(set(match_df.loc[match_df['link'] != 'hype', 'synset2'].tolist()))
    return (hypernym_synsets, other_synsets)


class CalculateWordDistanceWithWordNet(gokart.TaskOnKart):
    target_word: str = luigi.Parameter()
    traverse_depth: int = luigi.IntParameter(default=3)

    __version = luigi.IntParameter(default=0)

    def output(self):
        relative_file_path = os.path.join(self.__module__.replace('.', '/'),
                                          f'{type(self).__name__}_v{self.__version}_dep{self.traverse_depth}_{self.target_word}.pkl')
        return self.make_target(relative_file_path=relative_file_path, use_unique_id=False)

    def requires(self):
        synsets = word2synsets(word=self.target_word)
        return [CalculateSynsetDistanceWithWordNetBySynset(target_synset=synset, traverse_depth=self.traverse_depth) for synset in synsets]

    def run(self):
        synset_distances_by_synset: List[Dict[str, float]] = self.load()
        self.dump(self._get_distance_by_word(synset_distances_by_synset=synset_distances_by_synset))

    @classmethod
    def _get_distance_by_word(cls, synset_distances_by_synset: List[Dict[str, float]]) -> Dict[str, float]:
        synset_distance_by_synset = _use_closest_distance(distance_dicts_list=synset_distances_by_synset)
        word_distance_dicts = [{word: distance for word in synset2words(synset=synset)} for synset, distance in synset_distance_by_synset.items()]
        synset_distance_by_word = _use_closest_distance(distance_dicts_list=word_distance_dicts)
        return synset_distance_by_word


class CalculateSynsetDistanceWithWordNetBySynset(gokart.TaskOnKart):
    target_synset: str = luigi.Parameter()
    traverse_depth: int = luigi.IntParameter()

    __version = luigi.IntParameter(default=0)

    def output(self):
        relative_file_path = os.path.join(self.__module__.replace('.', '/'),
                                          f'{type(self).__name__}_v{self.__version}_{self.target_synset}_dep{self.traverse_depth}.pkl')
        return self.make_target(relative_file_path=relative_file_path, use_unique_id=False)

    def requires(self):
        hypernym_synsets, other_synsets = _get_connected_synsets(synset=self.target_synset)
        hypernym_synset_tasks = [
            CalculateSynsetDistanceWithWordNetBySynset(target_synset=synset, traverse_depth=self.traverse_depth - 1) for synset in hypernym_synsets
        ] if self.traverse_depth > 0 else []
        other_synset_tasks = [
            CalculateSynsetDistanceWithWordNetBySynset(target_synset=synset, traverse_depth=self.traverse_depth - 1) for synset in other_synsets
        ] if self.traverse_depth >= 3 else []
        return hypernym_synset_tasks + other_synset_tasks

    def run(self):
        connected_synsets: List[Dict[str, float]] = self.load()
        self.dump(self._run(target_synset=self.target_synset, connected_synsets=connected_synsets))

    @classmethod
    def _run(cls, target_synset: str, connected_synsets: List[Dict[str, float]]) -> Dict[str, float]:
        increased_distance = cls._increment_distance(connected_synsets=connected_synsets)
        distances_list = increased_distance + [{target_synset: 0.}]
        closest_distance = _use_closest_distance(distance_dicts_list=distances_list)
        return closest_distance

    @classmethod
    def _increment_distance(cls, connected_synsets: List[Dict[str, float]]) -> List[Dict[str, float]]:
        return [{synset: distance + 1. for synset, distance in distance_dict.items()} for distance_dict in connected_synsets]
