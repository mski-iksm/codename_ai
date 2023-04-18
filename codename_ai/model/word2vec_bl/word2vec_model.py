from typing import Dict, List, Tuple

import gokart
import luigi
import numpy as np
from tqdm import tqdm
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = 'jawiki_vector'
model_path = {'jawiki_vector': './data/entity_vector.model.bin'}
model = KeyedVectors.load_word2vec_format(model_path[MODEL_NAME], binary=True)


class CalculateWordDistanceWithWord2Vec(gokart.TaskOnKart):
    target_word: str = luigi.Parameter()
    candidate_words: List[str] = luigi.ListParameter()

    __dev = luigi.FloatParameter(default=0.03)

    def requires(self):
        return {
            'target_words': GetMultipleSentencesWord2VecVector(input_words=[self.target_word]),
            'candidate_words': GetMultipleSentencesWord2VecVector(input_words=self.candidate_words),
        }

    def run(self):
        target_word_vector: np.ndarray = self.load('target_words')[self.target_word]
        candidate_word_vectors: Dict[str, np.ndarray] = self.load('candidate_words')
        self.dump(self._run(target_word_vector=target_word_vector, candidate_word_vectors=candidate_word_vectors))

    @classmethod
    def _run(cls, target_word_vector: np.ndarray, candidate_word_vectors: Dict[str, np.ndarray]) -> Dict[str, float]:
        candidate_word_vector_keys = list(candidate_word_vectors.keys())
        candidate_word_vector_arrays = np.vstack([candidate_word_vectors[key] for key in candidate_word_vector_keys])
        distance = -cosine_similarity(target_word_vector.reshape([1, -1]), candidate_word_vector_arrays)
        assert distance[0].shape[0] == len(candidate_word_vector_keys)
        return {word: float(dist) for dist, word in zip(distance[0], candidate_word_vector_keys)}


class GetMultipleSentencesWord2VecVector(gokart.TaskOnKart):
    model_name: str = luigi.Parameter(default=MODEL_NAME)
    input_words: Tuple[str, ...] = luigi.ListParameter()

    def run(self):
        self.dump(self._words_to_vectors(input_words=list(self.input_words)))

    def _words_to_vectors(self, input_words: List[str]) -> Dict[str, np.ndarray]:
        all_vectors_dict = {}
        for word in tqdm(input_words):
            vector, err = self._get_single_vector(word=word)
            if not err:
                all_vectors_dict[word] = vector
        return all_vectors_dict

    @classmethod
    def _get_single_vector(cls, word: str) -> Tuple[np.ndarray, bool]:
        err = False
        if model.has_index_for(word):
            return model.get_vector(word), err
        el_word = f'[{word}]'
        if model.has_index_for(el_word):
            return model.get_vector(el_word), err
        return np.zeros_like(model.get_vector('イヌ')), True
