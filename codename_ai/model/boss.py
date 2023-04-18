import logging
from typing import Tuple

import gokart
from tqdm import tqdm

from codename_ai.model.bert_bl.bert_model import CalculateWordDistanceWithBERT
from codename_ai.model.candidate_words import (base_small_candidate_words, large_ipa_noun_candidate_words, small_noun_candidate_words)
from codename_ai.model.game import Game
from codename_ai.model.scoring import ScoringWithRedAndBlue

logging.config.fileConfig('./conf/logging.ini')


class BossModelBase:

    def next_hint(self, game: Game) -> Tuple[str, int, Tuple[str, ...]]:
        raise NotImplementedError()


# class FastTextBossModel(BossModelBase):

#     def __init__(self, my_color: str, fasttext_vectors: FastTextKeyedVectors) -> None:
#         self._my_color = my_color
#         self._fasttext_vectors = fasttext_vectors

#     @classmethod
#     def _get_model_path_name(cls) -> str:
#         return os.path.abspath(os.path.join(os.path.dirname(__file__), 'additional_data', 'cc.ja.300.bin'))

#     @classmethod
#     def setup_model(cls, my_color: str) -> 'FastTextBossModel':
#         assert my_color in ['red', 'blue']

#         fasttext_vectors = gensim.models.fasttext.load_facebook_vectors(path=cls._get_model_path_name())
#         return cls(my_color=my_color, fasttext_vectors=fasttext_vectors)

#     def next_hint(self, game: Game) -> Tuple[str, int]:
#         words_by_color = game.get_words_by_color()

#         my_color = self._my_color
#         other_color = ({'red', 'blue'} - {my_color}).pop()

#         positive_words = words_by_color[f'{my_color}_words']
#         negative_words = words_by_color[f'{other_color}_words'] + words_by_color['black_words'] + words_by_color['white_words']

#         print(self._fasttext_vectors.most_similar(positive=positive_words, negative=negative_words, topn=20))


class BaseLineBERTBossModel(BossModelBase):

    def __init__(self, my_color: str) -> None:
        self._my_color = my_color

    @classmethod
    def setup_model(cls, my_color: str) -> 'BaseLineBERTBossModel':
        assert my_color in ['red', 'blue']

        return cls(my_color=my_color)

    def next_hint(self, game: Game) -> Tuple[str, int, Tuple[str, ...]]:
        words_by_color = game.get_unopened_words_by_color()
        print(words_by_color)

        # TODO: 距離計算までをwordごとにやって、gokart化
        # そうするとキャッシュが使えるようになる
        target_words = words_by_color['blue_words'] + words_by_color['red_words'] + words_by_color['black_words'] + words_by_color['white_words']

        # 候補ワードをリストアップ
        # candidate_words = base_small_candidate_words()
        candidate_words = large_ipa_noun_candidate_words()
        # candidate_words = small_noun_candidate_words()

        distance_data_dict = {
            target_word: gokart.build(CalculateWordDistanceWithBERT(
                target_word=target_word,
                candidate_words=candidate_words,
            ), log_level=logging.ERROR)
            for target_word in tqdm(target_words)
        }

        # スコアリング
        my_target_words = words_by_color[f'{self._my_color}_words']
        opponent_target_words = words_by_color['red_words'] if self._my_color == 'blue' else words_by_color['blue_words']
        scoring_model = ScoringWithRedAndBlue.calculate_scores(my_target_words=my_target_words,
                                                               opponent_target_words=opponent_target_words,
                                                               distance_data_dict=distance_data_dict,
                                                               candidate_words=candidate_words)

        # ソート
        best_candidate_word, expect_count, expect_words = scoring_model.get_best_word_and_count()
        return (best_candidate_word, expect_count, expect_words)
