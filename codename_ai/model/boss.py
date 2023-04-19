import logging
from typing import Dict, Tuple

import gokart
from tqdm import tqdm

from codename_ai.model.bert_bl.bert_model import CalculateWordDistanceWithBERT
from codename_ai.model.candidate_words import (base_small_candidate_words, large_ipa_noun_candidate_words, small_noun_candidate_words)
from codename_ai.model.chatgpt_model.chatgpt_model import query_chatgpt
from codename_ai.model.game import Game
from codename_ai.model.scoring import ScoringWithRedAndBlue
from codename_ai.model.word2vec_bl.word2vec_model import CalculateWordDistanceWithWord2Vec

logging.config.fileConfig('./conf/logging.ini')


class BossModelBase:

    def __init__(self, my_color: str) -> None:
        self._my_color = my_color

    def next_hint(self, game: Game) -> Tuple[str, int, Tuple[str, ...]]:
        raise NotImplementedError()


class Word2VecBossModel(BossModelBase):

    @classmethod
    def setup_model(cls, my_color: str) -> 'Word2VecBossModel':
        assert my_color in ['red', 'blue']

        # fasttext_vectors = gensim.models.fasttext.load_facebook_vectors(path=cls._get_model_path_name())
        return cls(my_color=my_color)

    def next_hint(self, game: Game) -> Tuple[str, int, Tuple[str, ...]]:
        words_by_color = game.get_unopened_words_by_color()

        # 距離計算までをwordごとにやって、gokart化
        target_words = words_by_color['blue_words'] + words_by_color['red_words'] + words_by_color['black_words'] + words_by_color['white_words']

        # 候補ワードをリストアップ
        # candidate_words = base_small_candidate_words()
        candidate_words = large_ipa_noun_candidate_words()
        # candidate_words = small_noun_candidate_words()

        distance_data_dict = {
            target_word: gokart.build(CalculateWordDistanceWithWord2Vec(
                target_word=target_word,
                candidate_words=candidate_words,
            ), log_level=logging.ERROR)
            for target_word in tqdm(target_words)
        }

        # スコアリング
        my_target_words = words_by_color[f'{self._my_color}_words']
        other_target_words = words_by_color['red_words'] if self._my_color == 'blue' else words_by_color['blue_words']
        opponent_target_words = other_target_words + words_by_color['black_words'] + words_by_color['white_words']
        scoring_model = ScoringWithRedAndBlue.calculate_scores(my_target_words=my_target_words,
                                                               opponent_target_words=opponent_target_words,
                                                               distance_data_dict=distance_data_dict,
                                                               my_target_score_offset=0.1)

        # ソート
        best_candidate_word, expect_count, expect_words = scoring_model.get_best_word_and_count()
        return (best_candidate_word, expect_count, expect_words)


class BaseLineBERTBossModel(BossModelBase):

    @classmethod
    def setup_model(cls, my_color: str) -> 'BaseLineBERTBossModel':
        assert my_color in ['red', 'blue']

        return cls(my_color=my_color)

    def next_hint(self, game: Game) -> Tuple[str, int, Tuple[str, ...]]:
        words_by_color = game.get_unopened_words_by_color()

        # 距離計算までをwordごとにやって、gokart化
        target_words = words_by_color['blue_words'] + words_by_color['red_words'] + words_by_color['black_words'] + words_by_color['white_words']

        # 候補ワードをリストアップ
        # candidate_words = base_small_candidate_words()
        candidate_words = large_ipa_noun_candidate_words()
        # candidate_words = small_noun_candidate_words()

        distance_data_dict: Dict[str, Dict[str, float]] = {
            target_word: gokart.build(CalculateWordDistanceWithBERT(
                target_word=target_word,
                candidate_words=candidate_words,
            ), log_level=logging.ERROR)
            for target_word in tqdm(target_words)
        }

        # スコアリング
        my_target_words = words_by_color[f'{self._my_color}_words']
        other_target_words = words_by_color['red_words'] if self._my_color == 'blue' else words_by_color['blue_words']
        opponent_target_words = other_target_words + words_by_color['black_words'] + words_by_color['white_words']
        scoring_model = ScoringWithRedAndBlue.calculate_scores(my_target_words=my_target_words,
                                                               opponent_target_words=opponent_target_words,
                                                               distance_data_dict=distance_data_dict)

        # ソート
        best_candidate_word, expect_count, expect_words = scoring_model.get_best_word_and_count()
        return (best_candidate_word, expect_count, expect_words)


class ChatGPTBossModel(BossModelBase):

    @classmethod
    def setup_model(cls, my_color: str) -> 'ChatGPTBossModel':
        assert my_color in ['red', 'blue']
        return cls(my_color=my_color)

    def next_hint(self, game: Game) -> Tuple[str, int, Tuple[str, ...]]:
        words_by_color = game.get_unopened_words_by_color()

        my_target_words = words_by_color[f'{self._my_color}_words']
        other_target_words = words_by_color['red_words'] if self._my_color == 'blue' else words_by_color['blue_words']
        opponent_target_words = other_target_words + words_by_color['black_words'] + words_by_color['white_words']
        best_candidate_word, expect_count = query_chatgpt(my_target_words=my_target_words, opponent_target_words=opponent_target_words)
        expect_words: Tuple[str, ...] = tuple()
        return best_candidate_word, expect_count, expect_words
