import logging
from typing import Dict, Tuple

import gokart
from tqdm import tqdm

from codename_ai.model.bert_bl.bert_model import CalculateWordDistanceWithBERT
from codename_ai.model.candidate_words import (base_small_candidate_words, large_ipa_noun_candidate_words, small_noun_candidate_words)
from codename_ai.model.game import Game
from codename_ai.model.scorer import ChatGPT_WikiPMI_WordNet_SuppressedScorer, ChatGPTWikiPMISuppressedScorer, ChatGPTWordNetSuppressedScorer, WikiPMIScorer, WordNetScorer
from codename_ai.model.scoring import ScoringWithFieldWords, merge_scoring_model, merge_scoring_models
from codename_ai.model.word2vec_bl.word2vec_model import \
    CalculateWordDistanceWithWord2Vec
from codename_ai.model.wordnet_model.wordnet_model import \
    CalculateWordDistanceWithWordNet

logging.config.fileConfig('./conf/logging.ini')


class BossModelBase:

    def __init__(self, my_color: str) -> None:
        self._my_color = my_color

    @classmethod
    def setup_model(cls, my_color: str) -> 'BossModelBase':
        assert my_color in ['red', 'blue']
        return cls(my_color=my_color)

    def next_hint(self, game: Game) -> Tuple[str, int, Tuple[str, ...]]:
        raise NotImplementedError()


class Word2VecBossModel(BossModelBase):

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
        scoring_model = ScoringWithFieldWords.calculate_scores(my_target_words=my_target_words,
                                                               opponent_target_words=opponent_target_words,
                                                               distance_data_dict=distance_data_dict,
                                                               my_target_score_offset=0.1)

        # ソート
        best_candidate_word, expect_count, expect_words = scoring_model.get_best_word_and_count()
        return (best_candidate_word, expect_count, expect_words)


class BaseLineBERTBossModel(BossModelBase):

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
        scoring_model = ScoringWithFieldWords.calculate_scores(my_target_words=my_target_words,
                                                               opponent_target_words=opponent_target_words,
                                                               distance_data_dict=distance_data_dict)

        # ソート
        best_candidate_word, expect_count, expect_words = scoring_model.get_best_word_and_count()
        return (best_candidate_word, expect_count, expect_words)


class ChatGPTWithWordNetSuppressedBossModel(BossModelBase):

    def next_hint(self, game: Game) -> Tuple[str, int, Tuple[str, ...]]:
        filtered_scoring_model = ChatGPTWordNetSuppressedScorer.get_scoring_model(game=game, my_color=self._my_color)

        best_candidate_word, expect_count, expect_words = filtered_scoring_model.get_best_word_and_count()
        return (best_candidate_word, expect_count, expect_words)


class WordNetBossModel(BossModelBase):

    def next_hint(self, game: Game) -> Tuple[str, int, Tuple[str, ...]]:
        filtered_scoring_model = WordNetScorer.get_scoring_model(game=game, my_color=self._my_color)
        best_candidate_word, expect_count, expect_words = filtered_scoring_model.get_best_word_and_count()
        return (best_candidate_word, expect_count, expect_words)


class WikiPMIBossModel(BossModelBase):

    def next_hint(self, game: Game) -> Tuple[str, int, Tuple[str, ...]]:
        filtered_scoring_model = WikiPMIScorer.get_scoring_model(game=game, my_color=self._my_color)

        best_candidate_word, expect_count, expect_words = filtered_scoring_model.get_best_word_and_count()
        return (best_candidate_word, expect_count, expect_words)


class ChatGPTWithWordNetSuppressed_plus_WikiPMI_BossModel(BossModelBase):

    def next_hint(self, game: Game) -> Tuple[str, int, Tuple[str, ...]]:
        chatgpt_filtered_scoring_model = ChatGPTWordNetSuppressedScorer.get_scoring_model(game=game, my_color=self._my_color)
        wikipmi_filtered_scoring_model = WikiPMIScorer.get_scoring_model(game=game, my_color=self._my_color)

        merged_scoring_model = merge_scoring_model(chatgpt_filtered_scoring_model, wikipmi_filtered_scoring_model)

        best_candidate_word, expect_count, expect_words = merged_scoring_model.get_best_word_and_count()
        return (best_candidate_word, expect_count, expect_words)


class ChatGPTWithWikiPMISuppressed_plus_WikiPMI_BossModel(BossModelBase):

    def next_hint(self, game: Game) -> Tuple[str, int, Tuple[str, ...]]:
        chatgpt_filtered_scoring_model = ChatGPTWikiPMISuppressedScorer.get_scoring_model(game=game, my_color=self._my_color)
        wikipmi_filtered_scoring_model = WikiPMIScorer.get_scoring_model(game=game, my_color=self._my_color)

        merged_scoring_model = merge_scoring_model(chatgpt_filtered_scoring_model, wikipmi_filtered_scoring_model)

        best_candidate_word, expect_count, expect_words = merged_scoring_model.get_best_word_and_count()
        return (best_candidate_word, expect_count, expect_words)


class ChatGPTSuppressedWithWikiAndWordnet_plus_WikiAndWordnet_Model(BossModelBase):
    # chatgpt(wiki+wordnetで抑制) + wiki + wordnet

    def next_hint(self, game: Game) -> Tuple[str, int, Tuple[str, ...]]:
        chatgpt_filtered_scoring_model = ChatGPT_WikiPMI_WordNet_SuppressedScorer.get_scoring_model(game=game, my_color=self._my_color)
        wikipmi_filtered_scoring_model = WikiPMIScorer.get_scoring_model(game=game, my_color=self._my_color)
        wordnet_filtered_scoring_model = WordNetScorer.get_scoring_model(game=game, my_color=self._my_color)

        merged_scoring_model = merge_scoring_models(scoring_models=[
            chatgpt_filtered_scoring_model,
            wikipmi_filtered_scoring_model,
            wordnet_filtered_scoring_model,
        ])

        best_candidate_word, expect_count, expect_words = merged_scoring_model.get_best_word_and_count()
        return (best_candidate_word, expect_count, expect_words)
