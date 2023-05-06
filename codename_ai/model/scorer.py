import itertools
import json
import logging
from typing import Dict
from codename_ai.model.chatgpt_model.chatgpt_model import InvalidGPTOutputError, query_chatgpt_with_candidate_words
from codename_ai.model.game import Game
from codename_ai.model.pmi.wiki_pmi import CalculateWordDistanceWithWikiPMI
from codename_ai.model.scoring import FilteredScoringModel, ScoringWithFieldWords
import gokart
from tqdm import tqdm
from codename_ai.model.word2vec_bl.word2vec_model import CalculateWordDistanceWithWord2Vec

from codename_ai.model.wordnet_model.wordnet_model import CalculateWordDistanceWithWordNet


class Scorer:

    @classmethod
    def get_scoring_model(cls, game: Game, my_color: str) -> ScoringWithFieldWords:
        raise NotImplementedError()


class ChatGPTWordNetSuppressedScorer(Scorer):
    RETRY_COUNT = 5
    MAX_TRAVERSE_DEPTH = 3

    MIN_FREQUENCY = 200

    @classmethod
    def get_scoring_model(cls, game: Game, my_color: str) -> ScoringWithFieldWords:
        words_by_color = game.get_unopened_words_by_color()

        my_target_words = words_by_color[f'{my_color}_words']
        other_target_words = words_by_color['red_words'] if my_color == 'blue' else words_by_color['blue_words']
        opponent_target_words = other_target_words + words_by_color['black_words'] + words_by_color['white_words']

        for _ in range(cls.RETRY_COUNT):
            try:
                answer_df = query_chatgpt_with_candidate_words(my_target_words=my_target_words, opponent_target_words=opponent_target_words)
                break
            except (InvalidGPTOutputError, json.decoder.JSONDecodeError):
                continue

        # ngなヒントの行は除外: 時々gptがルールを無視してopponent_wordに近いヒントを出す
        def _is_gpt_following_rule(row):
            expecting_words = row['関係するA群の単語']
            if len(set(expecting_words) & set(opponent_target_words)) > 0:
                return False
            return True

        answer_df = answer_df.loc[answer_df.apply(_is_gpt_following_rule, axis=1)]

        target_words = my_target_words + opponent_target_words

        # filter with wordnet
        wordnet_distance_data_dict: Dict[str, Dict[str, float]] = {
            target_word: gokart.build(CalculateWordDistanceWithWordNet(target_word=target_word, traverse_depth=cls.MAX_TRAVERSE_DEPTH), log_level=logging.ERROR)
            for target_word in tqdm(target_words)
        }
        ng_words_from_wordnet = list(
            itertools.chain.from_iterable([wordnet_distance_data_dict[opponent_word].keys() for opponent_word in opponent_target_words]))

        filtered_answer_df = answer_df.loc[~answer_df['ヒント単語'].isin(ng_words_from_wordnet)]

        candidate_words = filtered_answer_df['ヒント単語'].tolist()
        expecting_my_target_words = filtered_answer_df['関係するA群の単語'].tolist()

        scoring_model = ScoringWithFieldWords.build_scoring_from_list_data(candidate_words=candidate_words, expecting_my_target_words=expecting_my_target_words)
        filtered_scoring_model = FilteredScoringModel.filer_words(scoring_model=scoring_model, min_frequency=cls.MIN_FREQUENCY, field_words=target_words)
        return filtered_scoring_model


class ChatGPTWikiPMISuppressedScorer(Scorer):
    RETRY_COUNT = 5
    MAX_TRAVERSE_DEPTH = 3
    OPPONENT_PMI_SCORE_CUTOFF = 3

    MIN_FREQUENCY = 200

    @classmethod
    def get_scoring_model(cls, game: Game, my_color: str) -> ScoringWithFieldWords:
        words_by_color = game.get_unopened_words_by_color()

        my_target_words = words_by_color[f'{my_color}_words']
        other_target_words = words_by_color['red_words'] if my_color == 'blue' else words_by_color['blue_words']
        opponent_target_words = other_target_words + words_by_color['black_words'] + words_by_color['white_words']

        for _ in range(cls.RETRY_COUNT):
            try:
                answer_df = query_chatgpt_with_candidate_words(my_target_words=my_target_words, opponent_target_words=opponent_target_words)
                break
            except (InvalidGPTOutputError, json.decoder.JSONDecodeError):
                continue

        # ngなヒントの行は除外: 時々gptがルールを無視してopponent_wordに近いヒントを出す
        def _is_gpt_following_rule(row):
            expecting_words = row['関係するA群の単語']
            if len(set(expecting_words) & set(opponent_target_words)) > 0:
                return False
            return True

        answer_df = answer_df.loc[answer_df.apply(_is_gpt_following_rule, axis=1)]

        target_words = my_target_words + opponent_target_words

        # filter with wiki pmi
        opponent_distance_data_dict: Dict[str, Dict[str, float]] = {
            target_word: gokart.build(CalculateWordDistanceWithWikiPMI(target_word=target_word, pmi_score_cutoff=cls.OPPONENT_PMI_SCORE_CUTOFF),
                                      log_level=logging.ERROR)
            for target_word in tqdm(opponent_target_words)
        }

        ng_words_from_wikipmi = list(
            itertools.chain.from_iterable([opponent_distance_data_dict[opponent_word].keys() for opponent_word in opponent_target_words]))

        filtered_answer_df = answer_df.loc[~answer_df['ヒント単語'].isin(ng_words_from_wikipmi)]

        candidate_words = filtered_answer_df['ヒント単語'].tolist()
        expecting_my_target_words = filtered_answer_df['関係するA群の単語'].tolist()

        scoring_model = ScoringWithFieldWords.build_scoring_from_list_data(candidate_words=candidate_words, expecting_my_target_words=expecting_my_target_words)
        filtered_scoring_model = FilteredScoringModel.filer_words(scoring_model=scoring_model, min_frequency=cls.MIN_FREQUENCY, field_words=target_words)
        return filtered_scoring_model


class ChatGPT_WikiPMI_WordNet_SuppressedScorer(Scorer):
    RETRY_COUNT = 5
    MAX_TRAVERSE_DEPTH = 3
    OPPONENT_PMI_SCORE_CUTOFF = 3

    MIN_FREQUENCY = 200

    @classmethod
    def get_scoring_model(cls, game: Game, my_color: str) -> ScoringWithFieldWords:
        words_by_color = game.get_unopened_words_by_color()

        my_target_words = words_by_color[f'{my_color}_words']
        other_target_words = words_by_color['red_words'] if my_color == 'blue' else words_by_color['blue_words']
        opponent_target_words = other_target_words + words_by_color['black_words'] + words_by_color['white_words']

        for _ in range(cls.RETRY_COUNT):
            try:
                answer_df = query_chatgpt_with_candidate_words(my_target_words=my_target_words, opponent_target_words=opponent_target_words)
                break
            except (InvalidGPTOutputError, json.decoder.JSONDecodeError):
                continue

        # ngなヒントの行は除外: 時々gptがルールを無視してopponent_wordに近いヒントを出す
        def _is_gpt_following_rule(row):
            expecting_words = row['関係するA群の単語']
            if len(set(expecting_words) & set(opponent_target_words)) > 0:
                return False
            return True

        answer_df = answer_df.loc[answer_df.apply(_is_gpt_following_rule, axis=1)]

        target_words = my_target_words + opponent_target_words

        # filter with wiki pmi
        opponent_distance_data_dict: Dict[str, Dict[str, float]] = {
            target_word: gokart.build(CalculateWordDistanceWithWikiPMI(target_word=target_word, pmi_score_cutoff=cls.OPPONENT_PMI_SCORE_CUTOFF),
                                      log_level=logging.ERROR)
            for target_word in tqdm(opponent_target_words)
        }
        ng_words_from_wikipmi = list(
            itertools.chain.from_iterable([opponent_distance_data_dict[opponent_word].keys() for opponent_word in opponent_target_words]))

        # filter with wordnet
        wordnet_distance_data_dict: Dict[str, Dict[str, float]] = {
            target_word: gokart.build(CalculateWordDistanceWithWordNet(target_word=target_word, traverse_depth=cls.MAX_TRAVERSE_DEPTH), log_level=logging.ERROR)
            for target_word in tqdm(target_words)
        }
        ng_words_from_wordnet = list(
            itertools.chain.from_iterable([wordnet_distance_data_dict[opponent_word].keys() for opponent_word in opponent_target_words]))

        filtered_answer_df = answer_df.loc[~answer_df['ヒント単語'].isin(ng_words_from_wordnet)]
        filtered_answer_df = filtered_answer_df.loc[~filtered_answer_df['ヒント単語'].isin(ng_words_from_wikipmi)]

        candidate_words = filtered_answer_df['ヒント単語'].tolist()
        expecting_my_target_words = filtered_answer_df['関係するA群の単語'].tolist()

        scoring_model = ScoringWithFieldWords.build_scoring_from_list_data(candidate_words=candidate_words, expecting_my_target_words=expecting_my_target_words)
        filtered_scoring_model = FilteredScoringModel.filer_words(scoring_model=scoring_model, min_frequency=cls.MIN_FREQUENCY, field_words=target_words)
        return filtered_scoring_model


class WikiPMIScorer(Scorer):
    MY_PMI_SCORE_CUTOFF = 2
    OPPONENT_PMI_SCORE_CUTOFF = 0
    MY_TARGET_SCORE_OFFSET = 0.3
    FILLNA_DISTANCE_FOR_ME = 9999
    FILLNA_DISTANCE_FOR_OPPONENT = 0

    MIN_FREQUENCY = 1000

    @classmethod
    def get_scoring_model(cls, game: Game, my_color: str) -> ScoringWithFieldWords:
        words_by_color = game.get_unopened_words_by_color()

        my_target_words = words_by_color[f'{my_color}_words']
        other_target_words = words_by_color['red_words'] if my_color == 'blue' else words_by_color['blue_words']
        opponent_target_words = other_target_words + words_by_color['black_words'] + words_by_color['white_words']

        # 自分チームと相手チームの単語でcutoffを変える
        my_distance_data_dict: Dict[str, Dict[str, float]] = {
            target_word: gokart.build(CalculateWordDistanceWithWikiPMI(target_word=target_word, pmi_score_cutoff=cls.MY_PMI_SCORE_CUTOFF),
                                      log_level=logging.ERROR)
            for target_word in tqdm(my_target_words)
        }
        opponent_distance_data_dict: Dict[str, Dict[str, float]] = {
            target_word: gokart.build(CalculateWordDistanceWithWikiPMI(target_word=target_word, pmi_score_cutoff=cls.OPPONENT_PMI_SCORE_CUTOFF),
                                      log_level=logging.ERROR)
            for target_word in tqdm(opponent_target_words)
        }

        distance_data_dict: Dict[str, Dict[str, float]] = my_distance_data_dict | opponent_distance_data_dict

        # スコアリング
        my_target_words = words_by_color[f'{my_color}_words']
        other_target_words = words_by_color['red_words'] if my_color == 'blue' else words_by_color['blue_words']
        opponent_target_words = other_target_words + words_by_color['black_words'] + words_by_color['white_words']
        scoring_model = ScoringWithFieldWords.calculate_scores(my_target_words=my_target_words,
                                                               opponent_target_words=opponent_target_words,
                                                               distance_data_dict=distance_data_dict,
                                                               my_target_score_offset=cls.MY_TARGET_SCORE_OFFSET,
                                                               fillna_distance_for_me=cls.FILLNA_DISTANCE_FOR_ME,
                                                               fillna_distance_for_opponent=cls.FILLNA_DISTANCE_FOR_OPPONENT)
        # フィルタ
        filed_words = my_target_words + opponent_target_words
        filtered_scoring_model = FilteredScoringModel.filer_words(scoring_model=scoring_model, min_frequency=cls.MIN_FREQUENCY, field_words=filed_words)
        return filtered_scoring_model


class WordNetScorer(Scorer):

    MIN_FREQUENCY = 200

    @classmethod
    def get_scoring_model(cls, game: Game, my_color: str) -> ScoringWithFieldWords:
        words_by_color = game.get_unopened_words_by_color()

        max_traverse_depth = 3

        my_target_words = words_by_color[f'{my_color}_words']
        other_target_words = words_by_color['red_words'] if my_color == 'blue' else words_by_color['blue_words']
        opponent_target_words = other_target_words + words_by_color['black_words'] + words_by_color['white_words']

        # wordnet で候補ワードを作る ========
        target_words = my_target_words + opponent_target_words
        distance_data_dict: Dict[str, Dict[str, float]] = {
            target_word: gokart.build(CalculateWordDistanceWithWordNet(target_word=target_word, traverse_depth=max_traverse_depth), log_level=logging.ERROR)
            for target_word in tqdm(target_words)
        }
        scoring_model = ScoringWithFieldWords.calculate_scores(
            my_target_words=my_target_words,
            opponent_target_words=opponent_target_words,
            distance_data_dict=distance_data_dict,
            my_target_score_offset=0,
            fillna_distance_for_me=999,
            fillna_distance_for_opponent=max_traverse_depth + 2,
        )
        word_candidate = scoring_model.get_candidate_words()

        # word2vecで順位づけ
        embedding_distance_data_dict = {
            target_word: gokart.build(CalculateWordDistanceWithWord2Vec(
                target_word=target_word,
                candidate_words=word_candidate,
            ), log_level=logging.ERROR)
            for target_word in tqdm(target_words)
        }
        scoring_model = ScoringWithFieldWords.calculate_scores(my_target_words=my_target_words,
                                                               opponent_target_words=opponent_target_words,
                                                               distance_data_dict=embedding_distance_data_dict,
                                                               my_target_score_offset=0.1)

        # フィルタ
        filed_words = my_target_words + opponent_target_words
        filtered_scoring_model = FilteredScoringModel.filer_words(scoring_model=scoring_model, min_frequency=cls.MIN_FREQUENCY, field_words=filed_words)
        return filtered_scoring_model
