import itertools
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from codename_ai.model.filtering import is_a_not_part_of_bs, is_bs_not_part_of_a, is_valid_hints


class ScoringWithFieldWords:

    def __init__(self, candidates_table: pd.DataFrame) -> None:
        self._candidates_table = candidates_table

    @classmethod
    def _build_score(cls, candidates_table: pd.DataFrame):
        candidates_table['total_score'] = candidates_table['score'] * candidates_table['count']
        assert set(candidates_table.columns) >= {
            'count', 'score', 'total_score', 'expecting_my_target_word'
        }, f"{set(['count', 'score', 'total_score','expecting_my_target_word']) - set(candidates_table.columns)}列が不足しています"
        return cls(candidates_table=candidates_table)

    @classmethod
    def build_scoring_from_list_data(cls, candidate_words: List[str], expecting_my_target_words: List[List[str]]):
        # スコアのないGPT用。スコアリングは別データで行う前提
        counts = [len(words) for words in expecting_my_target_words]
        candidates_table = pd.DataFrame(dict(count=counts, score=[3.] * len(counts), expecting_my_target_word=expecting_my_target_words),
                                        index=pd.Index(candidate_words))
        return cls._build_score(candidates_table=candidates_table)

    @classmethod
    def calculate_scores(
        cls,
        my_target_words: List[str],
        opponent_target_words: List[str],
        distance_data_dict: Dict[str, Dict[str, float]],
        my_target_score_offset: float = 0.02,
        fillna_distance_for_me: float = 1.0,
        fillna_distance_for_opponent: float = 1.0,
    ) -> 'ScoringWithFieldWords':
        print('calculating')
        available_candidates = sorted(list(set(itertools.chain.from_iterable([distance_dict.keys() for distance_dict in distance_data_dict.values()]))))

        opponent_target_single_word_scores = -pd.DataFrame(
            [[distance_data_dict[word].get(candidate_word, fillna_distance_for_opponent) for candidate_word in available_candidates]
             for word in opponent_target_words],
            index=opponent_target_words,
            columns=available_candidates,
        )
        opponent_scores = opponent_target_single_word_scores.max(axis=0)
        opponent_scores_dict_by_candidate_score = opponent_scores.loc[available_candidates].to_dict()

        my_target_single_word_scores = -pd.DataFrame(
            [[distance_data_dict[word].get(candidate_word, fillna_distance_for_me) for candidate_word in available_candidates] for word in my_target_words],
            index=my_target_words,
            columns=available_candidates,
        ) - my_target_score_offset
        scores = []
        counts = []
        expecting_my_target_words = []

        for candidate_word in tqdm(available_candidates):
            _score_series = my_target_single_word_scores[candidate_word]
            _greater_my_targets = _score_series.loc[_score_series >= opponent_scores_dict_by_candidate_score[candidate_word]]
            _score = _greater_my_targets.mean() - opponent_scores_dict_by_candidate_score[candidate_word]
            scores.append(_score)
            counts.append(len(_greater_my_targets))
            expecting_my_target_words.append(_greater_my_targets.index.tolist())

        candidates_table = pd.DataFrame(dict(score=scores, count=counts, expecting_my_target_word=expecting_my_target_words), index=available_candidates)

        return cls._build_score(candidates_table=candidates_table)

    def get_best_word_and_count(self, clip_max: int = 2) -> Tuple[str, int, Tuple[str, ...]]:
        scores = self._candidates_table
        scores['clip_count'] = scores['count'].clip(0, clip_max)
        sort_columns = ['clip_count', 'total_score']
        sorted_scores = scores.sort_values(sort_columns, ascending=False)
        # デバッグ
        pd.options.display.max_rows = 100
        pd.options.display.max_columns = 100
        print(sorted_scores.head(10))

        best_candidate_word = sorted_scores.iloc[0].name
        expect_count = sorted_scores.iloc[0]['count']
        expect_words = sorted_scores.iloc[0]['expecting_my_target_word']

        return (best_candidate_word, expect_count, expect_words)

    def get_candidate_words(self) -> List[str]:
        return self._candidates_table.index.to_list()


class FilteredScoringModel:

    @classmethod
    def filer_words(cls, scoring_model: ScoringWithFieldWords, min_frequency: int, field_words: List[str]) -> ScoringWithFieldWords:
        filtered_candidate_table = scoring_model._candidates_table

        filtered_candidate_table = cls._filter_low_frequent_words(filtered_candidate_table=filtered_candidate_table, min_frequency=min_frequency)
        filtered_candidate_table = cls._filter_english(filtered_candidate_table=filtered_candidate_table)
        filtered_candidate_table = cls._fliter_ng_words(filtered_candidate_table=filtered_candidate_table, field_words=field_words)
        filtered_candidate_table = cls._filter_partial_words(filtered_candidate_table=filtered_candidate_table, field_words=field_words)
        return ScoringWithFieldWords._build_score(candidates_table=filtered_candidate_table)

    @classmethod
    def _filter_low_frequent_words(cls, filtered_candidate_table: pd.DataFrame, min_frequency: int) -> pd.DataFrame:
        freq: Dict[str, int] = pd.read_pickle('./data/jawiki/frequency_dict_small.pkl')
        freq_df = pd.DataFrame.from_dict(freq, orient='index')
        freq_df.columns = ['total_freq']

        _df = filtered_candidate_table.join(freq_df)
        return _df.query(f'total_freq>={min_frequency}')

    @classmethod
    def _filter_partial_words(cls, filtered_candidate_table: pd.DataFrame, field_words: List[str]) -> pd.DataFrame:
        candidate_words = filtered_candidate_table.index
        valid_candidate_words = [
            candidate_word for candidate_word in candidate_words
            if is_a_not_part_of_bs(a_word=candidate_word, b_words=field_words) and is_bs_not_part_of_a(a_word=candidate_word, b_words=field_words)
        ]
        return filtered_candidate_table.loc[valid_candidate_words]

    @classmethod
    def _filter_english(cls, filtered_candidate_table: pd.DataFrame) -> pd.DataFrame:
        _df = filtered_candidate_table.copy()
        _df['isascii'] = [word.isascii() for word in _df.index]
        return _df.query('not isascii')

    @classmethod
    def _fliter_ng_words(cls, filtered_candidate_table: pd.DataFrame, field_words: List[str]) -> pd.DataFrame:
        _df = filtered_candidate_table.copy()
        _df['is_not_ng_word'] = is_valid_hints(hint_words=list(_df.index), target_words=field_words)
        _df = _df.query('is_not_ng_word')
        return _df


def merge_scoring_model(scoring_model_1: ScoringWithFieldWords, scoring_model_2: ScoringWithFieldWords) -> ScoringWithFieldWords:
    merged_candidates_table = scoring_model_1._candidates_table.join(scoring_model_2._candidates_table, how='outer', lsuffix='_1', rsuffix='_2')
    merged_candidates_table['expecting_my_target_word_1'] = merged_candidates_table['expecting_my_target_word_1'].apply(lambda d: d
                                                                                                                        if isinstance(d, list) else [])
    merged_candidates_table['expecting_my_target_word_2'] = merged_candidates_table['expecting_my_target_word_2'].apply(lambda d: d
                                                                                                                        if isinstance(d, list) else [])
    merged_candidates_table['score_1'] = merged_candidates_table['score_1'].fillna(0)
    merged_candidates_table['score_2'] = merged_candidates_table['score_2'].fillna(0)
    merged_candidates_table['count_1'] = merged_candidates_table['count_1'].fillna(0)
    merged_candidates_table['count_2'] = merged_candidates_table['count_2'].fillna(0)

    merged_candidates_table['expecting_my_target_word'] = [
        list(set(words)) for words in merged_candidates_table['expecting_my_target_word_1'] + merged_candidates_table['expecting_my_target_word_2']
    ]
    merged_candidates_table['count'] = [len(words) for words in merged_candidates_table['expecting_my_target_word']]
    merged_candidates_table['score'] = (merged_candidates_table['score_1'] * merged_candidates_table['count_1'] +
                                        merged_candidates_table['score_2'] * merged_candidates_table['count_2']) / merged_candidates_table['count']

    return ScoringWithFieldWords._build_score(candidates_table=merged_candidates_table)


def merge_scoring_models(scoring_models: List[ScoringWithFieldWords]) -> ScoringWithFieldWords:
    scoring_models_candidate_tables = [
        scoring_model._candidates_table.rename(columns={col: f'{col}_{i}'
                                                        for col in scoring_model._candidates_table.columns}) for i, scoring_model in enumerate(scoring_models)
    ]

    merged_candidates_table = pd.concat(scoring_models_candidate_tables, axis=1, join='outer')

    merged_counts = len(scoring_models)
    for i in range(merged_counts):
        merged_candidates_table[f'expecting_my_target_word_{i}'] = merged_candidates_table[f'expecting_my_target_word_{i}'].apply(
            lambda d: d if isinstance(d, list) else [])
        merged_candidates_table[f'score_{i}'] = merged_candidates_table[f'score_{i}'].fillna(0)
        merged_candidates_table[f'count_{i}'] = merged_candidates_table[f'count_{i}'].fillna(0)

    merged_candidates_table['expecting_my_target_word'] = [
        list(set(words)) for words in merged_candidates_table[[f'expecting_my_target_word_{i}' for i in range(merged_counts)]].sum(axis=1)
    ]
    merged_candidates_table['count'] = [len(words) for words in merged_candidates_table['expecting_my_target_word']]

    merged_candidates_table['score'] = merged_candidates_table[[f'total_score_{i}'
                                                                for i in range(merged_counts)]].sum(axis=1) / merged_candidates_table['count']

    return ScoringWithFieldWords._build_score(candidates_table=merged_candidates_table)
