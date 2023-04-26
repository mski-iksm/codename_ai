import itertools
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm


class ScoringWithRedAndBlue:

    def __init__(self, candidates_table: pd.DataFrame) -> None:
        self._candidates_table = candidates_table

    @classmethod
    def _is_word_valid(cls, candidate_word: str, ng_words: List[str]) -> bool:
        return all([ng_word not in candidate_word for ng_word in ng_words]) and all([candidate_word not in ng_word for ng_word in ng_words])

    @classmethod
    def build_scoring_from_list_data(cls, candidate_words: List[str], counts: List[int]):
        # スコアのないGPT用。スコアリングは別データで行う前提
        candidates_table = pd.DataFrame(dict(count=counts, total_score=[1] * len(counts), expecting_my_target_word=[''] * len(counts)),
                                        index=pd.Index(candidate_words))
        return cls(candidates_table=candidates_table)

    @classmethod
    def calculate_scores(
        cls,
        my_target_words: List[str],
        opponent_target_words: List[str],
        distance_data_dict: Dict[str, Dict[str, float]],
        my_target_score_offset: float = 0.02,
        fillna_distance_for_me: float = 1.0,
        fillna_distance_for_opponent: float = 1.0,
    ) -> 'ScoringWithRedAndBlue':
        print('calculating')
        available_candidates = sorted(list(set(itertools.chain.from_iterable([distance_dict.keys() for distance_dict in distance_data_dict.values()]))))
        valid_candidate_words = [
            candidate_word for candidate_word in available_candidates
            if cls._is_word_valid(candidate_word=candidate_word, ng_words=my_target_words + opponent_target_words)
        ]

        opponent_target_single_word_scores = -pd.DataFrame(
            [[distance_data_dict[word].get(candidate_word, fillna_distance_for_opponent) for candidate_word in valid_candidate_words]
             for word in opponent_target_words],
            index=opponent_target_words,
            columns=valid_candidate_words,
        )
        opponent_scores = opponent_target_single_word_scores.max(axis=0)
        opponent_scores_dict_by_candidate_score = opponent_scores.loc[valid_candidate_words].to_dict()

        my_target_single_word_scores = -pd.DataFrame(
            [[distance_data_dict[word].get(candidate_word, fillna_distance_for_me) for candidate_word in valid_candidate_words] for word in my_target_words],
            index=my_target_words,
            columns=valid_candidate_words,
        ) - my_target_score_offset
        scores = []
        counts = []
        expecting_my_target_words = []

        for candidate_word in tqdm(valid_candidate_words):
            _score_series = my_target_single_word_scores[candidate_word]
            _greater_my_targets = _score_series.loc[_score_series >= opponent_scores_dict_by_candidate_score[candidate_word]]
            _score = _greater_my_targets.mean() - opponent_scores_dict_by_candidate_score[candidate_word]
            scores.append(_score)
            counts.append(len(_greater_my_targets))
            expecting_my_target_words.append(_greater_my_targets.index.tolist())

        candidates_table = pd.DataFrame(dict(score=scores, count=counts, expecting_my_target_word=expecting_my_target_words), index=valid_candidate_words)
        # candidates_table['total_score'] = candidates_table['score'] * candidates_table['count']

        return cls(candidates_table=candidates_table)

    def get_best_word_and_count(self, second_table: Optional[pd.DataFrame] = None, count_cap: int = 3) -> Tuple[str, int, Tuple[str, ...]]:
        scores = self._candidates_table
        scores['capped_count'] = scores['count'].clip(0, count_cap)
        scores['total_score'] = scores['score'] * scores['count']
        sort_columns = ['capped_count', 'total_score']

        if second_table is not None:
            scores = self._candidates_table.join(second_table.rename(columns={'total_score': 'second_score'}))
            scores = scores.dropna()
            sort_columns = ['capped_count', 'total_score', 'second_score']

        sorted_scores = scores.sort_values(sort_columns, ascending=False)
        # デバッグ
        pd.options.display.max_rows = 1000
        print(sorted_scores.head(100))

        best_candidate_word = sorted_scores.iloc[0].name
        expect_count = sorted_scores.iloc[0]['count']
        expect_words = sorted_scores.iloc[0]['expecting_my_target_word']

        return (best_candidate_word, expect_count, expect_words)

    def get_candidate_words(self) -> List[str]:
        return self._candidates_table.index.to_list()
