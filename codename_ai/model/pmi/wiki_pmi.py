from typing import Dict
import luigi
import gokart
import pandas as pd
import numpy as np


class CalculateWordDistanceWithWikiPMI(gokart.TaskOnKart):
    target_word: str = luigi.Parameter()
    doc_size: int = luigi.IntParameter(default=21718516)
    pmi_score_cutoff: float = luigi.FloatParameter(default=-999)

    __version: int = luigi.IntParameter(default=0)

    FILL_NA_FOR_LOW_FREQUENT_WORD_COUNT = 10000

    def run(self):
        kyoki: Dict[str, Dict[str, int]] = pd.read_pickle('./data/jawiki/kyoki_frequency_dict_small.pkl')
        freq: Dict[str, int] = pd.read_pickle('./data/jawiki/frequency_dict_small.pkl')
        self.dump(self._run(kyoki=kyoki, freq=freq, target_word=self.target_word, doc_size=self.doc_size, pmi_score_cutoff=self.pmi_score_cutoff))

    @classmethod
    def _run(cls, kyoki: Dict[str, Dict[str, int]], freq: Dict[str, int], target_word: str, doc_size: int, pmi_score_cutoff: float) -> Dict[str, float]:
        if len(kyoki[target_word]) == 0:
            return {}

        freq_df = pd.DataFrame.from_dict(freq, orient='index')
        freq_df.columns = ['total_freq']

        kyoki_df = cls._build_pmi_df(kyoki=kyoki, target_word=target_word, freq_df=freq_df, doc_size=doc_size)

        kyoki_df_removed_low_relation = kyoki_df.query(f'pmi>{pmi_score_cutoff}')

        pmi_score_dict = kyoki_df_removed_low_relation['pmi_distance'].to_dict()

        # 自分自身を除く
        pmi_score_dict.pop(target_word)

        return pmi_score_dict

    @classmethod
    def _build_pmi_df(cls, kyoki: Dict[str, Dict[str, int]], target_word: str, freq_df: pd.DataFrame, doc_size: int):
        _kyoki_df = pd.DataFrame.from_dict(kyoki[target_word], orient='index')
        _kyoki_df.columns = ['count']
        _kyoki_df = _kyoki_df.join(freq_df, how='left').fillna(cls.FILL_NA_FOR_LOW_FREQUENT_WORD_COUNT)
        _kyoki_df['p_y_x'] = _kyoki_df['count'] / _kyoki_df['count'].max()
        _kyoki_df['p_y'] = _kyoki_df['total_freq'] / doc_size
        _kyoki_df['pmi'] = np.log(_kyoki_df['p_y_x'] / _kyoki_df['p_y'])
        _kyoki_df['pmi_distance'] = _kyoki_df['pmi'] * (-1)
        return _kyoki_df
