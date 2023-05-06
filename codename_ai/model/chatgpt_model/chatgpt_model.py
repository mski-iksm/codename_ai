import itertools
import json
import logging
import os
from typing import Any, Dict, List, Tuple

import gokart
import openai
import pandas as pd
from tqdm import tqdm
from codename_ai.model.bert_bl.bert_model import CalculateWordDistanceWithBERT
import numpy as np

from codename_ai.model.filtering import is_a_not_part_of_bs, is_bs_not_part_of_a, is_valid_hint
from codename_ai.model.scoring import ScoringWithFieldWords
from codename_ai.model.word2vec_bl.word2vec_model import \
    CalculateWordDistanceWithWord2Vec
from codename_ai.model.wordnet_model.wordnet_model import \
    CalculateWordDistanceWithWordNet


class InvalidGPTOutputError(ValueError):
    pass


def query_chatgpt_with_candidate_words(my_target_words: List[str], opponent_target_words: List[str]) -> pd.DataFrame:
    query = f"""あなたはボードゲーム「コードネーム」のspymasterです。
    今からA群とB群の単語群を与えますので、A群の単語のうち3つ以上の単語を連想できるようなヒントを1単語で出してください。
    ただしB群のいずれかの単語に関係するヒント単語は避けてください。

    また、A群の単語のうちどの単語に関係しているのかも教えて下さい。


    A群：{str(my_target_words)}
    B群：{str(opponent_target_words)}

    返答は以下のJSONの形式だけ受け付けます。
    [{{"ヒント単語": "XXXXXX", "関係するA群の単語": ["XXXXXX", "XXXXXX"]}}]

    ヒントの選定にあたっては次のルールを守ってください。
    - A群、B群の単語をカタカナ語に変換した単語をヒント単語にしてはいけません（リンゴ -> アップル は禁止です）
    - A群、B群の単語の一部を含む単語をヒント単語にしてはいけません（てんとう虫 -> 虫 は禁止です）

    候補を20個挙げてください
    """

    openai.api_key = os.getenv('OPENAI_API_KEY')

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {
                'role': 'system',
                'content': (query)
            },
        ],
    )
    answer = response['choices'][0]['message']['content']

    answers_list = json.loads(answer)

    # 形式をassert
    if not isinstance(answers_list, list):
        raise InvalidGPTOutputError('answer_listがlistでない')

    if any([not isinstance(answer_dict, dict) for answer_dict in answers_list]):
        raise InvalidGPTOutputError('answer_listにdictでないものがある')

    if any([not isinstance(answer_dict['ヒント単語'], str) for answer_dict in answers_list]):
        raise InvalidGPTOutputError('ヒント単語にstrでないものがある')

    if any([not isinstance(answer_dict['関係するA群の単語'], list) for answer_dict in answers_list]):
        raise InvalidGPTOutputError('関係するA群の単語にlistでないものがある')

    answer_df = pd.DataFrame(answers_list)
    return answer_df
