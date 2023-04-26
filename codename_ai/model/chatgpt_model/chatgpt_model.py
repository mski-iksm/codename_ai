import json
import logging
import os
from typing import Any, Dict, List, Tuple

import gokart
import openai
import pandas as pd
from codename_ai.model.scoring import ScoringWithRedAndBlue
from codename_ai.model.word2vec_bl.word2vec_model import CalculateWordDistanceWithWord2Vec

from codename_ai.model.wordnet_model.wordnet_model import CalculateWordDistanceWithWordNet
from tqdm import tqdm


def query_chatgpt(my_target_words: List[str], opponent_target_words: List[str]) -> Tuple[str, int]:
    query = f"""今から自分チームと相手チームの単語を与えますので、自分チームの単語のうちいくつかを連想できるようなヒントを1単語で出してください。
自分チームの単語をなるべく数多く連想できるヒント単語を挙げてください。
ただし、相手チームの単語を連想できる単語を挙げてはいけません。

ただし、以下のルールを守ってください。
- 自分チームの単語や相手チームの単語を日本語または英語に変換した単語を挙げてはいけません
- 自分チームの単語や相手チームの単語の全体または一部を含んだ単語を挙げてはいけません

また、そのヒント単語から自分チームのうち何個の単語を連想できるかも教えて下さい。
出せるヒントの数は1つだけですので、最も多く自分チームの単語を連想できそうなヒントを挙げてください。

自分チームの単語：{str(my_target_words)}
相手チームの単語：{str(opponent_target_words)}

返答は以下のJSONの形式だけ受け付けます。
{{"ヒント単語": "XXXXXX", "単語数": XXX}}
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
    json_answer = answer.split('\n')[0]
    dict_answer = json.loads(json_answer)

    return (dict_answer['ヒント単語'], dict_answer['単語数'])


class InvalidGPTOutputError(ValueError):
    pass


def query_enhanced_chatgpt(my_target_words: List[str], opponent_target_words: List[str]) -> pd.DataFrame:
    #     query = f"""今からA群とB群の単語群を与えますので、A群の単語のうち3つ以上の単語を連想できるようなヒントを1単語で出してください。
    # ただしB群を連想できるヒント単語は避けてください。

    # A群：{str(my_target_words)}
    # B群：{str(opponent_target_words)}

    # 返答は以下のJSONの形式だけ受け付けます。
    # [{{"ヒント単語": "XXXXXX", "単語数": XXX}}]

    # ヒントの選定にあたっては次のルールを守ってください。
    # - A群、B群の単語をカタカナ語に変換した単語をヒント単語にしてはいけません（リンゴ -> アップル は禁止です）
    # - A群、B群の単語の一部を含む単語をヒント単語にしてはいけません（てんとう虫 -> 虫 は禁止です）

    # 候補を20個挙げてください
    #     """

    #     openai.api_key = os.getenv('OPENAI_API_KEY')
    #     response = openai.ChatCompletion.create(
    #         model='gpt-3.5-turbo',
    #         messages=[
    #             {
    #                 'role': 'system',
    #                 'content': (query)
    #             },
    #         ],
    #     )

    #     # tryでjsonにしてだめならもう一回
    #     answer = response['choices'][0]['message']['content']
    answer = '''[{"ヒント単語": "家", "単語数": 5},
                {"ヒント単語": "アメリカン", "単語数": 2},
                {"ヒント単語": "エネルギー", "単語数": 2},
                {"ヒント単語": "緑", "単語数": 2},
                {"ヒント単語": "医療", "単語数": 2},
                {"ヒント単語": "交通事故", "単語数": 2},
                {"ヒント単語": "職場", "単語数": 2},
                {"ヒント単語": "車", "単語数": 4},
                {"ヒント単語": "音楽", "単語数": 2},
                {"ヒント単語": "球技", "単語数": 2},
                {"ヒント単語": "映画", "単語数": 2},
                {"ヒント単語": "飛行機", "単語数": 2},
                {"ヒント単語": "冷蔵庫", "単語数": 2},
                {"ヒント単語": "ドライブ", "単語数": 2},
                {"ヒント単語": "社会", "単語数": 2},
                {"ヒント単語": "音響", "単語数": 2},
                {"ヒント単語": "食品", "単語数": 2},
                {"ヒント単語": "経済", "単語数": 2},
                {"ヒント単語": "地球", "単語数": 2},
                {"ヒント単語": "事故", "単語数": 2}]'''
    print(answer)
    answers_list = json.loads(answer)

    # 形式をassert
    if not isinstance(answers_list, list):
        raise InvalidGPTOutputError('answer_listがlistでない')

    if any([not isinstance(answer_dict, dict) for answer_dict in answers_list]):
        raise InvalidGPTOutputError('answer_listにdictでないものがある')

    if any([not isinstance(answer_dict['ヒント単語'], str) for answer_dict in answers_list]):
        raise InvalidGPTOutputError('ヒント単語にstrでないものがある')

    if any([not isinstance(answer_dict['単語数'], int) for answer_dict in answers_list]):
        raise InvalidGPTOutputError('単語数にintでないものがある')

    answer_df = pd.DataFrame(answers_list)
    return answer_df


RETRY_COUNT = 5
MAX_TRAVERSE_DEPTH = 3


def filter_chatgpt_with_word2vec(my_target_words: List[str], opponent_target_words: List[str]) -> Tuple[str, int, Tuple[str, ...]]:
    for _ in range(RETRY_COUNT):
        try:
            answer_df = query_enhanced_chatgpt(my_target_words=my_target_words, opponent_target_words=opponent_target_words)
            break
        except (InvalidGPTOutputError, json.decoder.JSONDecodeError):
            continue

    # TODO: NGワードを除外する

    target_words = my_target_words + opponent_target_words
    scoring_model = ScoringWithRedAndBlue.build_scoring_from_list_data(
        candidate_words=answer_df['ヒント単語'].tolist(),
        counts=answer_df['単語数'].tolist(),
    )
    hint_candidate_words = answer_df['ヒント単語'].tolist()

    # word2vecで順位づけ
    embedding_distance_data_dict = {
        target_word: gokart.build(CalculateWordDistanceWithWord2Vec(
            target_word=target_word,
            candidate_words=hint_candidate_words,
        ), log_level=logging.ERROR)
        for target_word in tqdm(target_words)
    }
    embedding_scoring_model = ScoringWithRedAndBlue.calculate_scores(my_target_words=my_target_words,
                                                                     opponent_target_words=opponent_target_words,
                                                                     distance_data_dict=embedding_distance_data_dict,
                                                                     my_target_score_offset=0.0)
    print(embedding_scoring_model._candidates_table)

    best_candidate_word, expect_count, expect_words = scoring_model.get_best_word_and_count(
        second_table=embedding_scoring_model._candidates_table[['total_score']])

    print(best_candidate_word, expect_count, expect_words)
    return (best_candidate_word, expect_count, expect_words)
