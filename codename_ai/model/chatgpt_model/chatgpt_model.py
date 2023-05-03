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
from codename_ai.model.scoring import ScoringWithRedAndBlue
from codename_ai.model.word2vec_bl.word2vec_model import \
    CalculateWordDistanceWithWord2Vec
from codename_ai.model.wordnet_model.wordnet_model import \
    CalculateWordDistanceWithWordNet


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
    query = f"""今からA群とB群の単語群を与えますので、A群の単語のうち3つ以上の単語を連想できるようなヒントを1単語で出してください。
    ただしB群を連想できるヒント単語は避けてください。

    A群：{str(my_target_words)}
    B群：{str(opponent_target_words)}

    返答は以下のJSONの形式だけ受け付けます。
    [{{"ヒント単語": "XXXXXX", "単語数": XXX}}]

    ヒントの選定にあたっては次のルールを守ってください。
    - A群、B群の単語をカタカナ語に変換した単語をヒント単語にしてはいけません（リンゴ -> アップル は禁止です）
    - A群、B群の単語の一部を含む単語をヒント単語にしてはいけません（てんとう虫 -> 虫 は禁止です）

    候補を20個挙げてください
    """

    openai.api_key = os.getenv('OPENAI_API_KEY')
    # response = openai.ChatCompletion.create(
    #     model='gpt-3.5-turbo',
    #     messages=[
    #         {
    #             'role': 'system',
    #             'content': (query)
    #         },
    #     ],
    # )
    # answer = response['choices'][0]['message']['content']
    answer = '''[{"ヒント単語": "病院", "単語数": 4},
        {"ヒント単語": "鍵", "単語数": 3},
        {"ヒント単語": "水", "単語数": 3},
        {"ヒント単語": "郵便局", "単語数": 3},
        {"ヒント単語": "歌手", "単語数": 3},
        {"ヒント単語": "シンデレラ", "単語数": 3},
        {"ヒント単語": "メロン", "単語数": 3},
        {"ヒント単語": "ビューティー", "単語数": 3},
        {"ヒント単語": "グリーン", "単語数": 3},
        {"ヒント単語": "コンピューター", "単語数": 3},
        {"ヒント単語": "ラジオ", "単語数": 3},
        {"ヒント単語": "車", "単語数": 3},
        {"ヒント単語": "スイスイ", "単語数": 3},
        {"ヒント単語": "愛", "単語数": 3},
        {"ヒント単語": "ボトル", "単語数": 3},
        {"ヒント単語": "矢印", "単語数": 3},
        {"ヒント単語": "マップ", "単語数": 3},
        {"ヒント単語": "鉛筆", "単語数": 3},
        {"ヒント単語": "握手", "単語数": 3},
        {"ヒント単語": "糸", "単語数": 3}]'''
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


def filter_chatgpt_with_word2vec(my_target_words: List[str], opponent_target_words: List[str]) -> Tuple[str, int, Tuple[str, ...]]:
    RETRY_COUNT = 5
    MAX_TRAVERSE_DEPTH = 4

    for _ in range(RETRY_COUNT):
        try:
            answer_df = query_enhanced_chatgpt(my_target_words=my_target_words, opponent_target_words=opponent_target_words)
            break
        except (InvalidGPTOutputError, json.decoder.JSONDecodeError):
            continue

    target_words = my_target_words + opponent_target_words

    # NGワードを除外する
    print(answer_df)
    valid_answer_df = answer_df.loc[[is_valid_hint(hint_word=hint_word, target_words=target_words) for hint_word in answer_df['ヒント単語'].tolist()]]
    valid_answer_df = valid_answer_df.loc[[is_a_not_part_of_bs(a_word=hint_word, b_words=target_words) for hint_word in valid_answer_df['ヒント単語'].tolist()]]
    valid_answer_df = valid_answer_df.loc[[is_bs_not_part_of_a(a_word=hint_word, b_words=target_words) for hint_word in valid_answer_df['ヒント単語'].tolist()]]
    print(valid_answer_df)

    scoring_model = ScoringWithRedAndBlue.build_scoring_from_list_data(
        candidate_words=valid_answer_df['ヒント単語'].tolist(),
        counts=valid_answer_df['単語数'].tolist(),
    )
    hint_candidate_words = valid_answer_df['ヒント単語'].tolist()

    # TODO: word2vecとwordnetで許容できるやつの数を直してrerank
    word2vec_distance_data_dict = {
        target_word: gokart.build(CalculateWordDistanceWithWord2Vec(
            target_word=target_word,
            candidate_words=hint_candidate_words,
        ), log_level=logging.ERROR)
        for target_word in tqdm(target_words)
    }
    word2vec_scoring_model = ScoringWithRedAndBlue.calculate_scores(my_target_words=my_target_words,
                                                                    opponent_target_words=opponent_target_words,
                                                                    distance_data_dict=word2vec_distance_data_dict,
                                                                    my_target_score_offset=0.)
    print('word2vec')
    print(word2vec_scoring_model._candidates_table.sort_values('count', ascending=False))

    bert_distance_data_dict = {
        target_word: gokart.build(CalculateWordDistanceWithBERT(
            target_word=target_word,
            candidate_words=hint_candidate_words,
        ), log_level=logging.ERROR)
        for target_word in tqdm(target_words)
    }
    bert_scoring_model = ScoringWithRedAndBlue.calculate_scores(my_target_words=my_target_words,
                                                                opponent_target_words=opponent_target_words,
                                                                distance_data_dict=bert_distance_data_dict,
                                                                my_target_score_offset=0.)
    print('bert')
    print(bert_scoring_model._candidates_table.sort_values('count', ascending=False))

    wordnet_distance_data_dict: Dict[str, Dict[str, float]] = {
        target_word: gokart.build(CalculateWordDistanceWithWordNet(target_word=target_word, traverse_depth=MAX_TRAVERSE_DEPTH), log_level=logging.ERROR)
        for target_word in tqdm(target_words)
    }
    wordnet_scoring_model = ScoringWithRedAndBlue.calculate_scores(
        my_target_words=my_target_words,
        opponent_target_words=opponent_target_words,
        distance_data_dict=wordnet_distance_data_dict,
        my_target_score_offset=0,
        fillna_distance_for_me=999,
        fillna_distance_for_opponent=MAX_TRAVERSE_DEPTH + 2,
    )

    print('wordnet')
    print(wordnet_scoring_model._candidates_table.reindex(hint_candidate_words).dropna().sort_values('count', ascending=False))
    best_candidate_word, expect_count, expect_words = scoring_model.get_best_word_and_count(
        second_table=word2vec_scoring_model._candidates_table[['total_score']])

    print(best_candidate_word, expect_count, expect_words)
    return (best_candidate_word, expect_count, expect_words)


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

    ## 本番モード
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

    ## デバッグモード
    #     answer = '''[{"ヒント単語": "音楽", "関係するA群の単語": ["ピアノ", "フルート"]},
    #  {"ヒント単語": "水", "関係するA群の単語": ["川", "アトランティス"]},
    #  {"ヒント単語": "囲い", "関係するA群の単語": ["フェンス", "銀行"]},
    #  {"ヒント単語": "口", "関係するA群の単語": ["歯", "メール"]},
    #  {"ヒント単語": "動物", "関係するA群の単語": ["トキ"]},
    #  {"ヒント単語": "疾患", "関係するA群の単語": ["病気"]},
    #  {"ヒント単語": "王国", "関係するA群の単語": ["アトランティス", "エジプト"]},
    #  {"ヒント単語": "黒", "関係するA群の単語": ["シャドウ"]},
    #  {"ヒント単語": "建物", "関係するA群の単語": ["銀行", "スタジアム"]},
    #  {"ヒント単語": "書く", "関係するA群の単語": ["メール"]},
    #  {"ヒント単語": "科学", "関係するA群の単語": ["アトランティス"]},
    #  {"ヒント単語": "目", "関係するA群の単語": ["トキ"]},
    #  {"ヒント単語": "歴史", "関係するA群の単語": ["アトランティス"]},
    #  {"ヒント単語": "強盗", "関係するA群の単語": ["銀行"]},
    #  {"ヒント単語": "乗り物", "関係するA群の単語": ["川", "トキ"]},
    #  {"ヒント単語": "白", "関係するA群の単語": ["トキ"]},
    #  {"ヒント単語": "音", "関係するA群の単語": ["ピアノ", "フルート"]},
    #  {"ヒント単語": "治療", "関係するA群の単語": ["病気", "歯"]},
    #  {"ヒント単語": "海洋", "関係するA群の単語": ["アトランティス"]},
    #  {"ヒント単語": "化石", "関係するA群の単語": ["トキ"]}]'''
    print(answer)
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


def add_gpt_with_wordnet(my_target_words: List[str], opponent_target_words: List[str]) -> Tuple[str, int, Tuple[str, ...]]:
    RETRY_COUNT = 5
    MAX_TRAVERSE_DEPTH = 3

    for _ in range(RETRY_COUNT):
        try:
            answer_df = query_chatgpt_with_candidate_words(my_target_words=my_target_words, opponent_target_words=opponent_target_words)
            break
        except (InvalidGPTOutputError, json.decoder.JSONDecodeError):
            continue

    # ngなヒントは除外: 時々gptがルールを無視する
    def _filter_ng_gpt(row):
        expecting_words = row['関係するA群の単語']
        if len(set(expecting_words) & set(opponent_target_words)) > 0:
            return []
        return row['関係するA群の単語']

    answer_df['関係するA群の単語'] = answer_df.apply(_filter_ng_gpt, axis=1)
    print(answer_df)

    target_words = my_target_words + opponent_target_words

    # filter with wordnet
    wordnet_distance_data_dict: Dict[str, Dict[str, float]] = {
        target_word: gokart.build(CalculateWordDistanceWithWordNet(target_word=target_word, traverse_depth=MAX_TRAVERSE_DEPTH), log_level=logging.ERROR)
        for target_word in tqdm(target_words)
    }
    ng_words_from_wordnet = list(itertools.chain.from_iterable([wordnet_distance_data_dict[opponent_word].keys() for opponent_word in opponent_target_words]))
    print(ng_words_from_wordnet)

    filtered_answer_df = answer_df.loc[~answer_df['ヒント単語'].isin(ng_words_from_wordnet)]

    print('wordnet filtered')
    print(filtered_answer_df)

    # add words with wordnet
    wordnet_scoring_model = ScoringWithRedAndBlue.calculate_scores(
        my_target_words=my_target_words,
        opponent_target_words=opponent_target_words,
        distance_data_dict=wordnet_distance_data_dict,
        my_target_score_offset=0,
        fillna_distance_for_me=999,
        fillna_distance_for_opponent=MAX_TRAVERSE_DEPTH + 2,
    )

    print('wordnet')
    pd.options.display.max_rows = 100
    print(wordnet_scoring_model._candidates_table.sort_values(['count', 'total_score'], ascending=False).head(100))

    # merge
    chatgpt_outputs = filtered_answer_df.set_index('ヒント単語').rename(columns={'関係するA群の単語': 'expecting_my_target_word'})
    chatgpt_outputs['score'] = 3.

    wordnet_outputs = wordnet_scoring_model._candidates_table[['expecting_my_target_word', 'score']]

    merged_output = chatgpt_outputs.join(wordnet_outputs, how='outer', lsuffix='_chatgpt', rsuffix='_wordnet')
    merged_output['expecting_my_target_word_chatgpt'] = merged_output['expecting_my_target_word_chatgpt'].apply(lambda d: d if isinstance(d, list) else [])
    merged_output['expecting_my_target_word_wordnet'] = merged_output['expecting_my_target_word_wordnet'].apply(lambda d: d if isinstance(d, list) else [])
    merged_output['score_chatgpt'] = merged_output['score_chatgpt'].fillna(0)
    merged_output['score_wordnet'] = merged_output['score_wordnet'].fillna(0)
    merged_output['merged_expecting_words'] = [
        list(set(words)) for words in merged_output['expecting_my_target_word_chatgpt'] + merged_output['expecting_my_target_word_wordnet']
    ]
    merged_output['merged_count'] = [len(words) for words in merged_output['merged_expecting_words']]
    merged_output['merged_score'] = merged_output['score_chatgpt'] + merged_output['score_wordnet']

    merged_output = merged_output.sort_values(['merged_count', 'merged_score'], ascending=(False))
    print(merged_output.head(100))

    best_candidate_word = merged_output.iloc[0].name
    expect_count = merged_output.iloc[0]['merged_count']
    expect_words = merged_output.iloc[0]['merged_expecting_words']
    return best_candidate_word, expect_count, expect_words
