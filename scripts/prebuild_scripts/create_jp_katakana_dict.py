import json
import os
import re
from time import sleep
from typing import List

import gokart
import jaconv
import luigi
import openai
import pandas as pd
from tqdm import tqdm

from codename_ai.data.wordpack import CODENAME_WORDS
from codename_ai.model.chatgpt_model.chatgpt_model import InvalidGPTOutputError


def is_katakana(word) -> bool:
    re_katakana = re.compile(r'[\u30A1-\u30F4]+')
    return re_katakana.search(word) is not None


class ConvertJapaneseToKatakana(gokart.TaskOnKart):
    ja_word: str = luigi.Parameter()

    def run(self):
        self.dump(self.convert_to_katakana(ja_word=self.ja_word))

    @classmethod
    def convert_to_katakana(cls, ja_word: str) -> List[str]:
        RETRY_COUNT = 3
        for _ in range(RETRY_COUNT):
            try:
                answers_list = cls.query_gpt(ja_word=ja_word)
                break
            except (InvalidGPTOutputError, json.decoder.JSONDecodeError) as e:
                print(e, 'retry...')
                sleep(3)
                continue
        return answers_list

    @classmethod
    def query_gpt(cls, ja_word: str) -> List[str]:
        query = f"""これから与える日本語を外来語に直してください：{ja_word}
                次の形式で3種類考えてください: ["XXX", "YYY", "ZZZ"]
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

        if not isinstance(answers_list, list):
            raise InvalidGPTOutputError('answer_listがlistでない')
        return answers_list


class ConvertKatakanaToJapanese(gokart.TaskOnKart):
    katakana_word: str = luigi.Parameter()

    __version: int = luigi.IntParameter(default=1)

    def run(self):
        self.dump(self.convert_to_japanese(katakana_word=self.katakana_word))

    @classmethod
    def convert_to_japanese(cls, katakana_word: str) -> List[str]:
        RETRY_COUNT = 3
        for _ in range(RETRY_COUNT):
            try:
                answers_list = cls.query_gpt(katakana_word=katakana_word)
                break
            except (InvalidGPTOutputError, json.decoder.JSONDecodeError) as e:
                print(e, 'retry...')
                sleep(3)
                continue
        return answers_list

    @classmethod
    def query_gpt(cls, katakana_word: str) -> List[str]:
        query = f"""これから与える外来語をひらがな・漢字の日本語に直してください：{katakana_word}
                次の形式で3種類考えてください: ["XXX", "YYY", "ZZZ"]
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

        if not isinstance(answers_list, list):
            raise InvalidGPTOutputError('answer_listがlistでない')
        return answers_list


if __name__ == '__main__':
    # target単語を読み込み
    target_words = CODENAME_WORDS

    katakana_target_words = [target_word for target_word in target_words if is_katakana(target_word)]
    japanese_target_words = [target_word for target_word in target_words if not is_katakana(target_word)]

    # リンゴなどをひらがなに直す
    japanesed_katakana_target_words = [jaconv.kata2hira(target_word) for target_word in katakana_target_words]
    all_japanese_target_words = japanese_target_words + japanesed_katakana_target_words
    print(katakana_target_words)
    print(japanese_target_words)

    # convert japanese to katakana
    japanese2katakana_lists_dict = {
        ja_word: gokart.build(ConvertJapaneseToKatakana(ja_word=ja_word), reset_register=False)
        for ja_word in tqdm(all_japanese_target_words)
    }

    # convert katakana to japanese
    katakana2japanese_lists_dict = {
        katakana_word: gokart.build(ConvertKatakanaToJapanese(katakana_word=katakana_word), reset_register=False)
        for katakana_word in tqdm(katakana_target_words)
    }
    print(katakana2japanese_lists_dict)

    ng_word_dict = japanese2katakana_lists_dict | katakana2japanese_lists_dict

    with open('data/ng_words_v002.json', 'w') as f:
        json.dump(ng_word_dict, f, indent=4, ensure_ascii=False)
