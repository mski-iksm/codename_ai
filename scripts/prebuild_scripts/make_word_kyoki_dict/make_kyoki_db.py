import os
from typing import Dict
from codename_ai.data.wordpack import CODENAME_WORDS
import pickle

output_db_dir = 'output_db'

KEEP_LIMIT = 10


def cut_low_frequency_words(word_frequency_dict: Dict[str, Dict[str, int]], keep_limit: int) -> Dict[str, Dict[str, int]]:
    return {word: count for word, count in word_frequency_dict.items() if count >= keep_limit}


def cut_low_kyoki_frequency_words(output_dict: Dict[str, Dict[str, int]], keep_limit: int) -> Dict[str, Dict[str, int]]:
    return {
        key: {_inner_key: _inner_val
              for _inner_key, _inner_val in inner_dict.items() if _inner_val >= keep_limit}
        for key, inner_dict in output_dict.items()
    }


if __name__ == '__main__':
    codename_words_set = set(CODENAME_WORDS)

    output_dict: Dict[str, Dict[str, int]] = {codename_word: {} for codename_word in CODENAME_WORDS}
    word_frequency_dict: Dict[str, int] = {}

    with open('data/wiki_ngram/jawiki-tokenized-stem-sentence.txt', encoding='utf-8', mode='r') as f:
        while (True):
            line = f.readline()
            if (not line):
                break

            words = set(line.split(' '))

            for word in words:
                _new_val = word_frequency_dict.get(word, 0) + 1
                word_frequency_dict[word] = _new_val

            words_in_codenames = codename_words_set & words

            for key_word in words_in_codenames:
                for pair_word in words:
                    _inner_dict = output_dict[key_word]
                    _new_val = _inner_dict.get(pair_word, 0) + 1
                    output_dict[key_word][pair_word] = _new_val

    high_frequency_word_frequency_dict = cut_low_frequency_words(word_frequency_dict, keep_limit=10)
    high_frequency_output_dict = cut_low_kyoki_frequency_words(output_dict, keep_limit=10)

    os.makedirs(output_db_dir, exist_ok=True)

    with open(f'{output_db_dir}/frequency_dict_small.pkl', 'wb') as f:
        pickle.dump(high_frequency_word_frequency_dict, f)

    with open(f'{output_db_dir}/kyoki_frequency_dict_small.pkl', 'wb') as f:
        pickle.dump(high_frequency_output_dict, f)