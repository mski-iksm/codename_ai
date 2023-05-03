import itertools
import json
from typing import List

with open('data/ng_words_v002.json') as f:
    ng_words = json.load(f)


def is_a_not_part_of_bs(a_word: str, b_words: List[str]) -> bool:
    return all([a_word not in b_word for b_word in b_words])


def is_bs_not_part_of_a(a_word: str, b_words: List[str]) -> bool:
    return all([b_word not in a_word for b_word in b_words])


def is_valid_hint(hint_word: str, target_words: List[str]) -> bool:
    ng_words_for_targets = list(itertools.chain.from_iterable([ng_words.get(target_word, []) for target_word in target_words]))
    return hint_word not in ng_words_for_targets
