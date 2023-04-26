from typing import List


def is_a_not_part_of_bs(a_word: str, b_words: List[str]) -> bool:
    return all([a_word not in b_word for b_word in b_words])


def is_bs_not_part_of_a(a_word: str, b_words: List[str]) -> bool:
    return all([b_word not in a_word for b_word in b_words])
