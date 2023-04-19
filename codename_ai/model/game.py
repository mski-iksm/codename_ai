import itertools
import random
from typing import List, Set

from codename_ai.data.wordpack import CODENAME_WORDS


class Game:

    def __init__(self, blue_words: Set[str], red_words: Set[str], black_words: Set[str], white_words: Set[str]) -> None:
        self._blue_words = blue_words
        self._red_words = red_words
        self._black_words = black_words
        self._white_words = white_words

        self._opened_blue_words: Set[str] = set()
        self._opened_red_words: Set[str] = set()
        self._opened_white_words: Set[str] = set()

    @classmethod
    def setup_game(cls, random_seed: int) -> 'Game':
        random.seed(random_seed)
        reordered_words = random.sample(CODENAME_WORDS, len(CODENAME_WORDS))
        blue_words = set([reordered_words.pop() for _ in range(9)])
        red_words = set([reordered_words.pop() for _ in range(8)])
        black_words = set([reordered_words.pop() for _ in range(1)])
        white_words = set([reordered_words.pop() for _ in range(7)])

        return cls(blue_words=blue_words, red_words=red_words, black_words=black_words, white_words=white_words)

    def get_unopened_words_by_color(self):
        return dict(
            blue_words=list(self._blue_words - self._opened_blue_words),
            red_words=list(self._red_words - self._opened_red_words),
            black_words=list(self._black_words),
            white_words=list(self._white_words - self._opened_white_words),
        )

    def get_all_unopened_words_for_player(self):
        return sorted(list(itertools.chain.from_iterable(self.get_unopened_words_by_color().values())))
