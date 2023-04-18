import unittest

from codename_ai.model.game import Game


class TestGame(unittest.TestCase):

    def test_setup_game(self):
        game = Game.setup_game(3)
        self.assertListEqual(game._blue_words, ['トリップ', 'ボタン', 'パラシュート', 'シュート', '空気', 'マーチ', 'プール', 'ボルト', 'ビート'])
        self.assertListEqual(game._red_words, ['つり', 'カモノハシ', 'かわら', 'リンゴ', '葬儀屋', 'チョコレート', '科学者', 'スイング'])
        self.assertListEqual(game._black_words, ['毒'])
        self.assertListEqual(game._white_words, ['レモン', '潜水艦', 'ヤシ', 'ジム', '騎士', '飛行機', 'ストライク'])
