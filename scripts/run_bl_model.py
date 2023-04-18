from codename_ai.model.boss import BaseLineBERTBossModel
from codename_ai.model.game import Game

if __name__ == '__main__':
    game = Game.setup_game(random_seed=10)

    # 介入 ###
    # game._blue_words = {'レーザー', 'ルート', 'ラケット', 'コンサート', '土星'}
    # game._red_words = {'バミューダ', 'ルーム', 'アジ', 'ウサギ'}
    ##########
    print(game.get_unopened_words_by_color())

    boss_model = BaseLineBERTBossModel.setup_model(my_color='blue')
    best_candidate_word, expect_count, expect_words = boss_model.next_hint(game=game)
    print(best_candidate_word, expect_count, expect_words)
