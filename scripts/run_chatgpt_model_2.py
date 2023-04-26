from codename_ai.model.boss import ChatGPTWithWord2VecBossModel
from codename_ai.model.game import Game

if __name__ == '__main__':
    game = Game.setup_game(random_seed=0)
    print(game.get_unopened_words_by_color())

    boss_model = ChatGPTWithWord2VecBossModel.setup_model(my_color='blue')
    best_candidate_word, expect_count, expect_words = boss_model.next_hint(game=game)
