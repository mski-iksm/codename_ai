from codename_ai.model.boss import WikiPMIBossModel
from codename_ai.model.game import Game
from codename_ai.runner.validation import validation

if __name__ == '__main__':
    validation(model_name='wiki_pmi', random_base=1000, from_q_num=0)

    # DEBUG MODE =====================
    # game = Game.setup_game(random_seed=3)
    # print(game.get_unopened_words_by_color())

    # boss_model = WikiPMIBossModel.setup_model(my_color='blue')
    # best_candidate_word, expect_count, expect_words = boss_model.next_hint(game=game)
    # print(best_candidate_word, expect_count)
