from codename_ai.model.boss import Word2VecBossModel
from codename_ai.model.game import Game

if __name__ == '__main__':
    correct_counts = []
    wrong_counts = []

    for q_num, random_seed in enumerate(range(2)):
        print(f'第{q_num+1}問##############################')
        game = Game.setup_game(random_seed=random_seed)

        # 介入 ###
        # game._blue_words = {'レーザー', 'ルート', 'ラケット', 'コンサート', '土星'}
        # game._red_words = {'バミューダ', 'ルーム', 'アジ', 'ウサギ'}
        ##########

        # debug ###
        # print(game.get_unopened_words_by_color())
        # boss_model = Word2VecBossModel.setup_model(my_color='blue')
        # best_candidate_word, expect_count, expect_words = boss_model.next_hint(game=game)
        # print(best_candidate_word, expect_count, expect_words)
        ###

        boss_model = Word2VecBossModel.setup_model(my_color='blue')
        best_candidate_word, expect_count, expect_words = boss_model.next_hint(game=game)
        print('   '.join(game.get_all_unopened_words_for_player()))
        print(best_candidate_word, expect_count)

        selections = input(f'選択肢から{expect_count}個選んでください:').split()
        corrects = set(selections) & set(game._blue_words)
        wrongs = set(selections) - corrects

        print(f'正解数: {len(corrects)} {corrects}')
        print(f'誤答数: {len(wrongs)} {wrongs}')

        correct_counts.append(len(corrects))
        wrong_counts.append(len(wrongs))

    print('result ###############################################')
    print(f'平均正解数: {sum(correct_counts)/len(correct_counts)}')
    print(f'平均誤答数: {sum(wrong_counts)/len(wrong_counts)}')
