import os
from typing import Dict
from codename_ai.model.boss import BaseLineBERTBossModel, BossModelBase, ChatGPTBossModel, Word2VecBossModel, WordNetBossModel
from codename_ai.model.game import Game

model_name2class: Dict[str, BossModelBase] = {
    'word2vec': Word2VecBossModel,
    'bert': BaseLineBERTBossModel,
    'chatgpt': ChatGPTBossModel,
    'wordnet': WordNetBossModel,
}


def validation(model_name: str, random_base: int = 10000, from_q_num: int = 0):
    correct_counts = []
    wrong_counts = []

    os.makedirs('validation_output', exist_ok=True)
    file_path = f'validation_output/validation_output_{model_name}.txt'

    for q_num, random_seed in enumerate(range(from_q_num, 20)):
        q_num_print = f'第{q_num+1+from_q_num}問##############################'
        print(q_num_print)
        game = Game.setup_game(random_seed=random_seed + random_base)

        boss_model = model_name2class[model_name].setup_model(my_color='blue')
        best_candidate_word, expect_count, expect_words = boss_model.next_hint(game=game)
        print('\t'.join(game.get_all_unopened_words_for_player()))

        hint_print = f'hint: {best_candidate_word} count: {expect_count}'
        print(hint_print)

        corrects = []
        wrongs = []

        selections = input(f'選択肢から{expect_count}個選んでください:').split()
        for selection in selections:
            if selection not in game._blue_words:
                wrongs.append(selection)
                break
            corrects.append(selection)
        not_validated = list(set(selections) - set(corrects) - set(wrongs))

        correct_print = f'正解数: {len(corrects)} selected_correct: {corrects}'
        wrong_print = f'不正解数: {len(wrongs)} selected_correct: {wrongs}'
        not_validated_print = f'未評価数: {len(not_validated)} selected_correct: {not_validated}'
        print(correct_print)
        print(wrong_print)
        print(not_validated_print)

        with open(file_path, mode='a') as f:
            prints = [
                q_num_print,
                hint_print,
                correct_print,
                wrong_print,
                not_validated_print,
            ]
            f.writelines([item + '\n' for item in prints])

        correct_counts.append(len(corrects))
        wrong_counts.append(len(wrongs))

    result_prints = [
        'result ###############################################\n',
        f'平均正解数: {sum(correct_counts)/len(correct_counts)}\n',
        f'平均誤答数: {sum(wrong_counts)/len(wrong_counts)}\n',
    ]
    print(*result_prints)

    with open(file_path, mode='a') as f:
        f.writelines(result_prints)
