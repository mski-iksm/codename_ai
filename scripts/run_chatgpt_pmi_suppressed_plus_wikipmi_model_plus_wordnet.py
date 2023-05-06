from codename_ai.model.boss import WikiPMIBossModel
from codename_ai.model.game import Game
from codename_ai.runner.validation import validation

if __name__ == '__main__':
    # validation(model_name='chatgpt_pmi_suppressed_plus_pmi_plus_wordnet', random_base=1000, from_q_num=0)

    # DEBUG MODE =====================
    for i in range(10):
        game = Game.setup_game(random_seed=100 + i)

        text = f"""
        あなたはボードゲーム「コードネーム」のspymasterです。
        今からA群とB群の単語群を与えますので、A群の単語のうち3つ以上の単語を連想できるようなヒントを1単語で出してください。
        ただしB群のいずれかの単語に関係するヒント単語は避けてください。

        また、A群の単語のうちどの単語に関係しているのかも教えて下さい。


        A群：{list(game._blue_words)}
        B群：{list(game._red_words | game._black_words | game._white_words)}

        返答は以下のJSONの形式だけ受け付けます。
        [{{"ヒント単語": "XXXXXX", "関係するA群の単語": ["XXXXXX", "XXXXXX"]}}]

        ヒントの選定にあたっては次のルールを守ってください。
        - A群、B群の単語をカタカナ語に変換した単語をヒント単語にしてはいけません（リンゴ -> アップル は禁止です）
        - A群、B群の単語の一部を含む単語をヒント単語にしてはいけません（てんとう虫 -> 虫 は禁止です）
        """
        print(text)
        print("#########################################")
