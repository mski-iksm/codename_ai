import json
import os
from typing import List, Tuple
import openai


def query_chatgpt(my_target_words: List[str], opponent_target_words: List[str]) -> Tuple[str, int]:
    query = f"""今から自分チームと相手チームの単語を与えますので、自分チームの単語のうちいくつかを連想できるようなヒントを1単語で出してください。
自分チームの単語をなるべく数多く連想できるヒント単語を挙げてください。
ただし、相手チームの単語を連想できる単語を挙げてはいけません。
また、そのヒント単語から自分チームのうち何個の単語を連想できるかも教えて下さい。
出せるヒントの数は1つだけですので、最も多く自分チームの単語を連想できそうなヒントを挙げてください。

自分チームの単語：{str(my_target_words)}
相手チームの単語：{str(opponent_target_words)}

返答は以下のJSONの形式だけ受け付けます。
{{"ヒント単語": "XXXXXX", "単語数": XXX}}
    """

    openai.api_key = os.getenv('OPENAI_API_KEY')
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {
                'role': 'system',
                'content': (query)
            },
        ],
    )
    answer = response['choices'][0]['message']['content']
    json_answer = answer.split('\n')[0]
    dict_answer = json.loads(json_answer)

    return (dict_answer['ヒント単語'], dict_answer['単語数'])
