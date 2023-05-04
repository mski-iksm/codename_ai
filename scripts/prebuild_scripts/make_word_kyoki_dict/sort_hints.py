import pandas as pd

VERY_LARGE_INT = 99999999

if __name__ == '__main__':
    kyoki = pd.read_pickle('output_db/kyoki_frequency_dict_small.pkl')
    freq = pd.read_pickle('output_db/frequency_dict_small.pkl')

    for key_word in kyoki.keys():
        print(key_word, '=================')
        if len(kyoki[key_word]) == 0:
            continue

        _df = pd.DataFrame.from_dict({word: [count / freq.get(word, VERY_LARGE_INT)] for word, count in kyoki[key_word].items()}).T
        _df.columns = ['count']
        _df = _df.sort_values('count', ascending=False)
        print(_df.head(20))
