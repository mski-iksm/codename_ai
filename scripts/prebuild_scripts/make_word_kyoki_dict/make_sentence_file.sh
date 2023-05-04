export PYTHONPATH=~/github/codename_ai

# ファイルをコピー 本番は不要
# cp data/wiki_ngram/jawiki-tokenized-stem.tsv data/wiki_ngram/jawiki-tokenized-stem_small.tsv

# 1文1行にする
sed 's/\t\t*/\n/g' data/wiki_ngram/jawiki-tokenized-stem.tsv > data/wiki_ngram/jawiki-tokenized-stem-sentence.txt

python scripts/prebuild_scripts/make_word_kyoki_dict/make_kyoki_db.py