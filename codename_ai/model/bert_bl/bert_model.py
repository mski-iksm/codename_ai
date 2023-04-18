from typing import Dict, List, Tuple

import gokart
import luigi
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertJapaneseTokenizer, BertModel

# MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'
MODEL_NAME = 'cl-tohoku/bert-base-japanese'
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)


class CalculateWordDistanceWithBERT(gokart.TaskOnKart):
    target_word: str = luigi.Parameter()
    candidate_words: List[str] = luigi.ListParameter()

    __dev = luigi.FloatParameter(default=0.01)

    def requires(self):
        return {
            'target_words': GetMultipleSentencesBertVector(input_sentences=[self.target_word]),
            'candidate_words': GetMultipleSentencesBertVector(input_sentences=self.candidate_words),
        }

    def run(self):
        target_word_vector: torch.Tensor = self.load('target_words')[self.target_word]
        candidate_word_vectors: Dict[str, torch.Tensor] = self.load('candidate_words')
        self.dump(self._run(target_word_vector=target_word_vector, candidate_word_vectors=candidate_word_vectors))

    @classmethod
    def _run(cls, target_word_vector: torch.Tensor, candidate_word_vectors: Dict[str, torch.Tensor]) -> Dict[str, float]:
        candidate_word_vector_keys = list(candidate_word_vectors.keys())
        candidate_word_vectors_tensor = torch.stack([candidate_word_vectors[key] for key in candidate_word_vector_keys])
        distance = -torch.nn.functional.cosine_similarity(target_word_vector.to(torch.float32), candidate_word_vectors_tensor.to(torch.float32),
                                                          dim=1).detach().numpy().copy()
        assert distance.shape[0] == len(candidate_word_vector_keys)
        return {word: float(dist) for dist, word in zip(distance, candidate_word_vector_keys)}


class GetMultipleSentencesBertVector(gokart.TaskOnKart):
    model_name: str = luigi.Parameter(default=MODEL_NAME)
    input_sentences: Tuple[str, ...] = luigi.ListParameter()
    BATCH_SIZE = luigi.IntParameter(default=1000)

    def run(self):
        self.dump(self._sentences_to_vector(input_sentences=list(self.input_sentences)))

    def _sentences_to_vector(self, input_sentences: List[str]) -> Dict[str, torch.Tensor]:
        all_vectors_dict = {}
        for idx in tqdm(range(0, int((len(input_sentences) - 1) / self.BATCH_SIZE) + 1)):
            all_vectors_dict.update(self._bert_sentences_to_vector(sentences=input_sentences[self.BATCH_SIZE * idx:self.BATCH_SIZE * (idx + 1)]))
        return all_vectors_dict

    @classmethod
    def _bert_sentences_to_vector(cls, sentences: List[str]) -> Dict[str, torch.Tensor]:
        if len(sentences) == 0:
            return {}
        # 文を単語に区切って数字にラベル化
        token_obj = tokenizer(sentences, padding=True, truncation=True)
        target_word_tokens = token_obj['input_ids']

        # BERTモデルの処理のためtensor型に変換
        input_tensors = torch.tensor(target_word_tokens)
        attention_mask = torch.tensor(token_obj.attention_mask)

        # BERTモデルに入力し文のベクトルを取得
        model.eval()
        with torch.no_grad():
            outputs = model(input_tensors, output_hidden_states=True, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state
            averaged_hidden_state = torch.permute(torch.permute(last_hidden_state, (2, 0, 1)) * attention_mask, (1, 2, 0)).mean(1)

        vectors_dict = {key: state.detach().clone() for key, state in zip(sentences, averaged_hidden_state)}
        return vectors_dict
