from typing import List


def base_small_candidate_words() -> List[str]:
    from codename_ai.data.wordpack import CODENAME_WORDS
    return sorted(list(set(CODENAME_WORDS)))


def large_ipa_noun_candidate_words() -> List[str]:
    from codename_ai.data.ipa_noun import IPA_NOUNS
    return sorted(list(set(IPA_NOUNS)))


def small_noun_candidate_words() -> List[str]:
    from codename_ai.data.noun_small import NOUN_SMALL
    return sorted(list(set(NOUN_SMALL)))
