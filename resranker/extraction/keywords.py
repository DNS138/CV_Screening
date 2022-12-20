import re
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
nltk.download("omw-1.4")

import spacy

nlp = spacy.load("en_core_web_sm")


class POSKeywordExtractor:
    def __init__(self, text):
        self.raw_text = text
        self.preprocessed_text = ""
        self.nouns = []
        self.verbs = []
        self.adjective_nouns = []
        self.noun_verbs = []

    def _preprocess(self):
        preprocessed_text = self.raw_text.replace("/", " ")
        self.preprocessed_text = (
            re.sub(r"[^\w\s]", "", preprocessed_text)
            .replace("\n", "")
            .replace("  ", " ")
            .lower()
        )

    def _extract_keyword_groups(self):
        doc = nlp(self.preprocessed_text)

        doc = list(doc)

        prev_word = None
        prev_word_pos = None
        for word in doc:
            if word.pos_ == "NOUN" or word.pos_ == "PROPN":
                self.nouns.append(str(word))
            elif word.pos_ == "VERB":
                self.verbs.append(str(word))
            if prev_word is None:
                prev_word = word
                prev_word_pos = word.pos_
                continue
            if (prev_word_pos == "ADJ") and (
                word.pos_ == "NOUN" or word.pos_ == "PROPN"
            ):
                self.adjective_nouns.append((str(prev_word), str(word)))
            if (prev_word_pos == "NOUN" or prev_word_pos == "PROPN") and (
                word.pos_ == "VERB"
            ):
                self.noun_verbs.append((str(prev_word), str(word)))
            prev_word = word
            prev_word_pos = word.pos_

    def extract(self):
        self._preprocess()
        self._extract_keyword_groups()

    def get_all(self):
        return {
            "noun": self.nouns,
            "verb": self.verbs,
            "adjective_nouns": self.adjective_nouns,
            "noun_verbs": self.noun_verbs,
        }

    def get_unique(self):
        return {
            "noun": set(self.nouns),
            "verb": set(self.verbs),
            "adjective_nouns": set(self.adjective_nouns),
            "noun_verbs": set(self.noun_verbs),
        }

    def _df_convert_pairs_to_string(self, data):
        if data is not None:
            return " ".join(data)
        return None

    def get_df(self, pos_dict):
        df = pd.DataFrame.from_dict(pos_dict, orient="index").transpose()
        df["adjective_nouns"] = df["adjective_nouns"].apply(
            self._df_convert_pairs_to_string
        )
        df["noun_verbs"] = df["noun_verbs"].apply(self._df_convert_pairs_to_string)
        return df
