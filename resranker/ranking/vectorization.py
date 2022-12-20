import gensim.downloader as api
import numpy as np


class TFIDFWeightedWord2VecForPOS:
    def __init__(self, df_tfidf_weights, pos_dict):
        self.df_tfidf_weights = df_tfidf_weights
        self.pos_dict = pos_dict
        self.model = api.load("word2vec-google-news-300")
        self.group_with_pairs = ["adjective_nouns", "noun_verbs"]

    def get_pos_vectors(self):
        pos_vectors = {}

        for keyword_group in self.pos_dict:
            if keyword_group in self.group_with_pairs:
                pos_vectors[keyword_group] = self._vectorize_keyword_group_pair(
                    keyword_group
                )
            else:
                pos_vectors[keyword_group] = self._vectorize_keyword_group(
                    keyword_group
                )

        return pos_vectors

    def _vectorize_keyword_group(self, group_name):
        final_vector = np.zeros(300)
        total_weight = 0

        for word in self.pos_dict[group_name]:

            # get weight
            tfidf_weight = self._get_tfidf_weight(word)

            # get vector
            vector = self._get_vector(word)

            # weight vector
            if vector is not None:
                final_vector += tfidf_weight * vector
                total_weight += tfidf_weight

        # final mean
        return final_vector / total_weight

    def _vectorize_keyword_group_pair(self, group_name):
        final_vector = np.zeros(300)
        total_weight = 0

        for (word1, word2) in self.pos_dict[group_name]:
            tfidf_weight1 = self._get_tfidf_weight(word1)
            vector1 = self._get_vector(word1)

            tfidf_weight2 = self._get_tfidf_weight(word2)
            vector2 = self._get_vector(word2)

            if vector1 is not None and vector2 is not None:
                final_vector += (tfidf_weight1 * vector1) + (tfidf_weight2 * vector2)
                total_weight += tfidf_weight1 + tfidf_weight2

        # final mean
        return final_vector / total_weight

    def _get_vector(self, word):
        if word in self.model.key_to_index:
            return self.model[word]
        return None

    def _get_tfidf_weight(self, word):
        tfidf_weight = 0
        if word in self.df_tfidf_weights.columns:
            tfidf_weight = self.df_tfidf_weights[word].iloc[0]

        return tfidf_weight
