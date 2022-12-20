import os
from itertools import chain
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class CosineSimilarityForPOS:
    def __init__(self, pos_vector_dict1, pos_vector_dict2) -> None:
        self.pos_vector_dict1 = pos_vector_dict1
        self.pos_vector_dict2 = pos_vector_dict2
        self.keyword_group_weights = {
            "noun": 0.1,
            "verb": 0.1,
            "adjective_nouns": 0.45,
            "noun_verbs": 0.35,
        }

    def _similarity(self, vector1, vector2):
        return (
            cosine_similarity(vector1.reshape(1, -1), vector2.reshape(1, -1))
            .flatten()
            .item()
        )

    def final_score(self):
        total_score = 0
        for keyword_group in self.pos_vector_dict1:
            similarity_score = self._similarity(
                self.pos_vector_dict1[keyword_group],
                self.pos_vector_dict2[keyword_group],
            )
            total_score += self.keyword_group_weights[keyword_group] * similarity_score

        return total_score


class SkillsetOccurence:
    def __init__(self, jobwise_keywords_excel=None):
        self.jobwise_keywords_excel = jobwise_keywords_excel

        if jobwise_keywords_excel is None:
            self.jobwise_keywords_excel = os.path.join("data", "jobwise_keywords.xlsx")

        self.df_skillsets = pd.ExcelFile(self.jobwise_keywords_excel)

    def score(self, keywords, job_name):
        df_keywords = self.df_skillsets.parse(job_name)

        # extract skillsets from dataframe as a list
        skills_list = [
            list(df_keywords["Tools_Tech"].dropna().values),
            list(df_keywords["Programming_Languages"].dropna().values),
            list(df_keywords["Soft_Skills"].dropna().values),
            list(df_keywords["Hard_Skills"].dropna().values),
        ]
        skills_list_final = list(chain.from_iterable(skills_list))
        skills_list_final = list(map(lambda x: x.lower(), skills_list_final))

        # find common keywords
        common_keywords = set(skills_list_final) & set(keywords)

        return len(common_keywords) / len(skills_list_final)
