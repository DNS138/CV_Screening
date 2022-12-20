from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


class TFIDFWeightor:
    def __init__(self, resume_text, jd_text):
        self.resume_text = resume_text
        self.jd_text = jd_text
        self.vectorizer = TfidfVectorizer()

    def calculate_weights(self):
        vectors = self.vectorizer.fit_transform([self.resume_text, self.jd_text])
        feature_names = self.vectorizer.get_feature_names()
        denselist = vectors.todense().tolist()
        return pd.DataFrame(denselist, columns=feature_names)
