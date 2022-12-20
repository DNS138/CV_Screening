import os
import sys

from resranker.extraction.keywords import POSKeywordExtractor
from resranker.extraction.text import Pdf2Txt, Doc2Txt
from resranker.ranking.word_importance import TFIDFWeightor
from resranker.ranking.vectorization import TFIDFWeightedWord2VecForPOS
from resranker.ranking.scoring import CosineSimilarityForPOS, SkillsetOccurence


def extract_text(file_path):
    pdf_extractor = Pdf2Txt(file_path)
    docx_extractor = Doc2Txt(file_path)

    if pdf_extractor.is_valid():
        text = pdf_extractor.extract()
    elif docx_extractor.is_valid():
        text = docx_extractor.extract()
    else:
        print("Invalid file.")
        sys.exit(1)
    return text


def score_based_on_word2vec(resume_text, jd_text, pos_dict_resume, pos_dict_jd):
    # get tfidf weights
    tfidf_weights = TFIDFWeightor(resume_text, jd_text).calculate_weights()

    # get keyword vectors for resume
    vector_dict_resume = TFIDFWeightedWord2VecForPOS(
        tfidf_weights.iloc[:1], pos_dict_resume
    ).get_pos_vectors()

    # get keyword vectors for jd
    vector_dict_jd = TFIDFWeightedWord2VecForPOS(
        tfidf_weights.iloc[1:2], pos_dict_jd
    ).get_pos_vectors()

    # score resume for jd
    score = CosineSimilarityForPOS(vector_dict_resume, vector_dict_jd).final_score()
    return score


def score_based_on_skillset(pos_dict_resume, field):
    # get list of keyword
    all_keywords = []
    for keywords in pos_dict_resume.values():
        all_keywords += keywords

    score = SkillsetOccurence().score(all_keywords, field)
    return score


def final_score(score_w2v, score_skillset):
    return (0.55 * score_skillset) + (0.45 * score_w2v)


if __name__ == "__main__":
    jd_file_path = input("Enter the path of the Job Description file: ")
    resume_dir_path = input("Enter the path of the Resume directory: ")
    field = input("Enter the field: ")

    jd_text = extract_text(jd_file_path)
    # extract pos from jd
    pos_extractor_jd = POSKeywordExtractor(jd_text)
    pos_extractor_jd.extract()
    pos_dict_jd = pos_extractor_jd.get_unique()

    resume_scores = {}
    for resume_file_name in os.listdir(resume_dir_path):
        print("Processing", resume_file_name)

        resume_file_path = os.path.join(resume_dir_path, resume_file_name)

        resume_text = extract_text(resume_file_path)

        # extract pos from resume
        pos_extractor = POSKeywordExtractor(resume_text)
        pos_extractor.extract()
        pos_dict_resume = pos_extractor.get_unique()

        score_w2v = score_based_on_word2vec(
            resume_text, jd_text, pos_dict_resume, pos_dict_jd
        )
        # print("score_w2v:", score_w2v)

        score_skillset = score_based_on_skillset(pos_dict_resume, field)
        # print("score_skillset:", score_skillset)

        score = final_score(score_w2v, score_skillset)
        # print("Final Score: {}%".format(score * 100))

        resume_scores[resume_file_name] = score

    resume_sequence = dict(
        sorted(resume_scores.items(), key=lambda item: item[1], reverse=True)
    )
    print("Sequence of resumes from best to worst is:")
    for key, value in resume_sequence.items():
        print(key, ' : ', round(value*100,2), '%')

    print("**********************")
    print("The best matching resume is:", next(iter(resume_sequence)))
    print("Done \U0001F920")
    print("**********************")
