from rouge_score import rouge_scorer

r_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def get_rouge_score(s1, s2):
    if not (s1 and s2):
        return -1
    return r_scorer.score(s1, s2)["rougeL"].fmeasure
