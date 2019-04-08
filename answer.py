import sys
import math
from collections import Counter
import spacy

import warnings
warnings.filterwarnings("ignore")

def bestPara(article, questions):
    # sim: a 2D array
    # sim[i][j]: cos sim for question i and paragraph j
    sim = [[0]*len(article)]*len(questions)
    nlp = spacy.load("en_core_web_sm")

    # compute all values in sim
    for j in range(len(article)):
        # note that para is a string of several sentences
        parNLP = nlp(article[j])
        for i in range(len(questions)):
            sim[i][j] = parNLP.similarity(nlp(questions[i]))

    # find the best matching paragraph for each question
    # find the max index for each sim[i] (question i)

    # bstParagraph[i]: the index of best matching para for question i
    # iow, article[bstParagraph[i]]: best matching paragraph for question i
    bstParagraph = [0] * len(questions)
    for i in range(len(sim)):
        ind, v = max(enumerate(sim[i]), key = lambda x: x[1])
        bstParagraph[i] = ind
    return bstParagraph

def bestSent(parag, ques):
    # parag and ques are string
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(parag)
    sents = list(doc.sents)
    bstS = [0]*len(sents)

    qNLP = nlp(ques)
    bstS = [qNLP.similarity(nlp(s.text)) for s in sents]

    ind, v = max(enumerate(bstS), key = lambda x: x[1])
    return sents[ind].text

def whoAnswer(st, qt):
    # When modifying, consider both NER and dep
    ans = ""
    for i in range(len(st)):
        if st[i].dep_=="ROOT":
            chd = [ww.dep_ for ww in st[i].children]

            try:
                ind = chd.index("nsubj")
            except ValueError:
                ind = chd.index("nsubjpass")
            ans = list(st[i].children)[ind].text

            lfts = list(list(st[i].children)[ind].lefts)
            for l in range(len(lfts)-1, -1, -1):
                if lfts[l].dep_ == "nmod":
                    ans = lfts[l].text + " " + ans
            return ans
    return "None Found"

def whatAnswer(st, qt):
    # When modifying, consider both NER and dep
    ans = ""
    for i in range(len(st)):
        if st[i].dep_=="ROOT":
            chd = [ww.dep_ for ww in st[i].children]
            ind = chd.index("nsubj")
            ans = list(st[i].children)[ind].text

            lfts = list(list(st[i].children)[ind].lefts)
            for l in range(len(lfts)-1, -1, -1):
                if lfts[l].dep_ == "nmod":
                    ans = lfts[l].text + " " + ans
            return ans
    return "None Found"

def whenAnswer(st, qt):
    ans = ""
    #print([w.dep_ for w in st])
    return "None Found"

def whereAnswer(st, qt):
    return "None Found"

def whyAnswer(st, qt):
    return "None Found"

def howAnswer(st, qt):
    return "None Found"

def YNAnswer(st, qt):
    return "None Found"

def generateAns(s, q):
    # Who, where, when, what, which, why, how, yes/no
    nlp = spacy.load("en_core_web_sm")
    st = nlp(s)
    qt = nlp(q)
    ans = ""

    if qt[0].text == "Who":
        ans = whoAnswer(st, qt)
    elif qt[0].text == "What":
        ans = whatAnswer(st, qt)
    elif qt[0].text == "When":
        ans = whenAnswer(st, qt)
    elif qt[0].text == "Where":
        ans = whereAnswer(st, qt)
    elif qt[0].text == "Why":
        ans = whyAnswer(st, qt)
    elif qt[0].text == "How":
        ans = howAnswer(st, qt) 
    else:
        ans = YNAnswer(st, qt)
    return ans

def main():

    pathArticle, pathQuestions = sys.argv[1:3] # article and questions path

    ###############
    ### file IO ###
    ###############

    # text is a list where each element is a paragraph
    fArt = open(pathArticle)
    text = fArt.read()
    fArt.close()

    # qs is a list where each element is a question
    # len(qs) would be the number of questions asked
    fQ = open(pathQuestions)
    qs = fQ.readlines()
    fQ.close()

    ##############################
    ### Best Matching Sentence ###
    ##############################

    # find best matching paragraph for each question
    # loop through the text and compute each cosine sim
    
    #bestMatchingP = bestPara(text, qs)
    #print(bestMatchingP)

    # find the best matching sent for each q within the best mtaching paragraph

    #bestMatchingS = [0] * len(qs)
    #for i in range(len(bestMatchingP)):
    #    bestMatchingS[i] = bestSent(text[bestMatchingP[i]], qs[i])

    bestMatchingS = [0] * len(qs)
    for i in range(len(qs)):
        bestMatchingS[i] = bestSent(text, qs[i])

    ##################################################
    ### Finding Answer from Best Matching Sentence ###
    ##################################################

    # find the answer
    ans = []
    for i in range(len(bestMatchingS)):
        ans.append(generateAns(bestMatchingS[i], qs[i]))

    for a in ans:
        print(a)

main()