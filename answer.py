#!/usr/bin/env python3
import sys
import math
from collections import Counter
import spacy

import warnings
warnings.filterwarnings("ignore")

"""def allChildren(t):
    # input should be a token
    # return all the children as a string without considering the actual order
    # or content of the sentence
    #############################
    ### I personally like allChildrenIndex better
    #############################

    l = [w for w in t.children]
    if len(l) == 0:
        return t.text
    else:
        return " ".join([allChildren(c) for c in t.children])"""

def allChildrenIndexHelper(t):
    l = [w for w in t.children]
    if len(l) == 0:
        #print(l, str(t.i))
        return [t.i]
    else:
        #print(l, [allChildrenIndexHelper(c) for c in t.children])
        return [elem for elems in [allChildrenIndexHelper(c) for c in t.children]for elem in elems] + [t.i]

def allChildrenIndex(t):
    # input should be a token
    # Returning the index for the child clause
    # Output: start, end
    # child clause would be st[start, end] where st is the original sentence
    allIndexes = [elem for elem in allChildrenIndexHelper(t)]
    return min(allIndexes), max(allIndexes)+1

"""def bestPara(article, question, kword):
    # sim: a 2D array
    # sim[i][j]: cos sim for question i and paragraph j
    nlp = spacy.load("en_core_web_sm")
    article = article.split("\n")
    bstP = [0] * len(article)
    qNLP = nlp(question)

    bstP = [qNLP.similarity(nlp(art)) for art in article]
    #print(bstP)

    inds = [ind for ind, v in sorted(enumerate(bstP), key = lambda x: x[1], reverse=True)[:10]]
    for p in [article[ind] for ind in inds]:
        if kword in p:
            return p
    return [article[ind] for ind in inds][0]"""

def bestSent(parag, ques, kword):
    # parag and ques are string
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(parag)
    sents = list(doc.sents)

    qNLP = nlp(ques)
    bstS = [qNLP.similarity(nlp(s.text)) for s in sents]


    inds = [ind for ind, v in sorted(enumerate(bstS), key = lambda x: x[1], reverse=True)[:10]]
    for s in [sents[ind] for ind in inds]:
        if kword in s.text:
            return s
    return [sents[ind] for ind in inds][0]

    #ind, v = max(enumerate(bstS), key = lambda x: x[1])
    #return sents[ind].text

def whoAnswer(st, qt):
    r = ""
    for i in range(len(qt)):
        if qt[i].dep_ == "ROOT":
            r = qt[i]
            break

    s = st.text.split()
    if r.text in st.text and r.text in s:
        for i in range(s.index(r.text), len(st)):
            if st[i].text == r.text:
                #print(list(st.noun_chunks))
                children = [allChildrenIndex(c) for c in list(st[i].children) 
                if "nsubj" in c.dep_]
                children = [st[a:b] for a, b in children]

                if len(children)>0:
                    nlp = spacy.load("en_core_web_sm")
                    ind, v = max(enumerate([sub.similarity(qt) for sub in 
                        children]), key = lambda x: x[1])
                    return children[ind].text
                    
    # Otherwise
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
    return st.text

def whatAnswer(st, qt):
    r = ""
    for i in range(len(qt)):
        if qt[i].dep_ == "ROOT":
            r = qt[i]
            break

    s = st.text.split()
    if r.text in st.text and r.text in s:
        for i in range(s.index(r.text), len(st)):
            if st[i].text == r.text:
                children = [allChildrenIndex(c) for c in list(st[i].children) 
                if "nsubj" in c.dep_]
                children = [st[a:b] for a, b in children]

                if len(children)>0:
                    nlp = spacy.load("en_core_web_sm")
                    ind, v = max(enumerate([sub.similarity(qt) for sub in 
                        children]), key = lambda x: x[1])
                    return children[ind].text

    # Otherwise
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
    return st.text

def whenAnswer(st, qt):
    ans = ""
    #print([w.dep_ for w in st])
    return st.text

def whereAnswer(st, qt):
    return st.text

def whyAnswer(st, qt):
    because = -1
    subj = 0
    root = ""
    for token in qt:
        if token.dep_ == "ROOT":
            root = token
            # Find the subject of the sentence
            for child in token.children:
                if (child.dep_ == "nsubj" or child.dep_ == "nsubjpass"):
                    subj = child.i
                    break
    for token in st:
        if token.text.lower() == "because": because = token.i

    if (because == -1): return st.text

    if (subj > because):
        sentence = d[because:subject].text
        sentence.rstrip(",")
        return sentence
    else:
        #first_nounphrase = list(qt.noun_chunks)[0]
        sentence = "Because " + st[because+1:].text
        return sentence

def howAnswer(st, qt):
    return st.text

def YNAnswer(st, qt):
    if [t.dep_ for t in st].count('neg') % 2:
        # odd case, negation
        return "No"
    return "Yes"

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

def simp(s):
    nlp = spacy.load("en_core_web_sm")
    st = nlp(s)
    sts = []
    punctuations = []
    temp = 0
    roott = ""
    for i in range(len(st)):
        if st[i].dep_ == "ROOT":
            roott = st[i]
        if st[i].dep_ == 'punct':
            sts += [(temp, i)]
            punctuations += [i]
            temp = i+1

    sentReturn = []

    if len(sts) == len(punctuations):
        noooo = ['because']
        a, b = sts[0]
        short = st[a:b]
        flag = False
        for w in short:
            if w in roott.children:
                flag = True
                break
        if (flag or roott in short) and st[a].text.lower() != 'because':
            sentReturn += [[a, b]]

        for i in range(len(sts)-1):
            a, b = sts[i+1]
            if st[punctuations[i]].text not in [';', ','] and flag:
                sentReturn += [[punctuations[i], b]]
            else:
                short = st[a:b]
                flag = False
                for w in short:
                    if w in roott.children:
                        flag = True
                        break
                if (flag or roott in short) and st[a].text.lower() != 'because':
                    sentReturn += [[punctuations[i], b]]
    else:
        return s
    ans = ""
    for (a, b) in sentReturn:
        ans += st[a:b].text
    return ans + '.'


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

    bestMatchingP = [0] * len(qs)
    bestMatchingS = [0] * len(qs)
    for i in range(len(qs)):
        nlp = spacy.load("en_core_web_sm")
        qtn = nlp(qs[i])
        for j in range(len(qtn)):
            if qtn[j].dep_ == "ROOT":
                kword = qtn[j].text
        #bestMatchingP[i] = bestPara(text, qs[i], kword)
        bestMatchingS[i] = bestSent(text, qs[i], kword)


    ##################################################
    ### Finding Answer from Best Matching Sentence ###
    ##################################################

    # find the answer
    ans = []
    for i in range(len(bestMatchingS)):
        ans.append(generateAns(bestMatchingS[i].text, qs[i]).rstrip("\n"))

    for a in ans:
        print(a)

main()
