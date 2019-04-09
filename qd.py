#!/usr/bin/env python3
import sys
import spacy
import string

# Set the article file path and the number of questions we need to generate
pathArticle, nquestions = sys.argv[1:3]
f = open(pathArticle)
text = f.read()
f.close()


#define some parameters 
noisy_pos_tags = ["PROP"]
min_token_length = 2

#Function to check if the token is a noise or not 
def isNoise(token): 
    is_noise = False
    if token.pos_ in noisy_pos_tags:
        is_noise = True 
    elif token.is_stop == True:
        is_noise = True
    elif len(token.string) <= min_token_length:
        is_noise = True
    return is_noise 

def isNoun(token):
    return(token.pos_ == "NOUN" and len(token.string) > min_token_length)

def cleanup(token, lower = True):
    if lower:
        token = token.lower()
    return token.strip()

from collections import Counter

# Count the most frequent nouns in the corpus
def keywords_selection(doc, n=10):
    cleaned_list = [cleanup(word.string) for word in doc if not isNoise(word) and isNoun(word)]
    top = Counter(cleaned_list).most_common(n)
    return top

def sentences_selection(doc, keywords):
    sentences = doc.sents
    S = []
    for s in sentences:
        st = s.text
        if (st[-1] not in string.punctuation or '\n' in st):
            continue
        for w in keywords:
            if w[0] in st:
                S.append(s)
                break
    return S

def why_question_generator(sentence):
    doc = nlp(sentence)
    front = ""
    aux = False
    subj = doc[0]
    for token in doc:
        if token.dep_ == "ROOT":
            # When the predicate is any form of "be"
            if (token.text == "are" or token.text == "is" or token.text == "was" or token.text == "were"):
                front = token.text
                aux = True
            # Find the auxiliary word
            for child in token.children:
                if(child.dep_ == "nsubj" or child.dep_ =="nsubjpass"):
                    subj = child
                if (child.dep_ == "aux" or child.dep_ == "auxpass"):
                    front = child.text
                    aux = True
            # When there is no auxiliary word 
            if (aux == False):
                if (token.tag_ == "VBD"):
                    front = "did"
                elif (token.tag_ == "VBZ"):
                    front = "does"
                else:
                    front = "do"

    sentence = []
    sentence.append("Why")
    sentence.append(front)
    if (doc[0].text == "Because" or doc[0].text == "Since"):
        for token in doc[subj.i:]:
            if (token.dep_ == "ROOT" and aux == False):
                sentence.append(token.lemma_)
            elif (token.text != front and token.text != "."):
                sentence.append(token.text.lower())
    else:
        for token in doc:
            if (token.text == "because" or token.text == "since"):
                if (token.nbor(-1).is_punct):
                    sentence.pop()
                break
            if (token.dep_ == "ROOT" and aux == False):
                sentence.append(token.lemma_)
            elif (token.text != front and token.text != "."):
                sentence.append(token.text.lower())

    
    return (" ".join(sentence)+"?")


def binary_question_generator(sentence):
    doc = nlp(sentence)
    front = ""
    aux = False
    for token in doc:
        if token.dep_ == "ROOT":
            # When the predicate is any form of "be"
            if (token.text == "are" or token.text == "is" or token.text == "was" or token.text == "were"):
                front = token.text
                aux = True
                break
            # Find the auxiliary word
            for child in token.children:
                if (child.dep_ == "aux" or child.dep_ == "auxpass"):
                    front = child.text
                    aux = True
                    break
            # When there is no auxiliary word 
            if (aux == False):
                if (token.tag_ == "VBD"):
                    front = "did"
                elif (token.tag_ == "VBZ"):
                    front = "does"
                else:
                    front = "do"

    sentence = []
    sentence.append(front.capitalize())
    for token in doc:
        if (token.text == "."):
            break
        if (token.dep_ == "ROOT" and aux == False):
            sentence.append(token.lemma_)
        elif (token.text != front):
            if (token == doc[0] and token.ent_type_ == ""):
                sentence.append(token.text.lower())
            else:
                sentence.append(token.text)
    return (" ".join(sentence)+"?")

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
keywords = keywords_selection(doc, n=6)
keysents = sentences_selection(doc, keywords)
print(keywords)

Q = set()

for sent in keysents:
    labels = list(map(lambda x: x.label_, sent.ents))
    '''if 'PERSON' in labels:
        Q.add(who_question_generator(sent.text))
    if 'LOC' in labels:
        Q.add(where_question_generator(sent.text))
    if 'DATE' in labels:
        Q.add(when_question_generator(sent.text))'''
    if 'because' in sent.text or 'Because' in sent.text:
        Q.add((why_question_generator(sent.text), sent.text))
    Q.add((binary_question_generator(sent.text), sent.text))

for i in range(int(nquestions)):
    if Q: 
        x = Q.pop()
        print(x[0])
        #print(x[1] + "\n")    #the original sentence for debug info
    else: print("Try fewer questions?") 





