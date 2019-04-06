#!/usr/bin/env python3
from pntl.tools import Annotator
import spacy
import nltk

nlp = spacy.load("en_core_web_sm")

'''tokens = nltk.word_tokenize(text)
tagged = nltk.pos_tag(tokens)
entities = nltk.chunk.ne_chunk(tagged)
print(entities)

dep_parser = nltk.parse.corenlp.CoreNLPDependencyParser(url='http://localhost:9000')
parse = dep_parser.raw_parse(text)
print(parse.to_conll(4))'''
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
                    print(spacy.explain(child.tag_))
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

    
    print(" ".join(sentence)+"?")


def yesno_question_generator(sentence):
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
        if (token.dep_ == "ROOT" and aux == False):
            sentence.append(token.lemma_)
        elif (token.text != front and token.text != "."):
            sentence.append(token.text.lower())
    print(" ".join(sentence)+"?")

why_question_generator("I do need to finish the practice midterm tonight, because I want to watch movies all day tomorrow")

