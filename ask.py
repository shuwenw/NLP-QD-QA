#!/usr/bin/env python3
import sys
import spacy
import string
import re
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Set the article file path and the number of questions we need to generate
pathArticle = sys.argv[1]
nquestions = sys.argv[2]
f = open(pathArticle)
text = f.read()
f.close()


#define some parameters 
noisy_pos_tags = ["PROP"]
min_token_length = 2

#####################################################################################

from collections import Counter

# Count the most frequent nouns in the corpus
def keywords_selection(doc, n=10):
    cleaned_list = [cleanup(word.string) for word in doc if not isNoise(word) and isNoun(word)]
    top = Counter(cleaned_list).most_common(n)
    return top

# Select the sentences that contain the most frequent nouns
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

# Simplify a sentence
def simp(s):
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
        a, b = sts[0]
        short = st[a:b]
        flag = False
        for w in short:
            if w in roott.children:
                flag = True
                break
        if flag or roott in short:
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
                if flag or roott in short:
                    sentReturn += [[punctuations[i], b]]
    else:
        return s
    ans = ""
    for (a, b) in sentReturn:
        ans += st[a:b].text
    return ans + '.'

# Find the subject, root and any form of auxiliary word in the sentence
def find_front(doc):
    front = ""
    aux = False
    root = ""
    subj = 0
    for token in doc:
        if token.dep_ == "ROOT":
            root = token
            # Find the subject of the sentence
            for child in token.children:
                if (child.dep_ == "nsubj" or child.dep_ == "nsubjpass"):
                    subj = child.i
                    break
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
    return front, aux, root, subj

def why(sentence):
    doc = nlp(sentence)
    front, aux, root, subj = find_front(doc) 
    because = 0

    for token in doc:
        if token.text.lower() == "because": because = token.i
    
    question = []
    question.append("Why")
    question.append(front)

    if (subj > because):
        sentence = doc[subj:].text
        for word in sentence.split(" "):
            if (word == root.text and aux == False):
                question.append(root.lemma_)
            elif word != front:
                if word[-1] == "," or word[-1] == ".":
                    word = word[:-1]
                if (word == doc[0].text and doc[0].ent_type_ == ""):
                    question.append(word.lower())
                else:
                    question.append(word)
    else:
        first_nounphrase = list(doc.noun_chunks)[0]
        sentence = doc[first_nounphrase.start:].text
        for word in sentence.split(" "):
            pureword = re.sub(r'[^\w\s]','',word)
            if (pureword == "because" or pureword == "since"):
                break
            if (word == root.text and aux == False):
                question.append(root.lemma_)
            elif (word != front):
                if word[-1] == "," or word[-1] == ".":
                    word = word[:-1]
                if (word == doc[0].text and doc[0].ent_type_ == ""):
                    question.append(word.lower())
                else:
                    question.append(word)
    
    return (" ".join(question)+"?")


def binary(sentence):
    if(sentence[0] == ','): return
    doc = nlp(sentence)
    front, aux, root, subj = find_front(doc) 
    if doc[subj].text.lower() == "it": return
    if doc[subj].text.lower() == "this": return
    if doc[subj].text.lower() == "that": return

    for t in doc[:subj]:
        if t.text == ",":
            sentence = doc[t.i+1: -1].text + ", " + doc[:1].text.lower() + " " + doc[1:t.i].text


    question = []        
    question.append(front.capitalize())
    for word in sentence.split(" "):
        pureword = re.sub(r'[^\w\s]','',word)
        if (word[-1] == "."):
            question.append(word[:-1])
            break
        if (pureword == root.text and aux == False):
            question.append(root.lemma_)
        elif (pureword != front):
            if (pureword == doc[0].text and doc[0].ent_type_ == ""):
                question.append(word.lower())
            else:
                question.append(word)
    return (" ".join(question)+"?")

def when(sentence):
    if(sentence[0] == ','): return
    doc = nlp(sentence)
    start, end = 0, 0
    for ent in doc.ents:
        if ent.label_ == "DATE":
            start = ent.start
            end = ent.end
            break

    front, aux, root, subj = find_front(doc) 
    if doc[subj].text.lower() == "it": return
    if doc[subj].text.lower() == "this": return
    if doc[subj].text.lower() == "that": return

    question = []   
    if doc[start-1].text == "for":
        question.append("How long")
        sentence = doc[:start-1].text + " " + doc[end:-1].text
    elif doc[start-1].dep_ == "prep":
        question.append("When")  
        sentence = doc[:start-1].text + " " + doc[end:-1].text
    else: 
        question.append("How long")
        sentence = doc[:start].text + " " + doc[end:-1].text
    
    sentence = sentence.lstrip().rstrip()
    question.append(front)
    for word in sentence.split(" "):
        pureword = re.sub(r'[^\w\s]','',word)

        if (pureword == root.text and aux == False):
            question.append(root.lemma_)
        elif (pureword != front):
            if (pureword == doc[0].text and doc[0].ent_type_ == ""):
                question.append(word.lower())
            else:
                question.append(word)
    return (" ".join(question)+"?")

### What
#Can be anything as long as no "PERSON" or "LOC" or "GPE" in the sentence
def what(sentence):
    doc = nlp(sentence)
    word_bases = [token.lemma_ for token in doc]
    #ent_lbl = [ent.label_ for ent in doc.ents]
    #ent_txt = [ent.text for ent in doc.ents]
    #ent_dep = [ent.label_ for ent in doc.ents]
    deps = [token.dep_ for token in doc]
    root_idx = deps.index("ROOT")
    root = doc[root_idx]
    root_ch = [child for child in root.children] #Children
    root_ch_idx = [c.i for c in root_ch] #Children indices wrt doc
    root_ch_idx.append(root_idx)
    root_ch_idx.sort() #Add the root's index into the children list
    root_loc = root_ch_idx.index(root_idx) #Root index location wrt children list
    clauses = child_clause(root_ch,doc) #Clauses
    clauses.insert(root_loc, root.text) #Insert the root word for later merging    
    ch_dep = [child.dep_ for child in root.children]
    if("nsubj" in ch_dep or "nsubjpass" in ch_dep): #Sentence has a subject
        subj_loc = ["nsubj" in t for t in ch_dep]
        subj_idx = subj_loc.index(True) #One before the start clauses
        if(subj_idx >= root_loc):
            subj_idx += 1
        q = ["What"]
        stop = False
        i = subj_idx+1
        while(not stop and i < len(clauses)):
            if("," not in clauses[i]):
                q.append(clauses[i])
            else:
                c = clauses[i]
                q.append(c[0:c.index(",")])
                stop = True
            i += 1
        q = list(filter(None, q))
        q = " ".join(q).strip()
        return(q+"?")
    else:
        return

### Where
def where(sentence):
    doc = nlp(sentence)
    ent_lbl = [ent.label_ for ent in doc.ents]
    ent_dep = [ent.label_ for ent in doc.ents]
    deps = [token.dep_ for token in doc]
    #Get the place
    if("GPE" in ent_lbl):
        place = doc.ents[ent_lbl.index("GPE")]
    elif("LOC" in ent_lbl):
        place = doc.ents[ent_lbl.index("LOC")]
    i = place.start
    if(len(place) > 1):
        i = place.start - 1
        found = False
        while(not found and i <= place.end):
            i += 1 # The i is the index of the head of the root. If it's multiword, it finds the head of the noun phrase
            if(not doc[i].head.text in place.text):
                found = True
    #Get the root and its child clauses
    word_bases = [token.lemma_ for token in doc]
    root = doc[deps.index("ROOT")]
    ch = [child for child in root.children]
    ch_idx = [child.i for child in root.children]
    ch_clause = child_clause(ch, doc)
    root_ch_idx = insert_root(ch_idx, ch, root) #To find the clause right after the root verb
    ch_dep = [child.dep_ for child in ch]
    ch_subj = ["subj" in dep for dep in ch_dep]
    #If "was" is the verb
    if("be" in word_bases and True in ch_subj): #If sentence uses "is","was","are","were", etc
        be = doc[word_bases.index("be")]
        if(root.lemma_ == "be"):
            q = ["Where", root.text, ch_clause[ch_subj.index(True)]]
        else:
            q = ["Where", be.text, ch_clause[ch_subj.index(True)], root.text]
    elif(True in ch_subj):    #If "was" isn't the verb
        q = ["Where did", ch_clause[ch_subj.index(True)], root.lemma_]
        # Assume LOC's parent is a preposition, and that preposition's parent is the noun object
    else:
        return
    if(doc[i].head.dep_ == "prep"):
        prep = doc[i].head
        obj_place = prep.head
        phrase = child_clause([obj_place],doc)
        phrase = phrase[0].split(" ")
        stop_idx = phrase.index(prep.text)
        phrase_ofInt = " ".join(phrase[:stop_idx])
        q.append(phrase_ofInt)
    q.append("?")
    return(" ".join(q))

### Who
#Either "PERSON" in ent_lbls or "who" in the sentence
def who(sentence):
    #Clear of all (____) terms
    info = get_txt_info(sentence)
    if (info == None): return
    doc, txt_parse, lemma, deps, ent_lbl, ent_start, ent_end, ent_deps, nsubj_in, ent_ofInt, isSubj, root_idx, root_ch, root_ch_idx, root_loc, root_clause_idx, clauses = info
    #Making questions
    if("who" in doc.text):
        return(fullWhoClause(txt_parse))
    elif(doc[root_idx].lemma_ == "be"): #It's an OBJ "to be" SUBJ [descript] sentence
        root_ch_deps = [token.dep_ for token in root_ch]
        end_phrase = ""
        if("prep" in root_ch_deps):
            prep_idx = root_ch_deps.index("prep")
            end_phrase = " " + clauses[prep_idx]
        if(ent_ofInt.text in clauses[root_clause_idx - 1] and isSubj):
            return("Who " + doc[root_idx].text + " " + clauses[root_clause_idx+1] + end_phrase + "?")
        else:
            return("Who " + doc[root_idx].text + " " + clauses[root_clause_idx-1] + end_phrase + "?")
    else: #Root is not a 'to be' verb
        if("be" in lemma): #So the verb phrase represents "were constructed" or "is played"
            if(root_idx < ent_ofInt.start):
                d = clauses[root_loc+1].split(" ")[0]
                d_idx = txt_parse.index(d)
                if(doc[d_idx].dep_ == "prep"): #Entity clause start w/ preposition
                    q = ""
                    i = 0
                    while(i <= root_loc + 1):
                        if(i != root_loc + 1):
                            q = q + " " + clauses[i]
                        else:
                            q = q + " " + clauses[i].split(" ")[0] + " who"
                        i += 1
                    return((q[1:]+"?").capitalize())
                else:
                    clauses[root_loc+1] = "who"
                    return(" ".join(clauses)+"?")
            else:
                t = [ent_ofInt.text in clause for clause in clauses]
                if (True in t): t_ind = t.index(True)
                else: return
                q = "Who " + " ".join(clauses[t_ind+1:]) + "?"
                return(q)
        else:
            return

### Functions
def getData(dir, file):
    curDir = os.getcwd()
    filePath = "\\".join([curDir,dir,file])
    f =  open(filePath, "r")
    text = f.read()
    text = text.split("\n")
    return(" ".join(text))
def isNoise(token):     
    noisy_pos_tags = ["PROP"]
    min_token_length = 2
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
    
def get_txt_info(sentence):
    #Clear of all (____) terms
    while("(" in sentence and ")" in sentence):
        start_idx = sentence.index("(")-1
        end_idx = sentence.index(")")+1
        sentence = sentence.replace(sentence[start_idx:end_idx],"")
    #Generation
    doc = nlp(sentence)
    q = ""
    # Info about sentence #
    txt_parse = [token.text for token in doc]
    lemma = [token.lemma_ for token in doc]
    deps = [token.dep_ for token in doc] #Dependencies in the sentence. To find ROOT
    ent_lbl = [ent.label_ for ent in doc.ents] #Entity types
    ent_start = [ent.start for ent in doc.ents] #Start index of entities
    ent_end = [ent.end for ent in doc.ents]
    # If multiple people, find the one that's nsubj/nsubjpass or just the first one
    ent_deps = [[t.dep_ for t in doc[ent_start[i]:ent_end[i]]] for i in range(len(ent_start))]
    nsubj_in = ["nsubj" in t or "nsubjpass" in t for t in ent_deps]
    if(True in nsubj_in):
        ent_ofInt = doc.ents[nsubj_in.index(True)] #Finds entity that's a subject
        isSubj = True
    else:
        if("PERSON" in ent_lbl):
            ent_ofInt = doc.ents[ent_lbl.index("PERSON")] #Assumes only 1 person
        else: return
        isSubj = False
    # Working with the ROOT
    root_idx = deps.index("ROOT") #Get the index
    root_ch = [child for child in doc[root_idx].children] #Get all of the children of the root
    root_ch_idx = [c.i for c in root_ch] #And their indices
    #Getting index of root word wrt the clauses
    root_ch_idx.append(root_idx)
    root_ch_idx.sort()
    root_loc = root_ch_idx.index(root_idx)
    root_clause_idx = root_ch_idx.index(root_idx)
    clauses = child_clause(root_ch,doc)#,ent_ofInt.text) #And the clauses based off of them
    clauses.insert(root_clause_idx, doc[root_idx].text) #Insert the root word for later merging
    return((doc, txt_parse, lemma, deps, ent_lbl, ent_start, ent_end, ent_deps, nsubj_in, ent_ofInt, isSubj, root_idx, root_ch, root_ch_idx, root_loc, root_clause_idx, clauses))

def child_clause(children, doc, entity = None, types = ["PERSON"], bad = ["appos","punct"]):
    puncts = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    #bad = ["appos","punct","relcl"]
    goods = [t.text for t in doc.ents if t.label_ == "PERSON"]
    goods.append(",")
    clauses = []
    for ch in children:
        h_ofInt = [ch]
        idx_ofInt = [ch.i]
        heads = np.array([[doc[i],doc[i].head,i] for i in range(len(doc))])
        while(any(i in heads[:,1] for i in h_ofInt)):
            inds = [i for i in range(heads.shape[0]) if heads[i,1] in h_ofInt]
            #Find the heads values that have a head that's already in h_ofInt
            h_chosen = [heads[i,:] for i in range(heads.shape[0]) if heads[i,1] in h_ofInt and (heads[i,0].dep_ not in bad or heads[i,0].text in goods)]
            #Add what remains to the h_ofInt and idx_ofInt lists
            h_ofInt.extend([h_chosen[i][0] for i in range(len(h_chosen))])
            idx_ofInt.extend([h_chosen[i][2] for i in range(len(h_chosen))])
            heads = np.delete(heads,inds,axis=0)
        idx_ofInt.sort()
        clause = [doc[i].text for i in idx_ofInt]
        while(len(clause) > 0 and (clause[-1] in puncts or clause[-1] == " ")):
            clause = clause[:-1]
        clause = " ".join(clause)
        clause = re.sub(r' (?=\W)', '', clause)
        if(entity != None):
            if(entity not in clause):
                clauses.append(clause)
        else:
            clauses.append(clause)
    return(clauses)

def insert_root(ch_idx, ch_word, root):
    root_idx = root.i
    ch_idx.append(root_idx)
    ch_idx.sort()
    ch_root_idx = ch_idx.index(root_idx)
    return(ch_root_idx)

def fullWhoClause(doc_parse, i = -1):
    cap = False
    if(i == -1):
        cap = True
        i = doc_parse.index("who")
    if(doc_parse[i + 1] == "," or doc_parse[i + 1] == "."):
        return(doc_parse[i] + "?")
    else:
        if("'" in doc_parse[i+1]):
            q = doc_parse[i] + fullWhoClause(doc_parse, i+1)
        else:
            q = doc_parse[i] + " " + fullWhoClause(doc_parse, i+1)
        if(cap):
            return(q.capitalize())
        else:
            return(q)

#################################################################################

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
keywords = keywords_selection(doc, n=6)
keysents = [sent.text for sent in sentences_selection(doc, keywords)]

Q = set()

for sent in keysents:
    sent = simp(sent)
    doc = nlp(sent)
    labels = [ent.label_ for ent in doc.ents]
    if 'PERSON' in labels:
        t = who(sent)
        if t != None: 
            Q.add((t, sent))
    if 'LOC' in labels or 'GPE' in labels:
        t = where(sent)
        if t != None: 
            Q.add((t, sent))
    if 'DATE' in labels:
        t = when(sent)
        if t != None:
            Q.add((t, sent))
    if sent[-1] == '?':
        Q.add((sent, sent))
    elif 'because' in sent or 'Because' in sent:
        Q.add((why(sent), sent))
    else:
        others = ['PERSON','LOC','GPE','DATE','TIME']
        if(not any(i in labels for i in others)):
            #Q.add((what(sent.text), sent.text))
            t = binary(sent)
            if t!= None:
                Q.add((t, sent))
Q = set(filter(None,Q))
for i in range(int(nquestions)):
    if Q: 
        x = Q.pop()
        print(x[0])
        #print(x[1] + "\n")    #the original sentence for debug info
    else: print("Try fewer questions?") 





