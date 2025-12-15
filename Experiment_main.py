
import time

import random
import os
import pandas as pd
from generate_QA import generate_qa_with_ollama
import json
from RAG_agent_1 import rag_agent
from py_ecc.bn128 import G1, G2, pairing, add, multiply, eq,neg,curve_order
import random
import functools
from ecpy.curves import Curve
import libnum
import numpy as np 
import timeit
import math

"""
load docs, and sub_docs
"""


documents = []
folder_path_1 = '/Users/subhasisthakur/Desktop/2026/DEBSEC/all files RAG/archive-3/'
fileNames = os.listdir(folder_path_1)
for fileName in fileNames:
    if fileName[0]!= '.':
        folder_path_2 = folder_path_1 + fileName
        fileNames_3 = os.listdir(folder_path_2)
        for fileName_4 in fileNames_3:
            text_file = open(folder_path_2 + '/' +fileName_4,'r')
            documents.append(text_file.read())



sub_docs = random.sample(list(range(len(documents))),70)


"""
load questions
"""

questions = []

for i in range(1):
    random_doc_index = random.sample(sub_docs,1)
    source_text = documents[random_doc_index[0]]
    qa_results = generate_qa_with_ollama(source_text)
    print(qa_results)
    d = json.loads(qa_results)
    for x in d:
        questions.append(x['question'])



"""
load question 
"""

rag_agent_1 = rag_agent()
sin_cos_angle = rag_agent_1.docs_to_random_angle(sub_docs)
query = questions[0]
rag_agent_1.query_doc_poly_3(sin_cos_angle,query)
evals = rag_agent_1.evals
print(evals)
print('Q:', query)
start_time = time.time()
answer = rag_agent_1.generate_response(query)
print("A:", answer)
t = (time.time() - start_time)
print('query time',t)

"""
verify choice
"""
start_time = time.time()

k=4
x_1 = rag_agent_1.query_doc_poly_2(sin_cos_angle,query,k)
passed = 1
combined_poly = rag_agent_1.ret_combined_poly()

for i in range(3):
    ret_poly = combined_poly[i]
    x = x_1[i]
    polycommit = x[0]
    evalcommit = x[1]
    a = x[2]
    crs_g1 = rag_agent_1.ret_crs_g1()
    crs_g2 = rag_agent_1.ret_crs_g2()

    y = ret_poly(a)

    polyB = np.poly1d([1,-a])

    B = add(crs_g1[1],neg( multiply(G1,a)))

    C = add(polycommit,neg( multiply(crs_g2[0],y)))

    if(pairing(evalcommit,B) != pairing(C,G1)):
        passed = -1

print('passed',passed)

t = (time.time() - start_time)
print('choice verification time',t)


"""
verify use
"""
start_time = time.time()

k=4
x_1 = rag_agent_1.query_doc_poly_2(sin_cos_angle,answer,k)
passed = 1
combined_poly = rag_agent_1.ret_combined_poly()

for i in range(3):
    ret_poly = combined_poly[i]
    x = x_1[i]
    polycommit = x[0]
    evalcommit = x[1]
    a = x[2]
    crs_g1 = rag_agent_1.ret_crs_g1()
    crs_g2 = rag_agent_1.ret_crs_g2()

    y = ret_poly(a)

    polyB = np.poly1d([1,-a])

    B = add(crs_g1[1],neg( multiply(G1,a)))

    C = add(polycommit,neg( multiply(crs_g2[0],y)))

    if(pairing(evalcommit,B) != pairing(C,G1)):
        passed = -1

print('passed',passed)

t = (time.time() - start_time)
print('use verification time',t)


