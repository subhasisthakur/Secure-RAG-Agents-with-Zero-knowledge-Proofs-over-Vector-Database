import ollama
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import random
import string
import ollama
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import random
import string
from py_ecc.bn128 import G1, G2, pairing, add, multiply, eq,neg,curve_order
import random
import functools
from ecpy.curves import Curve
import libnum
import numpy as np 
import timeit
import math




class rag_agent:
    random_string = ''
    documents = []
    doc_to_rand_angle = []
    #ret_poly = np.poly1d([1,1])
    ret_comb_poly = []    
    ret_comb_poly_usage = []    

    ret_poly = []    
    ret_poly_eval = []
    polys = []
    evals = []
    sub_doc_ids = []

    def __init__(self) -> None:
        self.random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=20))
        folder_path_1 = '/Users/subhasisthakur/Desktop/2026/DEBSEC/all files RAG/archive-3/'
        fileNames = os.listdir(folder_path_1)
        for fileName in fileNames:
            if fileName[0]!= '.':
                folder_path_2 = folder_path_1 + fileName
                fileNames_3 = os.listdir(folder_path_2)
                for fileName_4 in fileNames_3:
                    text_file = open(folder_path_2 + '/' +fileName_4,'r')
                    self.documents.append(text_file.read())


    def get_crs_G2(self,tau):
        crs = []
        for i in range(30):
            x = (tau**(i))    
            e_point = multiply(G2,x)
            crs.append(e_point)
        return(crs)


    def get_crs_G1(self,tau):
        crs = []
        for i in range(30):
            x = (tau**(i))    
            e_point = multiply(G1,x)
            crs.append(e_point)
        return(crs)


    def poly_commit_G1(self,poly,crs):
        coef = list(poly.coef)
        coef.reverse()
        commit = multiply(G1,int(coef[0]) % curve_order)
        for i in range(len(coef)):
            if i>0:
                commit = add(commit,multiply(crs[i],int(coef[i]) % curve_order))
        return(commit)


    def poly_commit_G2(self,poly,crs):
        coef_1 = list(poly.coef)
        coef = []
        for i in range(len(coef_1)):
            coef.append(int(coef_1[i]))
        coef.reverse()
        commit = multiply(G2,int(coef[0]) % curve_order)
        #print('coef',coef)
        for i in range(len(coef)):
            if i>0:
                commit = add(commit,multiply(crs[i],int(coef[i]) % curve_order))
        return(commit)



    def eval_proof_g2(self,poly_1,a,crs_g2):
        #print('poly_1')
        #print(poly_1)
        y = poly_1(a)
        #print('y',y)
        #print('a',a)
        poly_2 = poly_1 - y
        poly_3 = np.poly1d([1,-a])
        poly_41 = np.polydiv(poly_2,poly_3)
        poly_4 = poly_41[0]
        eval_commit = self.poly_commit_G2(poly_4,crs_g2)
        return(eval_commit)


    def eval_proof_g1(self,poly_1,a,crs_g1):
        y = poly_1(a)
        poly_2 = poly_1 - y
        poly_3 = np.poly1d([1,-a])
        poly_41 = np.polydiv(poly_2,poly_3)
        poly_4 = poly_41[0]
        eval_commit = self.poly_commit_G1(poly_4,crs_g1)
        return(eval_commit)

    def ret_crs_g1(self):
        tau = 12
        crs_g1 = self.get_crs_G1(tau)
        return crs_g1

    def ret_crs_g2(self):
        tau = 12
        crs_g2 = self.get_crs_G2(tau)
        return crs_g2

    def execute_zkp_1(self,poly_1,a):

        tau = 12
        crs_g2 = self.get_crs_G2(tau)
        crs_g1 = self.get_crs_G1(tau)
        #a = .2
        y = poly_1(a)

        polycommit = self.poly_commit_G2(poly_1,crs_g2)
        evalcommit = self.eval_proof_g2(poly_1,a,crs_g2)

        polyB = np.poly1d([1,-a])

        B = add(crs_g1[1],neg( multiply(G1,a)))

        #polyBcommit = poly_commit_G1(polyB,crs_g1)

        C = add(polycommit,neg( multiply(crs_g2[0],y)))

        ret = 'xxx'
        if(pairing(evalcommit,B) == pairing(C,G1)):
            ret = 'aaa'
        else:
            ret = 'bbb'
        return(ret)


    def execute_zkp_2(self,poly_1,a):

        tau = 12
        crs_g2 = self.get_crs_G2(tau)
        crs_g1 = self.get_crs_G1(tau)
        #a = .2
        y = poly_1(a)

        polycommit = self.poly_commit_G2(poly_1,crs_g2)
        evalcommit = self.eval_proof_g2(poly_1,a,crs_g2)

        return([polycommit,evalcommit])





    def execute_zkp(self,poly_1,y,a):

        tau = 12
        crs_g2 = self.get_crs_G2(tau)
        crs_g1 = self.get_crs_G1(tau)
        
        #a = 2
        #y = poly_1(a)
        print('aa')
        print('a',a)
        print(poly_1)

        a = .2
        y = poly_1(a)

        polycommit = self.poly_commit_G2(poly_1,crs_g2)
        evalcommit = self.eval_proof_g2(poly_1,a,crs_g2)

        polyB = np.poly1d([1,-a])

        B = add(crs_g1[1],neg( multiply(G1,a)))

        #polyBcommit = poly_commit_G1(polyB,crs_g1)

        C = add(polycommit,neg( multiply(crs_g2[0],y)))

        ret = 'xxx'
        if(pairing(evalcommit,B) == pairing(C,G1)):
            ret = 'aaa'
        else:
            ret = 'bbb'
        return(ret)



    def docs_to_random_angle(self,doc_ids):
        self.sub_doc_ids = doc_ids
        print(len(self.documents))
        text1 = ollama.embeddings(model="llama3.2", prompt=self.random_string)
        vec1 = np.array(text1['embedding']).reshape(1, -1)
        doc_angle_sin = []
        doc_angle_cos = []
        sin_cos_data = []
        for i in range(len(doc_ids)):
            print(i)
            text2 = ollama.embeddings(model="llama3.2", prompt=self.documents[doc_ids[i]])
            vec2 = np.array(text2['embedding']).reshape(1, -1)
            similarity = cosine_similarity(vec1, vec2)
            self.doc_to_rand_angle.append(similarity[0])
            doc_angle_sin.append(math.sin(similarity))
            doc_angle_cos.append(math.cos(similarity))
        sin_cos_data.append(doc_angle_sin)
        sin_cos_data.append(doc_angle_cos)        
        return(sin_cos_data)
    
    def query_to_random_angle(self,query):
        text1 = ollama.embeddings(model="llama3.2", prompt=self.random_string)
        vec1 = np.array(text1['embedding']).reshape(1, -1)
        text2 = ollama.embeddings(model="llama3.2", prompt= query)
        vec2 = np.array(text2['embedding']).reshape(1, -1)
        similarity = cosine_similarity(vec1, vec2)
        return similarity
 

    def query_doc_poly_3(self,sin_cos_data,query):
        query_angle = self.query_to_random_angle(query)
        x= round(10*query_angle[0][0])
        for i in range(len(sin_cos_data[0])):
            a_1 = round(sin_cos_data[0][i]*100)
            a_2 = round(sin_cos_data[1][i]*100)
            poly_1 = np.poly1d([-round(a_2/6),-round(a_1/2),a_2,a_1])
            self.polys.append(poly_1)
            self.evals.append(int(poly_1(x)))

    def generate_response(self,query):
        min_angle = np.min(self.evals)
        doc_id_1 = self.evals.index(min_angle)
        doc_id = self.sub_doc_ids[doc_id_1]
        doc = self.documents[doc_id]
        prompt = f"""
            Answer the query solely on the following context,
            Query: \"{query}\"
            Context: \"{doc}\"
            """

        response = ollama.generate(model="llama3.2", prompt=prompt, stream=False)
        # The 'response' contains a 'response' field with the generated text
        generated_text = response['response']
        return generated_text




    def query_doc_poly_2(self,sin_cos_data,query,k):
        query_angle = self.query_to_random_angle(query)

        a_1 = round(sin_cos_data[0][0]*100)
        a_2 = round(sin_cos_data[1][0]*100)
        a_3 = round(10*query_angle[0][0])
        
        poly_1 = np.poly1d([-round(a_2/6),-round(a_1/2),a_2,a_1])
        for i in range(1,len(sin_cos_data[0])):
                if i> 0:
                    a_1 = round(sin_cos_data[0][i]*100)
                    a_2 = round(sin_cos_data[1][i]*100)
                    poly_2 = np.poly1d([-round(a_2/6),-round(a_1/2),a_2,a_1])
                    poly_1 = np.polyadd(poly_1,poly_2)
                    

        combine_polys_0 = poly_1
        self.ret_comb_poly.append(combine_polys_0)

        a_1 = round(sin_cos_data[0][0]*100)
        a_2 = round(sin_cos_data[1][0]*100)
        a_3 = round(10*query_angle[0][0])
        
        poly_1 = np.poly1d([-round(a_2/6),-round(a_1/2),a_2,a_1])
        
        if k > 2:
            for i in range(1,(k-1)):
                if i> 0:
                    a_1 = round(sin_cos_data[0][i]*100)
                    a_2 = round(sin_cos_data[1][i]*100)
                    poly_2 = np.poly1d([-round(a_2/6),-round(a_1/2),a_2,a_1])
                    poly_1 = np.polyadd(poly_1,poly_2)
                    
        combine_polys_1 = poly_1
        self.ret_comb_poly.append(combine_polys_1)

        a_1 = round(sin_cos_data[0][k]*100)
        a_2 = round(sin_cos_data[1][k]*100)
        a_3 = round(10*query_angle[0][0])
        
        poly_1_1 = np.poly1d([-round(a_2/6),-round(a_1/2),a_2,a_1])
        
        for i in range(k+1,len(sin_cos_data[0])):
            if i> 0:
                a_1 = round(sin_cos_data[0][i]*100)
                a_2 = round(sin_cos_data[1][i]*100)
                poly_2 = np.poly1d([-round(a_2/6),-round(a_1/2),a_2,a_1])
                poly_1_1 = np.polyadd(poly_1_1,poly_2)
                
        combine_polys_2 = poly_1
        self.ret_comb_poly.append(combine_polys_2)

        #print(poly_1)
        #print('y')
        #print(poly_1(query_angle))
        #print(y)
        #return self.execute_zkp(poly_1,y,query_angle[0][0])
        #return self.execute_zkp(poly_1,y,.2)
        ret_proofs = []
        x = self.execute_zkp_2(combine_polys_0,round(10*query_angle[0][0]))
        x.append(round(10*query_angle[0][0]))
        ret_proofs.append(x)

        x = self.execute_zkp_2(combine_polys_1,round(10*query_angle[0][0]))
        x.append(round(10*query_angle[0][0]))
        ret_proofs.append(x)
        
        x = self.execute_zkp_2(combine_polys_2,round(10*query_angle[0][0]))
        x.append(round(10*query_angle[0][0]))
        ret_proofs.append(x)
        
        return ret_proofs



    def ret_combined_poly(self):
        return self.ret_comb_poly

    def ret_poly(self):
        return self.ret_poly

    def ret_poly_eval(self):
        return self.ret_poly_eval





