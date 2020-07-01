import pickle
import numpy as np

referrence_vec = {} ## list of numpy average vectors to compare with

synonym_dict={}
names=["safety","products and services","technical skills","soft skills","orientation","onboarding"]
synonym_dict["safety"] = ["safety", "wellbeing","welfare","security"]
synonym_dict["products and services"] = ["products","services","resource","utility","system","amenity"]
synonym_dict["technical skills"] = ["technical","technological","applied"]
synonym_dict["soft skills"] = ["soft","communication","interpersonal"]
synonym_dict["orientation"]= ["orientation","accommodation","familiarization","acclimatization","introduction","initiation"]
synonym_dict["onboarding"]= ["induction","integration","onboarding"]


EMBEDDING_DIM=100

with open('embeddings_index.p', 'rb') as fp:
   embeddings_index = pickle.load(fp)

for name in names:
   synonyms = synonym_dict[name]
   vec_list = [embeddings_index[x] for x in synonyms]
   referrence_vec[name] = vec_list

with open('referrence_vec.p', 'wb') as fp:
    pickle.dump(referrence_vec, fp, protocol=pickle.HIGHEST_PROTOCOL)

        
    
