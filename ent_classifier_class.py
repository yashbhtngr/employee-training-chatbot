import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from nltk.corpus import wordnet

class ent_classifier:
    
    EMBEDDING_DIM=100
    

    with open('referrence_vec.p', 'rb') as fp:
        referrence_vec = pickle.load(fp)

    custom_stopwords = ["training","trainings","programs","program","and","of","to"]

    with open('word_index.p', 'rb') as fp:
        word_index = pickle.load(fp)

    with open('embeddings_index.p', 'rb') as fp:
        embeddings_index = pickle.load(fp)

    model=keras.models.load_model('entity_recognizer')

    def norm_dot(self,a,b): ## a and b are both numpy arrays
        prod = (np.dot(a,b))/(np.linalg.norm(a)* np.linalg.norm(b))
        return prod 

    def get_max_class(self,word):
        max_value = 0
        category = ""
        if word not in self.word_index:
            return [0,""]
        else:
            word_vec = (self.embeddings_index)[word]
            for training, vec_list in (self.referrence_vec).items():
                all_dots = np.asarray([(self.norm_dot)(word_vec,x) for x in vec_list],dtype="float")
                maximum = np.max(all_dots)
                if maximum>max_value:
                    max_value = maximum
                    category=training
            return [max_value,category]
            
    
    def classify(self,test_sentence):
        sent = test_sentence.lower()
        new_sent=""
        for ch in sent:
            if ch.isalnum()==True:
                new_sent=new_sent+ch
            else:
                new_sent= new_sent+" "+ch+" "
        sent=new_sent
        sent= sent.split()
        sequence=[]
        for x in sent:
            if x in self.word_index:
                sequence.append(self.word_index[x])
            else:
                sequence.append(0)
        test = [sequence]
        predictions = (self.model).predict(test)
        predict=predictions[0]
        #print(predict)
        predicted_sequence=[np.argmax(x) for x in predict]
        #print(predicted_sequence)
        predicted_sequence.append(0)
        target=[]
        temp_s=[]
        for i in range(len(predicted_sequence)):
            if predicted_sequence[i]==1:
                temp_s.append(sent[i])
            else:
                if len(temp_s)!=0:
                    temp_ent = ' '.join(temp_s)
                    target.append(temp_ent)
                    temp_s= []

        final_target = []
        all_programs=[]
        if len(target)!=0:
            for ent in target:
                temp_ent = ent.split()
                processed=[]
                for x in temp_ent:
                    if x.lower() not in self.custom_stopwords:
                        processed.append(x.lower())
                new_ent = ' '.join(processed)
                if new_ent!='':
                    final_target.append(new_ent)
            
            if len(final_target)!=0:
                for ent in final_target:
                    targets = ent.split()
                    max_value=0
                    category=""
                    for word in targets:
                        [value,training]=(self.get_max_class)(word)
                        if value > max_value:
                            max_value = value
                            category = training
                    if max_value > 0.6:
                        all_programs.append(category)
                    
        all_programs = list(set(all_programs))
        return all_programs