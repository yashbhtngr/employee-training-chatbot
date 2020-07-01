import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import csv
import pickle
import os
import re
from ent_classifier_class import *
from new_intent import *

classifier = ent_classifier()

def conv_flow(sentence):

    if sentence!='stop':

        mapped_intent=predictions(sentence)
        print("INTENT: "+mapped_intent)
        

        if mapped_intent == 'query_intent':
            all_programs=classifier.classify(sentence)
            next_res=""
            if len(all_programs)!=0:
                #Database_query
                for x in all_programs:
                    print(x)
                next_res= input("Anything else?\n") #Temporary
            else:
                next_res = input("Sorry, couldn't catch which training program you were referring to. Could you say it again?\n")
            conv_flow(next_res)
        else:
            start_date=input("Please specify the start date in DD-MM-YYYY format.\n")
            # if start date not correct ask again
            end_date=input("Please specify the end date in DD-MM-YYYY format.\n")
            # if end date not correct ask again
            if mapped_intent == "training_all_programs":
                info="queried-info from backend for all programs"
                print(info)
            else:
                if mapped_intent == "training_have_completed":
                    info="queried-info from backend for completed programs"
                    print(info)
                else:
                    info="queried-info from backend for pending programs"
                    print(info)
            next_res =  input("Anything else?\n")
            conv_flow(next_res)
    else:
        print("Okay, Bye!")

def conversation():
    with open('user_data.p', 'rb') as fp:
        user_data = pickle.load(fp)
    #print(user_data)
    serial_code = user_data["serial_code"]
    first_response=""
    if serial_code=="":
        print("------------------------------------")
        user_response = input("Hey there! I am training bot, here to help you! Can I get your serial code please?"+"\n")
        serial_codes_help = re.finditer(r"(\W|^)(\d *){8}(\W|$)", user_response)
        serial_codes=[]
        for i in serial_codes_help:
            serial_codes.append(i.group(0).replace(" ",""))
        got_it = False
        if len(serial_codes)!=0:
            got_it = True
        else:
            got_it = False
        while not got_it:
            user_response = input("Sorry, couldn't catch that. Can you give me your serial code once again?"+"\n")
            serial_codes_help = re.finditer(r"(\W|^)(\d *){8}(\W|$)", user_response)
            serial_codes=[]
            for i in serial_codes_help:
                serial_codes.append(i.group(0).replace(" ",""))
            if len(serial_codes)!=0:
                got_it = True
            else:
                got_it = False

        user_data["serial_code"] = serial_codes[0]
        with open('user_data.p', 'wb') as fp:
            pickle.dump(user_data, fp, protocol=pickle.HIGHEST_PROTOCOL)
        first_response = input("Great! So how can I help you?"+"\n")
    else:
        print("------------------------------------")
        first_response = input("Hey, nice to see you back! So how can I help you?"+"\n")

    conv_flow(first_response)

conversation()