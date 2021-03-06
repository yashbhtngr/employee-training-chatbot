# employee-training-chatbot
to make maintaining records of training programs which an employee has done or has yet to do, more convenient.

Often employees need to complete several training programs in order to brush up their skills so that all employees are in an equal footing. Training programs can involve soft skills training, hard skills training or orientation as well which introduces new hires to the company ethics and culture. Our main objective here is to build a bot that will be personalised and can provide information to the user regarding the training programs the person has completed or has yet to complete, along with answering queries related to the training programs. To answer queries related to the training programs we need to get a database of information regarding these programs, as training programs can vary from one company to another. To understand the scope of the bot first we would like to present a possible conversation between the user and the bot.

The details of the approach are as below,

Creating the Relevant Intents

From the user's side, he or she can ask questions regarding the training programs which he/she has completed or has yet to complete, basically information regarding training programs that concern that person, or he/she can query about specific training programs, like a query about safety training. So our first step is to divide these two kinds of questions into two categories namely query_intent and info_intent. The query intent will capture questions that pose a query regarding some training program and the info_intent will capture questions which ask personal progress in completion of training programs. The questions can be categorised into these two categories using a classification model that will be using a RNN-LSTM sequential architecture. Having categorised the question we move onto the next step.

Entity Recognition

After classifying the question into one of query_intent or info_intent, we need to identify some entities in the question. If the question falls in query_intent then it needs to also tell the chatbot which training program is the user asking about. Therefore names of training programs are the relevant entities here. Names of training programs will be part of the data given from the company's side. We'll have to extract the name from the user's query. In case of info_intent, the questions will contain entities which will help us identify as to which training programs the user wants to see. Is it the ones he has completed, or the ones which are pending or the ones which will occur in between two specific dates. Accordingly we will get the relevant information from the database and present it to the user. 

