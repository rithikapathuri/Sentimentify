from transformers import pipeline
import torch 
import requests
from textblob import TextBlob
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json

CHUNK_SIZE = 400
JSON_SCHEMA = """
{
  "bias_score": 85, 
  "direct_quotes" : ["1", "2", "3"], 
  "conclusion": "yayay"
}
"""
load_dotenv() 
classifier = pipeline("sentiment-analysis", model="SamLowe/roberta-base-go_emotions", top_k = None)

def chunkify(article_text): 
    blob_object = TextBlob(article_text)
    #Divides Word Chunks into 512-character lists
    chunks = []
    chunk = [] 
    for word in blob_object.words: 
        if len(chunk) == CHUNK_SIZE: 
            chunks.append(chunk)
            chunk = [] 
        else: 
            chunk.append(word)
    if len(chunk) > 0: 
        chunks.append(chunk)
    sentence_chunks = [] 
    for chunk in chunks: 
        sentence_chunks.append(' '.join(chunk))
    return sentence_chunks
    
def gemini_eval (article_text, score_array): 
    genai.configure(api_key=os.getenv("GEM_API_KEY"))
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")


    prompt = f"Using this text input: {article_text}, I have analyzed several specific values (neutrality, subjectivity, and presence of approval/disapproval; all are on a scale of 0 to 1) present in the input and their respective numerical values and placed them into the given array. Can you please take this text input and also interpret it according to your sentiment analysis methods, and using both the values present in the array and your own interpretation, can you give me a bias score from 0 to 100 where higher score is higher bias. If there is a level of bias over 50, can you also give me direct quotes from the article that indicate bias (at least 3), and give a concluding summary of the potential biases and background of the writer of the article and their potential stance on the issue. Make sure not to mention the actual input text or array that I am giving you. Please put these in a PROPER JSON format, with a bias_score key, direct_quotes key with a list of direct_quotes (if < 50 bias, this list can be empty), and a conclusion key with the concluding summary. Please provide a response in a structured JSON format that matches the following model: {JSON_SCHEMA}"

    response = model.generate_content([str(score_array), str(prompt)])

    response = (response.text)[(len("json") + 3):-4]
    response = response.strip("json")
    
    return response

    
def gen_sentiments(article_text): 
    emotion_dict = {}
    sentence_chunks = chunkify(article_text)
    for sentence in sentence_chunks:
        try: 
            dict1 = classifier(sentence)[0]
            dict1 = {item['label']: item['score'] for item in dict1} #converts list to dictionary
            if len(emotion_dict) == 0: 
                emotion_dict = dict1
            else: 
                for label in dict1: 
                    emotion_dict[label] = (dict1[label] + emotion_dict[label])/2
        except: 
            print("Couldn't evaluate model")
    
    emotion_dict['subjectivity'] = TextBlob(article_text).sentiment.subjectivity
    neutrality_score = emotion_dict['neutral']
    subjectivity_score = emotion_dict['subjectivity']
    ad_score = 0
    
    if emotion_dict['approval'] >= emotion_dict['disapproval']: 
        ad_score = emotion_dict['approval']
    else: 
        ad_score = emotion_dict['approval']

    score_array = [neutrality_score, subjectivity_score, ad_score]
    return gemini_eval(article_text, score_array)
    
    
