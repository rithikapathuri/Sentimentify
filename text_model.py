from transformers import pipeline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import pipeline
import google.generativeai as genai
import os
from dotenv import load_dotenv
from PIL import Image
import json

load_dotenv() 
genai.configure(api_key=os.getenv("GEM_API_KEY"))
model = genai.GenerativeModel(model_name="gemini-1.5-flash")


def emotion_sents(input): 
    classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
    emotion_array = []
    try: 
        emotion_array = classifier(input)
    except: 
        print("Couldn't generate sentiment array.")
        return []
    return emotion_array

def gemini_parse_text(image): 
    image_conv = Image.open(image) 
    return model.generate_content([image_conv, "Parse the conversation from this text message stream, and return only the exact content/words in the image, nothing extra."]).text

def gemini_eval_text(text):
    emotion_array = emotion_sents(text = text)
    prompt = "Using this text input \""+text+"\", I have analyzed several specific emotions present in the input and their respective numerical values and placed them into the given array. Can you please take this text input and also interpret it according to your sentiment analysis methods, and using both the emotions present in the array and your own interpretation, can you give me specific details about the deeper meaning behind the text input in a short blurb. Make sure not to mention the actual input text or array that I am giving you, only give a short description of the hidden meaning of the input text. Also, make sure to consider that the input may include generational slang and lots of sarcasm, so be extra careful to watch out for any signs of sarcasm and classify the input text appropriately. Expect lots of sarcasm in the text message conversations, and even seemingly neutral messages can have an underlying sarcastic/passive aggressive tone, so please be careful."
    response = model.generate_content([str(emotion_array), str(prompt)])
    dict = {"response" : response.text}
    return json.dumps(dict)
    
def gemini_eval_image(image):
    parsed_image = gemini_parse_text(image)
    emotion_array = emotion_sents(parsed_image)
    prompt = "Using this text input \""+parsed_image+"\", I have analyzed several specific emotions present in the input and their respective numerical values and placed them into the given array. Can you please take this text input and also interpret it according to your sentiment analysis methods, and using both the emotions present in the array and your own interpretation, can you give me specific details about the deeper meaning behind the text input in a short blurb. Make sure not to mention the actual input text or array that I am giving you, only give a short description of the hidden meaning of the input text. Also, make sure to consider that the input may include generational slang and lots of sarcasm, so be extra careful to watch out for any signs of sarcasm and classify the input text appropriately. Expect lots of sarcasm in the text message conversations, and even seemingly neutral messages can have an underlying sarcastic/passive aggressive tone, so please be careful."
    response = model.generate_content([str(emotion_array), str(prompt)])
    dict = {"response" : response.text}
    return json.dumps(dict)