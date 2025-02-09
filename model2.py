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

URL = "https://www.foxnews.com/media/rwandan-president-praises-unconventional-trump-says-we-might-learn-some-lessons-usaid-shutdown"
r = requests.get(URL)

classifier = pipeline("sentiment-analysis", model="SamLowe/roberta-base-go_emotions", top_k = None)
load_dotenv() 
prompt = """
   "Why Coffee Drinkers Are Clearly Superior to Tea Drinkers"

It is an undeniable fact that coffee drinkers are simply more intelligent, more productive, and more cultured than those who opt for the weak, uninspiring beverage known as tea. While coffee fuels the world's greatest minds—scientists, entrepreneurs, and artists—tea drinkers are often found aimlessly sipping their lukewarm leaf water, contributing little to society.

Studies (which we won’t bother citing) have clearly shown that coffee enhances brain function, while tea only serves to lull people into a false sense of sophistication. Every major innovation of the past century? Powered by coffee. Tea, on the other hand, has been the beverage of indecision and mediocrity, embraced by those who prefer inactivity over action.

Let’s be real—when was the last time you heard someone say, “I need a cup of tea to power through this work”? Never. Because it doesn’t happen. Tea drinkers are just delaying the inevitable realization that coffee is king and the only true beverage of champions.
"""

prompt1 = """
The Impact of Climate Change: An Overview of Scientific Findings
Introduction
Climate change, driven primarily by human activities, is one of the most widely discussed scientific issues of the 21st century. It is characterized by long-term shifts in temperature, weather patterns, and sea levels. While the phenomenon is often debated, a large body of scientific research confirms the significant impact of greenhouse gases, particularly carbon dioxide (CO₂), on global warming.

Scientific Consensus
According to the Intergovernmental Panel on Climate Change (IPCC), over 97% of climate scientists agree that human activities, including burning fossil fuels and deforestation, contribute to the rise in global temperatures. The increased concentration of greenhouse gases in the atmosphere traps heat and leads to changes in weather patterns. This is consistent with data from ice cores, tree rings, and satellite observations.

Evidence of Climate Change

Global Temperature Increases
Since the late 19th century, global temperatures have risen by approximately 1.2°C. The warmest years on record have occurred since the turn of the century, with 2020 being tied with 2016 as the hottest year globally. Temperature records from weather stations, ocean buoys, and satellite data consistently show a rise in global average temperatures.

Melting Ice and Rising Sea Levels
The melting of glaciers and ice sheets, particularly in Greenland and Antarctica, contributes to rising sea levels. Since 1880, sea levels have risen by about 20 cm globally, with an accelerated rate observed since the 1990s. These changes pose risks to coastal communities and ecosystems.

Extreme Weather Events
An increase in the frequency and intensity of extreme weather events, such as hurricanes, droughts, and wildfires, has been linked to climate change. Warmer ocean temperatures are fueling more intense storms, while rising global temperatures are contributing to longer and more severe droughts in certain regions.

Future Projections
The IPCC's 2021 report outlines multiple future scenarios depending on greenhouse gas emission levels. If emissions continue at current rates, global temperatures could rise by 3°C or more by 2100, leading to more extreme weather events and significant disruptions to ecosystems and human societies. However, if global efforts to reduce emissions succeed, the rise in temperatures could be limited to 1.5°C, significantly reducing the risk of severe impacts.

Conclusion
Climate change is a global challenge that requires coordinated action across nations. While uncertainty remains regarding the exact pace and magnitude of its effects, the overwhelming body of evidence points to significant and ongoing changes in the climate. Effective mitigation strategies, such as transitioning to renewable energy, improving energy efficiency, and protecting natural carbon sinks like forests, are essential for addressing these challenges.

This article includes scientific data and projections, provides an overview of findings from various research bodies like the IPCC, and explains the issue with careful consideration of evidence. It's not promoting any particular viewpoint but rather explaining the phenomenon based on available research, which is a hallmark of neutral, unbiased reporting.
"""

prompt3 = """President Donald Trump announced on Truth Social on Thursday that he has uncovered what he says “could be the biggest scandal of them all, perhaps the biggest in history!” He alleged (in capital letters) that “billions of dollars have been stollen [sic] at USAID, and other agencies, much of it going to the fake news media as a ‘payoff’ for creating good stories about the Democrats.” The president went on to claim Politico had received $8 million from the federal government, and questioned if The New York Times and other media outlets had also received payments.

What Trump is describing as scandalous state funding of media outlets is in fact the banal business of government agencies paying for subscriptions from those outlets.

Even for a man with a long record of spreading misinformation and disinformation, this ranks up there as one of his most stupefying conspiracy theories to date.

What Trump is describing as scandalous state funding of media outlets is in fact the banal business of government agencies paying for subscriptions from those outlets. Not only is this not corrupt, it’s a good thing for a functioning democracy that federal workers stay up to date on the news. The administration’s subsequent decision to cancel all subscriptions to media outlets, billed as a way to make the government more “efficient” and less corrupt, distills the militantly know-nothing attitude of the contemporary American right.

Watching the Politico “scandal” unfold online was at turns hilarious and horrifying. Prominent pro-Trump voices shared the “news” that money from USAID and other government agencies was going toward media outlets and appending them with statements like “everything makes sense now” and “If this is not prima facie corruption, what is it?” They apparently believed that they had discovered the smoking gun proving Democrats were covertly “funding” Politico and other media outlets. Trump and X CEO Elon Musk then turbocharged the smear, and White House press secretary Karoline Leavitt said the government had been “subsidizing” Politico “on the American taxpayers’ dime.”"""

EXAMPLE_SCHEME = """
{
  "bias_score": 85, 
  "direct_quotes" : ["1", "2", "3"], 
  "conclusion": "yayay"
}
"""

blob_object = TextBlob(prompt3)

words = blob_object.words

#Divides Word Chunks into 512-character lists
chunks = []
chunk = [] 
for word in blob_object.words: 
    if len(chunk) == 400: 
        chunks.append(chunk)
        chunk = [] 
    else: 
        chunk.append(word)

if len(chunk) > 0: 
    chunks.append(chunk)

    
sentence_chunks = [] 

for chunk in chunks: 
    sentence_chunks.append(' '.join(chunk))
    
emotion_dict = {}


for sentence in sentence_chunks: 
    try: 
        dict1 = classifier(sentence)[0]
        dict1 = {item['label']: item['score'] for item in dict1} #converts list to dictionary
        print(dict1, "\n")
        if len(emotion_dict) == 0: 
            emotion_dict = dict1
            print(emotion_dict, "\n")
        else: 
            print(dict1, "\n")
            for label in dict1: 
                emotion_dict[label] = (dict1[label] + emotion_dict[label])/2
    except: 
        print("Couldn't evaluate model")

emotion_dict['subjectivity'] = TextBlob(prompt3).sentiment.subjectivity

neutrality_score = emotion_dict['neutral']
subjectivity_score = emotion_dict['subjectivity']
ad_score = 0

if emotion_dict['approval'] >= emotion_dict['disapproval']: 
    ad_score = emotion_dict['approval']
else: 
    ad_score = emotion_dict['approval']

score_array = [neutrality_score, subjectivity_score, ad_score]

genai.configure(api_key=os.getenv("GEM_API_KEY"))
model = genai.GenerativeModel(model_name="gemini-1.5-flash")


prompt = f"Using this text input: {prompt3}, I have analyzed several specific values (neutrality, subjectivity, and presence of approval/disapproval; all are on a scale of 0 to 1) present in the input and their respective numerical values and placed them into the given array. Can you please take this text input and also interpret it according to your sentiment analysis methods, and using both the values present in the array and your own interpretation, can you give me a bias score from 0 to 100 where higher score is higher bias. If there is a level of bias over 50, can you also give me direct quotes from the article that indicate bias (at least 3), and give a concluding summary of the potential biases and background of the writer of the article and their potential stance on the issue. Make sure not to mention the actual input text or array that I am giving you. Please put these in a PROPER JSON format, with a bias_score key, direct_quotes key with a list of direct_quotes (if < 50 bias, this list can be empty), and a conclusion key with the concluding summary. Please provide a response in a structured JSON format that matches the following model: {EXAMPLE_SCHEME}"

response = model.generate_content([str(score_array), str(prompt)])



response = (response.text)[(len("json") + 3):-4]
response = response.strip("json")

print(response)

data = json.loads(response)

print(data["bias_score"])
print(data["direct_quotes"][0])
print(data["conclusion"])




'''
emotion_dict['neutral'] = 1 - emotion_dict['neutral'] #invert neutrality -- higher neutrality = lower bias 

bias_score = 0



print("Subjectivity: ", emotion_dict['subjectivity'])

keys = list(emotion_dict.keys())
keys.remove("neutral")
keys.remove("approval")
keys.remove("disapproval")
keys.remove("subjectivity")

labels = ['approval', 'disapproval', 'subjectivity', *keys]
values = [emotion_dict[label] for label in labels]

left_over_values = len(emotion_dict.keys()) - 4

#neutrality_weight = 0.35
subjectivity_weight = 0.75
ad_weight = 0.20


left_over_weight = 1 - ad_weight - subjectivity_weight

if emotion_dict['approval'] >= emotion_dict['disapproval']: 
    values.remove(emotion_dict['disapproval'])
else: 
    values.remove(emotion_dict['approval'])

weights = [ad_weight, subjectivity_weight, *[left_over_weight/left_over_values] * left_over_values]

print("VALUES", values)
print("WEIGHTS", weights)

bias_score = np.average(values, weights = weights)

print("The bias score for this article is: ", bias_score * 100)

                
            
        

    
'''






'''
response = classifier(prompt)

print(type(response))

for r in response[0]: 
    emotion = r.get("label", None)
    score = r.get("score", None)
    print(f"Emotion: {emotion} Score: {score}")
'''