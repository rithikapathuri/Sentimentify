# HackHERS2025

## Inspiration
Sentimentify was inspired by all of our struggles to decipher the media that we are bombarded with everyday. Sometimes you might read news articles that have persuasive agendas instead of factual reporting. Other times you may have to decipher the underlying sentiments of a text message to better understand what someone is really trying to say. Sentimentify is a solution to these everyday problems that are common to all of us. 

## What it does
Sentimentify has 2 main features: analyzing articles and analyzing text messages. 

### Articles
Using a distilled BART model, TextBlob, and Gemini, our app can evaluate the sentiments and biases within articles by generating an array of scores in terms of neutrality, subjectivity, and presence of approval/disapproval. These 3 relevant output metrics are important when determining the presence of opinions within an article. These are also used to generate an overall bias score.

### Texts
Also using distilled BART model and Gemini, our app analzyes text screenshots and directly inputed text messages, generating sentiment metrics and a conclusive statement about the text message in consideration. 

## How we built it
We built Sentimentify using many different python libraries such as TextBlob, a sentiment analyzer model from HuggingFace's Open Source BART models, and google Gemini's extensive capabilities for OCR and natural language processing.  

## Challenges we ran into
Initially, we wanted to train and fine-tune our own model for the purposes of bias-analysis, but we quickly realized that there weren't many good datasets available for this case, and considering the time constraints, this wouldn't be possible. To address this, we looked into pre-set open-source models that conducted sentiment analyses and found a model that worked for us. 

Other challenges included making our frontend dynamic so it could support a variety of different sizes of text using flex box. 

## Accomplishments that we're proud of
We're proud that we were able to make the webapp fully functional and support both text messages and articles. We're also proud of the dynamicness of our webapp and the ease of use with the UI. 

## What we learned
We learned many different skills including using libraries for complex text-parsing in python, supporting image uploads via HTML, and making our webapp asynchronously dynamic using JavaScript. We also learned how to run and use an AI model, as well as gained some background on what would be required to train our own model. 

## What's next for Sentimentify 
