from transformers import pipeline
import torch 

classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")
print(classifier("Jeez. You're a real piece of work."))
#"Wow. I can't believe you're actually good at something.