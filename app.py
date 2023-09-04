# Installing Gradio
!pip install gradio transformers -q

# Import the required Libraries
import gradio as gr
import numpy as np
import pandas as pd
import pickle
import transformers
from transformers import AutoTokenizer 
from transformers import AutoConfig
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import pipeline
from scipy.special import softmax

# Requirements
model_path ="HOLYBOY/Sentiment_Analysis_distilBERT"
tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)

# ---- Function to process the input and return prediction
def sentiment_analysis(text):
    text = preprocess(text)

    encoded_input = tokenizer(text, return_tensors = "pt") # for PyTorch-based models
    output = model(**encoded_input)
    scores_ = output[0][0].detach().numpy()
    scores_ = softmax(scores_)
    
    # Format output dict of scores
    labels = ["Negative", "Neutral", "Positive"]
    scores = {l:float(s) for (l,s) in zip(labels, scores_) }
    
    return scores


# ---- Gradio app interface
app = gr.Interface(fn = sentiment_analysis,
                   inputs = gr.Textbox("Write your text or tweet here or use the example listed below..."),
                   outputs = "label",
                   title = "Public Perceiption of Tweets on COVID-19 Vaccines",
                   description  = "This app analyzes Public Perceiption of text based on tweets about COVID-19 Vaccines using a fine-tuned distilBERT model",
                   interpretation = "default",
                   examples = [["The idea of a vaccine to curb the spread of COVID 19 is good"], 
                               ["I am not going to take this vaccine because it has serious health implication"],
                              ["I dont like the idea of the vaccine"]]
                   )

app.launch()