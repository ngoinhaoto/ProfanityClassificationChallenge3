import torch
from transformers import AutoModel, AutoTokenizer,TFBertForSequenceClassification
import tensorflow as tf
import numpy as np
import os
import pandas as pd



phobert_tokenizer = AutoTokenizer.from_pretrained("path-to-save/Tokenizer")
 
phobert_model = TFBertForSequenceClassification.from_pretrained("path-to-save/Model")

label = {
    0: 'Clean',
    1: 'Offensive',
    2: 'Hate'
}


def Get_sentiment(Review, Tokenizer=phobert_tokenizer, Model=phobert_model):
    # Convert Review to a list if it's not already a list
    if not isinstance(Review, list):
        Review = [Review]
 
    Input_ids, Token_type_ids, Attention_mask = Tokenizer.batch_encode_plus(Review,
                                                                             padding=True,
                                                                             truncation=True,
                                                                             max_length=128,
                                                                             return_tensors='tf').values()
    

    prediction = Model.predict([Input_ids, Token_type_ids, Attention_mask-1])
 
    # Use argmax along the appropriate axis to get the predicted labels
    pred_labels = tf.argmax(prediction.logits, axis=1)
 
    # Convert the TensorFlow tensor to a NumPy array and then to a list to get the predicted sentiment labels
    pred_labels = [label[i] for i in pred_labels.numpy().tolist()]
    return pred_labels

