import torch
from transformers import AutoTokenizer,TFBertForSequenceClassification
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import streamlit as st


phobert_tokenizer = AutoTokenizer.from_pretrained("path-to-save/Tokenizer")
 
phobert_model = TFBertForSequenceClassification.from_pretrained("path-to-save/Model")

label = {
    0: 'Bình thường',
    1: 'Phản cảm',
    2: 'Mang tính thù ghét một cá nhân hay tập thể'
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


def main():
    st.title('Website xác nhận là câu có chứa nội dung phản cảm hay không?')
    
    user_input = st.text_area("Điền câu ở đây:", "Ghi thoải mái, tục tĩu cũng được...")

    # Create a placeholder for the spinner
    with st.spinner('Đang xử lý...'):
        # Button to trigger sentiment analysis
        if st.button('Kiểm tra'):
            if user_input != "Ghi thoải mái, tục tĩu cũng được...":
                sentiment_label = Get_sentiment(user_input)
                st.write(f"Câu trên {sentiment_label[0]}!")
                if sentiment_label[0] == 'Bình thường':
                    st.success('Đã xử lý xong! Câu này bình thường.')
                elif sentiment_label[0] == 'Phản cảm':
                    st.warning('Đã xử lý xong! Câu này có chứa nội dung phản cảm.')
                elif sentiment_label[0] == 'Mang tính thù ghét một cá nhân hay tập thể':
                    st.error('Đã xử lý xong! Câu này có nội dung mang tính thù ghét.')

                st.balloons()
                


if __name__ == '__main__':
    main()