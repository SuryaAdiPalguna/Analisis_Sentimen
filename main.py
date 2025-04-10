import streamlit as st
import pandas as pd
import numpy as np
from modul._2_preprocessing_data import case_folding, normalization, tokenizing, stopword_removal, stemming
from modul._3_feature_extraction import tfidf, punctuation, pos_tagging
from modul._7_modelling import DWKNN

try:
  df = pd.read_parquet('output/4_feature_engineering/train.parquet')
except FileNotFoundError as e:
  df = pd.read_parquet('Hasil_Pembahasan/output/4_feature_engineering/train.parquet')

st.set_page_config(page_title="DISTANCE-WEIGHTED K-NEAREST NEIGHBOR", layout="centered")

# Judul
st.title("ANALISIS SENTIMEN TWITTER MENGENAI PENGARUH TOKOH POLITIK DENGAN METODE DISTANCE-WEIGHTED K-NEAREST NEIGHBOR")

# Input
user_input = st.text_input("Masukkan ulasan:")

# Tombol untuk menampilkan hasil
if st.button("Proses"):
  # Pengumpulan Data
  df_user_input = pd.DataFrame({'full_text': [user_input], 'sentiment': [""]})
  df_user_input['clean_full_text'] = df_user_input['full_text'].apply(lambda data: normalization(text=data, remove_punctuation_number=False))
  df_user_input['word_count'] = df_user_input["clean_full_text"].str.split().str.len()

  # Preprocessing Data
  df_user_input['text_preprocessed'] = df_user_input['full_text'].apply(case_folding)
  df_user_input['text_preprocessed'] = df_user_input['text_preprocessed'].apply(normalization)
  df_user_input['text_preprocessed'] = df_user_input['text_preprocessed'].apply(tokenizing)
  df_user_input['text_preprocessed'] = df_user_input['text_preprocessed'].apply(stopword_removal)
  df_user_input['text_preprocessed'] = df_user_input['text_preprocessed'].apply(stemming)
  df_user_input['text_preprocessed'] = df_user_input['text_preprocessed'].apply(lambda text: ' '.join(text))

  # Feature Extraction
  tfidf_df_user_input = tfidf(df_user_input)
  punctuation_df_user_input = punctuation(df_user_input)
  pos_tagging_df_user_input = pos_tagging(df_user_input)

  # Feature Engineering
  df_user_input = pd.concat([tfidf_df_user_input, punctuation_df_user_input, pos_tagging_df_user_input, df_user_input[['sentiment']]], axis=1)

  # Modelling
  X_train = df.drop(columns=['sentiment'])
  X_test = df_user_input.drop(columns=['sentiment'])
  y_train = df['sentiment']
  y_test = df_user_input['sentiment']

  X_train = X_train[list(set(X_test.columns) & set(X_train.columns))]
  for col in list(set(X_test.columns) - set(X_train.columns)):
    X_train[col] = 0
  X_train = X_train[sorted(X_train.columns)]
  X_test = X_test[sorted(X_test.columns)]

  X_train = np.array(X_train)
  X_test = np.array(X_test)
  y_train = np.array(y_train)
  y_test = np.array(y_test)

  result = pd.DataFrame()
  dwknn = DWKNN(k=50)
  dwknn.fit(X_train, y_train)
  y_predict = dwknn.predict(X_test)
  result['predict'] = pd.Series(y_predict)
  
  # Output
  sentiment = result['predict'][0]
  sentiment_result = "Negatif" if sentiment == -1 else "Positif" if sentiment == 1 else "Netral"

  st.write(f"Ulasan ini memiliki sentimen **{sentiment_result}**")

# Footer
st.sidebar.title("Disusun Oleh:")
st.sidebar.info("I Made Surya Adi Palguna")
st.sidebar.info("2108561067")