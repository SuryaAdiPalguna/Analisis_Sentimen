import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from nltk.tag import CRFTagger
from collections import Counter

def tfidf(df: pd.DataFrame) -> pd.DataFrame:
  vectorizer = CountVectorizer()
  X = vectorizer.fit_transform(df['text_preprocessed'])
  features = vectorizer.get_feature_names_out()
  tftd = csr_matrix(X).toarray()
  tf = np.where(tftd > 0, 1 + np.log(tftd), 0)
  N = len(df['full_text'])
  dft = np.array(tftd > 0).sum(axis=0)
  idf = np.log(N / dft)
  tfidf = tf * idf
  tfidf_df = pd.DataFrame(tfidf, columns=features, index=df.index)
  return tfidf_df

def punctuation(df: pd.DataFrame) -> pd.DataFrame:
  punctuation_df = pd.DataFrame()
  punctuation_df['_length'] = df['clean_full_text'].apply(lambda text: len(str(text).split()))
  punctuation_df['_exclamation'] = df['clean_full_text'].apply(lambda text: str(text).count('!'))
  punctuation_df['_question'] = df['clean_full_text'].apply(lambda text: str(text).count('?'))
  punctuation_df['_quote'] = df['clean_full_text'].apply(lambda text: str(text).count('\'') + str(text).count('\"'))
  punctuation_df['_capital'] = df['clean_full_text'].apply(lambda text: len(re.findall(r'\b[A-Z]+\b', (re.sub(r'[^\w\s]', '', text)))))
  return punctuation_df

def pos_tagging(df: pd.DataFrame) -> pd.DataFrame:
  ct = CRFTagger()
  try:
    ct.set_model_file('data/feature/all_indo_man_tag_corpus_model.crf.tagger')
  except FileNotFoundError:
    ct.set_model_file('Hasil_Pembahasan/data/feature/all_indo_man_tag_corpus_model.crf.tagger')
  df['pos_tagging'] = df['clean_full_text'].apply(lambda text: ct.tag(str(text).split()))
  df['pos_counts'] = df['pos_tagging'].apply(lambda row: Counter([tag for _, tag in row]))
  pos_tagging_df = pd.json_normalize(df['pos_counts']).fillna(0).astype(int).reindex(columns=["NN", "JJ", "VB", "RB", "UH"], fill_value=0)
  for col in ["NN", "JJ", "VB", "RB", "UH"]:
    pos_tagging_df[f'{col}_percentage'] = pos_tagging_df[col] / df['word_count']
  return pos_tagging_df