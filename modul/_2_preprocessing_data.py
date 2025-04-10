import pandas as pd
import re
import json
import nltk
from nltk.corpus import stopwords
from mpstemmer import MPStemmer

def cleaning_data(df: pd.DataFrame) -> pd.DataFrame:
  # drop missing value
  df = df.dropna(ignore_index=True)
  # drop duplicated
  df = df.drop_duplicates(subset=['full_text'], ignore_index=True)
  return df

def case_folding(text: str) -> str:
  return text.lower()

def normalization(text: str, remove_punctuation_number: bool = True) -> str:
  # remove urls, hashtags, and mentions
  text = re.sub(r'http[s]?://[\S]+', '', text)
  text = re.sub(r'[\S]+[\.|\s]com', '', text)
  text = re.sub(r'#[\w]+', '', text)
  text = re.sub(r'@[\w]+', '', text)
  text = re.sub(r'RT[\s]+', '', text)
  # remove punctuation
  if remove_punctuation_number:
    text = re.sub(r'\&[\#]?[\w]+\;', '', text)
    text = re.sub(r'[\_|\'|\.|\,]', '', text)
    text = re.sub(r'[\-]', ' ', text)
    text = re.sub(r'[\W]', ' ', text)
  # convert slang words
  normals = dict()
  try: # from dataset
    with open('data/preprocessing/slang_words.json', 'r') as file:
      normals.update(json.load(file))
  except FileNotFoundError as e:
    with open('Hasil_Pembahasan/data/preprocessing/slang_words.json', 'r') as file:
      normals.update(json.load(file))
  try: # from manual
    with open('data/preprocessing/slang_words_manual.json', 'r') as file:
      normals.update(json.load(file))
  except FileNotFoundError as e:
    with open('Hasil_Pembahasan/data/preprocessing/slang_words_manual.json', 'r') as file:
      normals.update(json.load(file))
  text = f' {text} ' # implement
  for i in normals:
    text = text.replace(i, normals[i])
  # remove number
  if remove_punctuation_number:
    text = re.sub(r'\d+', '', text)
  return text

def tokenizing(text: str) -> list:
  return text.split()

def stopword_removal(text: list) -> list:
  nltk.download('stopwords')
  stop_words = set(stopwords.words('indonesian'))
  try:
    with open('data/preprocessing/stopwords.txt', 'r') as file:
      more_stopwords = file.read().split('\n')
  except FileNotFoundError as e:
    with open('Hasil_Pembahasan/data/preprocessing/stopwords.txt', 'r') as file:
      more_stopwords = file.read().split('\n')
  stop_words.update(more_stopwords)
  return [word for word in text if word not in stop_words]

def stemming(text: list) -> list:
  stemmer = MPStemmer()
  return [stemmer.stem_kalimat(word) for word in text]