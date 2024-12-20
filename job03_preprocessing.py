import pickle
import pandas as pd
import numpy as np
from keras.src.legacy.preprocessing.text import one_hot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from konlpy.tag import Okt, Kkma



df = pd.read_csv('./crawling_data_2/naver_news_titles_20241220_0925.csv')
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.head())
df.info()
print(df.category.value_counts())

X = df['title']  #안돼면s붙여
Y = df['category']

print(X[0])
okt = Okt()
okt_x = okt.morphs(X[0], stem=True)
print('Okt :', okt_x)

encoder = LabelEncoder()
labeled_y = encoder.fit_transform(Y)  #트랜스폼한 이후에 엔코더를 저장해놔야됨
print(labeled_y[:3])

label = encoder.classes_
print(label)

with open('./models/encoder.pickle', 'wb') as f:
    pickle.dump(encoder, f)

onehot_Y = to_categorical(labeled_y)
print(onehot_Y)

for i in range(len(X)):
    X[i] = okt.morphs(X[i], stem=True)
print(X)
