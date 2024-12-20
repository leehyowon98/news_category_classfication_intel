import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from konlpy.tag import Okt  # 한국어 형태소 분석기 (Okt 사용)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. 데이터 불러오기
df = pd.read_csv('./crawling_data_2/naver_news_titles_20241220_0925.csv')  # 크롤링한 뉴스 데이터 읽기
df.drop_duplicates(inplace=True)  # 중복 데이터 제거
df.reset_index(drop=True, inplace=True)  # 인덱스 초기화
print(df.head())  # 데이터 샘플 출력
df.info()  # 데이터 요약 정보 출력
print(df.category.value_counts())  # 카테고리별 데이터 수 확인

# 2. 데이터 열 분리
X = df['title']  # 뉴스 제목 데이터
Y = df['category']  # 뉴스 카테고리 데이터

# 3. 형태소 분석 예제 (첫 번째 제목만 처리)
print(X[0])  # 첫 번째 제목 출력
okt = Okt()
okt_x = okt.morphs(X[0], stem=True)  # 형태소 분석 및 어간 추출
print('Okt :', okt_x)

# 4. 카테고리 데이터 레이블 인코딩
encoder = LabelEncoder()
labeled_y = encoder.fit_transform(Y)  # 문자열 형태의 카테고리를 숫자로 변환
print(labeled_y[:3])  # 인코딩된 레이블 데이터 일부 출력

label = encoder.classes_  # 레이블 클래스 확인
print(label)

# 5. 레이블 인코더 저장
with open('./models/encoder.pickle', 'wb') as f:
    pickle.dump(encoder, f)  # 레이블 인코더를 파일로 저장

# 6. 원-핫 인코딩
onehot_Y = to_categorical(labeled_y)  # 숫자형 레이블 데이터를 원-핫 인코딩 형태로 변환
print(onehot_Y)

# 7. 모든 뉴스 제목을 형태소 단위로 분해
for i in range(len(X)):
    X[i] = okt.morphs(X[i], stem=True)  # 형태소 분석 및 어간 추출
print(X)

# 8. 불용어(Stopwords) 처리
stopwords = pd.read_csv('./crawling_data/stopwords.csv', index_col=0)  # 불용어 파일 읽기
print(stopwords)

# 불용어 제거 및 한 글자 단어 제거
for sentence in range(len(X)):  # 각 문장에 대해 반복
    words = []
    for word in range(len(X[sentence])):  # 각 단어에 대해 반복
        if len(X[sentence][word]) > 1:  # 두 글자 이상인 단어만 선택
            if X[sentence][word] not in list(stopwords['stopword']):  # 불용어 목록에 없을 경우
                words.append(X[sentence][word])  # 단어 추가
    X[sentence] = ' '.join(words)  # 단어를 공백으로 연결하여 문장으로 변환

print(X[:5])  # 전처리된 데이터 일부 출력

# 9. 토큰화(Tokenization)
token = Tokenizer()  # 토크나이저 생성
token.fit_on_texts(X)  # 텍스트 데이터를 토크나이저에 학습
tokened_X = token.texts_to_sequences(X)  # 텍스트를 시퀀스 데이터로 변환
wordsize = len(token.word_index) + 1  # 단어 집합 크기 계산
print(wordsize)  # 단어 집합 크기 출력

print(tokened_X[:5])  # 변환된 시퀀스 데이터 일부 출력

# 10. 가장 긴 문장의 길이 찾기
max = 0  # 최대 길이를 0으로 초기화
for i in range(len(tokened_X)):  # 모든 시퀀스를 확인
    if max < len(tokened_X[i]):  # 현재 문장의 길이가 기존 최대 길이보다 길다면
        max = len(tokened_X[i])  # 최대 길이를 업데이트
print(max)  # 가장 긴 문장의 길이 출력

# 11. 시퀀스 데이터 패딩
X_pad = pad_sequences(tokened_X, max)  # 패딩 함수로 길이를 동일하게 맞춤 (짧은 문장은 0으로 채움)
print(X_pad)  # 패딩된 데이터 출력
print(len(X_pad[0]))  # 각 문장의 길이 확인 (모두 동일)

# 12. 학습 데이터와 테스트 데이터 분리
X_train, X_test, Y_train, Y_test = train_test_split(X_pad, onehot_Y, test_size=0.1)  # 데이터 분리
print(X_train.shape, Y_train.shape)  # 학습 데이터 크기 출력
print(X_test.shape, Y_test.shape)  # 테스트 데이터 크기 출력

# 13. 데이터 저장
np.save('./crawling_data/news_data_X_train_max_{}_wordsize_{}'.format(max, wordsize), X_train)  # 학습용 X 데이터 저장
np.save('./crawling_data/news_data_Y_train_max_{}_wordsize_{}'.format(max, wordsize), Y_train)  # 학습용 Y 데이터 저장
np.save('./crawling_data/news_data_X_test_max_{}_wordsize_{}'.format(max, wordsize), X_test)  # 테스트용 X 데이터 저장
np.save('./crawling_data/news_data_Y_test_max_{}_wordsize_{}'.format(max, wordsize), Y_test)  # 테스트용 Y 데이터 저장
