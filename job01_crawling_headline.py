from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import datetime

category = ['Politics', 'Economic', 'Social', 'Culture', 'World', 'IT']


url ='https://news.naver.com/section/100'

# resp = requests.get(url)
# print(list(resp))  # 브라우저가 해당하는 주소에 요청을 하고 응답을 한다 응답을 페이지로 응답한다.
#
# soup = BeautifulSoup(resp.text, 'html.parser') #parser가 html형식으로 바꿔준다
# print(soup)
# title_tags = soup.select('.sa_text_strong')
# print(len(title_tags))
# for title_tags in title_tags:
#     print(title_tags.text)

df_titles = pd.DataFrame()

#코드설명 주소별 하나씩 받고 한글만 하고 받아서 카테고리마다 라벨붙이고 df_title에 빈거에 채워넣어서 출력

for i in range(6):
    url = 'https://news.naver.com/section/10{}'.format(i)
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    title_tags = soup.select('.sa_text_strong')
    titles = []
    for title_tags in title_tags:
        title = title_tags.text
        title = re.compile('[^가-힣 ]').sub('',title) #한글만 남게 전처리 [^가-힣 ] 띄어쓰기해야됨  뺴고 널문자로 채워라
        titles.append(title)
    df_section_titles = pd.DataFrame(titles, columns=['titles'])
    df_section_titles['category'] = category[i]
    df_titles = pd.concat([df_titles, df_section_titles], axis='rows', ignore_index=True)
print(df_titles.head())
df_titles.info()
print(df_titles['category'].value_counts())
df_titles.to_csv('./crawling_data/naver_headline_news_{}.csv'.format(
    datetime.datetime.now().strftime('%Y%m%d')), index=False)  #현재시간으로 알려줌 엄청큰값이고 strtime로 우리가쓰는 값으로 알려줌 시간,분,초까지 나올수있음 인데스폴스안하면 0,1,2...붙어서