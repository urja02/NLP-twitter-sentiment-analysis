# NLP-twitter-sentiment-analysis
import re
import numpy as np
import pandas as pd
import string

#plotting
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

#nltk
from nltk.stem import WordNetLemmatizer

#sklearn
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
import sklearn.metrics as metrics
#importing the dataset
dataset_col=['target','ids','date','flag','user','text']
dataset_encoding="ISO-8859-1"
df=pd.read_csv('twitter_data.csv',encoding=dataset_encoding,names=dataset_col)
print(df.sample(5))

#print(df['ids'].sample(n=3))
#print(df.isnull().sum())
#mathematical operation on boolean pandas converts true to 1 and false to 0
#sums across the rows which means down the column rowwise
#print(df.sum())
#print(df['target'].nunique())
#print(df['target'].unique())

#data visualization for target variables
labels=['Negative','Positive']
fig,ax=plt.subplots()
#ax=df.groupby('target').count()
df.groupby('target').count().plot(kind='bar',ax=ax,title='Distribution of data',legend=False)
#ax.plot(kind='bar',title='Distribution of data',legend=False)

#ax=plt.subplot()
ax.set_xticklabels(labels,rotation=0)
plt.xlabel('Target')

#plt.show()
text=list(df['text'])
sentiment=list(df['target'])

import seaborn as sns
sns.countplot(x='target',data=df)
#plt.show()

data=df.loc[:,['text','target']]
#replace positive sentiment 4 to 1
#print(data.head())

data['target']=data['target'].replace(4,1)
print(data['target'].unique())


data_pos=data.loc[data['target']==1,:]


data_neg=data[data['target']==0]


#data_pos=data_pos.iloc[0:20000,:]
#print(data_pos.head())
#data_neg=data_neg[:int(20000)]
# print(data_neg.head())

dataset=pd.concat([data_pos,data_neg])
print(dataset.shape)
print(dataset.tail())

dataset['text']=dataset['text'].str.lower()
print("lowered")
print(dataset['text'].tail())
print("whole")
print(dataset.head(40000))


import nltk

from nltk.corpus import stopwords

stop_words=set(stopwords.words('english'))
not_stopwords={'not'}
final_stopword=stop_words-not_stopwords
stop_words1=sorted(final_stopword)
# print(stop_words1)

# dataset['text_length']=dataset['text'].str.len()
# print("yo")
# print(dataset.head())
# for t in dataset.text:
#     s=len(t)


def clean_stopwords(text):
    filter=[]
    for w in text.split():
        if w not in stop_words1:
            filter.append(w)
    return ' '.join(filter)

#def remove_stopwords(text):
#    return ' '.join(word for word in text.split() if word not in stop_words)

dataset['text'] = dataset['text'].apply(clean_stopwords)

print(dataset.head())
# for i in range(len(dataset)):
#     if(i==0):
#         dataset['text_rep']=cleaning(df.loc[i,"text"])

#dataset['stopword_removed']=dataset['text'].apply(cleaning)
#print(dataset.head())

def clean_usernames(data):
    text = re.sub('@[\S]+','',data)
    return text
dataset['text'] = dataset['text'].apply(clean_usernames)

print("cleaning")
english_punctuations=string.punctuation
punctuation_list=english_punctuations
def clean_punctuations(text):
    translator=str.maketrans('','',punctuation_list)
    return text.translate(translator)
dataset['text']=dataset['text'].apply(clean_punctuations)
#print(dataset.tail())
def clean_repeating_char(text):
    repeat_pattern=re.compile(r'(\w)\1+')
    word=repeat_pattern.sub(r'\1\1',text)
    return word
dataset['text']=dataset['text'].apply(clean_repeating_char)
#print(dataset.tail())

def clean_url(data):
    urls=re.sub(r'(www.[^\s]+)|(https?://[^\s]+)',' ',data)
    return urls
dataset['text']=dataset['text'].apply(clean_url)
#print(dataset.tail())

def clean_numbers(data):
    number=re.sub(r'[0-9]+','',data)
    return number
dataset['text']=dataset['text'].apply(clean_numbers)
print(dataset.head())


from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
#tokenizer=word_tokenize()
#tokenizer=RegexpTokenizer(r' \w+ ')
#dataset['tokens']=dataset['text'].apply(tokenizer.tokenize)
dataset['tokens']=dataset['text'].apply(word_tokenize)
print(dataset.head())

st=nltk.PorterStemmer()
def stemming_on_text(data):
    text=[]
    for word in data:
        text.append(st.stem(word))
    return text
dataset['tokens']=dataset['tokens'].apply(stemming_on_text)
print(dataset.head())

print("lemmatized")
lm=nltk.WordNetLemmatizer()
def lemmatizer_on_text(data):
     text=[lm.lemmatize(word) for word in data]
     return text
dataset['tokens']=dataset['tokens'].apply(lemmatizer_on_text)
print(dataset.head())



X=dataset['tokens']
y=dataset.target
print(X.head())
print(dataset.tail())
#plotting cloud of words for negative
print("negative and positive")
data_neg1 = dataset['tokens'][20000:]
print(data_neg1)
data_pos1=dataset['tokens'][:20000]
print(data_pos1)



#X1=pd.DataFrame(response.toarray())

#print(X1.head(5))

# plt.figure(figsize=(20,20))
# # allwords=''.join(map(str,twts) for twts in data_neg1)
# wc=WordCloud(max_words=1000,width=1600,height=800,collocations=False).generate(data_neg.to_string())
# plt.imshow(wc,interpolation='bilinear')
# plt.show()
#
# plt.figure(figsize=(20,20))
# allwords=''.join(map(str,twts) for twts in data_neg1)
# wc=WordCloud(max_words=1000,width=1600,height=800,collocations=False).generate(data_pos.to_string())
# plt.imshow(wc,interpolation='bilinear')
# plt.show()

#training and testing the data
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.05,)
print(x_train.head())
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect=TfidfVectorizer(analyzer = lambda x : x,ngram_range=(1,2),max_features=500000)
X1=tfidf_vect.fit_transform(x_train)
X_test = tfidf_vect.transform(x_test)

# print("length:",len(tfidf_vect.get_feature_names_out()))
# print(response.shape)


#vectoriser=TfidfVectorizer(analyzer=)
# response=tfidf_vect.fit(cord)
# print(response.vocabulary_)
# print(tfidf_vect.get_feature_names())
# response=tfidf_vect.transform(cord)
# print(response.shape)
# print(response.toarray())
# df1=pd.DataFrame(response.toarray(),columns=tfidf_vect.get_feature_names())
#print('no. of input features:',len(response.get_feature_names()))
#print(df1)

#print(x_train)
#print(X1)
MNB = LogisticRegression()
MNB.fit(X1,y_train)

MNB2 = BernoulliNB()
MNB2.fit(X1,y_train)

y_predict_test2 = MNB2.predict(X_test)
y_predict_test2_prob = MNB2.predict_proba(X_test)
print(y_predict_test2_prob[0:10,:])

y_predict_test=MNB.predict(X_test)
y_predict_test_prob = MNB.predict_proba(X_test)
lr = y_predict_test_prob[:,1]
lr2 = y_predict_test2_prob[:,1]

cm=confusion_matrix(y_test,y_predict_test)
cm2 = confusion_matrix(y_test,y_predict_test2)
sns.heatmap(cm,annot=True)
#sns.heatmap(cm2,annot=True)
#plt.show()
print(classification_report(y_test,y_predict_test))
print(classification_report(y_test,y_predict_test))

fpr,tpr,thresholds =metrics.roc_curve(y_test,lr)
plt.plot(fpr,tpr)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.title('roc curve for logistic regression')
plt.ylabel('sensitivity')
plt.xlabel('false positive rate')
plt.grid('true')
plt.show()

print(metrics.roc_auc_score(y_test,lr))


fpr2,tpr2,thresholds2 =metrics.roc_curve(y_test,lr2)
plt.plot(fpr2,tpr2)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.title('roc curve for naive bayes')
plt.ylabel('sensitivity')
plt.xlabel('false positive rate')
plt.grid('true')
plt.show()

print(metrics.roc_auc_score(y_test,lr2))
