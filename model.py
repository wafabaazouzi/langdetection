import pandas as pd
import re          #(regular expression)
import warnings
import neattext as nt
import neattext.functions as nfx
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import pickle





df= pd.read_csv("Mergedfile.csv")
print(df.head())
print(df.dtypes)





df['sent'].apply(lambda x:nt.TextFrame(x).noise_scan())
df['sent'].apply(lambda x:nt.TextExtractor(x).extract_stopwords())
df['sent'].apply(nfx.remove_stopwords)
corpus = df['sent'].apply(nfx.remove_stopwords)




#df["lang"].value_counts()
y = df["lang"]
X = df["sent"]
le = LabelEncoder()
y = le.fit_transform(y)

'''
X = df["sent"]
y = df["lang"]




le = LabelEncoder()
y = le.fit_transform(y)



# cleaning the data

text_list = []

# iterating through all the text
for sent in X:
    sent = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ',str(sent)) # removes all the symbols and numbers
    sent = re.sub(r'[[]]', ' ', sent)
    sent = re.sub("[إأٱآا]", "ا", sent)

    sent = sent.lower()# converts all the text to lower case
    text_list.append(sent)       # appends the text to the text_list

'''
#tfidf = TfidfVectorizer()
#Xfeatures = tfidf.fit_transform(corpus).toarray()
#print(Xfeatures)

print(y)
cv = CountVectorizer()

X = cv.fit_transform(corpus)# tokenize a collection of text documents and store it
                                            #in an array


# Modeling the data

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.15, random_state=30)



model = MultinomialNB()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


# Model evaluation

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Accuracy is :",acc)
print(":",cm)


def predict(txt):
     texte = cv.transform([txt]).toarray() # convert text to bag of words model (Vector)
     lang = model.predict(texte) # predict the language
     lang=le.inverse_transform(lang)
     print ("The language is in",lang[0]) # printing the language


predict('كيف ما تشوفو في التصويرة ')  # Call the function


pickle.dump(cv, open("_vectorizer.pickle", "wb"))
pickle.dump(model, open('_model.pickle','wb'))
pickle.dump(le, open('_le.pickle','wb'))