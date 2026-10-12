import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

df=pd.read_csv('spam.csv', encoding='latin-1')
df=df[['v1','v2']]
df.columns=["label","message"]
df['label']=df['label'].map({
    'ham':0,
    'spam':1
})

X=df['message']
y=df['label']


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

tfidf_vectorizer=TfidfVectorizer(lowercase=True, stop_words='english')
X_train_tfidf=tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf=tfidf_vectorizer.transform(X_test)

model=LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf,y_train)

predictions = model.predict_proba(X_test_tfidf)[:,1]
y_pred = (predictions > 0.2).astype(int)

print("------model performance metrics-----")
accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy :   {accuracy:.4f}")
precision = precision_score(y_test, y_pred)
print(f"precision :  {precision:.4f}")
recall= recall_score(y_test, y_pred)
print(f"Recall_score :  {recall:.4f}")
f1 = f1_score(y_test, y_pred)
print(f"f1_score :  {f1:.4f}")
print("------confusion matrix-----")
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)

print("------predicting new data------")
new_data=input("Enter a message to classify:")
new_data_tfidf=tfidf_vectorizer.transform([new_data])
new_prediction=model.predict_proba(new_data_tfidf)[0][1]

if new_prediction > 0.2:
    print("The message is classified as : Spam")
else:
    print("The message is classified as Ham(legitimate)")



