from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from GetInput import getCorpus
from Gmail_api import getMails
from Pineline import frequency_pipeline
import time
import matplotlib.pyplot as plt


def SVM(X, y):
    
    vectorizer = TfidfVectorizer()
    X_augmented = vectorizer.fit_transform(X)

    start = time.time()
    clf = svm.SVC(kernel='rbf', C=1000)
    scores = 100 * cross_val_score(clf, X_augmented, y, cv=5).mean()
    end = time.time()

    print("SVM:")
    print("Rate: {:.2f}%".format(scores))
    print("Time(second): ", end - start)
    return scores, end - start

def MNB(X, y):
    
    X_augmented = frequency_pipeline.fit_transform(X).toarray()
    
    start = time.time()
    clf = MultinomialNB()
    scores = 100 * cross_val_score(clf, X_augmented, y, cv=5).mean()
    end = time.time()

    print("MNB")
    print("Rate: {:.2f}%".format(scores))
    print("Time(second): ", end - start)
    return scores, end - start

def TrainModel():
    n = [200, 500, 1000, 2000, 5000]
    rates_svm = []
    times_svm = []
    rates_mnb = []
    times_mnb = []
    for i in n:
        X,y = getCorpus(url = 'input/emails_dataset.csv', number = i)
        print("N = ", y.shape)
        rate_svm, time_svm = SVM(X, y)
        rates_svm.append(rate_svm)
        times_svm.append(time_svm)
        rate_mnb, time_mnb = MNB(X, y)
        rates_mnb.append(rate_mnb)
        times_mnb.append(time_mnb)
    plt.plot(n, rates_svm, n, rates_mnb)
    plt.axis([0, 6000, 70, 100])
    plt.show()
    plt.plot(n, times_svm, n, times_mnb)
    plt.axis([0, 6000, 0, 60])
    plt.show()
    

def MyEmail(category = 'CATEGORY_UPDATES'):
    #['CATEGORY_UPDATES']
    X,y = GetInput.getInput(url = 'input/emails_dataset.csv', number = 5000)
    mails = getMails(cate = category)
    if mails is None:
        print("No mail in  category ",category)
        return

    vectorizer = TfidfVectorizer()
    X_augmented = vectorizer.fit_transform(X)
    Mails_augmented= vectorizer.transform(mails)

    clf = svm.SVC(kernel='linear', C=1000)
    clf.fit(X_augmented, y)

    y_pred = clf.predict(Mails_augmented)
    rate = y_pred.tolist().count(1)/len(y_pred)
    print("Spam rate: ", rate)

def main():
    MyEmail('SPAM')

if __name__ == "__main__":
    main()
