# SMS SPAM detection

A simple SPAM detector for SMS messages.

---

## Background

### Bayes' Theorem

* Foundation of deductive reasoning.
* Focuses on determining the probability of an event occurring based on prior knowledge of conditions that might be related to the event.

**P(H|E) = (P(E|H) * P(H)) / P(E)**

* P(H|E) is the probability of hypothesis H given the event E.
* P(E|H) is the probability of event E given that the hypothesis H is true.
* P(H) is the probability of hypothesis H being true.
* P(E) is the probability of the event E occurring.

Example: To know whether an SMS that contains the word ***clearance*** (event) is spam (hypothesis).  

**P(class=SPAM|contains="clearance") = (P(contains="clearance"|class=SPAM) * P(class=SPAM)) / P(contains="clearance")**

The probability of an e-mail containing the word 'clearance' being spam is equal to the proportion of spam emails that contain the word 'clearance' multiplied by the proportion of e-mails being spam and divided by the proportion of e-mails containing the word 'clearance'.

* P(class=SPAM|contains="clearance") is the probability of an SMS being SPAM given that this e-mail contains the word "clearance". This is what we are interested in predicting.
* P(contains="clearance"|class=SPAM) is the probability of an SMS containing the word "clearance" given that this SMS has been recognized as SPAM. This is our training data, which represents the correlation between an SMS being considered SPAM and such SMS containing the word "clearance".
* P(class=SPAM) is the probability of an SMS being SPAM. This is simply the proportion of SMS being SPAM in our entire training set. We multiply by this value because we are interested in knowing how significant is information concerning SPAM SMSs. If this value is low, the significance of any events related to SPAM SMSs will also be low.
* P(contains="clearance") is the probability of an SMS containing the word "clearance". This is simply the proportion of SMSs containing the word "clearance" in our entire training set. We divide by this value because the more exclusive the word "clearance" is, the more important is the context in which it appears. Thus, if this number is low. i.e. the word appears very rarely, it can be a great indicator that in the cases it does appear, it is a relevant feature to analyze.

There are **two types of probabilities** that appear in Bayes' Theorm:

* **Class Probabilities**

**P(class=SPAM) = count(class=SPAM) / (count(class=notSPAM) + count(class=SPAM))**

* **Conditional Probabilities**

**P(class=SPAM|contains="clearance") = count(class=SPAM & contains="clearance") / count(contains="clearance")**

### Naive Bayes' Classifier

* Based on Bayes' Theorem.
* Simple and easy to implement algorithm.
* Outperforms more complex models when the amount of data is limited.
* Works well with numerical and categorical data.
* Works well as long as the categories are kept simple.
  * Works well for problems involving keywords as features e.g. spam detection.
  * Does NOT work when the relationship between words is important e.g. sentiment analysis.  
* Does NOT work well when there are certain missing combination of values in the training data.
  * If you have no occurrences of a class label and a certain attribute value together then the frequency-based probability estimate will be zero. Given Naive-Bayes' conditional independence assumption, in such a case when all the probabilities are multiplied you will get zero.
* Can also be used to perform regression by using Gaussian Naive Bayes.  

---

## Goal 

Build a classification model that detects if an SMS is a SPAM using **Naive Bayes's Classifier**.

## Dependencies

* Pandas
* Nltk
* Scikit-learn
* Numpy

`pip install -r requirements.txt`

Download the tokenizer *punkt* from the python console:

```python
import nltk
nltk.download()
```

An installation window will appear. Go to the "Models" tab and select "punkt" from the "Identifier" column. Then click "Download" and it will install the necessary files.

## Dataset

SPAM SMS collect data from UCI archive: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection<br>
Saved in: *data/SMSSpamCollection*

## Data Preprocessing

* Check for null values and remove those records
* Convert the labels to binary values
  * spam = 1, not spam = 0
* Convert the message to lower case
* Remove punctuations in the message
* Tokenize the message to words
* Perform word stemming to remove the word variations and normalize
  * *Porter Stemmer*
* Transform the data into word occurences
  * Convert list of words into space-separated strings
  * Word count per message: *Count Vectorizer*
  * Term Irequency Inverse Document Frequency: *TF-IDF*
* Split the data to Training and Test sets
  * Test size = 10%

## Create and Train Model

Use ***Multinomial* Naive Bayes' Classifier**

`MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)`

## Evaluate the model

* Accuracy of the simple Naive Bayes' classifier is 94.8%.
  * Accuracy is not enough if the datset is imbalanced.
  * The classifier might be overfitting.
* Confusuion matrix solves the uncertainty.

```
Confusion Matrix:
 [[482   0]
 [ 29  47]]
```

## Conclusion

* 29 Spam messages are classified as legitimate. This is NOT a good indicator.
* The model needs to be improved as there is a significant number of **False Negatives** in this case.


