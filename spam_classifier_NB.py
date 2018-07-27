import os
import io
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#READ FILES FROM PATH
def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)

            inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
            f.close()
            message = '\n'.join(lines)
            yield path, message

# MAKE DATAFRAME CONTAINING MESSAGE AND CLASS
def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)

    return DataFrame(rows, index=index)

# CREATING A DATAFRAME
data = DataFrame({'message': [], 'class': []})

# ADDING DATA INTO THE
data = data.append(dataFrameFromDirectory('E:/NJIT/Data Science, Deep Learning and Machine Learning with Python/DataScience/DataScience-Python3/emails/spam','spam'))
data = data.append(dataFrameFromDirectory('E:/NJIT/Data Science, Deep Learning and Machine Learning with Python/DataScience/DataScience-Python3/emails/ham','ham'))

data.head()

vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data['message'].values)

classifier = MultinomialNB()
targets = data['class'].values
classifier.fit(counts, targets)

examples = ['Slim Down - Guaranteed to lose 10-12 lbs in 30 days!!!', "Hi Bob, how about a free game of golf tomorrow?"]
example_counts = vectorizer.transform(examples)
predictions = classifier.predict(example_counts)
print(predictions)
