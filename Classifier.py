import os
import io
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Find all files in dir, build/read path 
def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)

            inBody = False
            lines = []
            
            f = io.open(path, 'r', encoding='latin1')

            #Skip header
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
            
            f.close()
            
            message = '\n'.join(lines)

            yield path, message


# Create DF obj of emails with training data, columns msg and classification
def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)

    return DataFrame(rows, index=index)

data = DataFrame({'message': [], 'class': []})

data = data.append(dataFrameFromDirectory('emails/spam', 'spam'))
data = data.append(dataFrameFromDirectory('emails/valid', 'valid'))

# Split each message into list of words
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data['message'].values)

classifier = MultinomialNB()
targets = data['class'].values

# Pass in actual data, targets
classifier.fit(counts, targets)
