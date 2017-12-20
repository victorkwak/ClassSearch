import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import fastText
from gensim import utils


ENGLISH_STOP_WORDS = frozenset([
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
    "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill",
    "find", "fire", "first", "five", "for", "former", "formerly", "forty",
    "found", "four", "from", "front", "full", "further", "get", "give", "go",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
    "latterly", "least", "less", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "move", "much", "must", "my", "myself", "name", "namely", "neither",
    "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
    "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
    "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "rather", "re", "same", "see", "seem", "seemed",
    "seeming", "seems", "serious", "several", "she", "should", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such",
    "system", "take", "ten", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "three", "through", "throughout",
    "thru", "thus", "to", "together", "too", "top", "toward", "towards",
    "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves"])

def format_for_fastext(X, y, filename):
    prefix = '__label__'
    f = open(''.join(['data/', filename]), 'w')
    for title, label in zip(X, y):
        title = title.lower()
        tokens = utils.simple_preprocess(title)
        tokens = [token for token in tokens if token not in ENGLISH_STOP_WORDS]
        string = ''.join([prefix, str(label), ' ', ' '.join(tokens), '\n'])
        f.write(string)
    f.close()

def test_fasttext(y, X, classifier, n=1):
    match = []
    for true, string in zip(y, X):
        predictions = list(classifier.predict(string, n)[0])
        for i in range(n):
            predictions[i] = int(predictions[i].split('__label__')[1])
        match.append(int(true in predictions))
    return np.array(match)

print('Importing data...')
data = pd.read_csv('data/cs_subs.csv').dropna().drop_duplicates()

print('Filtering out uncommon and unpopular title posts...')
counts = data['subreddit'].value_counts()
counts = counts[counts > 150]
top_values = list(counts.index)
data = data[data['subreddit'].isin(top_values)]

means = {}
for subreddit in data['subreddit'].unique():
    means[subreddit] = data[data['subreddit'] == subreddit]['score'].mean()

filtered = []

for subreddit in data['subreddit'].unique():
    filtered.append(data.loc[(data['subreddit'] == subreddit) & (data['score'] >= means[subreddit])])

filtered_data = pd.concat(filtered)

print('Total number of rows:', filtered_data.shape[0])

X = filtered_data['title']
y = filtered_data['subreddit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=17)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=31)

label_encoder = LabelEncoder()
label_encoder.fit(data['subreddit'])
y_train = label_encoder.transform(y_train)
y_val = label_encoder.transform(y_val)
y_test = label_encoder.transform(y_test)
print('Number of unique subreddits (categories):', len(label_encoder.classes_))

print('Formatting data for FastText consumption...')
format_for_fastext(X_train, y_train, 'reddit_fasttext_train.txt')

print('Training classifier..')
classifier = fastText.train_supervised(input='data/reddit_fasttext_train.txt',
                                       lr=0.1,
                                       epoch=30,
                                       dim=64,
                                       minn=2,
                                       maxn=5
                                       )

correct = test_fasttext(y_val, X_val, classifier)
print('Validation accuracy:', correct.sum() / y_val.size)

correct = test_fasttext(y_test, X_test, classifier)
print('Test accuracy:', correct.sum() / y_test.size)

path = 'models/fasttext.bin'
print('Saving model to', path)
classifier.save_model('models/fasttext.bin')
