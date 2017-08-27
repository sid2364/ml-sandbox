import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
positive_text = 'pos.txt'
negative_text = 'neg.txt'
n_lines = 1000000
sentiment_pkl = "pos-neg-sentiment.pkl"

def create_lexicon(positive, negative):
	lexicon = []
	for filen in [positive, negative]:
		with open(filen, 'r') as f:
			contents = f.readlines()
			for l in contents[:n_lines]:
				nl = l.decode("utf8").lower()
				all_words = word_tokenize(nl)
				lexicon += list(all_words)

	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	w_counts = Counter(lexicon) # is a dictionary with {word: count}
	
	l2 = []
	for w in w_counts:
		if 10 < w_counts[w] < 500: # more occurances than 50 and less than 1000
			l2.append(w)

	return l2

def sample_handling(sample, lexicon, classification):
	featureset = []

	with open(sample, 'r') as f:
		contents = f.readlines()
		for l in contents[:n_lines]:
			nl = l.decode("utf8").lower()
			current_words = word_tokenize(nl)
			current_words = [lemmatizer.lemmatize(i) for i in current_words]
			features = np.zeros(len(lexicon))
			for word in current_words:
				if word.lower() in lexicon:
					index = lexicon.index(word.lower())
					features[index] += 1

			features = list(features)
			featureset.append([features, classification])
	#print(featureset)
	return featureset

def create_feature_sets_and_labels(positive_text, negative_text, test_size=0.1):
	lexicon = create_lexicon(positive_text, negative_text)
	features = []
	features += sample_handling(positive_text, lexicon, [1,0])
	features += sample_handling(negative_text, lexicon, [0,1])
	random.shuffle(features)

	features = np.array(features)
	testing_size = int(test_size*len(features))

	train_X = list(features[:,0][:-testing_size])
	train_y = list(features[:,1][:-testing_size])

	test_X = list(features[:,0][:-testing_size])
	test_y = list(features[:,1][:-testing_size])

	return train_X, train_y, test_X, test_y

if __name__ == '__main__':
	train_X, train_y, test_X, test_y = create_feature_sets_and_labels(positive_text, negative_text)
	with open(sentiment_pkl, 'wb') as f:
		pickle.dump([train_X, train_y, test_X, test_y], f)