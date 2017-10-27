import json
from pprint import pprint
import sklearn_crfsuite

def readFile(file):
	with open(file) as data_file:
		data = json.load(data_file)

	return data

# helper function
# creates features for a simple crf model - only uses word and pos tag
def simple_crf(sent, i):
	word = sent[i]
	features = {
		'token': word[0],
		'postag': word[1]
	}
	return features


def crf2(sent, i):
	word = sent[i]
	features = {
		'token': word[0],
		'postag': word[1],
		'start': (i == 0),
		'capitals': (word[0].lower() != word[0])
	}
	if i > 0:
		features['prev_postag'] = sent[i-1][1]
		features['prev_token'] = sent[i-1][0]
	if i < len(sent) - 1:
		features['next_postag'] = sent[i+1][1]
		features['next_token'] = sent[i+1][0]
	return features


# helper function
# for a given sentence, creates the features for each word
def sentence_features(sent, create_features):
	return [create_features(sent, i) for i in range(len(sent))]


# helper function
# for a given sentence, returns the labels that we are predicting
def sentence_labels(sent):
	return [label[2:] if len(label) > 1 else label for word, pos, label in sent]


# creates CRF model, trains it on the train_file and returns predictions for the test_file
def crf(train_file, test_file, create_features):
	crf = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
	train = group_tokens_tags(train_file)
	x_train = [sentence_features(s, create_features) for s in train]
	y_train = [sentence_labels(s) for s in train]
	crf.fit(x_train, y_train)
	
	test = group_tokens_tags(test_file)
	x_test = [sentence_features(s, create_features) for s in test]
	y_test = [sentence_labels(s) for s in test]
	y_test_flat = [word for sent in y_test for word in sent]

	y_pred = crf.predict(x_test)
	y_pred_flat = [word for sent in y_pred for word in sent]
	return y_pred_flat

