import json
from pprint import pprint
from textblob import TextBlob
import random
from collections import defaultdict
import pandas as pd
from nltk.tag.stanford import StanfordNERTagger


####Main method, gets called when file run as script. Where the magic happens####
def main():

	runscript = input("Do the thing? (y/n) ")
	if not (runscript == "y" or runscript == "Y"):
		return "Done."

	tagger = StanfordNERTagger("stanford-ner-2014-06-16\classifiers\english.conll.4class.distsim.crf.ser.gz",
		"stanford-ner-2014-06-16\stanford-ner.jar")
	evalFile = "testing.json"

	data = readFile(evalFile)
	paragraphs = data["data"][0]["paragraphs"]
	baseDict = {}

	for i in range(len(paragraphs)):
		context = paragraphs[i]["context"]
		

#####BASELINE MODEL:

#reads file, returns data of file as json
def readFile(file):
	with open(file) as data_file:
		data = json.load(data_file)

	return data


#Random choose model
def baseLine(data):
	paragraphs = data["data"][0]["paragraphs"]
	baseDict = {}

	for i in range(len(paragraphs)):
		context = paragraphs[i]["context"]
		noun_phrases = TextBlob(context).noun_phrases
		qas = paragraphs[i]["qas"]

		for j in range(len(qas)):
			num = random.randrange(len(noun_phrases))
			randanswer = noun_phrases[num]
			current_id = qas[j]["id"]
			baseDict[current_id] = randanswer

	return baseDict

#write out prediction json
def writeJson(dict):
	with open('pred.json', 'w') as fp:
		json.dump(dict, fp)


####### Actual Method #######

#question is the string query to which we will find an answer.
#Returns one of the following strings:
#"person" "place" "organization" "time" "thing" "number" "unknown" 
#This string indicates which type of answer the question is looking for
def guessAnswerType(question):
	personIdentifiers = ["person", "Who", "who", "Whom", "whom", "person", "individual"]
	locationIdentifiers = ["location", "Where", "where", "location", "place", "at", "country", "state",
	 "city", "county", "province"]
	organizationIdentifiers = ["organization", "Which", "which", "What", "what", "organization", 
	 "team", "business", "company"]
	timeIdentifiers = ["time", "When", "when", "time", "What time", "what time", "in"]
	dateIdentifiers = ["date", "When", "when", "what date", "What date", "year"]
	thingIdentifiers = ["thing", "What", "what", "Which", "which"]
	numberIdentifiers = ["number", "How many", "how many", "What number", "what number", "How much",
	 "how much", "number", "number of", "count"]

	identifiersList = [personIdentifiers,locationIdentifiers,organizationIdentifiers,
	timeIdentifiers,dateIdentifiers,thingIdentifiers,numberIdentifiers]
	
	runningGuesses = []

	for identifiers in identifiersList: #Loops through every phrase in each set of
		for phrase in identifiers[1:]:   #identifiers, and adds the corresponding
			if phrase + " " in question:      #tag to the guesses list as it goes
				runningGuesses.append(identifiers[0])
	
	if runningGuesses == []: #if there is no guess, return unknown
		return "unknown"

	uniqueTags = [] #---------------Getting most frequent tag------------------
	for tag in runningGuesses:
		if tag not in uniqueTags:
			uniqueTags.append(tag)

	tagCounts = []
	for tag in uniqueTags:
		tagCounts.append([tag,runningGuesses.count(tag)])

	mostFreq = 0
	mostTag = ""
	tie = ""
	for tag,count in tagCounts:
		if count > mostFreq:
			mostTag = tag
			mostFreq = count
		elif count == mostFreq:
			tie = tag #--------------/Getting most frequent tag-----------------

	if mostTag == "":
		return "unknown"

	return mostTag #if there is a tie, just use one of them, it's more helpful than nothing


#############################

#######IGNORE, THIS IS FOR POSSIBLE IOG TAG METHODS FOR USE LATER IN PROJECT


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


#create bigram/unigram table
def store_counts(filename):
	
	data = readFile(filename)
	types = defaultdict(lambda: defaultdict(int))
	seen = set()  # instantiate seen word set

	paras = data["data"][0]["paragraphs"]

	for i in range(len(paras)):

		currentp = paras[i]['context']

		currentp = currentp.replace(" n't", "n 't")  # Standardize the contractions ('t is a separate word)
		currentp = currentp.replace('-', ' ')  # Get rid of hyphens

		#split into words, add start/end tokens
		tokens = ['<s>'] + currentp.split() + ['</s>']

		count = len(tokens)
		for i in range(count-1):
			# treat upper and lower case words the same
			word1 = tokens[i].lower()
			word2 = tokens[i+1].lower()

			#if words NOT in set, add to set, and change current word to <unk>
			if(not(word1 in seen)):
				seen.add(word1)
				word1 = "<unk>"
			
			if(not(word2 in seen)):
				seen.add(word2)
				word2 = "<unk>"

			types[word1][word2] += 1

	# convert dictionary to table
	table = pd.DataFrame(types).T
	# add totals
	table['SUM'] = table.sum(axis=1)
	#table.loc['</s>', 'SUM'] = int(table.loc['<s>', 'SUM'])
	table = table.fillna(0).applymap(lambda x: int(x))
	return table

# return the unigram P(word) for a given word, with a given table of counts
def unigram(word, table):
	try:
		return float(table.loc[word, 'SUM'])/float(table['SUM'].sum())
	except KeyError:
		print("This word doesn't exist in the corpus.")

def random_paragraph(filename):
	data = readFile(filename)
	paragraphs = data["data"][0]["paragraphs"]
	p = random.randrange(len(paragraphs))
	return paragraphs[p]["context"]

def noun_phrases(paragraph):
	return TextBlob(paragraph).noun_phrases

def pos_tagger(paragraph):
	return TextBlob(paragraph).tags


if __name__ == "__main__":
	main()
