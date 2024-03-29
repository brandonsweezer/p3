import json
from pprint import pprint
from textblob import TextBlob
import random
from collections import defaultdict
import pandas as pd
from nltk.tag.stanford import StanfordNERTagger
import re

import nltk
import re
from nltk.tokenize import sent_tokenize
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter

####Main method, gets called when file run as script. Where the magic happens####
def main():
	# runscript = input("Do the thing? (y/n) ")
	# if not (runscript == "y" or runscript == "Y"):
	# 	return "Done."

	tagger = StanfordNERTagger("stanford-ner-2014-06-16/classifiers/english.conll.4class.distsim.crf.ser.gz",
							   "stanford-ner-2014-06-16/stanford-ner.jar")
	evalFile = "development.json"

	data = readFile(evalFile)
	QADict = {}
	for k in range(len(data["data"])):
		print(k)
		paragraphs = data["data"][k]["paragraphs"]
		for i in range(len(paragraphs)):
			print(i)
			context = paragraphs[i]["context"]
			nps = noun_phrases(context) #noun phrases
			qs = paragraphs[i]["qas"] #questions
			taggedContext = tagger.tag(context.split())
			taggedPhrases = getNERPhrases(taggedContext)
			answerSentences = answerquestions(paragraphs[i])

			for j in range(len(qs)):
				current_id = qs[j]["id"]
				question = qs[j]["question"]
				answerSentence = answerSentences.get(question)
				answerType = guessAnswerType(question)
				taggedSentence = tagger.tag(answerSentence)

				possibleAnswers = searchForAnswer(taggedSentence,answerType)

				#possibleAnswers = narrowPhrases(answerType,taggedPhrases,question) #WE NEED TO TAKE INTO ACCOUNT VERB PHRASES
				
				# bestAnswer = ""
				# bestScore = 2^31
				# for answer in possibleAnswers:
				# 	answerStatement = rephrase(answer,question) #Rephrase the answer as a statement
				# 	answerScore = calculatePerplexity(answerStatement,context) #Calculate the perplexity of the generated statement based on the context
				# 	if answerScore < bestScore: #Find answer with least perplexity
				# 		bestAnswer = answer
				# 		bestScore = answerScore
				
				bestAnswer = getBestAnswer(possibleAnswers,answerSentence)

				QADict[current_id] = bestAnswer

	writeJson(QADict)
	print("Done.")


"""Searches a given TAGGED sentence for a certain answer type
returns a list of possible answers of that type, or all possible answers
if that type is not found.
"""
def searchForAnswer(sentence,answerType):
	panswers = []
	allanswers = []

	s = getNERPhrases(sentence)
	for phrase, tag in s:
		allanswers.append(phrase)
		if tag.lower() == answerType:
			panswers.add(phrase)
		elif tag == "MISC":
			if answerType == "number" or answerType == "date":
				for n,t in sentence:
					if n.isdigit():
						panswers.append(n)

	if panswers = []:
		panswers = allanswers

	return panswers
		

"""Finds the best answer given a list of possible answers and a sentence"""
def getBestAnswer(possibleAnswers, answerSentence):
	pass


def tester(para):
	QADict = {}
	tagger = StanfordNERTagger("stanford-ner-2014-06-16/classifiers/english.conll.4class.distsim.crf.ser.gz",
		"stanford-ner-2014-06-16/stanford-ner.jar")
	context = para[0]["context"]
	nps = noun_phrases(context) #noun phrases
	qs = para[0]["qas"] #questions
	taggedContext = tagger.tag(context.split())
	taggedPhrases = getNERPhrases(taggedContext)

	for j in range(len(qs)):
		current_id = qs[j]["id"]
		question = qs[j]["question"]
		answerType = guessAnswerType(question)
		possibleAnswers = narrowPhrases(answerType,taggedPhrases,question) #WE NEED TO TAKE INTO ACCOUNT VERB PHRASES

		bestAnswer = ""
		bestScore = 2^31
		for answer in possibleAnswers:
			answerStatement = rephrase(answer,question) #Rephrase the answer as a statement
			answerScore = calculatePerplexity(answerStatement,context) #Calculate the perplexity of the generated statement based on the context
			if answerScore < bestScore: #Find answer with least perplexity
				bestAnswer = answer
				bestScore = answerScore

		QADict[current_id] = bestAnswer

	return QADict

"""Rephrases an answer (a noun or verb phrase) as a statement, given the question
For example, if the question was "How many hours in the day are there?" and the answer
provided was "24", rephrase("24","How many hours in the day are there?") would output
"There are 24 hours in the day."
"""
def rephrase(answer, question):
	unwanted = ["Did", "Who", "When", "What", "Where", "Why", "How", "Whom", "did", "who", "when",
				"what", "where", "why", "how", "whom", "much", "many", "kind", "sort", " "]

	newq = question
	for item in unwanted:
		newq.strip(item)

	return answer + question


"""Calculates the perplexity of a statement given a context. This statement will be the 
answer statement from the rephrase function above. This will be used to evaluate which answer
is the best out of a group of possible answers. Context must be a string. 
"""
def calculatePerplexity(answerStatement,context):
	uni = unigram(context)
	bi = bigram(context)

	return(calcPer(answerStatement, uni, bi))

"""Returns the unigram value, P(word), for a given word with a given dictionary of unigrams"""
def unigramValue(word, dic):
	return float(dic.get(word,1/len(dic))) / sum(dic.values())

"""Returns the unigram dictionary given the context paragraph"""
def unigram(para):
	token = nltk.word_tokenize(para)
	unidict = {}
	for i in range(len(token)):
		word1 = token[i].lower()
		if not(word1 in unidict):
			unidict[word1] = 1
		else:
			unidict[word1] = unidict[word1] + 1
	return unidict

"""Returns the bigram counter given the context paragraph"""
def bigram(para):
	token = nltk.word_tokenize(para)
	token = [x.lower() for x in token]
	bigrams = ngrams(token, 2)
	return Counter(bigrams)

"""Returns calculated perplexity given a context string, unigram dictionary, and bigram counter"""
def calcPer(context, uni, bi):
	contextBi = bigram(context)

	#if answer is a single word return unigram perplexity
	if(len(contextBi) == 0):
		return (1/unigramValue(context, uni))

	#otherwise calculate bigram perplexity
	elements = list(contextBi.elements())
	perplex = 1
	for x in elements:
		prev = x[0]
		if(x in bi.keys()):
			perplex = perplex * (unigramValue(prev, uni) / (float(bi[x]) / sum(bi.values())))
		else:
			perplex * (unigramValue(prev, uni) / ((float(1)/len(bi.keys())) / sum(bi.values())))
	N = len(elements)
	return (perplex)**(1/float(N))


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
def writebaseJson(dict):
	with open('pred.json', 'w') as fp:
		json.dump(dict, fp)

#write out prediction json
def writeJson(dict):
	with open('pred.json', 'w') as fp:
		json.dump(dict, fp)


####### Actual Method #######

#question is the string query to which we will find an answer.
#Returns one of the following strings:
#"person" "location" "organization" "time" "date" "thing" "number" "unknown" 
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
			if not question.find(phrase + " ") == -1:      #tag to the guesses list as it goes
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


def narrowPhrases(aType, taggedPhrases, question):
	narrowedList = []
	allphrases = []

	for phrase, tag in taggedPhrases:
		allphrases.append(phrase)
		t = tag.lower()
		if aType == t:
			narrowedList.append(phrase)

		if t == "misc":
			if aType == "number":
				if bool(re.search(r'\d', phrase)): #see if there's a digit in the phrase
				 	narrowedList.append(phrase)

			elif not (aType == "person" or aType == "organization" or aType == "location"):
				narrowedList.append(phrase)

	if narrowedList == []:
		narrowedList = allphrases

	for phrase in narrowedList:
		if not (question.find(phrase) == -1):
			narrowedList.remove(phrase)

	return narrowedList


def getNERPhrases(taggedWords):
	taggedPhrases = []

	runningPhrase = ""
	runningTag = "O"
	prevword = ""

	for word, tag in taggedWords:
		#print(tag)
		#print(runningTag)
		if(endOfSent(word) and tag != "O"):
			runningPhrase = runningPhrase + " " + word[:-1]
			taggedPhrases.append((runningPhrase.strip(' '),runningTag))
			runningPhrase = ""
			prevword = word
			continue

		if tag == "O":
			if (word[0].isupper()):
				runningPhrase = runningPhrase + " " + word
				prevword = word
			elif not (runningTag == "O"):
				taggedPhrases.append((runningPhrase.strip(' '),runningTag))
				runningPhrase = ""
				runningTag = "O"
				prevword = word
		elif not (tag == runningTag):
			taggedPhrases.append((runningPhrase.strip(' '),runningTag))
			runningPhrase = word
			runningTag = tag
			prevword = word
		else:
			if endOfSent(word):
				runningPhrase = runningPhrase + " " + word
				taggedPhrases.append((runningPhrase.strip(' '),runningTag))
				runningPhrase = ""
			else:
				runningPhrase = runningPhrase + " " + word


	for p, t in taggedPhrases:
		if p == "" or t == "O":
			taggedPhrases.remove((p,t))

	return taggedPhrases

def endOfSent(word):
    return word[-1] == "."

def answerquestions(para):
	sent_tokenize_list = sent_tokenize(para['context'])

	qsent = {}
	maxindex = 0
	maxcount = 0
	for q in para['qas']:
		question = q["question"]
		#words = re.findall(r'(?<!\.\s)\b[A-Z][a-z]*\b', question)
		words = question.split()
		for i in range(len(sent_tokenize_list)):
			count = 0
			for word in words:
				if word in sent_tokenize_list[i]:
					count = count + 1
			if (count > maxcount):
				maxcount = count
				maxindex = i
		qsent[question] = sent_tokenize_list[maxindex]
		maxindex = 0
		maxcount = 0
	return qsent


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

#returns unigram dictionary based off training data
def training_unigram():
	data = readFile("training.json")
	seen = set()
	unigrams = {}
	paras = data["data"][0]["paragraphs"]
	length = len(paras)

	for i in range(length):
		currentp = paras[i]["context"]
		currentp = currentp.replace(" n't", "n 't")
		currentp = currentp.replace("-", " ")
		tokens = ['<s>'] + currentp.split() + ['</s>']
		count = len(tokens)
		for j in range(count):
			word1 = tokens[j].lower()
			if(not(word1 in seen)):
				seen.add(word1)
				word1 = "<unk>"

			if(not(word1 in unigrams.keys())):
				unigrams[word1] = 1
			else:
				unigrams[word1] = unigrams[word1] + 1

	return unigrams

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
