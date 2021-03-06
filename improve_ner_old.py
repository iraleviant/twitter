import re, math, collections, itertools, os
import nltk, nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk import precision
from nltk import recall
import random
from haversine import haversine
from collections import Counter
from nltk.tag.stanford import StanfordNERTagger
#from difflib import SequenceMatcher
#http://andybromberg.com/sentiment-analysis-python/
#https://github.com/abromberg/sentiment_analysis_python

#===============================================================================
# st = StanfordNERTagger('stanford-ner/english.all.3class.distsim.crf.ser.gz', 'stanford-ner/stanford-ner.jar')
# 
# print (st.tag('You can call me Billiy Bubu and I live in Amsterdam.'.split()))
# 
# sen= u"Twenty miles east of Reno, Nev., " \
#     "where packs of wild mustangs roam free through " \
#     "the parched landscape, Tesla Gigafactory 1 " \
#     "sprawls near Interstate 80."
# print (st.tag  (nltk.word_tokenize(sen) ) )
#===============================================================================

jar = '/home/ira/Dropbox/twitter/stanford-ner/stanford-ner.jar' 
model = '/home/ira/Dropbox/twitter/stanford-ner/dummy-ner-model-twitter.ser.gz'  # the trained model on twitter train

ner_tagger = StanfordNERTagger(model, jar, encoding='utf8')


def extract_GPS(tweet):
    #gps=(-70.2, 40.77)
    first=tweet.index('\t')
    last=tweet.rindex('\t')
    subtweet=tweet[first+1:last] #check
    gps=subtweet.split('\t')
    gps_tuple=(float(gps[0]), float(gps[1]))
    return  gps_tuple#returns tuple with two elements

#POLARITY_DATA_DIR = os.path.join('polarityData', 'rt-polaritydata')
#RT_POLARITY_POS_FILE = '/home/ira/Dropbox/twitter/contain_location_tweets.txt'
NEG_FILE_TRAIN = '/home/ira/Dropbox/twitter/no_location_for_train.txt'
NEG_FILE_TEST = '/home/ira/Dropbox/twitter/no_location_for_test.txt'
POS_FILE_TRAIN ='/home/ira/Dropbox/twitter/train_with_location.txt'
POS_FILE_TEST ='/home/ira/Dropbox/twitter/test_with_location.txt'
ALL_POS_FILE ='/home/ira/Dropbox/twitter/with_location_all.txt'
ALL_NEG_FILE ='/home/ira/Dropbox/twitter/no_location_all.txt'
test_file= "/home/ira/Dropbox/twitter/test_all_non_tagged.txt"

#########  build a dictionary, key is a tweet and the value is the list of geographical closest tweets
Close_Twt_Dic={}

with open(test_file, 'rb') as ofh:
    twitts = ofh.readlines()    

for twt in twitts:
    twtn=(twt.decode("utf-8")).strip()  # convert bytes to strings
    #str1 = ''.join(twtn) #check if needed
    gps1=extract_GPS(twtn) 
    close_tweets=[]
    twitts_list=twitts[:]
    twitts_list.remove(twt)
    for twt2 in twitts_list:
        twtn2=(twt2.decode("utf-8")).strip()
        gps2=extract_GPS(twtn2)
        dis=haversine(gps1, gps2 )
        if dis<=0.05: #tweets whithin 50 meters from the initial tweet
            close_tweets.append(twtn2)   
    Close_Twt_Dic[twtn]=close_tweets
    
#########################################################################################################    
#this function takes a feature selection mechanism and returns its performance in a variety of metrics
Train_twit_Dic={}
Test_twit_Dic={}

def evaluate_features(feature_select):
    posFeatures_train = []
    negFeatures_train = []
    posFeatures_test = []
    negFeatures_test = []
    #http://stackoverflow.com/questions/367155/splitting-a-string-into-words-and-punctuation
    #breaks up the sentences into lists of individual words (as selected by the input mechanism) and appends 'pos' or 'neg' after each list
    with open(POS_FILE_TRAIN, "rb") as f:
        posSentences_train = f.readlines()
        #posSentences = pos_data.split('\n')
    random.shuffle(posSentences_train)
    
    with open(POS_FILE_TEST, "rb") as f:
        posSentences_test = f.readlines()
        #posSentences = pos_data.split('\n')
    random.shuffle(posSentences_test)
    
    with open(NEG_FILE_TRAIN, "rb") as f:
        #negSentences = f.read().split('\n')
        negSentences_train = f.readlines()
    random.shuffle(negSentences_train)
    
    with open(NEG_FILE_TEST, "rb") as f:
        #negSentences = f.read().split('\n')
        negSentences_test = f.readlines()
    random.shuffle(negSentences_test)
    
    #with open(RT_POLARITY_POS_FILE, 'r') as posSentences:
    for i in posSentences_train:
        posWords = re.findall(r"[\w']+|[.,!?;@#]", str(i).rstrip())
        posWords = [feature_select(posWords), 'pos'] #pos = contains location
        posFeatures_train.append(posWords)
        str_i=(i.decode("utf-8")).strip()
        Train_twit_Dic[frozenset(posWords[0].items())]=str_i
        
    for i in posSentences_test:
        posWords_test = re.findall(r"[\w']+|[.,!?;@#]", str(i).rstrip())
        posWords_test = [feature_select(posWords_test), 'pos'] #pos = contains location
        posFeatures_test.append(posWords_test)
        str1=(i.decode("utf-8")).strip()
        Test_twit_Dic[frozenset(posWords_test[0].items())]=str1
    
    #with open(RT_POLARITY_NEG_FILE, 'r') as negSenBufferedReader: <_io.BufferedReader name='/home/ira/Dropbox/twitter/contain_location_tweets.txt'>tences:
    for i in negSentences_train:
        negWords = re.findall(r"[\w']+|[.,!?;@#]", str(i).rstrip())
        negWords = [feature_select(negWords), 'neg'] #neg = doesn't contain location 
        negFeatures_train.append(negWords)
        str2=(i.decode("utf-8")).strip()
        Train_twit_Dic[frozenset(negWords[0].items())]=str2

    for i in negSentences_test:
        negWords_test = re.findall(r"[\w']+|[.,!?;@#]", str(i).rstrip())
        negWords_test = [feature_select(negWords_test), 'neg'] #neg = doesn't contain location 
        negFeatures_test.append(negWords_test)
        str3=(i.decode("utf-8")).strip()
        Test_twit_Dic[frozenset(negWords_test[0].items())]=str3
    #selects 3/4 of the features to be used for training and 1/4 to be used for testing
    #posCutoff = int(math.floor(len(posFeatures)*3/4))
    #negCutoff = int(math.floor(len(negFeatures)*3/4))
    #trainFeatures = posFeatures[:posCutoff] + negFeatures[:negCutoff] ###need to understand what is test here
    #testFeatures = posFeatures[posCutoff:] + negFeatures[negCutoff:]
    trainFeatures = posFeatures_train+negFeatures_train ###need to understand what is test here
    testFeatures = posFeatures_test + negFeatures_test


##############################################################################3
    #trains a Naive Bayes Classifier
    classifier = NaiveBayesClassifier.train(trainFeatures)    

    #initiates referenceSets and testSets
    referenceSets = collections.defaultdict(set)
    testSets = collections.defaultdict(set)    

    #puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
    for i, (features, label) in enumerate(testFeatures):
        referenceSets[label].add(i)
        predicted = classifier.classify(features)
        testSets[predicted].add(i)    
####################   MINE   ####################################
        if predicted=="pos":  ##the twit according to the classifier contains a location
            twiit=Test_twit_Dic[frozenset(features.items())]
            list_close_twits=Close_Twt_Dic[twiit]
            words = nltk.word_tokenize(twiit)
            tagged_words=ner_tagger.tag(words)
            lbl=""
            for tag_w in tagged_words:
                if tag_w[1]=="LOCATION":
                    lbl=lbl+tag_w[0]+" "  #found a label for our twiit
                final_lbl=lbl
            ### employ satnford trained classifier on all of the close twiits to find 
            if lbl=="":  #couldnt find the location (lable) for our twiit, lets try to find it wihitn its physical neiborhood twwits
                lbl_list=[]
                for s in list_close_twits:
                    words = nltk.word_tokenize(s)
                    tagged_words=ner_tagger.tag(words)
                    lbl=""
                    for tag_w in tagged_words:
                        if tag_w[1]=="LOCATION":
                            lbl=lbl+tag_w[0]+" "
                    if lbl!="": 
                        lbl_list.append(lbl)        
                ## find most common str (label) in lbl_list
                c = Counter(lbl_list)
                c.most_common(1)
                final_lbl=c[0][0]
                
                
                
    #prints metrics to show how well the feature selection did
    print ('train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures)) )
    print ('accuracy:', nltk.classify.util.accuracy(classifier, testFeatures))
    #print 'pos precision:', nltk.metrics.precision(referenceSets['pos'], testSets['pos'])
    print ('pos precision:', precision(referenceSets['pos'], testSets['pos']) )
    print ('pos recall:', recall(referenceSets['pos'], testSets['pos']) )
    #print 'neg precision:', nltk.metrics.precision(referenceSets['neg'], testSets['neg'])
    print ('neg precision:',precision(referenceSets['neg'], testSets['neg']))
    print ('neg recall:', recall(referenceSets['neg'], testSets['neg']))
    classifier.show_most_informative_features(10)

#creates a feature selection mechanism that uses all words
def make_full_dict(words):
    return dict(  [(word, True) for word in words]      )

#tries using all words as the feature selection mechanism
print ('using all words as features')
evaluate_features(make_full_dict)

#scores words based on chi-squared test to show information gain (http://streamhacker.com/2010/06/16/text-classification-sentiment-analysis-eliminate-low-information-features/)
def create_word_scores():
    #creates lists of all positive and negative words
    posWords = []
    negWords = []
    with open(ALL_POS_FILE, 'r') as posSentences:
        for i in posSentences:
            posWord = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
            posWords.append(posWord)
    with open(ALL_NEG_FILE, 'r') as negSentences:
        for i in negSentences:
            negWord = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
            negWords.append(negWord)
    posWords = list(itertools.chain(*posWords))
    negWords = list(itertools.chain(*negWords))

    #build frequency distibution of all words and then frequency distributions of words within positive and negative labels
    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in posWords:
        word_fd[word.lower()] += 1
        cond_word_fd['pos'][word.lower()] += 1
    for word in negWords:
        word_fd[word.lower()] += 1
        cond_word_fd['neg'][word.lower()] += 1

    #finds the number of positive and negative words, as well as the total number of words
    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count

    #builds dictionary of word scores based on chi-squared test
    word_scores = {}
    for word, freq in word_fd.items():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score

    return word_scores

#finds word scores
word_scores = create_word_scores()

#finds the best 'number' words based on word scores
def find_best_words(word_scores, number):
    best_vals = sorted(word_scores.items(), key=lambda ws:(-ws[1], ws[0]) )[:number]
    #best_vals = sorted(word_scores.items(), key=lambda (w, s): s, reverse=True)[:number]
    best_words = set([w for w, s in best_vals])  #lambda kv: (-kv[1], kv[0])
    return best_words

#creates feature selection mechanism that only uses best words
def best_word_features(words):
    return dict([(word, True) for word in words if word in best_words])

#numbers of features to select
numbers_to_test = [10, 100, 1000, 10000, 15000]
#tries the best_word_features mechanism with each of the numbers_to_test of features
for num in numbers_to_test:
    print ('evaluating best %d word features' % (num) )
    best_words = find_best_words(word_scores, num)
    evaluate_features(best_word_features)
    
#################################################################################################
    
print ("I'm here")
