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
from itertools import islice

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
test_file="/home/ira/Dropbox/twitter/test_file_tagged5.txt"

###Fixed
ALL_POS_FILE ='/home/ira/Dropbox/twitter/with_location_all.txt'
ALL_NEG_FILE ='/home/ira/Dropbox/twitter/no_location_all.txt'
#test_file= "/home/ira/Dropbox/twitter/test_all_non_tagged.txt" #not sure if neccesary


#########  build a dictionary, key is a tweet and the value is the list of geographical closest tweets
Close_Twt_Dic={}

with open(test_file, 'rb') as ofh:
    twitts = ofh.readlines()    

for twt in twitts:
    twts=(twt.decode("utf-8")).strip()  # convert bytes to strings
    loc_index=twts.rfind('::::::::::::::::::::::')
    if loc_index==-1:  #no location in the twit
        twtn=twts
    else:
        twtn=twts[:twts.find('::::::::::::::::::::::')]
    gps1=extract_GPS(twtn) 
    close_tweets=[]
    twitts_list=twitts[:]
    twitts_list.remove(twt)
    for twt2 in twitts_list:
        twts2=(twt2.decode("utf-8")).strip()
        loc_index=twts2.rfind('::::::::::::::::::::::')
        if loc_index==-1:  #no location in the twit
            twtn2=twts2
        else:
            twtn2=twts2[:twts2.find('::::::::::::::::::::::')]
        gps2=extract_GPS(twtn2)
        dis=haversine(gps1, gps2 )
        if dis<=0.04: #tweets whithin 50 meters from the initial tweet
            close_tweets.append(twtn2)   
    Close_Twt_Dic[twtn]=close_tweets

location_dic={}


for twt in twitts: 
    twts=(twt.decode("utf-8")).strip()
    loc_index=twts.rfind('::::::::::::::::::::::')
    if loc_index==-1:  #no location in the twit
        location=""
        twit=twts
    else:
        location=twts[loc_index+22:]
        twit=twts[:twts.find('::::::::::::::::::::::')]
    #twit=re.sub("\s*", " ", twit)
    twit="".join(twit.split())
    location_dic[twit]=location

def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result    
    for elem in it:
        result = result[1:] + (elem,)
        yield result

   
#########################################################################################################    
#this function takes a feature selection mechanism and returns its performance in a variety of metrics
Train_twit_Dic={}
Test_twit_Dic={}

def evaluate_features(fn, tp, fp,feature_select):
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
        i1=(i.decode("utf-8")).strip()
        loc_index=i1.rfind('::::::::::::::::::::::')
        if loc_index==-1:  #no location in the twit
            i1=i1
        else:
            i1=i1[:i1.find('::::::::::::::::::::::')]
        posWords = re.findall(r"[\w']+|[.,!?;@#]", str(i1).rstrip())
        posWords = [feature_select(posWords), 'pos'] #pos = contains location
        posFeatures_train.append(posWords)
        #str_i=(i.decode("utf-8")).strip()
        #Train_twit_Dic[frozenset(posWords[0].items())]=str_i
        
    for i in posSentences_test:
        i1=(i.decode("utf-8")).strip()
        loc_index=i1.rfind('::::::::::::::::::::::')
        if loc_index==-1:  #no location in the twit
            i1=i1
        else:
            i1=i1[:i1.find('::::::::::::::::::::::')]
        posWords_test = re.findall(r"[\w']+|[.,!?;@#]", str(i1).rstrip())
        posWords_test = [feature_select(posWords_test), 'pos'] #pos = contains location
        posFeatures_test.append(posWords_test)
        #str1=(i.decode("utf-8")).strip()
        Test_twit_Dic[frozenset(posWords_test[0].items())]=i1
    
    #with open(RT_POLARITY_NEG_FILE, 'r') as negSenBufferedReader: <_io.BufferedReader name='/home/ira/Dropbox/twitter/contain_location_tweets.txt'>tences:
    for i in negSentences_train:
        i1=(i.decode("utf-8")).strip()
        loc_index=i1.rfind('::::::::::::::::::::::')
        if loc_index==-1:  #no location in the twit
            i1=i1
        else:
            i1=i1[:i1.find('::::::::::::::::::::::')]
        negWords = re.findall(r"[\w']+|[.,!?;@#]", str(i1).rstrip())
        negWords = [feature_select(negWords), 'neg'] #neg = doesn't contain location 
        negFeatures_train.append(negWords)
        #str2=(i.decode("utf-8")).strip()
        #Train_twit_Dic[frozenset(negWords[0].items())]=str2

    for i in negSentences_test:
        i1=(i.decode("utf-8")).strip()
        loc_index=i1.rfind('::::::::::::::::::::::')
        if loc_index==-1:  #no location in the twit
            i1=i1
        else:
            i1=i1[:i1.find('::::::::::::::::::::::')]
        negWords_test = re.findall(r"[\w']+|[.,!?;@#]", str(i1).rstrip())
        negWords_test = [feature_select(negWords_test), 'neg'] #neg = doesn't contain location 
        negFeatures_test.append(negWords_test)
        #str3=(i1.decode("utf-8")).strip()
        Test_twit_Dic[frozenset(negWords_test[0].items())]=i1
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
    testSetsMy=  collections.defaultdict(set)
    
    #puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
    for i, (features, label) in enumerate(testFeatures):
        referenceSets[label].add(i)
        predicted = classifier.classify(features)
        testSets[predicted].add(i)    
####################   MINE   ####################################
        if predicted=="pos":  ##the twit according to the classifier contains a location
            twiit=Test_twit_Dic[frozenset(features.items())]
            #tw=twiit.replace("\t", " ")
            twitn="".join(twiit.split())
            #twitn=re.sub("\s\s+", " ", twitn)
            if twitn in location_dic.keys():
                location=location_dic[twitn]
                #print ("NO ERR")
            else:
                print (" Error", twitn, "TWIIT", twiit)
                continue
            list_close_twits=Close_Twt_Dic[twiit]
            list_close_twits.append(twiit)

            minimum_size = 2
            counts = Counter() 
            for s in list_close_twits:
                words = nltk.word_tokenize(s)
                counts.update(' '.join(sp) 
                              for size in range(minimum_size, len(words) + 1)
                              for sp in window(words, size)          )
            #print( counts.most_common(5))
            loc_test=counts.most_common(1)[0][0]  #[('modern art', 1306), ('museum modern', 1302), ('museum modern art', 1302), ('art (', 1299), ('modern art (', 1297)]

            del counts
            
            if len(loc_test)==0 and len(location)==0:
                continue
            elif loc_test==location and len(location)>=2:
                tp=tp+1
            elif loc_test!=location and len(loc_test)==0 and len(location)!=0:    #the classifier didnt label anything
                fn=fn+1 #in the ground truth labeled but the classifier didnt label
            elif loc_test!=location and len(loc_test)!=0 and len(location)!=0: # the classifier didn't label correctly (fn) and even more labeled something incorrectly (fp)
                fn=fn+1
                fp=fp+1   
    
    percision_m=tp / float(tp+fp)
    recall_m= tp / float(tp+fn)
    print ("the number of tp is:", tp)
    print ("the number of fp is:", fp)
    print ("the number of fn is:", fn)
    print("The percision is:", percision_m)
    print("The recall is:",recall_m)
    print ("im here")
    
     
                #print ('{:<20} {:>3d} times'.format(substring, count) )
            #===================================================================
            #An attempt to employ the stanford ner on close twwits to help-- but it didint work well on the close twiits as well
            # tagged_words=ner_tagger.tag(words)
            # lbl=""
            # for tag_w in tagged_words:
            #     if tag_w[1]=="LOCATION":
            #         lbl=lbl+tag_w[0]+" "  #found a label for our twiit
            #     #final_lbl=lbl
            # ### employ satnford trained classifier on all of the close twiits to find 
            # #if lbl=="":  #couldnt find the location (lable) for our twiit, lets try to find it wihitn its physical neiborhood twwits
            # lbl_list=[lbl]
            # for s in list_close_twits:
            #     words = nltk.word_tokenize(s)
            #     tagged_words=ner_tagger.tag(words)
            #     lbl=""
            #     for tag_w in tagged_words:
            #         if tag_w[1]=="LOCATION":
            #             lbl=lbl+tag_w[0]+" "
            #     if lbl!="": 
            #         lbl_list.append(lbl)        
            # ## find most common str (label) in lbl_list
            # if len(lbl_list)>=1:
            #     c = Counter(lbl_list)
            #     final_lbl=c.most_common(1)[0][0]
            # else:
            #     final_lbl=""
            #===================================================================
                
                
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
tp=0
fp=0
fn=0
evaluate_features(tp, fp, fn, make_full_dict)

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
    fn=0 
    tp=0 
    fp=0
    evaluate_features(fn, tp, fp,best_word_features)
    
#################################################################################################
    
print ("I'm here")
