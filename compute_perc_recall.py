import nltk
from nltk.tag import StanfordNERTagger
from nltk.metrics.scores import accuracy
from nltk.metrics.scores import precision
from nltk.metrics.scores import recall
from nltk.metrics.scores import f_measure
from collections import Counter
import re

jar = '/home/ira/Dropbox/twitter/stanford-ner/stanford-ner.jar' 
model = '/home/ira/Dropbox/twitter/stanford-ner/dummy-ner-model-twitter.ser.gz'  # the trained model on twitter train

ner_tagger = StanfordNERTagger(model, jar, encoding='utf8')

test_file="/home/ira/Dropbox/twitter/test_all_dup.txt"
with open(test_file, "rb") as f:
    Sentences = f.readlines()

tp=0
fn=0
fp=0

for twt in Sentences: 
    twts=(twt.decode("utf-8")).strip()
    loc_index=twts.rfind('::::::::::::::::::::::')
    if loc_index==-1:  #no location in the twit
        location=""
        twit=twts
    else:
        location=twts[loc_index+22:]
        twit=twts[:twts.find(':')]
    words = nltk.word_tokenize(twit)
    #words=re.findall(r"[\w']+|[.,!?;@#]", str(twit).rstrip())
    tagged=ner_tagger.tag(words) 
    loc_test_list=[t[0] for t in tagged if t[1] == "LOCATION"]
    loc_test=' '.join(loc_test_list)
    #if len(loc_test_list)!=0 and len(location)!=0:
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
print(percision_m)
print(recall_m)
       
    #         twts=(twt.decode("utf-8")).strip()  # convert bytes to strings
    #         loc_index=twts.rfind(':')
    #         location=twts[loc_index+1:]
    #         loc_list=location.split()
    #         loc_tagged=''
    #         for l in loc_list:
    #             loc_tagged=loc_tagged+l+' LOCATION '
    #         new_twt=twts.replace(location, loc_tagged, 1) #replaces only the first appearence
    #         ofh.write(new_twt)
    #         ofh.write('\n')





#===============================================================================
# #raw_annotations = open("/home/ira/Dropbox/twitter/test_per.txt").read()
# raw_annotations = open("/home/ira/Dropbox/twitter/stanford-ner/train/twitter_corpus_test.txt").read()
# 
# split_annotations = raw_annotations.split()
# 
# 
# #test_sentences=open("/home/ira/Dropbox/twitter/test_all_non_tagged.txt").read()
#     #with open(test_file, "rb") as f:
#     #    locSentences = f.readlines()
#         
# #words = nltk.word_tokenize(test_sentences)
# #tagged_words=ner_tagger.tag(words)   #len=26,338
# 
# # Amend class annotations to reflect Stanford's NERTagger
# #===============================================================================
# # for n,i in enumerate(split_annotations):
# #     if i == "I-PER":
# #         split_annotations[n] = "PERSON"
# #     if i == "I-ORG":
# #         split_annotations[n] = "ORGANIZATION"
# #     if i == "I-LOC":
# #         split_annotations[n] = "LOCATION"
# #===============================================================================
# 
# # Group NE data into tuples
# def group(lst, n):
#   for i in range(0, len(lst), n):
#     if lst[i]=="EOS":
#         continue
#     val = lst[i:i+n]
#     if len(val) == n:
#       yield tuple(val)
# 
# reference_annotations = list(group(split_annotations, 2))
# 
# #test_file="/home/ira/Dropbox/twitter/test_all_non_tagged.txt"
# 
# #with open(test_file, 'rb') as ofh:
# #    twitts = ofh.readlines()    
# loc_ref_list=[(t[0],t[1]) for t in reference_annotations if t[1] == "LOCATION"]
# 
#     
# pure_tokens = split_annotations[::2]
# tagged_words=ner_tagger.tag(pure_tokens) 
# 
# #tp=set(loc_ref_list) & set(tagged_words)  #num of exact match
# ca = Counter(loc_ref_list)
# cb = Counter(tagged_words)
# 
# intersection = ca & cb #tp
# #tagged_words = nltk.pos_tag(pure_tokens)
# #nltk_unformatted_prediction = nltk.ne_chunk(tagged_words)
# 
# #Convert prediction to multiline string and then to list (includes pos tags)
# #===============================================================================
# # multiline_string = nltk.chunk.tree2conllstr(nltk_unformatted_prediction)
# # listed_pos_and_ne = multiline_string.split()
# # 
# # # Delete pos tags and rename
# # del listed_pos_and_ne[1::3]
# # listed_ne = listed_pos_and_ne
# # 
# # # Amend class annotations for consistency with reference_annotations
# # for n,i in enumerate(listed_ne):
# #     if i == "B-PERSON":
# #         listed_ne[n] = "PERSON"
# #     if i == "I-PERSON":
# #         listed_ne[n] = "PERSON"    
# #     if i == "B-ORGANIZATION":
# #         listed_ne[n] = "ORGANIZATION"
# #     if i == "I-ORGANIZATION":
# #         listed_ne[n] = "ORGANIZATION"
# #     if i == "B-LOCATION":
# #         listed_ne[n] = "LOCATION"
# #     if i == "I-LOCATION":
# #         listed_ne[n] = "LOCATION"
# #     if i == "B-GPE":
# #         listed_ne[n] = "LOCATION"
# #     if i == "I-GPE":
# #         listed_ne[n] = "LOCATION"
# # 
# # # Group prediction into tuples
# # nltk_formatted_prediction = list(group(listed_ne, 2))
# #===============================================================================
# 
# nltk_accuracy = accuracy(reference_annotations, tagged_words) #19/25
# nltk_precision = precision(set(reference_annotations), set(tagged_words)) #reference and test should be sets
# nltk_recall = recall(set(reference_annotations), set(tagged_words)) #reference and test should be sets
# nltk_f_measure = f_measure(set(reference_annotations), set(tagged_words))  #reference and test should be sets
# 
# print(nltk_accuracy)
# print(nltk_precision)
# print(nltk_recall)
# print(nltk_f_measure)
#===============================================================================
