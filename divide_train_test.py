import nltk
from nltk.tag.stanford import StanfordNERTagger
import os
import re

def main():
    
    locs_for_test1=['trump tower','balthazar restaurant', 'cornerstone tavern', 'bowery hotel',
                        'jane hotel','stern school', 'bank america tower']
    #########################################################
    tagged_file="all_data_tagged.txt"
    
    with open(tagged_file, "rb") as f:
        locSentences = f.readlines()
    
    output_train="train_file_tagged1.txt"
    output_test="test_file_tagged1.txt"
    
    #no_loc_count_test=0
    no_loc_count_train=0
    with open(output_train, 'w') as fa, open(output_test, 'w') as fb:
        for twt in locSentences:       
            twts=(twt.decode("utf-8")).strip()  # convert bytes to strings
            loc_index=twts.rfind('::::::::::::::::::::::')
            if loc_index==-1: #no lacation in the twiit
                if no_loc_count_train<3300:  #no location in the twit, 3300 precomputed number
                    fa.write(twts)
                    fa.write("\n")
                    no_loc_count_train+=1
                else:
                    fb.write(twts)
                    fb.write("\n")
            else:
                if bool(re.search(r"(?=("+'|'.join(locs_for_test1)+r"))",twts)):
                    fb.write(twts)
                    fb.write("\n")
                else:
                    fa.write(twts)
                    fa.write("\n")
                    
                    
    #########################################
    ## divide the train to pos_train and neg_train
    ## divide the test to pos_test and neg_test
    #######################################################3
    NEG_FILE_TRAIN = '/home/ira/Dropbox/twitter/no_location_for_train.txt'
    NEG_FILE_TEST = '/home/ira/Dropbox/twitter/no_location_for_test.txt'
    POS_FILE_TRAIN ='/home/ira/Dropbox/twitter/train_with_location.txt'
    POS_FILE_TEST ='/home/ira/Dropbox/twitter/test_with_location.txt'
    
    with open(output_train, "rb") as f:
        trainSentences = f.readlines()
    
    with open(NEG_FILE_TRAIN, 'w') as fa, open(POS_FILE_TRAIN, 'w') as fb:
        for twt in trainSentences:       
            twts=(twt.decode("utf-8")).strip()  # convert bytes to strings
            loc_index=twts.rfind('::::::::::::::::::::::')
            if loc_index==-1: #no lםcation in the twiit
                fa.write(twts)
                fa.write("\n")
            else:
                fb.write(twts)
                fb.write("\n")
                
    ######################################################
    ####                test                 #####
    ###########################################################
    with open(output_test, "rb") as f2:
        testSentences = f2.readlines()
    
    with open(NEG_FILE_TEST, 'w') as a, open(POS_FILE_TEST, 'w') as b:
        for twt in testSentences:       
            twts=(twt.decode("utf-8")).strip()  # convert bytes to strings
            loc_index=twts.rfind('::::::::::::::::::::::')
            if loc_index==-1: #no lacation in the twiit
                a.write(twts)
                a.write("\n")
            else:
                b.write(twts)
                b.write("\n")
                
    print ("I'm here")
    
    





#################################################################     
########################

if __name__ == '__main__':
    main()
