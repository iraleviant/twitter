import nltk
from nltk.tag.stanford import StanfordNERTagger
import os
import re

def main():
    
    loc_words=['minetta tavern','cornerstone tavern','balthazar restaurant', "rolf's german restaurant",'presidential pizza',
  "ollie's sichuan restaurant",'bar boulud','hibernia bar','bar boulud','museum modern art','columbia university','hearst tower',
  'freedom tower','bank america tower','times square tower','trump tower','st. patricks cathedral','jane hotel', 
  'bowery hotel', 'palace hotel', 'tribecagrand hotel', 'soho grand hotel', 'plaza hotel', "maxwell's bar", 'stern school']
    
    #locs_for_test1=['trump tower','balthazar restaurant', 'cornerstone tavern', 'bowery hotel',
    #                    'jane hotel','stern school', 'bank america tower'] #959
    
    #locs_for_test2=['museum modern art'] #1335, fails in both cases
    
    #locs_for_test3=['columbia university','freedom tower','soho grand hotel'] #1029
    
    #locs_for_test4=['trump tower','bowery hotel',"rolf's german restaurant",'balthazar restaurant', 'stern school',
    #                'cornerstone tavern', "maxwell's bar", 'palace hotel'] #946
    
    locs_for_test5=['jane hotel','columbia university' 'minetta tavern', 'stern school', 
                    'bank america tower', 'palace hotel', 'freedom tower', "rolf's german restaurant",
                    "ollie's sichuan restaurant", 'bar boulud', 'hibernia bar','st. patricks cathedral',
                    'presidential pizza'] #961
    
    #########################################################
    tagged_file="all_data_tagged.txt"
    
    with open(tagged_file, "rb") as f:
        locSentences = f.readlines()
    
    output_train="train_file_tagged5.txt"
    output_test="test_file_tagged5.txt"
    
    count_test=0
    no_loc_count_train=0
    with open(output_train, 'w') as fa, open(output_test, 'w') as fb:
        for twt in locSentences:       
            twts=(twt.decode("utf-8")).strip()  # convert bytes to strings
            loc_index=twts.rfind('::::::::::::::::::::::')
            if loc_index==-1: #no lacation in the twiit
                if no_loc_count_train<3330:  #no location in the twit, 3300 precomputed number
                    fa.write(twts)
                    fa.write("\n")
                    no_loc_count_train+=1
                else:
                    fb.write(twts)
                    fb.write("\n")
            else:
                if bool(re.search(r"(?=("+'|'.join(locs_for_test5)+r"))",twts)):
                    count_test+=1
                    fb.write(twts)
                    fb.write("\n")
                else:
                    fa.write(twts)
                    fa.write("\n")
                    
    print("The number of twiits with location in test: ", count_test)
                     
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
