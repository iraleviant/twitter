import csv
import re
import nltk
#from haversine import haversine
#from difflib import SequenceMatcher

def main():
    ############################################################################
    ###########        PART ONE      ###########################################
    ############################################################################
     
    test_file="/home/ira/Dropbox/twitter/train_file_tagged5.txt"    
    #test_file="/home/ira/Dropbox/twitter/test_file_tagged5.txt"

    
    with open(test_file, "rb") as f:
        locSentences = f.readlines()
       
    output_file="train_with_LOCATION5.txt"
    #output_file="test_with_LOCATION5.txt"
    with open(output_file, 'w') as ofh:
        for twt in locSentences:       
            twts=(twt.decode("utf-8")).strip()  # convert bytes to strings
            loc_index=twts.rfind('::::::::::::::::::::::')
            if loc_index!=-1:
                location=twts[loc_index+22:]
                loc_list=location.split()
                loc_tagged=''
                for l in loc_list:
                    loc_tagged=loc_tagged+l+' LOCATION '
                new_twt=twts.replace(location, loc_tagged, 1) #replaces only the first appearence
                ofh.write(new_twt)
                ofh.write('\n')
            else:
                ofh.write(twts)
                ofh.write('\n')
          
     
    #############################################################################
    ############        PART TWO     ###########################################
    #############################################################################
    #input_file="/home/ira/Dropbox/twitter/test_all.txt"
    with open(output_file, 'rb') as ofh:
        tagSentences = ofh.readlines()
        
    corpus_file='twitter_corpus_train5.tsv'
    #corpus_file='twitter_corpus_test5.tsv'
    with open(corpus_file, 'w') as h:
        for twt in tagSentences:
            twts=(twt.decode("utf-8")).strip()  # convert bytes to strings
            loc_index=twts.find(':')
            ntwt=twts[0:loc_index]
            twt_list=ntwt.split()
            #twt_list=nltk.word_tokenize(ntwt)
            for i, word in enumerate(twt_list):
                if i==len(twt_list)-1: #if we reached the last word in the tweet
                    h.write(word)
                    h.write("\t") #tab space
                    h.write("O")
                    h.write("\n")
                    #h.write("EOS") #adding end of twit
                    #h.write("\t") #tab space
                    #h.write("O")
                    #h.write("\n")
                else:  # not in the end
                    if word=="LOCATION":
                        continue
                    if twt_list[i+1]=="LOCATION":  # if next word is LOACION
                        h.write(word)
                        h.write("\t") #tab space
                        h.write("LOCATION")
                        h.write("\n")
                    else:
                        h.write(word)
                        h.write("\t") #tab space
                        h.write("O")
                        h.write("\n")
                          
    print (" i'm here")
    
    
    
#################################################################     
########################

if __name__ == '__main__':
    main()
