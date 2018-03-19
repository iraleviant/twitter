import csv
import re
from haversine import haversine
from difflib import SequenceMatcher

def main():
    
    RT_POLARITY_POS_FILE = '/home/ira/Dropbox/crowdPlanning/manhattan/contain_location_tweets.txt'
    
    loc_words=[
        'minetta tavern','cornerstone tavern','balthazar restaurant', "rolf's german restaurant",'presidential pizza',
        "ollie's sichuan restaurant",'bar boulud','hibernia bar','bar boulud','museum modern art','columbia university','hearst tower',
        'freedom tower','bank america tower','times square tower','trump tower','st. patricks cathedral','jane hotel', 
        'bowery hotel', 'palace hotel', 'tribecagrand hotel', 'soho grand hotel', 'plaza hotel', "maxwell's bar", 'stern school'
        ]
    
    with open(RT_POLARITY_POS_FILE, "rb") as f:
        posSentences = f.read().split('\n')
    
    dic={}
    for twt in posSentences:
        for loc in loc_words:
            if loc in twt:
                dic[twt]=loc
                continue
            
    #write dic to file
    output_file="Tagged_twitts_with_location.txt"
    with open(output_file, 'w') as ofh:
        for k in dic:
            ofh.write(k)
            ofh.write('::::::::::::::::::::::')
            ofh.write(dic[k])
            ofh.write('\n')

    print " i'm here"
    
    
    
#################################################################     
########################

if __name__ == '__main__':
    main()
