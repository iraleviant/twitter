import csv
import re
from haversine import haversine
from difflib import SequenceMatcher



def jaccard_distance(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    #print(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return round(float(intersection / float(union)),3)


def extract_GPS(tweet):
    #gps=(-70.2, 40.77)
    first=tweet.index('\t')
    last=tweet.rindex('\t')
    subtweet=tweet[first+1:last] #check
    gps=subtweet.split('\t')
    gps_tuple=(float(gps[0]), float(gps[1]))
    return  gps_tuple#returns tuple with two elements

def main():
    
    file_to_read='/home/ira/Dropbox/crowdPlanning/manhattanClean.csv'
    
    #loc_words=['new york', 'ny', 'city', 'center', 'street', 'st', 'hotel', 'museum', 'park']

    loc_words=['museum modern art']#['columbia university']#['hearst tower']#['freedom tower']#['bank america tower']#['times square tower'] #['trump tower']    
    #loc_words=['st. patricks cathedral','jane hotel', 'bowery hotel', 'palace hotel', 'tribecagrand hotel', 'soho grand hotel', 'plaza hotel', "maxwell's bar", 'stern school']
    
    #below words for finding tweets that do not contain spesific location
    #loc_words=['restaurant', 'church', 'hotel', 'park', 'museum', 'school', 'street', 'st', 'tower', 'lounge', 'square', 'cafe','@', "i'm", 'theatre', 'university', 'site', 'center']
    
    #===========================================================================
    # lyon = (45.7597, 4.8422)
    # paris = (48.8567, 2.3508)
    # d=haversine(lyon, paris) #result in kilometers
    #===========================================================================
    
    #initial_tweet='dinner @denbanaag! @ park avenue tavern\t40.75027309\t-73.97886784\t1382585133'
    #initial_tweet="i'm saks fifth avenue - @saks (new york", ' ny)\t40.75801773\t-73.97692323\t1363180094'
    #initial_tweet="i'm spitzer's corner w/ @notbillsaturday\t40.72006791\t-73.98831323\t1386384896"
    #initial_tweet="i'm apple store", ' fifth avenue (new york', ' ny) w/ 40 others\t40.76382254\t-73.97301\t1379685986'
    initial_tweet="while avenue', ' i should go around block 1oak ask my mac charger i left behind.\t40.82526229\t-73.95330789\t1390095945"
    initial_tweet=''.join(initial_tweet) #make sure the tweet is a string
    
    initial_gps=extract_GPS(initial_tweet)
    
    text_file = open("contain_location_tweets1.txt", "w")
    close_tweets=[]
    n_loc=0
    n_tweets=0
    with open(file_to_read, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            n_tweets+=1
            str1 = ''.join(row)
            if bool(re.search(r"(?=("+'|'.join(loc_words)+r"))",str1)) : #replace with findall to find all the locations
                n_loc+=1
                print str1
                text_file.write(str1)
                text_file.write('\n')
            #===================================================================
            # cnt=0
            # for loc in loc_words:
            #     if loc not in str1:
            #         cnt+=1
            # if cnt==len(loc_words):
            #     n_loc+=1
            #     print str1
            #     #print str1
            #     #text_file.write(str1)
            #     #text_file.write('\n')
            #     if n_loc==4000:
            #         break
            #===================================================================
            #gps=extract_GPS(str1)
            #dis=haversine(initial_gps, gps )
            #if dis<=0.02: #tweets whithin 20 meters from the initial tweet, 50 meters= 151 tweets; 20meters= 44 tweets/948066 total; saks: 325/total, corner=212, apple=1566/total
            #    close_tweets.append(str1)
    
    
    text_file.close()

    
    ######### compute jaccard measure to find textually most similar tweets
    jac_dis=[]
    first=initial_tweet.index('\t')
    initial_tweet=initial_tweet[0:first]
    initial_tweet_list=initial_tweet.split(" ")
    for twt in close_tweets:
        first_t=twt.index('\t')
        twt=twt[0:first]
        twt_list=twt.split(" ")
        jac_dis.append(jaccard_distance(twt_list,initial_tweet_list))
    
    
    ############ find common substring from the most similar tweets
    sorted_jac_dis_indexes=[b[0] for b in sorted(enumerate(jac_dis),key=lambda i:-i[1])] #decreasing order, the first one with dis=1 is the tweet itself
    #find the common substring with the top 5 similar tweets  sorted_jac_dis_indexes[1:6]
    common_sub_string=[]
    string1=initial_tweet
    for ind in sorted_jac_dis_indexes[1:6]:
        str2=close_tweets[ind]
        first=str2.index('\t')
        string2=str2[0:first]
        match = SequenceMatcher(None, string1, string2).find_longest_match(0, len(string1), 0, len(string2))
        common_sub_string.append(string1[match.a: match.a + match.size]) # in all I got ' park avenue tavern'; in part i got:'i'm saks fifth avenue - @saks (new york ny)'
        # and in others 'saks fifth avenue'; 'i'm spitzer's corner w/ @' and one 'i'm spitzer's corner'; all= i'm apple store fifth avenue (new york ny) w/
    
    print " i'm here"
    
    
    
#################################################################     
########################

if __name__ == '__main__':
    main()
