import numpy as np


import nltk

from nltk import word_tokenize
from nltk import pos_tag

from nltk.corpus import stopwords
nltk.download('stopwords')


import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB


all_words = []
clf = 0


def calcSentiment_train(trainFile):
    import json
    import re
    
    # list to store all reviews
    data = []
    
    # collect data from json file
    with open(trainFile, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    # set the stopwords criteria
    stops = set(stopwords.words('english'))
    
    # coprora for true (positive) reviews and false (negative) reviews
    corpus_t = []
    corpus_f = []
    
    # lists to store word frequencies of each corpus
    words_freq_t = []
    words_freq_f = []
    
    # for loop to process all positive reviews (first 401 reviews in the training file)
    for i in range(0,401):
        # text stores the review part only, without the sentiment
        # some text processing to clean up the text
        text = data[i]['review']
        text = re.sub("_"," ",text)
        text = re.sub(" +"," ",text)
        text = re.sub("'re"," are",text)
        text = re.sub("n't"," not",text)
        text = re.sub("'ll"," will",text)
        
        # create tokens of the current review being processed
        stopped_text = text.split()
        
        # remove stopwords
        for w in stopped_text:
            if w in stops:
                stopped_text.remove(w)
        
        space = " "
        
        # regenerate text without the stop words
        clean_text = space.join(stopped_text)
        
        # create pos tags
        tags = pos_tag(word_tokenize(text))
        
        # list to store relevant tags
        rele = []
        
        # for loop that only keeps words tagged as adjectives or adverbs; these describe things so they are most useful 
        for tag in tags:
            if tag[1] in ['JJ','JJR','JJS','RB','RBR','RBS']:
                rele.append(tag[0])
        
        # regenerate review using only the adjectives and adverbs selected
        rele_text = space.join(rele)
        
        # add this processed text to the corpus of positive reviews
        corpus_t.append(rele_text)
    
    # process the negative reviews (last 401 reviews in training file) in the exact same way
    for i in range(401,802):
        text = data[i]['review']
        text = re.sub("_"," ",text)
        text = re.sub(" +"," ",text)
        text = re.sub("'re","are",text)
        text = re.sub("n't","not",text)
        text = re.sub("'ll","will",text)
        
        stopped_text = text.split()
        
        for w in stopped_text:
            if w in stops:
                stopped_text.remove(w)
        
        space = " "
        
        clean_text = space.join(stopped_text)
        
        tags = pos_tag(word_tokenize(text))
        
        rele = []
        
        for tag in tags:
            if tag[1] in ['JJ','JJR','JJS','RB','RBR','RBS']:
                rele.append(tag[0])
        
        rele_text = space.join(rele)
        corpus_f.append(rele_text)
    
    # create bag of words and sum up word occurrences to find most frequent words in positive reviews
    vec_t = CountVectorizer().fit(corpus_t)
    bag_of_words_t = vec_t.transform(corpus_t)
    sum_words_t = bag_of_words_t.sum(axis=0)
    # only keep words that have a count within a certain range(15-200)
    words_freq_t = [(word, sum_words_t[0, idx]) for word, idx in vec_t.vocabulary_.items() if sum_words_t[0,idx] >= 15 and sum_words_t[0,idx] <=200]
    words_freq_t =sorted(words_freq_t, key = lambda x: x[1], reverse=True)
    
    # create bag of words and sum up word occurrences to find most frequent words in negative reviews
    vec_f = CountVectorizer().fit(corpus_f)
    bag_of_words_f = vec_f.transform(corpus_f)
    sum_words_f = bag_of_words_f.sum(axis=0)
    # only keep words that have a count within a certain range(15-200)
    words_freq_f = [(word, sum_words_f[0, idx]) for word, idx in vec_f.vocabulary_.items() if sum_words_f[0,idx] >= 15 and sum_words_f[0,idx] <=200]
    words_freq_f =sorted(words_freq_f, key = lambda x: x[1], reverse=True)
    
    # create list of all words filtered from positive and negative reviews
    just_words_t = [w[0] for w in words_freq_t]
    just_words_f = [w[0] for w in words_freq_f]

    # find words common to both positive and negative reviews
    common = []

    for w in just_words_t:
        if w in just_words_f:
            common.append(w)
    
    # remove words common to both classes, from their word lists
    for w in common:
        just_words_t.remove(w)
        just_words_f.remove(w)
    
    # combine these word lists to form a larger list of all those words
    global all_words
    all_words = just_words_t + just_words_f
    
    # add back some of the common word, if they occured more relatively more frequently in one class of reviews
    for w in common:
        tc = 0
        fc = 0

        for tw in words_freq_t:
            if w == tw[0]:
                tc = tw[1]

        for fw in words_freq_f:
            if w == fw[0]:
                fc = fw[1]
        
        # add word to the list of all words, if the difference in its occurences in the two classes is more than 20
        if abs(tc - fc) > 20:
            all_words.append(w)
    
    # lists to store training data and target labels/classes
    train = []
    target = []

    # iterate through all reviews
    for r in data:
        # list for vector representaton of a single review
        vec_rep = []
        
        # clean data as before
        text = r['review']
        text = re.sub("_"," ",text)
        text = re.sub(" +"," ",text)
        text = re.sub("'re"," are",text)
        text = re.sub("n't"," not",text)
        text = re.sub("'ll"," will",text)
        
        # check if review contains any of the words in the list of all filtered words
        # if a word is present, append 1 to the vector representation, else append 0
        for w in all_words:
            if text.find(w) > -1:
                vec_rep.append(1)
            else:
                vec_rep.append(0)
        
        # add this vector representation to the list of vector representations for all the training examples
        train.append(vec_rep)
        
        # if the sentiment of the current review is positive append 1 to the target vector/list, else append 0
        if r['sentiment'] == True:
            target.append(1)
        else:
            target.append(0)
    
    # structure data for model training
    X = np.array(train)
    Y = np.array(target)
    
    # initialize and train the model
    global clf
    clf = GaussianNB()
    clf.fit(X,Y) 


# In[5]:


def calcSentiment_test(text):
    import re
    import numpy as np
    
    # list to store vector representation of the given sentence
    vec_rep = []

    # clean up the given sentence
    text = re.sub("_"," ",text)
    text = re.sub(" +"," ",text)
    text = re.sub("'re","are",text)
    text = re.sub("n't","not",text)
    text = re.sub("'ll","will",text)
    
    # check if review contains any of the words in the list of all filtered words
    # if a word is present, append 1 to the vector representation, else append 0
    for w in all_words:
        if text.find(w) > -1:
            vec_rep.append(1)
        else:
            vec_rep.append(0)
    
    # structure the vector for testing, and then predict a label using our model
    test = np.array(vec_rep)
    pred = clf.predict([test])
    
    # return the prediction
    if pred == 1:
        return True
    else:
        return False

