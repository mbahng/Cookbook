from nltk.tokenize import sent_tokenize, word_tokenize
import random, time 

data = open("/home/mbahng/Desktop/Cookbook/NLP/NLTK/alice.txt")

f = data.read().lower().replace("\n", " ") 

dict = {} 

tokens = word_tokenize(f)

for i in range(len(tokens)-1): 
    if tokens[i] not in dict.keys(): 
        dict[tokens[i]] = [] 
    
    dict[tokens[i]].append(tokens[i+1])
    
phrase = "i"
word = "i" 

for i in range(100): 
    time.sleep(0.1)
    print(word, end = " ")
    word = random.choice(dict[word])

