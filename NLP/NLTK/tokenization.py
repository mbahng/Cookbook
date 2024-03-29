import nltk 
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import gensim 
from syllables import estimate 
import string

example_string = """Natural Language Processing (NLP) is an interdisciplinary field that empowers machines to understand, interpret, and generate human language. Its applications span across various domains, including chatbots, language translation, sentiment analysis, and information extraction. We're going to rock'n'roll in the long-term. """

print(word_tokenize(example_string))

