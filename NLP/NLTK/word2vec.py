from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
warnings.filterwarnings(action = 'ignore')
import gensim

# Reads ‘alice.txt’ file
sample = open("/home/mbahng/Desktop/Cookbook/NLP/NLTK/alice.txt")
f = sample.read().replace("\n", " ")

data = []
for i in sent_tokenize(f):
	temp = []
	
	# tokenize the sentence into words
	for j in word_tokenize(i):
		temp.append(j.lower())

	data.append(temp)

# Create CBOW model
model1 = gensim.models.Word2Vec(data, min_count = 1,
							vector_size = 100, window = 5)

print(model1.wv["alice"])

# Print results
print("Cosine similarity between 'alice' " +
			"and 'wonderland' - CBOW : ",
	model1.wv.similarity('alice', 'wonderland'))
	
print("Cosine similarity between 'alice' " +
				"and 'machines' - CBOW : ",
	model1.wv.similarity('alice', 'machines'))

# Create Skip Gram model
model2 = gensim.models.Word2Vec(data, min_count = 1, vector_size = 100,
											window = 5, sg = 1)

# Print results
print("Cosine similarity between 'alice' " +
		"and 'wonderland' - Skip Gram : ",
	model2.wv.similarity('alice', 'wonderland'))
	
print("Cosine similarity between 'alice' " +
			"and 'machines' - Skip Gram : ",
	model2.wv.similarity('alice', 'machines'))
