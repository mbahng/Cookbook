import torch 
from torch import nn
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import sent_tokenize, word_tokenize
import random, time 
import numpy as np 
import gensim

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class AliceData(Dataset): 
    
    def __init__(self): 
        
        self.data = []
        
        text = open("/home/mbahng/Desktop/Cookbook/NLP/NLTK/alice.txt")
        f = text.read().lower().replace("\n", " ") 

        tokens = word_tokenize(f)
        
        self.class_map = {}
        class_num = 0

        for i in range(len(tokens)-2): 
            inpt = (tokens[i], tokens[i+1])
            otpt = tokens[i+2] 
            self.data.append([inpt, otpt])
            
            if otpt not in self.class_map.keys(): 
                self.class_map[otpt] = class_num
                class_num += 1
        
    def __len__(self): 
        return len(self.data) 
    
    def init_Word2Vec(self): 
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
        self.embedding = gensim.models.Word2Vec(data, min_count = 1,
                                    vector_size = 100, window = 5)
    
    def __getitem__(self, idx): 
        
        self.init_Word2Vec()
        
        prev_words, next_words = self.data[idx]
        
        word_tensor = np.hstack((self.embedding.wv[prev_words[0]], self.embedding.wv[prev_words[-1]]))
        word_tensor = torch.from_numpy(word_tensor)
        
        class_id = self.class_map[next_words]
        class_id = torch.tensor(class_id)
        return word_tensor, class_id 
        
        
dataset = AliceData() 
data_loader = DataLoader(dataset, batch_size=64, shuffle=True) 
    
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(200, 512),
            nn.ReLU(), 
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3512)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),     # which parameters to optimize
    lr=1e-3                 # learning rate 
)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)  # size of train dataset 
    correct = 0
    model.train()
    
    for X, y in dataloader:
        
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)
    
        loss.backward()
        optimizer.step()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        
    print(f"Training Loss: {loss.item():>7f}")
    print(f"Training Accuracy: {correct/size * 100:>0.1f}%")
    

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1} ---------------------------")
    train(data_loader, model, loss_fn, optimizer)
print("Done!")