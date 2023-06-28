import torch 
import torch.nn as nn 
import os 
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 

torch.manual_seed(0)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

root = os.path.join('/home/mbahng/Desktop/Cookbook/data')

training_data = datasets.CIFAR10(
    root=root,            
    train=True,            
    download=False,          
    transform=ToTensor()    
)
test_data = datasets.CIFAR10(
    root=root,
    train=False,
    download=False,
    transform=ToTensor()
)

training_data.data = training_data.data[:,:, :, :]
training_data.targets = training_data.targets[:]
test_data.data = test_data.data[:,:, :, :]
test_data.targets = test_data.targets[:]

train_dataloader = DataLoader(training_data,    # our dataset
                              batch_size=64,    # batch size
                              shuffle=True      # shuffling the data
                            )
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

for X, y in train_dataloader: 
    print(X)
    print(y)
    assert False

train_features, train_labels = next(iter(train_dataloader))

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(32*32*3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(), 
            nn.Linear(512, 10)
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

class LassoRegularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, model):
        lasso_reg = 0.0
        for param in model.parameters():
            lasso_reg += torch.norm(param, 1)
        return self.alpha * lasso_reg
    
lasso_reg = LassoRegularizer(alpha=1e-5)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)  # size of train dataset 
    correct = 0
    model.train()
    
    for X, y in dataloader:
        
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y) + lasso_reg(model)
    
        loss.backward()
        optimizer.step()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        
    print(f"Training Loss: {loss.item():>7f}")
    print(f"Training Accuracy: {correct/size * 100:>0.1f}%")
            
def test(dataloader, model, loss_fn):
    
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item() + lasso_reg(model).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg Test loss: {test_loss:>8f} \n")
    
    return (correct, test_loss) 


epochs = 100
for t in range(epochs):
    print(f"Epoch {t+1} ---------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")