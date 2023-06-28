import torch 
import torch.nn as nn 

input_features = 10
hidden_features = 20
num_layers = 2
sequence_length = 5

rnn = nn.RNN(input_features, hidden_features, num_layers)
input = torch.randn(sequence_length, input_features)
h0 = torch.randn(num_layers, hidden_features)
print(input.size(), h0.size()) 

print([weight.data.size() for weights in rnn.all_weights for weight in weights])
        
output, hn = rnn(input, h0) 
print(output.size(), hn.size())