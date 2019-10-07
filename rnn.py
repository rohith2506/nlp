'''
Implementation of Vanilla Recurrent Neural Network

Limitations
1) Overfitting
2) Lack of word embeddings
3) Error handling of missing data

@Author: Rohith Uppala
'''

import torch
from torch import nn
import numpy as np

def convert_from_char_to_int(text):
    chars = set(''.join(text))
    int2char = dict(enumerate(chars))
    char2int = { char : ind for ind, char in int2char.items() }
    return char2int, int2char

text = ['Hey How are you', 'good i am fine', 'have a nice day']

maxlen = len(max(text, key=len))
for i in range(len(text)):
    while len(text[i]) < maxlen:
        text[i] += ' '

char2int, int2char = convert_from_char_to_int(text)

input_seq, target_seq = [], []
for i in range(len(text)):
    input_seq.append(text[i][:-1])
    target_seq.append(text[i][1:])

for i in range(len(text)):
    input_seq[i] = [char2int[char] for char in input_seq[i]]
    target_seq[i] = [char2int[char] for char in target_seq[i]]


dict_size, seq_len, batch_size = len(char2int), maxlen - 1, len(text)


def one_hot_encode(sequence, dict_size, seq_len, batch_size):
    features = np.zeros((batch_size, seq_len, dict_size), dtype=np.float32)
    for i in range(batch_size):
        for j in range(seq_len):
            features[i, j, sequence[i][j]] = 1
    return features


input_seq = one_hot_encode(input_seq, dict_size, seq_len, batch_size)

input_seq = torch.from_numpy(input_seq)
target_seq = torch.Tensor(target_seq)


is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        out, hidden = self.rnn(x, hidden)
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden


model = Model(input_size=dict_size, output_size=dict_size, hidden_dim = 12, n_layers = 1)
model.to(device)


n_epochs, lr = 100, 0.01
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(n_epochs):
    optimizer.zero_grad()
    input_seq.to(device)
    output, hidden = model(input_seq)
    loss = criterion(output, target_seq.view(-1).long())
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print("Loss: ", loss.item())



def predict(model, character):
    character = np.array([[char2int[c] for c in character]])
    character = one_hot_encode(character, dict_size, character.shape[1], 1)
    character = torch.from_numpy(character)
    character.to(device)
    out, hidden = model(character)
    prob = nn.functional.softmax(out[-1], dim=0).data
    char_ind = torch.max(prob, dim=0)[1].item()
    return int2char[char_ind], hidden


def complete_sentence(model, out_len, start='hello'):
    model.eval()
    start = start.lower()
    chars = [ch for ch in start]
    size = out_len - len(chars)
    for ii in range(size):
        char, h = predict(model, chars)
        chars.append(char)
    return ''.join(chars)

input_sent = input()
output = complete_sentence(model, 15, input_sent)
print(output)
