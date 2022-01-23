import torch
import torch.nn as nn
import torchvision.models as models
from data_loader import get_loader

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        ## TODO: define the LSTM
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers,
                            batch_first=True)
        
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
          


    def forward(self, features, captions):
        
        # the first value returned by LSTM is all of the hidden states throughout
        # the sequence. the second is just the most recent hidden state
        embeds = self.embedding(captions[:, :-1])
        inputs = torch.cat((features.view(len(features), 1, -1), embeds), dim=1)
        out, hidden = self.lstm(inputs)
        
        ## TODO: put x through the fully-connected layer
        out = self.fc(out)
        
        return out

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        output = []
        
        states = (torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device),
                  torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device))
        
        for i in range(max_len):
            
            out, states = self.lstm(inputs, states)
            out = self.fc(out)
            _, out = out.max(2)
            output.append(out.item())
            
            inputs = self.embedding(out)
        

        return output
        
        