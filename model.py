import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

"""For the encoder part, the pretrained CNN extracts the feature vector from a given input image. The feature vector is linearly transformed to have the same dimension as the input dimension of the LSTM network. For the decoder part, source and target texts are predefined. For example, if the image description is "Giraffes standing next to each other", the source sequence is a list containing ['<start>', 'Giraffes', 'standing', 'next', 'to', 'each', 'other'] and the target sequence is a list containing ['Giraffes', 'standing', 'next', 'to', 'each', 'other', '<end>']. Using these source and target sequences and the feature vector, the LSTM decoder is trained as a language model conditioned on the feature vector."""

class EncoderCNN(nn.Module):
  def __init__(self, embed_size):
    super(EncoderCNN, self).__init__()
    resnet = models.resnet152(pretrained=True)
    modules = nn.Sequential(*list(resnet.children())[:-1])
    self.resnet = nn.Sequential(*modules)
    self.linear = nn.Linear(resnet.fc.in_features, embed_size)
    self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
  
  def forward(self, images):
    with torch.no_grad():
      features = self.resnet(images)
    features = features.reshape(features.size(0), -1)
    features = self.bn(self.linear(features))
    return features


class DecoderLSTM(nn.Module):
  def __init__(self, vocab_size, embed_size, hidden_size, num_layers, max_seq_length=20):
    super(DecoderLSTM, self).__init__()
    self.embed = nn.Embedding(vocab_size, embed_size)
    self.LSTM = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
    self.linear = nn.Linear(hidden_size, vocab_size)
    self.max_seq_length = max_seq_length
  
  def forward(self, features, captions, lengths):
    embeddings = self.embed(captions)
    embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
    packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
    hiddens, _ = self.LSTM(packed)
    outputs = self.linear(hiddens[0])
    return outputs
 
  def sample(self, features, states=None):
    """Generate captions for given image features using greedy search."""
    sampled_ids = []
    inputs = features.unsqueeze(1)
    for i in range(self.max_seq_length):
      hiddens, states = self.LSTM(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
      outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
      _, predicted = outputs.max(1)                        # predicted: (batch_size)
      sampled_ids.append(predicted)
      inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
      inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
    sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
    return sampled_ids
