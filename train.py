import torch
import torch.nn as nn
import math
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
from data_loader import get_loader
from model import EncoderCNN, DecoderLSTM


image_ids = os.listdir(args.image_dir)
## lower all the sentences

with open(args.text_path, 'r') as f:
    w = f.readlines()
w_dict = {}
for l in w:
if l.split('#')[0] in list(w_dict.keys()):
    w_dict[l.split('#')[0]].append(l.split('#')[1][3:-4])
else:
    w_dict[l.split('#')[0]] = [l.split('#')[1][3:-4]]
image_ids = [ids for ids in image_ids if ids in list(w_dict.keys())]
valid_size = 0.2
train_ids = image_ids[:math.floor(len(image_ids)*valid_size)]
valid_ids = image_ids[math.floor(len(image_ids)*valid_size):]

with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vocab_size = len(vocab)

# Image preprocessing, normalization for the pretrained resnet
transform = transforms.Compose([ 
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(), 
    transforms.ToTensor(), 
    transforms.Normalize((0.485, 0.456, 0.406), 
                          (0.229, 0.224, 0.225))])
    

# Build data loader
train_dataloader = get_loader(args.image_dir, vocab,  train_ids,w_dict, args.batch_size, True,1, transform = transform) 
valid_dataloader = get_loader(args.image_dir, vocab,  valid_ids,w_dict, args.batch_size, True,1, transform = transform)
# Build the models
encoder = EncoderCNN(embed_size).to(device)
decoder = DecoderLSTM(len(vocab), args.embed_size, args.hidden_size, num_layers).to(device)
    
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
# Train the models
#total_step = len(dataloader)
for epoch in range(args.num_epochs):
  encoder.train()
  decoder.train()
  train_loss = 0
  valid_loss = 0
  for i, (images, captions, lengths) in enumerate(train_dataloader):

    # Set mini-batch dataset
    images = images.to(device)
    captions = captions.to(device)
    targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

    # Forward, backward and optimize
    features = encoder(images)
    outputs = decoder(features, captions, lengths)
    loss = criterion(outputs, targets)
    train_loss += loss.item()
    decoder.zero_grad()
    encoder.zero_grad()
    loss.backward()
    optimizer.step()
    ## validation
  encoder.eval()
  decoder.eval()
  for j, (images_, captions_, lengths_) in enumerate(valid_dataloader):

    # Set mini-batch dataset
    images_ = images_.to(device)
    captions_ = captions_.to(device)
    targets_ = pack_padded_sequence(captions_, lengths_, batch_first=True)[0]

    # Forward, backward and optimize
    features_ = encoder(images_)
    outputs_ = decoder(features_, captions_, lengths_)
    valid_los = criterion(outputs_, targets_)
    valid_loss += valid_los.item()


    # Print log info
   # if i % log_step == 0:
  print('Epoch [{}/{}], train_Perplexity: {:5.4f}, valid_perplexity: {:5.4f}'
              .format(epoch, num_epochs, np.exp(train_loss/i), np.exp(valid_loss/j)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='data/resized2014', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='data/annotations/captions_train2014.json',
                        help='path for train annotation json file')
    parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=256, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=3, help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)

