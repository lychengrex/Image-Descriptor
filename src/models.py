import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        self.attention = nn.Linear(hidden_size+embed_size, 2048)
        self.attended = nn.Linear(2048+embed_size, embed_size)
        self.softmax = nn.Softmax()
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        self.attention.weight.data.uniform_(-0.1, 0.1)
        self.attention.bias.data.fill_(0)
        self.attended.weight.data.uniform_(-0.1, 0.1)
        self.attended.bias.data.fill_(0)
        
    # def forward(self, features, captions, lengths):
    def forward(self, features, cnn_features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        # packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        # hiddens, _ = self.lstm(packed)
        # outputs = self.linear(hiddens[0])
        # return outputs

        packed, batch_sizes = pack_padded_sequence(embeddings, lengths, batch_first=True)
        
        hiddenStates = None
        start = 0
        for batch_size in batch_sizes:
            in_vector = packed[start:start+batch_size].view(batch_size, 1, -1)
            start += batch_size
            if hiddenStates is None:
                hiddenStates, (h_n , c_n) = self.lstm(in_vector)
                hiddenStates = torch.squeeze(hiddenStates)
            else:
                h_n, c_n = h_n[:,0:batch_size,:], c_n[:,0:batch_size,:]
                info_vector = torch.cat((in_vector, h_n.view(batch_size, 1, -1)), dim=2)
                attention_weights = self.attention(info_vector.view(batch_size, -1))
                attention_weights = self.softmax(attention_weights)
                attended_weights = cnn_features[0:batch_size] * attention_weights
                attended_info_vector = torch.cat((in_vector.view(batch_size, -1), attended_weights), dim=1)
                attended_in_vector = self.attended(attended_info_vector)
                attended_in_vector = attended_in_vector.view(batch_size, 1, -1)
                out, (h_n , c_n) = self.lstm(attended_in_vector, (h_n, c_n))
                hiddenStates = torch.cat((hiddenStates, out.view(batch_size, -1)))
        hiddenStates = self.linear(hiddenStates)
        return hiddenStates
    
    # def sample(self, features, states=None):
    def sample(self, features, cnn_features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)

        batch_size = features.size(0)

        for i in range(self.max_seg_length):
            if states is None:
                hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            else:
                h_n, c_n = states
                info_vector = torch.cat((inputs, h_n.view(batch_size, 1, -1)), dim=2)
                attention_weights = self.attention(info_vector.view(batch_size, -1))
                attention_weights = self.softmax(attention_weights)
                attended_weights = cnn_features[0:batch_size] * attention_weights
                attended_info_vector = torch.cat((inputs.view(batch_size, -1), attended_weights), dim=1)
                attended_in_vector = self.attended(attended_info_vector)
                attended_in_vector = attended_in_vector.view(batch_size, 1, -1)
                hiddens, states = self.lstm(attended_in_vector, states)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        # sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        sampled_ids = torch.cat(sampled_ids, 0)                  # (batch_size, 20)
        return sampled_ids.squeeze()


class ImageDescriptor():
    def __init__(self, args):
        self.__args = args
        self.__history = []

        if not os.path.exists(args.model_path):
            os.makedirs(args.model_path)

        self.__config_path = os.path.join(args.model_path, "config.txt")

        # Device configuration
        self.__device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Load vocabulary wrapper
        with open(args.vocab_path, 'rb') as f:
            self.__vocab = pickle.load(f)

        # coco dataset
        train_set = 

        # Build data loader
        self.__data_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                         batch_size=args.batch_size,
                                                         shuffle=True,
                                                         num_workers=args.num_workers,
                                                         collate_fn=collate_fn)

        # Build the models
        self.__encoder = EncoderCNN(args.embed_size).to(self.__device)
        self.__decoder = DecoderRNN(args.embed_size, args.hidden_size,
                                    len(self.__vocab), args.num_layers).to(self.__device)

        # Loss and optimizer
        self.__criterion = nn.CrossEntropyLoss()
        self.__params = list(self.__decoder.parameters(
        )) + list(self.__encoder.linear.parameters()) + list(self.__encoder.bn.parameters())
        self.__optimizer = torch.optim.Adam(
            self.__params, lr=args.learning_rate)

        # Load checkpoint and check compatibility
        if os.path.isfile(self.__config_path):
            self.__model_path = max(
                glob.iglob(os.path.join(args.model_path, '*.ckpt')), key=os.path.getctime)
            with open(self.__config_path, 'r') as f:
                content = f.read()[:-1]
                if content != repr(self):
                    print(f'f.read():\n{content}')
                    print(f'repr(self):\n{repr(self)}')
                    raise ValueError(
                        "Cannot create this experiment: "
                        "I found a checkpoint conflicting with the current setting.")
            self.load()
        # else:
        #     self.save()
    
    def fit(self, dataset):


    def setting(self):
        '''
        Return the setting of the experiment.
        '''
        return {'Net': (self.__encoder, self.__decoder),
                'Optimizer': self.__optimizer,
                'BatchSize': self.__args.batch_size}

    @property
    def epoch(self):
        return len(self.__history)

    def __repr__(self):
        """Pretty printer showing the setting of the experiment. This is what
        is displayed when doing ``print(experiment)``. This is also what is
        saved in the ``config.txt`` file.
        """
        string = ''
        for key, val in self.setting().items():
            string += '{}({})\n'.format(key, val)
        return string

    def state_dict(self):
        """Returns the current state of the experiment."""
        return {'Net': (self.__encoder.state_dict(), self.__decoder.state_dict()),
                'Optimizer': self.__optimizer.state_dict(),
                'History': self.__history}

    def save(self):
        # torch.save(self.__decoder.state_dict(), os.path.join(
        #     self.__args.model_path, 'decoder-{}-{}.ckpt'.format(self.__epoch+1, self.__i+1)))
        # torch.save(self.__encoder.state_dict(), os.path.join(
        #     self.__args.model_path, 'encoder-{}-{}.ckpt'.format(self.__epoch+1, self.__i+1)))
        self.__model_path = os.path.join(
            self.__args.model_path, 'epoch-{}.ckpt'.format(self.epoch))
        torch.save(self.state_dict(), self.__model_path)
        with open(self.__config_path, 'w') as f:
            print(self, file=f)

    def load(self):
        """Loads the experiment from the last checkpoint saved on disk."""
        checkpoint = torch.load(self.__model_path,
                                map_location=self.__device)
        self.load_state_dict(checkpoint)
        del checkpoint

    def load_state_dict(self, checkpoint):
        """Loads the experiment from the input checkpoint."""
        self.__encoder.load_state_dict(checkpoint['Net'][0])
        self.__decoder.load_state_dict(checkpoint['Net'][1])
        self.__optimizer.load_state_dict(checkpoint['Optimizer'])
        self.__history = checkpoint['History']

        # The following loops are used to fix a bug that was
        # discussed here: https://github.com/pytorch/pytorch/issues/2830
        # (it is supposed to be fixed in recent PyTorch version)
        for state in self.__optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.__device)

    def run(self):
        total_step = len(self.__data_loader)
        start_epoch = self.epoch
        print("Start/Continue training from epoch {}".format(start_epoch))
        for epoch in range(start_epoch, self.__args.num_epochs):
            t_start = time.time()
            for i, (images, captions, lengths) in enumerate(self.__data_loader):
                # Set mini-batch dataset
                images = images.to(self.__device)
                captions = captions.to(self.__device)
                targets = pack_padded_sequence(
                    captions, lengths, batch_first=True)[0]

                # Forward, backward and optimize
                features = self.__encoder(images)
                outputs = self.__decoder(features, captions, lengths)
                loss = self.__criterion(outputs, targets)
                self.__decoder.zero_grad()
                self.__encoder.zero_grad()
                loss.backward()
                self.__optimizer.step()

                # Print log info
                if i % self.__args.log_step == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                          .format(epoch+1, self.__args.num_epochs, i, total_step, loss.item(), np.exp(loss.item())))

            self.__history.append(loss)
            # Save the model checkpoints
            print("Epoch {} (Time: {:.2f}s)".format(
                self.epoch, time.time() - t_start))
            self.save()