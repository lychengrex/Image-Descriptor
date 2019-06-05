import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class ResNet(nn.Module):
    def __init__(self, embed_size, ver=101, attention_mechanism=False):
        '''
        Load the pretrained ResNet and replace top fc layer.

        Args:
            embed_size (int): dimension of word embedding vectors
            ver (int): version of the pretrained network
            attention_mechanism (bool): use attention layers in decoder network or not
        '''
        super(ResNet, self).__init__()

        if ver == 152:
            resnet = models.resnet152(pretrained=True)
        elif ver == 101:
            resnet = models.resnet101(pretrained=True)
        elif ver == 50:
            resnet = models.resnet50(pretrained=True)
        else:
            raise ModuleNotFoundError()

        # delete the last fc layer.
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        if attention_mechanism:
            # Initialize the weights
            self.linear.weight.data.normal_(0.0, 0.02)
            self.linear.bias.data.fill_(0)
            self.attention_size = resnet.fc.in_features
        self.attention_mechanism = attention_mechanism

    def forward(self, images):
        '''
        Extract feature vectors from input images.

        Args:
            images (tensor)
        '''
        with torch.no_grad():
            features = self.resnet(images)
        if self.attention_mechanism:
            # features = features.data
            features = features.reshape(features.size(0), -1)
            cnn_features = features
            features = self.bn(self.linear(features))
            return features, cnn_features
        else:
            features = features.reshape(features.size(0), -1)
            features = self.bn(self.linear(features))
            return features


class VGG(nn.Module):
    def __init__(self, embed_size, ver=19, attention_mechanism=False):
        '''
        Load the pretrained VGG and replace fc layer.

        Args:
            embed_size (int): dimension of word embedding vectors
            ver (int): version of the pretrained network
            attention_mechanism (bool): use attention layers in decoder network or not
        '''
        super(VGG, self).__init__()
        if ver == 19:
            vgg = models.vgg19_bn(pretrained=True)
        else:
            raise ModuleNotFoundError()

        self.features = vgg.features
        self.classifier = vgg.classifier
        # change input node and output node for output FC # 4096
        # self.classifier[3] = nn.Linear(vgg.classifier[3].in_features, 2048)
        self.classifier[6] = nn.Linear(
            vgg.classifier[6].in_features, embed_size)
        self.linear = self.classifier[6]
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

        if attention_mechanism:
            """Initialize the weights."""
            self.classifier[3].weight.data.normal_(0.0, 0.02)
            self.classifier[3].bias.data.fill_(0)
            self.classifier[6].weight.data.normal_(0.0, 0.02)
            self.classifier[6].bias.data.fill_(0)
            self.attention_size = vgg.classifier[6].in_features

        self.attention_mechanism = attention_mechanism

    def forward(self, images):
        '''
        Extract feature vectors from input images.
        '''
        # features extraction
        with torch.no_grad():
            features = self.features(images)

        # flattern
        features = features.view(features.size(0), -1)

        # for no attention layers in RNN
        if not self.attention_mechanism:
            features = self.classifier(features)
            features = self.bn(features)
            return features

        # for using attention layers in RNN
        # for i in range(4):
        #     features = self.classifier[i](features)
        # # cnn_features = features
        # for i in range(4, len(self.classifier)):
        #     features = self.classifier[i](features)
        features = self.classifier[0](features)
        cnn_features = features
        for i in range(1, len(self.classifier)):
            features = self.classifier[i](features)
        features = self.bn(features)
        return features, cnn_features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, attention_size,
                 num_layers, max_seq_length=20, attention_mechanism=False):
        '''
        Set the hyper-parameters and build the layers.

        Args:
            embed_size (int): dimension of word embedding vectors
            hidden_size (int): dimension of lstm hidden states
            vocab_size (int): number of vocabulary in the caption dictionary
            attention_size (int): number of features before the output of CNN
            num_layers (int): number of layers in lstm
            max_seq_length (int)
            attention_mechanism (bool): use attention layer or not
        '''
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size,
                            num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        if attention_mechanism:
            self.attention = nn.Linear(hidden_size+embed_size, attention_size)
            self.attended = nn.Linear(attention_size+embed_size, embed_size)
            self.softmax = nn.Softmax(dim=1)
            self.init_weights()
        self.attention_mechanism = attention_mechanism

    def init_weights(self):
        '''
        Initialize weights.
        '''
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        self.attention.weight.data.uniform_(-0.1, 0.1)
        self.attention.bias.data.fill_(0)
        self.attended.weight.data.uniform_(-0.1, 0.1)
        self.attended.bias.data.fill_(0)

    def forward(self, features, captions, lengths, cnn_features=None):
        '''
        Decode image feature vectors and generates captions.

        Args:
            features
            captions
            lengths
            cnn_features
        '''
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)

        if not self.attention_mechanism:
            packed = pack_padded_sequence(
                embeddings, lengths, batch_first=True)
            hiddens, _ = self.lstm(packed)
            outputs = self.linear(hiddens[0])
            return outputs

        packed_seq = pack_padded_sequence(
            embeddings, lengths, batch_first=True)
        packed = packed_seq.data
        batch_sizes = packed_seq.batch_sizes

        hiddenStates = None
        start = 0
        for batch_size in batch_sizes:
            in_vector = packed[start:start+batch_size].view(batch_size, 1, -1)
            start += batch_size
            if hiddenStates is None:
                hiddenStates, (h_n, c_n) = self.lstm(in_vector)
                hiddenStates = torch.squeeze(hiddenStates)
            else:
                h_n, c_n = h_n[:, 0:batch_size, :], c_n[:, 0:batch_size, :]
                info_vector = torch.cat(
                    (in_vector, h_n.view(batch_size, 1, -1)), dim=2)
                attention_weights = self.attention(
                    info_vector.view(batch_size, -1))
                attention_weights = self.softmax(attention_weights)
                attended_weights = cnn_features[0:batch_size] * \
                    attention_weights
                attended_info_vector = torch.cat(
                    (in_vector.view(batch_size, -1), attended_weights), dim=1)
                attended_in_vector = self.attended(attended_info_vector)
                attended_in_vector = attended_in_vector.view(batch_size, 1, -1)
                out, (h_n, c_n) = self.lstm(attended_in_vector, (h_n, c_n))
                hiddenStates = torch.cat(
                    (hiddenStates, out.view(batch_size, -1)))
        hiddenStates = self.linear(hiddenStates)
        return hiddenStates

    def sample(self, features, cnn_features=None, states=None):
        '''
        Generate captions for given image features using greedy search.

        Args:
            features
            cnn_features
            states
        '''
        sampled_ids = []
        inputs = features.unsqueeze(1)

        if not self.attention_mechanism:
            for i in range(self.max_seg_length):
                # hiddens: (batch_size, 1, hidden_size)
                hiddens, states = self.lstm(inputs, states)
                # outputs:  (batch_size, vocab_size)
                outputs = self.linear(hiddens.squeeze(1))
                # predicted: (batch_size)
                _, predicted = outputs.max(1)
                sampled_ids.append(predicted)
                # inputs: (batch_size, embed_size)
                inputs = self.embed(predicted)
                # inputs: (batch_size, 1, embed_size)
                inputs = inputs.unsqueeze(1)
            # sampled_ids: (batch_size, max_seq_length)
            sampled_ids = torch.stack(sampled_ids, 1)
            return sampled_ids
        batch_size = features.size(0)

        for i in range(self.max_seg_length):
            if states is None:
                # hiddens: (batch_size, 1, hidden_size)
                hiddens, states = self.lstm(inputs, states)
            else:
                h_n, c_n = states
                info_vector = torch.cat(
                    (inputs, h_n.view(batch_size, 1, -1)), dim=2)
                attention_weights = self.attention(
                    info_vector.view(batch_size, -1))
                attention_weights = self.softmax(attention_weights)
                attended_weights = cnn_features[0:batch_size] * \
                    attention_weights
                attended_info_vector = torch.cat(
                    (inputs.view(batch_size, -1), attended_weights), dim=1)
                attended_in_vector = self.attended(attended_info_vector)
                attended_in_vector = attended_in_vector.view(batch_size, 1, -1)
                hiddens, states = self.lstm(attended_in_vector, states)
            # outputs:  (batch_size, vocab_size)
            outputs = self.linear(hiddens.squeeze(1))
            # predicted: (batch_size)
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            # inputs: (batch_size, embed_size)
            inputs = self.embed(predicted)
            # inputs: (batch_size, 1, embed_size)
            inputs = inputs.unsqueeze(1)
        # (batch_size, 20)
        sampled_ids = torch.cat(sampled_ids, 0)
        return sampled_ids.squeeze()
