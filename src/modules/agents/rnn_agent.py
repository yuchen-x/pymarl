import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.LSTMCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.leaky_relu(self.fc1(inputs))
        h_in = (hidden_state[0].reshape(-1, self.args.rnn_hidden_dim),
                hidden_state[1].reshape(-1, self.args.rnn_hidden_dim))
        h, c = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, (h,c)
