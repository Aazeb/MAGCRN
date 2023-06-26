import torch      #final model
import torch.nn as nn
from model.AGCRNCell import AGCRNCell
from torch.nn import functional as F, Parameter
import math
from torch.autograd import Variable

class Hypernetwork(nn.Module):
    def __init__(self, cheb_k, hidden_dim, h_filter, h_filter_size, num_node, horizon):
        super(Hypernetwork, self).__init__()
        self.cheb_k = cheb_k
        self.hidden_dim = hidden_dim
        self.h_filter = h_filter
        self.h_filter_size = h_filter_size
        self.num_node = num_node
        self.horizon = horizon
        self.hyp = nn.Linear((self.cheb_k*self.hidden_dim)*(self.cheb_k*self.hidden_dim), self.h_filter)

    def forward(self, output, weights):
        weights = weights.view(output.size(2), -1)
        output = output[:, -1:, :, :]                                                              
        output = output.permute(1, 2, 0, 3)                                                             
        h = self.hyp(weights)
        h = h.view(h.size(0), 1, self.horizon, 1, self.h_filter_size)
        h = h.view(h.size(0)*h.size(2), 1, 1, self.h_filter_size)
        h = F.conv2d(output, h, dilation=0, groups=self.num_node)
        h = h.view(output.size(1), 1, self.horizon, h.size(2), h.size(3))
        h = h.permute(0, 3, 4, 1, 2)
        h = torch.sum(h, dim=3)
        h = h.permute(0, 3, 1, 2).contiguous()
        h = h.permute(2, 1, 0, 3)
        return h

class Transform(nn.Module):
    def __init__(self, outfea, d):
        super(Transform, self).__init__()
        self.qff = nn.Linear(outfea, outfea)
        self.kff = nn.Linear(outfea, outfea)
        self.vff = nn.Linear(outfea, outfea)

        self.ln = nn.LayerNorm(outfea)
        self.lnff = nn.LayerNorm(outfea)

        self.ff = nn.Sequential(
            nn.Linear(outfea, outfea),
            nn.ReLU(),
            nn.Linear(outfea, outfea)
        )

        self.d = d

    def forward(self, x, bias):
        query = self.qff(x)
        key = self.kff(x)
        value = self.vff(bias)
        query = torch.cat(torch.split(query, self.d, -1), 0).permute(0,2,1,3)
        key = torch.cat(torch.split(key, self.d, -1), 0).permute(0,2,3,1)
        value = torch.cat(torch.split(value, self.d, -1), 0).permute(0,2,1,3)
        A = torch.matmul(query, key)
        A /= (self.d ** 0.5)
        A = torch.softmax(A, -1)
        value = torch.matmul(A ,value)
        value = torch.cat(torch.split(value, x.shape[0], 0), -1).permute(0,2,1,3)
        value += x
        value = self.ln(value)
        x = self.ff(value) + value
        return self.lnff(x)

class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state, weights = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden, weights

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)

class MAGCRN(nn.Module):
    def __init__(self, args):
        super(MAGCRN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.cheb_k = args.cheb_k
        self.h_filter_size = args.h_filter_size
        self.num_layers = args.num_layers
        self.default_graph = args.default_graph
        self.decoder = args.decoder
        self.L = args.att_block_num_L
        self.d = args.att_head_dim_d
        self.h_filter_size = args.h_filter_size
        self.h_filter = args.horizon * args.h_filter_size
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)
        self.encoder = AVWDCRNN(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                                args.embed_dim, args.num_layers)

        #predictor
        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        self.hypernetwork = Hypernetwork(self.cheb_k, self.hidden_dim, self.h_filter, self.h_filter_size, self.num_node, self.horizon)
        self.transform = nn.ModuleList([Transform(self.hidden_dim , self.d) for i in range(self.L)])
        self.fc = torch.nn.Linear(self.hidden_dim , 1)

    def forward(self, source, targets, teacher_forcing_ratio=0.5):

        init_state = self.encoder.init_hidden(source.shape[0])
        output, _, weights = self.encoder(source, init_state, self.node_embeddings)     
        h =  self.hypernetwork(output, weights)
        for i in range(self.L):
            output_h = self.transform[i](output, h)
        output = self.fc(output_h)                    
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node) 
        output = output.permute(0, 1, 3, 2)                                                     
        return output


