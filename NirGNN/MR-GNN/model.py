import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
import scipy.spatial.distance as dist
from torch import distributions
import time

from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence


class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
    # Ensure hidden matches the shape expected for matrix multiplication
        if A.shape[1] != hidden.shape[1]:
            hidden = hidden[:, :A.shape[1], :]  # Adjust hidden size if needed

    # Compute input_in and input_out
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
    
    # Concatenate the inputs
        inputs = torch.cat([input_in, input_out], 2)

    # Apply gates
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)

    # Split gates
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)

    # Update hidden state using gates
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
    
        return hy




    def forward(self, A, hidden):
    # Ensure hidden matches the shape expected for matrix multiplication
        if A.shape[1] != hidden.shape[1]:
            hidden = hidden[:, :A.shape[1], :]

    # Perform multiple iterations of the GNNCell
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size) #num_embeddings, embedding_dim
        self.gnn = GNN(self.hidden_size, step=opt.step) 
        self.weight = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_zero = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True) #？
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()
        self.nn = nn.Linear(in_features=2 * self.hidden_size, out_features=self.hidden_size)
        self.nn1 = nn.Linear(in_features=4 * self.hidden_size, out_features=self.hidden_size)
        self.nnb = nn.Linear(in_features=3 * self.hidden_size, out_features=self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.logic = nn.LogSoftmax(dim=1)
        self.fc1 = nn.Linear(self.hidden_size, 512)
        self.bn1_fc = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, self.hidden_size)
        self.bn_fc2 = nn.BatchNorm1d(self.hidden_size)
        self.W = nn.Parameter(torch.zeros(size=(self.hidden_size, self.hidden_size)))
        # self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask,topk):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)#sg
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1)) 
        # b = self.embedding.weight[1:]  # n_nodes x latent_size
        b = topk
        scores = torch.matmul(a, b.transpose(1, 0))

        return scores

    def compute_intent(self, item):
        q2 =self.linear_two(item)
        alpha = self.linear_three(torch.sigmoid(q2))
        intent = alpha*item
        # intent = torch.sum(alpha * item, 1)  # sg
        return intent

    def compute_beta_scores(self, hidden, mask,topk,beta):
        beta_std = torch.std(beta,dim=2).unsqueeze(dim=2)
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]
        beta1 = beta[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]
        q2 = beta*hidden
        q1 = (beta1*ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
          # batch_size x seq_length x latent_size
        avg =  torch.mean(q1+q2,dim=2)
        avg = torch.unsqueeze(avg,2)
        alpha = self.linear_three((q1+q2-avg)/beta_std)
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)#sg
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1)) 
        # b=torch.sum(a,dim=-1)
        # print(b[18])
        # b = self.embedding.weight[1:]  # n_nodes x latent_size
        b = topk
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores


    def beta_intent(self,hidden,taxo1,taxo2,taxo3):
        taxo = self.nnb(torch.cat((taxo1,taxo2,taxo3),dim =-1))
        taxo = torch.nn.functional.softplus(taxo)
        hidden = torch.nn.functional.softplus(hidden)
        B = torch.distributions.beta.Beta(hidden, taxo)
        beta = B.sample()
        return beta

    def forward(self, inputs, A_list):
        hidden = self.embedding(inputs)
        for A in A_list:
            hidden = self.gnn(A, hidden)
        return hidden




        attr_intent = self.nn(torch.cat((attr1_intent, attr2_intent), dim=-1))

        # attr_intent = attr1_intent #torch.cat((attr1_intent,attr2_intent),2)
        ca = self.nn(torch.cat((ca1,ca2),dim=-1))

        return hidden, hidt, attr_intent,ca, hidt_beta


    def map(self,x):
        # x = torch.cat((x,attr),1)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = self.fc2(x)
        return x

    def bhattacharyya_distance_1(self, vector1, vector2):
        a = vector1.detach().cpu().numpy() * vector2.detach().cpu().numpy()
        b = np.sqrt(a)
        for i in range(len(b)):
            for j in range(len(b[i])):
                if math.isnan(b[i][j]):
                    b[i][j] = 0
        # for i in range(len(b)):
        #     if math.isnan(b[i][j]):
        #         b[i][j] = 0

        bc = np.sum(b,axis =1)
        # BC = np.sum(np.sqrt(vector1.detach().numpy() * vector2.detach().numpy()))
        # dt = np.ones(100, dtype=float)#
        return -np.log(bc)#, np.sqrt(dt-bc)

    def bhattacharyya_distance_2(self, vector1, vector2):
        a = vector1.detach().cpu().numpy() * vector2.detach().cpu().numpy()
        b = np.sqrt(a)
        # for i in range(len(b)):
        #     for j in range(len(b[i])):
        #         if math.isnan(b[i][j]):
        #             b[i][j] = 0
        for i in range(len(b)):
            if math.isnan(b[i]):
                b[i] = 0

        bc = np.sum(b, axis=0)
        # BC = np.sum(np.sqrt(vector1.detach().numpy() * vector2.detach().numpy()))
        # dt = np.ones(100, dtype=float)#
        return np.log(bc)  # , np.sqrt(dt-bc)

    def Mahalanobis_distance_2(self,x, y):
        x =x.detach().numpy()
        y = y.detach().numpy()
        X = np.vstack([x, y])
        XT = X.T
        d2 = dist.pdist(XT, 'mahalanobis')
        return d2

    def zeroshot(self, intent_item , intent_attribute, candidate_attribute,mask,alias_inputs):
        get = lambda i: intent_item[i][alias_inputs[i]]
        seq_item = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
        oriitem = torch.sum(seq_item * mask.view(mask.shape[0], -1, 1).float(), 1)

        get = lambda i: intent_attribute[i][alias_inputs[i]]
        seq_intent = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
        intent = torch.sum(seq_intent * mask.view(mask.shape[0], -1, 1).float(), 1)

        # xx = self.linear_zero(intent)


        distanceb = self.bhattacharyya_distance_1(self.linear_zero(intent),oriitem)
        # distancem = self.Mahalanobis_distance_2(self.linear_zero(intent),oriitem)
        lossB = torch.LongTensor(distanceb)
        # lossh = torch.LongTensor(distanceh)

        loss = torch.sum(lossB)#+torch.sum(lossh)
        #------------topk------------------
        # intent2 = torch.unsqueeze(intent, 1)
        # intent2 = intent2.repeat(1,4,1)
        # sim_score = F.cosine_similarity(intent2, candidate_attribute, dim=2).clamp(min=-1, max=1)
        # score = torch.Tensor(sim_score)
        # top_val, idx = torch.topk(score, 1) #candidate_value
        # getcan = lambda i:candidate_attribute[i][idx[i]]
        # seq_can = torch.stack([getcan(i) for i in torch.arange(len(alias_inputs)).long()])
        # seq_can= torch.squeeze(seq_can,1)
        #----------------no topk----------------
        #candidate 0

        candidate_item_embedding = self.linear_zero(candidate_attribute)
        return candidate_item_embedding, loss

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data, attr_data, taxo_data):
    alias_inputs, A_list, items, mask, targets = data.get_slice(i, attr_data, taxo_data)

    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A_list = [trans_to_cuda(torch.Tensor(A).float()) for A in A_list]
    mask = trans_to_cuda(torch.Tensor(mask).long())

    hidden, hidt, attr_intent, ca, beta = model(items, A_list)

    get = lambda i: hidt[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])

    candidate_item_embedding, zeroloss = model.zeroshot(hidden, attr_intent, ca, mask, alias_inputs)
    score = model.compute_scores(seq_hidden, mask, candidate_item_embedding)

    return targets, score, zeroloss





def train_test(model, train_data, test_data, attr_data, taxo_data):
    model.scheduler.step()
    print('start training: ', time.strftime("%Y-%m-%d %H:%M:%S"))
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    
    for i, j in zip(slices, range(len(slices))):
        model.optimizer.zero_grad()
        targets, scores, zeroloss = forward(model, i, train_data, attr_data, taxo_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        
        # Combine cross-entropy loss and zero-shot loss
        loss = model.loss_function(scores, targets - 1) * 0.7 + zeroloss * 0.3
        loss.backward()
        model.optimizer.step()
        
        total_loss += loss.item()
        if j % int(len(slices) / 5 + 1) == 0:
            print(f'[{j}/{len(slices)}] Loss: {loss.item():.4f}')
    
    print('start predicting: ', time.strftime("%Y-%m-%d %H:%M:%S"))
    model.eval()
    hit10, mrr10, hit20, mrr20 = [], [], [], []
    
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores, zeroloss = forward(model, i, test_data, attr_data, taxo_data)
        
        # Top-10 predictions
        sub_scores10 = scores.topk(10)[1].cpu().detach().numpy()
        for score, target in zip(sub_scores10, targets):
            hit10.append(np.isin(target - 1, score))
            if target - 1 in score:
                mrr10.append(1 / (np.where(score == target - 1)[0][0] + 1))
            else:
                mrr10.append(0)
        
        # Top-20 predictions
        sub_scores20 = scores.topk(20)[1].cpu().detach().numpy()
        for score, target in zip(sub_scores20, targets):
            hit20.append(np.isin(target - 1, score))
            if target - 1 in score:
                mrr20.append(1 / (np.where(score == target - 1)[0][0] + 1))
            else:
                mrr20.append(0)
    
    # Calculate metrics as percentages
    precision10 = np.mean(hit10) * 100
    mrr10 = np.mean(mrr10) * 100
    precision20 = np.mean(hit20) * 100
    mrr20 = np.mean(mrr20) * 100

    return precision10, mrr10, precision20, mrr20