import torch as T
from torch_geometric.nn import Linear
from utilities.sage_online import StatefulSAGEConv as SAGEConv

class GCN(T.nn.Module):
    def __init__(self, feat, hidden, classes):
        super(GCN, self).__init__()
        self.conv1 = SAGEConv(feat, int(hidden/2),  project=True)
        self.conv2 = SAGEConv(int(hidden/2), hidden, project=True)
        self.lin1 = Linear(hidden, int(hidden*1.5))
        self.lin2 = Linear(int(hidden*1.5), int(hidden/2))
        self.lin3 = Linear(int(hidden/2), int(hidden/2))
        self.lin4 = Linear(int(hidden/2), int(hidden/2))
        self.lin5 = Linear(int(hidden/2), int(hidden*1.5))
        self.lin6 = Linear(int(hidden*1.5), int(hidden/2))
        self.conv3 = SAGEConv(int(hidden/2), int(hidden/2), project=True)
        self.conv4 = SAGEConv(int(hidden/2), classes, project=True)


    def forward(self, data):
        [x, edge_index] = data
        h = self.conv1(x, edge_index)
        h = h.relu()
        #print(h.shape, 'conv1')
        h = self.conv2(h, edge_index)
        h = h.relu()
        #print(h.shape, 'conv2')
        h = self.lin1(h)
        h = h.relu()
        #print(h.shape, 'lin1')  
        h = self.lin2(h)
        h = h.relu()
        #print(h.shape, 'lin2')  
        h = self.lin3(h)
        h = h.relu()
        #print(h.shape, 'lin3')
        h = self.lin4(h)
        h = h.relu()  
        #print(h.shape, 'lin4')
        h = self.lin5(h)
        h = h.relu()  
        #print(h.shape, 'lin5')
        h = self.lin6(h)
        h = h.relu()
        #print(h.shape, 'lin6')
        h = self.conv3(h, edge_index)
        h = h.relu()
        #print(h.shape, 'conv3')
        h = self.conv4(h, edge_index)
        #print(h.shape, 'conv4')
        return h