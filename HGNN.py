import torch.nn as nn
import torch
from torch_geometric.nn.conv import MessagePassing,HypergraphConv
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot, zeros
import torch.nn.functional as F
from layers import MLP,HalfNLHconv,Linear

#
class hgnn(nn.Module):
    def __init__(self,args,n_dim):
        super(hgnn, self).__init__()
        self.args=args
        self.hyperedge_weight = nn.Parameter(torch.ones(1000))
        self.hyperedge_attr1 = nn.Parameter(torch.rand(n_dim))
        self.hyperedge_attr2 = nn.Parameter(torch.rand(n_dim))
        self.EW_weight = nn.Parameter(torch.ones(5200))
        self.hyperconv=HyConv(n_dim,n_dim,dropout=0.5)
        # self.hcha=HypergraphConv(n_dim,n_dim,True)

    def forward(self,features, hyperedge_index,hyperedge_type,bi_weight=None):
        hyperedge_weight = self.hyperedge_weight[0:hyperedge_index[1].max().item() + 1]  # 450：1
        hyperedge_attr = self.hyperedge_attr1 * hyperedge_type + self.hyperedge_attr2 * (1 - hyperedge_type)  # （450，1024）
        norm = self.EW_weight[0:hyperedge_index.size(1)]
        if bi_weight != None:
            norm = bi_weight
        out,hy=self.hyperconv(features, hyperedge_index, hyperedge_weight,hyperedge_attr,norm)
        # out, hy = self.hyperconv(out, hyperedge_index, hyperedge_weight, hyperedge_attr, norm)
        # out, hy = self.hyperconv(out, hyperedge_index, hyperedge_weight, hyperedge_attr, norm)
        # out=self.hcha(features,hyperedge_index,hyperedge_weight,hyperedge_attr)
        # hy=0      
        return out,hy

class HyConv(MessagePassing):
    """Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        use_attention (bool, optional): If set to :obj:`True`, attention
            will be added to this layer. (default: :obj:`False`)
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels,use_attention=False,attention_mode: str = 'edge', heads=1,
                 concat=True, negative_slope=0.2, dropout=0.1, bias=True,**kwargs):
        kwargs.setdefault('aggr', 'add')
        super(HyConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        self.attention_mode=attention_mode

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.weight = Parameter(torch.Tensor(in_channels, heads * out_channels))
            self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
            self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.edgeweight = Parameter(torch.Tensor(in_channels, out_channels))
        self.edgefc = torch.nn.Linear(in_channels, out_channels)

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.edgeweight)
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)

    def forward(self, x, hyperedge_index,hyperedge_weight=None,hyperedge_attr=None,EW_weight=None ,dia_len=None):

        num_nodes, num_edges = x.size(0), 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1
        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)
        alpha = None

        if self.use_attention:
            assert hyperedge_attr is not None
            x = x.view(-1, self.heads, self.out_channels)
            hyperedge_attr = torch.matmul(hyperedge_attr, self.weight)
            hyperedge_attr = hyperedge_attr.view(-1, self.heads, self.out_channels)
            x_i = x[hyperedge_index[0]]
            x_j = hyperedge_attr[hyperedge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            if self.attention_mode == 'node':
                alpha = softmax(alpha, hyperedge_index[1], num_nodes=x.size(0))
            else:
                alpha = softmax(alpha, hyperedge_index[0], num_nodes=x.size(0))
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # 537：0.5
        D = scatter_add(hyperedge_weight[hyperedge_index[1]],hyperedge_index[0], dim=0, dim_size=num_nodes)  # [num_nodes]
        D = 1.0 / D  # all 0.5 if hyperedge_weight is None
        D[D == float("inf")] = 0

        # 227
        if EW_weight is None:
            B = scatter_add(x.new_ones(hyperedge_index.size(1)),
                        hyperedge_index[1], dim=0, dim_size=num_edges)      #[num_edges]
        else:
            B = scatter_add(EW_weight[hyperedge_index[0]],
                        hyperedge_index[1], dim=0, dim_size=num_edges)
        B = 1.0 / B
        B[B == float("inf")] = 0

        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,  # hyperedge_attr[hyperedge_index[1]],
                             size=(num_nodes, num_edges))  # num_edges,1,100
        out = out.view(num_edges, -1)
        hyperedge_attr=out




        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha,
                             size=(num_nodes, num_edges))  # num_nodes,1,100
        if self.concat is True and out.size(1) == 1:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return torch.nn.LeakyReLU()(out) ,torch.nn.LeakyReLU()(hyperedge_attr.view(num_edges, -1)) #

    def message(self, x_j, norm_i, alpha):
        H, F = self.heads, self.out_channels

        if x_j.dim() == 2:
            out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)
        else:
            out = norm_i.view(-1, 1, 1) * x_j
        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out

        return out

    # def update(self, aggr_out):
    #    return aggr_out

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class SetGNN(nn.Module):
    def __init__(self, args, dim,norm=None, sig=False):
        super(SetGNN, self).__init__()
        """
        args should contain the following:
        V_in_dim, V_enc_hid_dim, V_dec_hid_dim, V_out_dim, V_enc_num_layers, V_dec_num_layers
        E_in_dim, E_enc_hid_dim, E_dec_hid_dim, E_out_dim, E_enc_num_layers, E_dec_num_layers
        All_num_layers,dropout
        !!! V_in_dim should be the dimension of node features
        !!! E_out_dim should be the number of classes (for classification)
        """
        #         V_in_dim = V_dict['in_dim']
        #         V_enc_hid_dim = V_dict['enc_hid_dim']
        #         V_dec_hid_dim = V_dict['dec_hid_dim']
        #         V_out_dim = V_dict['out_dim']
        #         V_enc_num_layers = V_dict['enc_num_layers']
        #         V_dec_num_layers = V_dict['dec_num_layers']

        #         E_in_dim = E_dict['in_dim']
        #         E_enc_hid_dim = E_dict['enc_hid_dim']
        #         E_dec_hid_dim = E_dict['dec_hid_dim']
        #         E_out_dim = E_dict['out_dim']
        #         E_enc_num_layers = E_dict['enc_num_layers']
        #         E_dec_num_layers = E_dict['dec_num_layers']

        #         Now set all dropout the same, but can be different
        self.All_num_layers = args.All_num_layers
        self.dropout = args.dropout
        self.aggr = args.aggregate
        self.NormLayer = args.normalization
        self.InputNorm =True
        self.GPR = False
        self.args = args
        self.sig = sig
        #         Now define V2EConvs[i], V2EConvs[i] for ith layers
        #         Currently we assume there's no hyperedge features, which means V_out_dim = E_in_dim
        #         If there's hyperedge features, concat with Vpart decoder output features [V_feat||E_feat]
        self.V2EConvs = nn.ModuleList()
        self.E2VConvs = nn.ModuleList()

        if self.All_num_layers == 0:
            self.classifier = MLP(in_channels=dim,
                                  hidden_channels=args.Classifier_hidden,
                                  out_channels=args.num_classes,
                                  num_layers=args.Classifier_num_layers,
                                  dropout=self.dropout,
                                  Normalization=self.NormLayer,
                                  InputNorm=False)
        else:
            self.V2EConvs.append(HalfNLHconv(in_dim=dim,
                                             hid_dim=args.MLP_hidden,
                                             out_dim=args.MLP_hidden,
                                             num_layers=args.MLP_num_layers,
                                             dropout=self.dropout,
                                             Normalization=self.NormLayer,
                                             InputNorm=self.InputNorm,
                                             heads=args.heads,
                                             attention=args.PMA))

            self.E2VConvs.append(HalfNLHconv(in_dim=args.MLP_hidden,
                                             hid_dim=args.MLP_hidden,
                                             out_dim=args.MLP_hidden,
                                             num_layers=args.MLP_num_layers,
                                             dropout=self.dropout,
                                             Normalization=self.NormLayer,
                                             InputNorm=self.InputNorm,
                                             heads=args.heads,
                                             attention=args.PMA))

            for _ in range(self.All_num_layers - 1):
                self.V2EConvs.append(HalfNLHconv(in_dim=args.MLP_hidden,
                                                 hid_dim=args.MLP_hidden,
                                                 out_dim=args.MLP_hidden,
                                                 num_layers=args.MLP_num_layers,
                                                 dropout=self.dropout,
                                                 Normalization=self.NormLayer,
                                                 InputNorm=self.InputNorm,
                                                 heads=args.heads,
                                                 attention=args.PMA))
                self.E2VConvs.append(HalfNLHconv(in_dim=args.MLP_hidden,
                                                 hid_dim=args.MLP_hidden,
                                                 out_dim=args.MLP_hidden,
                                                 num_layers=args.MLP_num_layers,
                                                 dropout=self.dropout,
                                                 Normalization=self.NormLayer,
                                                 InputNorm=self.InputNorm,
                                                 heads=args.heads,
                                                 attention=args.PMA))

    def reset_parameters(self):
        for layer in self.V2EConvs:
            layer.reset_parameters()
        for layer in self.E2VConvs:
            layer.reset_parameters()
        for layer in self.bnV2Es:
            layer.reset_parameters()
        for layer in self.bnE2Vs:
            layer.reset_parameters()
        self.classifier.reset_parameters()
        if self.GPR:
            self.MLP.reset_parameters()
            self.GPRweights.reset_parameters()
        if self.LearnMask:
            nn.init.ones_(self.Importance)

    def forward(self,x,edge_index,norm,aug_weight=None):
        """
        The data should contain the follows
        data.x: node features
        data.edge_index: edge list (of size (2,|E|)) where data.edge_index[0] contains nodes and data.edge_index[1] contains hyperedges
        !!! Note that self loop should be assigned to a new (hyper)edge id!!!
        !!! Also note that the (hyper)edge id should start at 0 (akin to node id)
        data.norm: The weight for edges in bipartite graphs, correspond to data.edge_index
        !!! Note that we output final node representation. Loss should be defined outside.
        """
        #             The data should contain the follows
        #             data.x: node features
        #             data.V2Eedge_index:  edge list (of size (2,|E|)) where
        #             data.V2Eedge_index[0] contains nodes and data.V2Eedge_index[1] contains hyperedges

        cidx = edge_index[1].min()
        edge_index[1] -= cidx  # make sure we do not waste memory
        reversed_edge_index = torch.stack(
            [edge_index[1], edge_index[0]], dim=0)
        # false
        if self.GPR:
            xs = []
            xs.append(F.relu(self.MLP(x)))
            for i, _ in enumerate(self.V2EConvs):
                x = F.relu(self.V2EConvs[i](x, edge_index, norm, self.aggr))
                #                 x = self.bnV2Es[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.E2VConvs[i](x, reversed_edge_index, norm, self.aggr)
                x = F.relu(x)
                xs.append(x)
                #                 x = self.bnE2Vs[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = torch.stack(xs, dim=-1)
            x = self.GPRweights(x).squeeze()
        else:
            # if self.dropout:

            x = F.dropout(x, p=0.2, training=self.training)  # Input dropout
            for i, _ in enumerate(self.V2EConvs):
                # print(edge_index[0].unique())
                # print(x.shape)
                x = F.relu(self.V2EConvs[i](x, edge_index, norm, self.aggr, aug_weight))
                # print(x.shape)
                #                 x = self.bnV2Es[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(self.E2VConvs[i](
                    x, reversed_edge_index, norm, self.aggr, aug_weight))
                # print(x.shape)
                #                 x = self.bnE2Vs[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

