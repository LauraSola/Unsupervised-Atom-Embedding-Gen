# Copyright Universitat Politècnica de Catalunya 2024 https://imatge.upc.edu
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import torch
import torch_geometric.nn as pyg_nn
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_scatter import scatter
from models.utils import ExpNormalSmearing, CosineCutoff


class CartNet(torch.nn.Module):
    """
    CartNet model from Cartesian Encoding Graph Neural Network for Crystal Structures Property Prediction: Application to Thermal Ellipsoid Estimation.
    Args:
        dim_in (int): Dimensionality of the input features.
        dim_rbf (int): Dimensionality of the radial basis function embeddings.
        num_layers (int): Number of CartNet layers in the model.
        radius (float, optional): Radius cutoff for neighbor interactions. Default is 5.0.
        invariant (bool, optional): If `True`, enforces rotational invariance in the encoder. Default is `False`.
        temperature (bool, optional): If `True`, includes temperature information in the encoder. Default is `True`.
        use_envelope (bool, optional): If `True`, applies an envelope function to the interactions. Default is `True`.
        cholesky (bool, optional): If `True`, uses a Cholesky head for the output. If `False`, uses a scalar head. Default is `True`.
    Methods:
        forward(batch):
            Performs a forward pass of the model.
            Args:
                batch: A batch of input data.
            Returns:
                pred: The model's predictions.
                true: The ground truth values corresponding to the input batch.
    """


    def __init__(self, 
        dim_in: int, 
        dim_rbf: int, 
        num_layers: int,
        radius: float = 5.0,
        invariant: bool = False,
        temperature: bool = True, 
        use_envelope: bool = True,
        atom_types: bool = True,
        cholesky: bool = True):
        super().__init__()
    
        self.encoder = Encoder(dim_in, dim_rbf=dim_rbf, radius=radius, invariant=invariant, temperature=temperature, atom_types=atom_types)
        self.dim_in = dim_in

        layers = []
        for _ in range(num_layers):
            layers.append(CartNet_layer(
                dim_in=dim_in,
                use_envelope=use_envelope,
            ))
        self.layers = torch.nn.Sequential(*layers)

        #if cholesky:
        #    self.head = Cholesky_head(dim_in)
        #else:
        self.head = Scalar_head(dim_in)


        ## UNSUPERVISED LEARNING
        self.feature_reconstr = Node_and_Edge_Reconstruction(dim_in, 6, 119) #(dim in, dim out edge, dim out node)
        
        ## GRAPH EMBEDDINGS (SELF-SUPERVISED)
        self.graph_embedding = GraphEmbeddings(dim_in, dim_in) #dim_graph_emb
        
    def forward(self, batch, graph_emb = False, reconstr = False):
        batch = self.encoder(batch)

        for layer in self.layers:
            batch = layer(batch)
        
        if graph_emb:
            pred, true = self.graph_embedding(batch)
        
        elif reconstr:
            pred, true = self.feature_reconstr(batch)
        
        else:
            pred,true = self.head(batch)
        
        return pred,true,batch

class Encoder(torch.nn.Module):
    """
    Encoder module for the CartNet model.
    This module encodes node and edge features for input into the CartNet model, incorporating optional temperature information and rotational invariance.
    Args:
        dim_in (int): Dimension of the input features after embedding.
        dim_rbf (int): Dimension of the radial basis function used for edge attributes.
        radius (float, optional): Cutoff radius for neighbor interactions. Defaults to 5.0.
        invariant (bool, optional): If True, the encoder enforces rotational invariance by excluding directional information from edge attributes. Defaults to False.
        temperature (bool, optional): If True, includes temperature data in the node embeddings. Defaults to True.
    Attributes:
        dim_in (int): Dimension of the input features.
        invariant (bool): Indicates if rotational invariance is enforced.
        temperature (bool): Indicates if temperature information is included.
        embedding (nn.Embedding): Embedding layer mapping atomic numbers to feature vectors.
        temperature_proj_atom (pyg_nn.Linear): Linear layer projecting temperature to embedding dimensions (used if temperature is True).
        bias (nn.Parameter): Bias term added to embeddings (used if temperature is False).
        activation (nn.Module): Activation function (SiLU).
        encoder_atom (nn.Sequential): Sequential network encoding node features.
        encoder_edge (nn.Sequential): Sequential network encoding edge features.
        rbf (ExpNormalSmearing): Radial basis function for encoding distances.
    """
    
    def __init__(
        self,
        dim_in: int,
        dim_rbf: int,
        radius: float = 5.0,
        invariant: bool = False, 
        temperature: bool = True,
        atom_types: bool = True
    ):
        super(Encoder, self).__init__()
        self.dim_in = dim_in
        self.invariant = invariant
        self.temperature = temperature
        self.atom_types = atom_types
        if self.atom_types:
            self.embedding = nn.Embedding(119, self.dim_in*2)
            torch.nn.init.xavier_uniform_(self.embedding.weight.data)  # Default init
        elif not self.temperature:
            self.embedding = nn.Embedding(1, self.dim_in)

        if self.temperature:
            self.temperature_proj_atom = pyg_nn.Linear(1, self.dim_in*2, bias=True)
        elif self.atom_types:
            self.bias = nn.Parameter(torch.zeros(self.dim_in*2))
        self.activation = nn.SiLU(inplace=True)
        
        if self.temperature or self.atom_types:
            self.encoder_atom = nn.Sequential(self.activation,
                                        pyg_nn.Linear(self.dim_in*2, self.dim_in),
                                        self.activation)
        if self.invariant:
            dim_edge = dim_rbf
        else:
            dim_edge = dim_rbf + 3
        
        self.encoder_edge = nn.Sequential(pyg_nn.Linear(dim_edge, self.dim_in*2),
                                        self.activation,
                                        pyg_nn.Linear(self.dim_in*2, self.dim_in),
                                        self.activation)

        self.rbf = ExpNormalSmearing(0.0,radius,dim_rbf,False)  

    def forward(self, batch):

        if self.temperature and self.atom_types:
            x = self.embedding(batch.x) + self.temperature_proj_atom(batch.temperature.unsqueeze(-1))[batch.batch]
        elif not self.temperature and self.atom_types:
            x = self.embedding(batch.x) + self.bias
        elif self.temperature and not self.atom_types:
            x = self.temperature_proj_atom(batch.temperature.unsqueeze(-1))[batch.batch]
        else:
            batch.x = self.embedding.weight.repeat(batch.x.shape[0],1)
        
        if self.temperature or self.atom_types:
            batch.x = self.encoder_atom(x)

        if cfg.invariant:
            batch.edge_attr = self.encoder_edge(self.rbf(batch.cart_dist))
        else:
            batch.edge_attr = self.encoder_edge(torch.cat([self.rbf(batch.cart_dist), batch.cart_dir], dim=-1))

        return batch

class CartNet_layer(pyg_nn.conv.MessagePassing):
    """
    The message-passing layer used in the CartNet architecture.
    Parameters:
        dim_in (int): Dimension of the input node features.
        use_envelope (bool, optional): If True, applies an envelope function to the distances. Defaults to True.
    Attributes:
        dim_in (int): Dimension of the input node features.
        activation (nn.Module): Activation function (SiLU) used in the layer.
        MLP_aggr (nn.Sequential): MLP used for aggregating messages.
        MLP_gate (nn.Sequential): MLP used for computing gating coefficients.
        norm (nn.BatchNorm1d): Batch normalization applied to the gating coefficients.
        norm2 (nn.BatchNorm1d): Batch normalization applied to the aggregated messages.
        use_envelope (bool): Indicates if the envelope function is used.
        envelope (CosineCutoff): Envelope function applied to the distances.
    """
    
    def __init__(self, 
        dim_in: int, 
        use_envelope: bool = True
    ):
        super().__init__()
        self.dim_in = dim_in
        self.activation = nn.SiLU(inplace=True) 
        self.MLP_aggr = nn.Sequential(
            pyg_nn.Linear(dim_in*3, dim_in, bias=True),
            self.activation,
            pyg_nn.Linear(dim_in, dim_in, bias=True),
        )
        self.MLP_gate = nn.Sequential(
            pyg_nn.Linear(dim_in*3, dim_in, bias=True),
            self.activation,
            pyg_nn.Linear(dim_in, dim_in, bias=True),
        )
        
        self.norm = nn.BatchNorm1d(dim_in)
        self.norm2 = nn.BatchNorm1d(dim_in)
        self.use_envelope = use_envelope
        self.envelope = CosineCutoff(0, cfg.radius)
        

    def forward(self, batch):

        x, e, edge_index, dist = batch.x, batch.edge_attr, batch.edge_index, batch.cart_dist
        """
        x               : [n_nodes, dim_in]
        e               : [n_edges, dim_in]
        edge_index      : [2, n_edges]
        dist            : [n_edges]
        batch           : [n_nodes]
        """
        
        x_in = x
        e_in = e

        x, e = self.propagate(edge_index,
                                Xx=x, Ee=e,
                                He=dist,
                            )
 
        batch.x = self.activation(x) + x_in
        
        batch.edge_attr = e_in + e 

        return batch


    def message(self, Xx_i, Ee, Xx_j, He):
        """
        x_i           : [n_edges, dim_in]
        x_j           : [n_edges, dim_in]
        e             : [n_edges, dim_in]
        """

        e_ij = self.MLP_gate(torch.cat([Xx_i, Xx_j, Ee], dim=-1))
        e_ij = F.sigmoid(self.norm(e_ij))
        
        if self.use_envelope:
            sigma_ij = self.envelope(He).unsqueeze(-1)*e_ij
        else:
            sigma_ij = e_ij
        
        self.e = sigma_ij
        return sigma_ij

    def aggregate(self, sigma_ij, index, Xx_i, Xx_j, Ee, Xx):
        """
        sigma_ij        : [n_edges, dim_in]  ; is the output from message() function
        index           : [n_edges]
        x_j           : [n_edges, dim_in]
        """
        dim_size = Xx.shape[0]  

        sender = self.MLP_aggr(torch.cat([Xx_i, Xx_j, Ee], dim=-1))
        

        out = scatter(sigma_ij*sender, index, 0, None, dim_size,
                                   reduce='sum')

        return out

    def update(self, aggr_out):
        """
        aggr_out        : [n_nodes, dim_in] ; is the output from aggregate() function after the aggregation
        x             : [n_nodes, dim_in]
        """
        x = self.norm2(aggr_out)
       
        e_out = self.e
        del self.e

        return x, e_out

class Cholesky_head(torch.nn.Module):
    """
    The Cholesky head used in the CartNet model.
    It enforce the positive definiteness of the output covariance matrix.

    Args:
        dim_in (int): The input dimension of the features.
    """
    
    def __init__(self, 
        dim_in: int
    ):
        super(Cholesky_head, self).__init__()
        self.MLP = nn.Sequential(pyg_nn.Linear(dim_in, dim_in//2),
                                nn.SiLU(inplace=True), 
                                pyg_nn.Linear(dim_in//2, 6))

    def forward(self, batch):
        pred = self.MLP(batch.x[batch.non_H_mask])

        diag_elements = F.softplus(pred[:, :3])

        i,j = torch.tensor([0,1,2,0,0,1]), torch.tensor([0,1,2,1,2,2])
        L_matrix = torch.zeros(pred.size(0),3,3, device=pred.device, dtype=pred.dtype)
        L_matrix[:,i[:3], i[:3]] = diag_elements
        L_matrix[:,i[3:], j[3:]] = pred[:,3:]

        U = torch.bmm(L_matrix.transpose(1, 2), L_matrix)
        
        return U, batch.y

class Scalar_head(torch.nn.Module):
    """
    A head to predict scalar values.
    Args:
        dim_in (int): The dimension of the input features.
    """
    
    def __init__(self,
        dim_in
    ):
        super(Scalar_head, self).__init__()

        self.MLP = nn.Sequential(pyg_nn.Linear(dim_in, dim_in//2), 
                                nn.SiLU(inplace=True), 
                                pyg_nn.Linear(dim_in//2, 1))

    def forward(self, batch):
        dim_size = int(batch.batch.max().item() + 1)
        batch.x = scatter(batch.x, batch.batch, dim=0, reduce="mean", dim_size=dim_size)
        batch.x = self.MLP(batch.x).squeeze(-1)
        
        return batch.x, batch.target


class Node_and_Edge_Reconstruction(torch.nn.Module):
    """
    A head to predict scalar values.
    Args:
        dim_in (int): The dimension of the input features.
    """

    def __init__(self, dim_in, dim_out_edge, dim_out_node):
        super(Node_and_Edge_Reconstruction, self).__init__()

        # Bilinear layer for edge prediction
        self.bilinear = nn.Bilinear(dim_in, dim_in, dim_out_edge)
        self.linear_edge = nn.Linear(dim_out_edge, dim_out_edge)  # Predict {0,1,2,3,4,5+} edges

        # Fully connected layer for node feature reconstruction
        self.linear_node = nn.Linear(dim_in, dim_out_node) # Predict atomic number 1-118


    def forward(self, batch):

        dim_size = batch.batch_size
        edge_prob_list = []
        atom_feature_list = []
        sg_pred_list=[]

        for i in range(dim_size): #iterem sobre cada cristall
            atom_fea_sense_norm= batch.x[batch.batch == i]

            atom_fea_ = F.normalize(atom_fea_sense_norm, dim=1, p=2) #normalització L2

            N = atom_fea_.size(0)  # Number of nodes
            dim = atom_fea_.size(1)  # Feature dimension

            # Repeat feature N times : (N,N,dim)
            atom_nbr_fea = atom_fea_.repeat(N, 1, 1) #(N, N, dim)
            atom_nbr_fea = atom_nbr_fea.contiguous().view(-1, dim) #(N*N, dim)

            # Expand N times : (N,N,dim)
            atom_adj_fea = torch.unsqueeze(atom_fea_, 1).expand(N, N, dim) # (N, N, dim)
            atom_adj_fea = atom_adj_fea.contiguous().view(-1, dim) # (N*N, dim)

            # Decoder

            # Bilinear Layer : Adjacency List Reconstruction
            edge_p = self.bilinear(atom_adj_fea, atom_nbr_fea)
            edge_p = self.linear_edge(edge_p)
            edge_p=F.log_softmax(edge_p, dim=1)
            edge_prob_list.append(edge_p)

            # Node Feature Reconstruction
            if cfg.normalize == "False":
                atom_feature_list.append(self.linear_node(atom_fea_sense_norm))
            else:
                atom_feature_list.append(self.linear_node(atom_fea_))

        atom_feature_list = torch.cat(atom_feature_list, dim=0)
        edge_prob_list = torch.cat(edge_prob_list, dim=0)

        return (atom_feature_list, edge_prob_list), (batch.y, batch.adj) #at_nums


class GraphEmbeddings(torch.nn.Module):
    """
    A head to generate graph embeddings from node embeddings.
    Args:
        dim_in (int): The dimension of the input features.
    """
    
    def __init__(self,
        dim_in,
        dim_graph_emb
    ):
        super(GraphEmbeddings, self).__init__()

        #self.linear = nn.Linear(dim_in, dim_graph_emb)
        #self.silu = nn.SiLU()
        self.MLP = nn.Sequential(pyg_nn.Linear(dim_in, dim_in), 
                                nn.SiLU(inplace=True), #crysatom fan softplus
                                pyg_nn.Linear(dim_in, dim_in))                

    def forward(self, batch):
        # Get the number of graphs in the batch
        dim_size = int(batch.batch.max().item() + 1) #batch.batch_size
        
        pooled_node_embeddings = scatter(batch.x, batch.batch, dim=0, reduce="mean", dim_size=dim_size).squeeze(-1) #perform pooling of node features by graphs

        #graph_embedding = self.linear(self.silu(pooled_node_embeddings))
        #graph_embedding = self.silu(graph_embedding)
        graph_embedding = self.MLP(pooled_node_embeddings)

        return graph_embedding, None#lo segon hauria de ser true label, arreglo pochillo de moment

