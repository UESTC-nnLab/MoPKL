import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .darknet import BaseConv, CSPDarknet, CSPLayer, DWConv
from einops import rearrange

class Motion_Relation_Mining(nn.Module):
    def __init__(self, num_regions=64, region_size=8):
        super(Motion_Relation_Mining, self).__init__()
        self.num_regions = num_regions
        self.region_size = region_size
        self.tau = 13
        self.eta = 0.8

    def preprocess(self, feature_map):
        batch_size, frames, channels, height, width = feature_map.shape
        grid_h, grid_w = self.region_size, self.region_size
        feature_map = feature_map.view(batch_size, frames, channels, height // grid_h, grid_h, width // grid_w, grid_w)
        feature_map = feature_map.permute(0, 1, 2, 3, 5, 4, 6)
        feature_map = feature_map.reshape(batch_size, frames, channels, -1, grid_h, grid_w)
        return feature_map

    def calculate_motion_information(self, feature_map):
        batch_size, frames, channels, num_grids, grid_h, grid_w = feature_map.shape
        motion = feature_map[:, 1:] - feature_map[:, :-1]
        
        for b in range(batch_size):
            x_t_flat = feature_map[b, :-1].reshape(1, frames - 1, channels, num_grids, -1)
            x_t1_flat = feature_map[b, 1:].reshape(1, frames - 1, channels, num_grids, -1)
            
            joint_hist = torch.histc((x_t_flat * x_t1_flat).flatten(), bins=100)
            p_x = torch.histc(x_t_flat.flatten(), bins=100) / x_t_flat.numel()
            p_x1 = torch.histc(x_t1_flat.flatten(), bins=100) / x_t1_flat.numel()
            p_joint = joint_hist / joint_hist.sum()
            
            p_x = p_x / p_x.sum()
            p_x1 = p_x1 / p_x1.sum()
            mutual_info = (p_joint * (torch.log(p_joint + 1e-10) - torch.log(p_x * p_x1 + 1e-10))).sum()
            motion[b] = motion[b] + math.e**self.eta/torch.tanh(torch.log(mutual_info))
        return motion

    def select_top_regions(self, motion_scores):
        batch_size, num_frames, channels, num_grids, grid_h, grid_w = motion_scores.shape
        motion_magnitude = motion_scores.view(batch_size, num_frames, channels, -1).sum(dim=2)
        _, top_indices = motion_magnitude.view(batch_size, num_frames, -1).topk(self.num_regions, dim=-1)
        return top_indices

    def calculate_spatial_distances(self, top_indices):
        y_indices = torch.div(top_indices, self.num_regions, rounding_mode='floor')
        x_indices = top_indices % self.num_regions
        center_y = y_indices * self.region_size + self.region_size // 2
        center_x = x_indices * self.region_size + self.region_size // 2
        centers = torch.stack([center_y, center_x], dim=-1).float()

        batch_size, num_frames, num_regions, _ = centers.shape
        centers_flat = centers.view(batch_size * num_frames, num_regions, 2)
        mean_centers = centers_flat.mean(dim=1, keepdim=True)
        diff = centers_flat - mean_centers
        cov_matrices = torch.bmm(diff.transpose(1, 2), diff) / (num_regions - 1)
        cov_matrices = cov_matrices.view(batch_size, num_frames, 2, 2)

        inv_cov_matrices = torch.linalg.inv(cov_matrices + torch.eye(2, device=centers.device) * 1e-6)
        diff = centers.unsqueeze(2) - centers.unsqueeze(3)
        mahalanobis_distances = torch.sqrt((diff @ inv_cov_matrices.unsqueeze(2)) * diff).sum(dim=-1)
        return mahalanobis_distances

    def construct_graph(self, distances):
        batch_size, num_frames, num_regions, _ = distances.shape
        num_relations = self.tau**2
        flat_distances = distances.view(batch_size, num_frames, -1)
        _, topk_indices = torch.topk(flat_distances, num_relations, largest=False, dim=-1)
        new_distances = torch.zeros_like(flat_distances)
        new_distances.scatter_(2, topk_indices, 1)
        distances = new_distances.view(batch_size, num_frames, num_regions, num_regions)
        return distances

    def forward(self, input_tensor):
        feat_map = self.preprocess(input_tensor)
        motion_scores = self.calculate_motion_information(feat_map)
        selected_regions = self.select_top_regions(motion_scores)
        distances = self.calculate_spatial_distances(selected_regions)
        motion_graph = self.construct_graph(distances)
        return motion_graph




def normalize(features):
    min_val = features.min(dim=2, keepdim=True)[0]
    max_val = features.max(dim=2, keepdim=True)[0]
    normalized_features = (features - min_val) / (max_val - min_val)
    return normalized_features

def pairwise_distance(x, y, p=2): #p>=2
    return torch.sum((x - y) ** p, dim=-1)**(1/(p-1))

def weight(x, y, p=2):
    batch_size, num_frames, num_nodes, feature_dim = x.size()
    weights = torch.zeros(batch_size, device=x.device)
    for b in range(batch_size):
        weight = 0.0
        for f in range(num_frames):
            x_flat = x[b, f].unsqueeze(1)  
            y_flat = y[b, f].unsqueeze(0) 
            dist = pairwise_distance(x_flat, y_flat, p)
            weight += dist.min(dim=1)[0].mean()
        weights[b] = weight / num_frames
    return weights

def motion_vision_align(x, y):
    batch_size = x.size(0)
    weights_x = 1 / (1 + weight(x, y)) 
    weights_y = 1 / (1 + weight(y, x))
    normalized_x = normalize(x)
    normalized_y = normalize(y)
    
    aligned_features = torch.zeros_like(x)
    for b in range(batch_size):
        aligned_features[b] = weights_x[b].unsqueeze(-1).unsqueeze(-1) * normalized_x[b] + weights_y[b].unsqueeze(-1).unsqueeze(-1) * normalized_y[b]
    
    return aligned_features



class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.kaiming_uniform_(self.W.data, nonlinearity='leaky_relu')
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.kaiming_uniform_(self.a.data, nonlinearity='leaky_relu')

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
    
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

        
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nout, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x

 

class Feature_Extractor(nn.Module):
    def __init__(self, depth = 1.0, width = 1.0, in_features = ("dark3", "dark4", "dark5"), in_channels = [256, 512, 1024], depthwise = False, act = "silu"):
        super().__init__()
        Conv                = DWConv if depthwise else BaseConv
        self.backbone       = CSPDarknet(depth, width, depthwise = depthwise, act = act)
        self.in_features    = in_features
        self.upsample       = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0  = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )  
        self.reduce_conv1   = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )

    def forward(self, input):
        out_features            = self.backbone.forward(input)
        [feat1, feat2, feat3]   = [out_features[f] for f in self.in_features]
        P5          = self.lateral_conv0(feat3)
        P5_upsample = self.upsample(P5)
        P5_upsample = torch.cat([P5_upsample, feat2], 1)
        P5_upsample = self.C3_p4(P5_upsample)
        P4          = self.reduce_conv1(P5_upsample) 
        P4_upsample = self.upsample(P4) 
        P4_upsample = torch.cat([P4_upsample, feat1], 1) 

        P3_out      = self.C3_p3(P4_upsample)  
        
        return P3_out


class MoPKL(nn.Module):
    def __init__(self, num_classes, num_frame=5):
        super(MoPKL, self).__init__()
        
        self.num_frame = num_frame
        self.backbone = Feature_Extractor(0.33,0.50)
        self.fusion = Fusion_Module(channels=[128], num_frame=num_frame)
        self.head = YOLOXHead(num_classes=num_classes, width = 1.0, in_channels = [128], act = "silu")
    
        self.des1 = nn.Linear(300,4096)
        self.des2 = nn.Linear(130,64)
    
        self.wt = nn.Linear(4,1)
        self.wn = nn.Linear(128,64)
        self.wc = nn.Linear(64,128)
        self.fc =  nn.Linear(8192,4096)
        
        self.conv =  nn.Sequential(
            BaseConv(256,128,3,1),
            BaseConv(128,128,3,1))

        self.convT3d = nn.Sequential(nn.ConvTranspose3d(5,5,(1,4,4),(1,2,2),(0,1,1)),
                                    nn.ConvTranspose3d(5,5,(1,4,4),(1,2,2),(0,1,1)),
                                    nn.Conv3d(5,5,1,1,0))
        self.GAT = GAT(nfeat=4096,nhid=512,nout=4096,dropout=0.2,alpha=0.2,nheads=8)
        self.motion_graph = Motion_Relation_Mining(num_regions=64, region_size=8)
        
        
    def forward(self, inputs, descriptions=None):  
        feat = []
        feat2 = []
        outputs = []
        
        for i in range(self.num_frame):
            feats = self.backbone(inputs[:,:,i,:,:]) 
            feat.append(feats)

        feats = torch.stack(feat, 1) 

        B, T, N, W, H = feats.shape  
        
        feats_up = self.convT3d(feats) 
        motion_adj = self.motion_graph(feats_up)
        nodes = rearrange(feats, 'b t n w h -> b t n (w h)', b=B, t=T, n=N) 
        nodes = self.wn(nodes.transpose(-2,-1)).transpose(-2,-1)[:,1:5,:,:] 

        if self.training:
            descriptions = self.des1(descriptions) 
            descriptions = self.des2(descriptions.transpose(-2,-1)).transpose(-2,-1)[:,1:5,:,:] 
            nodes = motion_vision_align(nodes, descriptions)
        
        batch_relation = []
        for b in range(B):
            motion_relation = []
            for t in range(T-1):
                motion_relation.append(self.GAT(nodes[b,t,:,:], motion_adj[b,t,:,:])) 
            motion = torch.stack(motion_relation,0) 
            batch_relation.append(motion)
        motion = torch.stack(batch_relation,0) 
        motion = self.wt(motion.transpose(1,-1)).transpose(1,-1).squeeze(1)
        motion = self.wc(motion.transpose(1,-1)).transpose(1,-1) 
        motion = rearrange(motion, 'b n (w h) -> b n w h', b=B, n=N, w=W, h=H)
        motion = torch.cat([feat[-1], motion],1)
        motion = self.conv(motion)

        feat = self.fusion(feat,motion)
        outputs  = self.head(feat) 
        
        return  outputs    
            
        
class Fusion_Module(nn.Module):
    def __init__(self, channels=[128,256,512] ,num_frame=5):
        super().__init__()
        self.num_frame = num_frame
        
        self.conv_mv = nn.Sequential(
            BaseConv(channels[0]*2, channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1)
        )
        self.conv_cr_mix = nn.Sequential(
            BaseConv(channels[0]*2, channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1)
        )
        self.conv_final = nn.Sequential(
            BaseConv(channels[0]*2, channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1)
        )
        self.conv3d = nn.Conv3d(in_channels=self.num_frame, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, feat, motion):
        f_feats = []
        r_feats = torch.stack([self.conv_mv(torch.cat([feat[i], motion], dim=1)) for i in range(self.num_frame)], dim=1)
        r_feats = self.conv3d(r_feats).squeeze(1)
        c_feat = self.conv_final(torch.cat([r_feats,feat[-1]], dim=1))
        f_feats.append(c_feat)

        return f_feats

    

class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width = 1.0, in_channels = [16, 32, 64], act = "silu"):
        super().__init__()
        Conv            =  BaseConv
        
        self.cls_convs  = nn.ModuleList()
        self.reg_convs  = nn.ModuleList()
        self.cls_preds  = nn.ModuleList()
        self.reg_preds  = nn.ModuleList()
        self.obj_preds  = nn.ModuleList()
        self.stems      = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(BaseConv(in_channels = int(in_channels[i] * width), out_channels = int(256 * width), ksize = 1, stride = 1, act = act))
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = num_classes, kernel_size = 1, stride = 1, padding = 0)
            )
            
            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 4, kernel_size = 1, stride = 1, padding = 0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 1, kernel_size = 1, stride = 1, padding = 0)
            )

    def forward(self, inputs):
        
        outputs = []
        for k, x in enumerate(inputs):
            x       = self.stems[k](x)
            cls_feat    = self.cls_convs[k](x)
            cls_output  = self.cls_preds[k](cls_feat)
            reg_feat    = self.reg_convs[k](x)
            reg_output  = self.reg_preds[k](reg_feat)
            obj_output  = self.obj_preds[k](reg_feat)
            output      = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs


