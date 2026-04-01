import torch
import torch.nn as nn
import torch.nn.functional as F



class SiameseNetSingleSubject(nn.Module):
    """
    Siamese network for learning similarity between pairs of time-series segments.
    
    Uses a shared CNN encoder to embed each segment and compares them using distance.
    
    Args:
        embedding_dim (int): Dimensionality of the latent embedding space.
    """
    def __init__(self, embedding_dim, dropout=0.3, normalized_output=False):
        super(SiameseNetSingleSubject, self).__init__()
        self.encoder = CNNEncoderSingleSubject(embedding_dim, dropout, normalized_output)

    def forward(self, x1, x2):
        embed1 = self.encoder(x1)
        embed2 = self.encoder(x2)
        distance = (embed1 - embed2).norm(p=2, dim=1)   # shape: (batch,)
        
        return distance 
    

class SiameseNetMultiSubject(nn.Module):
    """
    Siamese network for learning similarity between pairs of time-series segments.
    
    Uses a shared CNN encoder to embed each segment and compares them using distance.
    
    Args:
        embedding_dim (int): Dimensionality of the latent embedding space.
    """
    def __init__(self, embedding_dim, dropout=0.3, normalized_output=True):
        super(SiameseNetMultiSubject, self).__init__()
        self.encoder = CNNEncoderMultiSubject(embedding_dim, dropout, normalized_output)

    def forward(self, x1, x2):
        embed1 = self.encoder(x1)
        embed2 = self.encoder(x2)
        distance = F.pairwise_distance(embed1, embed2)
        return distance 
    

class SupConNetMultiSubject(nn.Module):
    """
    Encodes two views and returns L2-normalized projection vectors.
    """
    def __init__(self, embedding_dim=32, proj_dim=32, dropout=0.1, normalized_output=True):
        super().__init__()
        self.encoder = CNNEncoderMultiSubject(embedding_dim, dropout, normalized_output)
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x,return_raw=False):
        """
        x: (B, 1, T)
        return: z (B, D) L2-normalized projection
        """
        h = self.encoder.embed(x)          # (B, emb)
        z_raw = self.proj(h)               
        z = F.normalize(z_raw, p=2, dim=1) # (B, proj) in case additional projection is needed
        return h
    
    def embed(self,x):
        return self.forward(x)
       

class CNNEncoderSingleSubject(nn.Module): 
    """
    CNN-based encoder for embedding 1D neural time-series signals into a functional latent space.
    
    This version uses sequential 1D convolutions, GELU activations, dropout, and pooling layers.
    The output is a fixed-size embedding vector.
    
    Args:
        embedding_dim (int): The dimensionality of the final embedding vector.
    """
    def __init__(self, embedding_dim, dropout=0.3, normalized_output = False):
        super(CNNEncoderSingleSubject, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv1d(1, 64, kernel_size=15, stride=4, padding=0),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(3, stride=2),
            nn.Dropout(dropout),

            nn.Conv1d(64, 64, kernel_size=11, stride=2, padding=0), 
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(3, stride=2),
            nn.Dropout(dropout),
            
            nn.Conv1d(64, 64, kernel_size=7, stride=2, padding=0),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(3, stride=2),
            nn.Dropout(dropout),
            
            nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(3, stride=2),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(10)
            
        ])

        self.fc = nn.Sequential(
            nn.Linear(10*64, embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),

        )

        self.normalized_output = normalized_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.normalized_output:
            return F.normalize(x, p=2, dim=1)
        else:
            return x
    
    def embed(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.normalized_output:
            return F.normalize(x, p=2, dim=1)
        else:
            return x
        

class CNNEncoderMultiSubject(nn.Module):
    """
    CNN-based encoder for embedding 1D neural time-series signals into a functional latent space.
    
    This version uses sequential 1D convolutions, GELU activations, dropout, and pooling layers.
    The output is a fixed-size embedding vector.
    
    Args:
        embedding_dim (int): The dimensionality of the final embedding vector.
    """
    def __init__(self, embedding_dim, dropout=0.3, normalized_output=True):
        super(CNNEncoderMultiSubject, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv1d(1, 64, kernel_size=15, stride=4, padding=0),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(3, stride=2),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=11, stride=2, padding=0), # trying 16 now
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(3, stride=2),
            nn.Dropout(dropout),

            nn.Conv1d(128, 128, kernel_size=11, stride=2, padding=0), # trying 16 now
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(3, stride=2),
            nn.Dropout(dropout),

            nn.Conv1d(128, 128, kernel_size=11, stride=2, padding=0), # trying 16 now
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(3, stride=2),
            nn.Dropout(dropout),
            
            nn.Conv1d(128, 128, kernel_size=7, stride=2, padding=0),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(128, 128, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(10)
        ])

        self.fc = nn.Sequential(
            nn.Linear(10*128, embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.normalized_output = normalized_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.normalized_output:
            return F.normalize(x, p=2, dim=1)
        else:
            return x
    
    def embed(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.normalized_output:
            return F.normalize(x, p=2, dim=1)
        else:
            return x
