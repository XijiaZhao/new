import torch
import torch.nn as nn
import math
import torch.nn.functional as F
torch.manual_seed(0)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a matrix of [max_len, d_model] representing the positional encodings
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to each element in the input sequence
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                  dim_feedforward=dim_feedforward, 
                                                  dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

    def forward(self, src):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output

class TimeSeriesTransformer(nn.Module):
    def __init__(self, num_features, num_classes, d_model, nhead, num_encoder_layers, dim_feedforward, dropout):
        super(TimeSeriesTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.encoder = nn.Linear(num_features, d_model)
        self.transformer_encoder = TransformerEncoder(d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
        self.decoder = nn.Linear(d_model, num_classes)

    def forward(self, src):
        src = self.encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output.mean(dim=1))
        return output
    
class SSLTransformer(nn.Module):
    """
    A self-supervised Transformer model for learning representations.

    :param num_features: Number of input features.
    :param d_model: Dimension of the model.
    :param nhead: Number of heads in the multiheadattention models.
    :param num_encoder_layers: Number of sub-encoder-layers in the encoder.
    :param dim_feedforward: Dimension of the feedforward network model.
    :param dropout: Dropout value.
    """
    def __init__(self, num_features, d_model, nhead, num_encoder_layers, dim_feedforward, proj_dimension,dropout):
        super().__init__()
        self.model_type = 'Transformer'
        self.encoder = nn.Linear(num_features, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.transformer_encoder = TransformerEncoder(d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
        # self.projection_head = nn.Linear(d_model, proj_dimension)  # Optional: Projection head

    def forward(self, src):
        """
        Forward pass of the SSLTransformer.

        :param src: Input tensor.
        :return: Transformed tensor.
        """
        src = self.encoder(src)
        src = self.norm(src)
        src = self.dropout(src)
        output = self.transformer_encoder(src)
        # output = self.projection_head(output)  # Optional: Apply projection head
        output = output.mean(dim=1)
        return output
    
class ContrastiveLosses:
    def __init__(self, margin=1.0):
        """
        Initialize the ContrastiveLosses class.

        :param margin: Margin for contrastive loss. Default is 1.0.
        """
        self.margin = margin

    def standard_contrastive_loss(self, out1, out2, label):
        """
        Standard Contrastive Loss Function

        :param out1: Output from the network for the first input of the pair.
        :param out2: Output from the network for the second input of the pair.
        :param label: The label indicating if the pair is similar (1) or dissimilar (0).
        :return: Calculated loss.
        """
        distance = F.pairwise_distance(out1, out2)
        loss = torch.mean((1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2) +
                          label * torch.pow(distance, 2))
        return loss

class ClassificationModel(nn.Module):
    def __init__(self, ssl_model, num_classes, feature_size=256):
        super(ClassificationModel, self).__init__()
        self.ssl_model = ssl_model
        if not isinstance(ssl_model, nn.Module):
            raise TypeError("ssl_model must be an instance of nn.Module")
        
        for param in self.ssl_model.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 1024),  # Adjust the sizes as needed
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.ssl_model(x)
        # Assuming the SSL model output is the feature vector we want to use
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x