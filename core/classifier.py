import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional


class SimpleCNN(nn.Module):
    """
    Simple CNN classifier with 2 conv layers and adaptive pooling.
    """

    def __init__(self, num_classes: int = 2) -> None:
        """
        Parameters
        ----------
        num_classes : int, optional
            Number of output classes, by default 2
        """
        super(SimpleCNN, self).__init__()

        # Conv2d layers with max pooling
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Global average pooling layer
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 4, H, W)

        Returns
        -------
        torch.Tensor
            Output logits of shape (B, num_classes)
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.gap(x)
        return self.classifier(x)
    

class BetterCNN(nn.Module):
    """
    CNN classifier with 4 residual-style blocks and adaptive pooling.
    """

    def __init__(self, num_classes: int = 2, drop: float = 0.3) -> None:
        """
        Parameters
        ----------
        num_classes : int, optional
            Number of output classes, by default 2
        drop : float, optional
            Dropout rate, by default 0.3
        """
        super().__init__()
        self.block1 = self._make_block(4, 64)
        self.block2 = self._make_block(64, 128)
        self.block3 = self._make_block(128, 256)
        self.block4 = self._make_block(256, 512)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(256, num_classes)
        )

    @staticmethod
    def _make_block(in_ch: int, out_ch: int) -> nn.Sequential:
        """
        Create a conv block with batch norm and max pooling.

        Parameters
        ----------
        in_ch : int
            Input channels
        out_ch : int
            Output channels

        Returns
        -------
        nn.Sequential
            Residual-style conv block
        """
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 4, H, W)

        Returns
        -------
        torch.Tensor
            Output logits of shape (B, num_classes)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x)
    

class FusionCNN(nn.Module):
    """
    Dual-path feature fusion with GAP and GMP pooling.
    Concatenates global average and max pooled features for classification.
    """

    def __init__(self, num_classes: int = 2, drop: float = 0.4) -> None:
        """
        Parameters
        ----------
        num_classes : int, optional
            Number of output classes, by default 2
        drop : float, optional
            Dropout rate, by default 0.4
        """
        super(FusionCNN, self).__init__()

        self.block1 = self._make_block(4, 64)
        self.block2 = self._make_block(64, 128)
        self.block3 = self._make_block(128, 256)
        self.block4 = self._make_block(256, 512)

        # GAP branch
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gap_fc = nn.Linear(512, 256)

        # GMP branch
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.gmp_fc = nn.Linear(512, 256)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(128, num_classes)
        )

    @staticmethod
    def _make_block(in_ch: int, out_ch: int) -> nn.Sequential:
        """
        Create a conv block with batch norm and max pooling.

        Parameters
        ----------
        in_ch : int
            Input channels
        out_ch : int
            Output channels

        Returns
        -------
        nn.Sequential
            Residual-style conv block
        """
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dual-path fusion.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 4, H, W)

        Returns
        -------
        torch.Tensor
            Output logits of shape (B, num_classes)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # GAP branch
        g = self.gap(x).flatten(1)
        g = self.gap_fc(g)

        # GMP branch
        f = self.gmp(x).flatten(1)
        f = self.gmp_fc(f)

        # Fusion and classification
        fuse = torch.cat([g, f], dim=1)
        return self.classifier(fuse)
    


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Combines channel and spatial attention mechanisms.
    """

    def __init__(self, in_ch: int, r: int = 16, kernel_size: int = 7) -> None:
        """
        Parameters
        ----------
        in_ch : int
            Input channels
        r : int, optional
            Reduction ratio for channel attention, by default 16
        kernel_size : int, optional
            Kernel size for spatial attention, by default 7
        """
        super(CBAM, self).__init__()

        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // r, in_ch, 1, bias=False)
        )

        # Spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W)

        Returns
        -------
        torch.Tensor
            Output tensor with attention applied, shape (B, C, H, W)
        """
        # Channel attention
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        ca = torch.sigmoid(avg_out + max_out)
        x = x * ca

        # Spatial attention
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        x = x * sa
        return x


class AttentionCNN(nn.Module):
    """
    CNN classifier with CBAM attention module.
    Combines convolutional blocks with channel and spatial attention.
    """

    def __init__(self, num_classes: int = 2, drop: float = 0.3) -> None:
        """
        Parameters
        ----------
        num_classes : int, optional
            Number of output classes, by default 2
        drop : float, optional
            Dropout rate, by default 0.3
        """
        super(AttentionCNN, self).__init__()

        self.block1 = self._make_block(4, 64)
        self.block2 = self._make_block(64, 128)
        self.block3 = self._make_block(128, 256)
        self.block4 = self._make_block(256, 512)

        # CBAM attention module
        self.att = CBAM(512)

        # Global average pooling layer
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(128, num_classes)
        )

    @staticmethod
    def _make_block(in_ch: int, out_ch: int) -> nn.Sequential:
        """
        Create a conv block with batch norm and max pooling.

        Parameters
        ----------
        in_ch : int
            Input channels
        out_ch : int
            Output channels

        Returns
        -------
        nn.Sequential
            Residual-style conv block
        """
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 4, H, W)

        Returns
        -------
        torch.Tensor
            Output logits of shape (B, num_classes)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.att(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x)


class AdvancedSpatialClassifier(nn.Module):
    """
    Advanced CNN with multi-scale features, SE attention, and residual connections.
    """

    def __init__(
        self,
        input_channels: int = 4,
        conv_channels: List[int] = None,
        fc_dims: List[int] = None,
        dropout: float = 0.3,
        use_se: bool = True,
        use_residual: bool = True,
        use_bn: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        input_channels : int, optional
            Number of input channels, by default 4
        conv_channels : List[int], optional
            Channel sizes for conv layers, by default [32, 64, 128]
        fc_dims : List[int], optional
            Fully connected layer dimensions, by default [256, 128]
        dropout : float, optional
            Dropout rate, by default 0.3
        use_se : bool, optional
            Whether to use SE attention, by default True
        use_residual : bool, optional
            Whether to use residual connections, by default True
        use_bn : bool, optional
            Whether to use batch normalization, by default True
        """
        super().__init__()

        if conv_channels is None:
            conv_channels = [32, 64, 128]
        if fc_dims is None:
            fc_dims = [256, 128]

        self.use_residual = use_residual
        self.use_se = use_se

        def _block(
            in_ch: int,
            out_ch: int,
            skip_conv: Optional[nn.Module] = None,
        ) -> tuple:
            """
            Create a conv block with optional residual connection.

            Parameters
            ----------
            in_ch : int
                Input channels
            out_ch : int
                Output channels
            skip_conv : Optional[nn.Module]
                Optional skip connection layer

            Returns
            -------
            tuple
                (Sequential block, skip connection layer)
            """
            layers = [
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=not use_bn),
                nn.BatchNorm2d(out_ch) if use_bn else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=not use_bn),
                nn.BatchNorm2d(out_ch) if use_bn else nn.Identity(),
            ]
            return nn.Sequential(*layers), skip_conv

        # Multi-scale conv blocks
        self.conv1, self.skip1 = _block(
            input_channels,
            conv_channels[0],
            nn.Conv2d(input_channels, conv_channels[0], 1)
            if use_residual
            else None,
        )
        self.pool1 = nn.MaxPool2d(2)

        self.conv2, self.skip2 = _block(
            conv_channels[0],
            conv_channels[1],
            nn.Conv2d(conv_channels[0], conv_channels[1], 1)
            if use_residual
            else None,
        )
        self.pool2 = nn.MaxPool2d(2)

        self.conv3, self.skip3 = _block(
            conv_channels[1],
            conv_channels[2],
            nn.Conv2d(conv_channels[1], conv_channels[2], 1)
            if use_residual
            else None,
        )

        # SE attention module
        if use_se:
            c = conv_channels[2]
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c, c // 16, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(c // 16, c, 1, bias=False),
                nn.Sigmoid(),
            )

        # Global average pooling and classifier
        self.gap = nn.AdaptiveAvgPool2d(1)
        classifier_list = []
        prev_dim = conv_channels[2]
        for hid in fc_dims:
            classifier_list += [
                nn.Linear(prev_dim, hid),
                nn.BatchNorm1d(hid) if use_bn else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev_dim = hid
        classifier_list.append(nn.Linear(prev_dim, 2))
        self.classifier = nn.Sequential(*classifier_list)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights using kaiming normal."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, input_channels, H, W)

        Returns
        -------
        torch.Tensor
            Output logits of shape (B, 2)
        """
        # Block 1
        out = self.conv1(x)
        if self.use_residual:
            x = out + (self.skip1(x) if self.skip1 is not None else x)
        x = F.relu(x)
        x = self.pool1(x)

        # Block 2
        out = self.conv2(x)
        if self.use_residual:
            x = out + (self.skip2(x) if self.skip2 is not None else x)
        x = F.relu(x)
        x = self.pool2(x)

        # Block 3
        out = self.conv3(x)
        if self.use_residual:
            x = out + (self.skip3(x) if self.skip3 is not None else x)
        x = F.relu(x)

        # SE attention
        if self.use_se:
            x = x * self.se(x)

        # Global pooling and classification
        x = self.gap(x).flatten(1)
        return self.classifier(x)
class TransformerSpatialClassifier(nn.Module):
    """
    Transformer-based image classifier with dynamic positional encoding.
    Converts spatial features to patches and processes via transformer encoder.
    """

    def __init__(
        self,
        input_channels: int = 4,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        fc_dims: Optional[List[int]] = None,
        dropout: float = 0.3,
        use_batchnorm: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        input_channels : int, optional
            Number of input channels, by default 4
        embed_dim : int, optional
            Embedding dimension, by default 128
        num_heads : int, optional
            Number of transformer attention heads, by default 4
        num_layers : int, optional
            Number of transformer encoder layers, by default 2
        fc_dims : Optional[List[int]], optional
            Fully connected layer dimensions, by default [256, 128]
        dropout : float, optional
            Dropout rate, by default 0.3
        use_batchnorm : bool, optional
            Whether to use batch normalization, by default True
        """
        super().__init__()

        if fc_dims is None:
            fc_dims = [256, 128]

        self.embed_dim = embed_dim

        # Patch embedding layer with stride=2 downsampling
        self.patch_embed = nn.Conv2d(
            input_channels,
            embed_dim,
            kernel_size=3,
            stride=2,
            padding=1
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Classification head
        classifier = []
        prev_dim = embed_dim
        for hid in fc_dims:
            classifier += [
                nn.Linear(prev_dim, hid),
                nn.BatchNorm1d(hid) if use_batchnorm else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev_dim = hid
        classifier.append(nn.Linear(prev_dim, 2))
        self.classifier = nn.Sequential(*classifier)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Parameter):
                nn.init.normal_(m, 0, 0.02)

    @staticmethod
    def _pos_encoding(
        num_patches: int,
        embed_dim: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generate sinusoidal positional encoding.

        Parameters
        ----------
        num_patches : int
            Number of patches
        embed_dim : int
            Embedding dimension
        device : torch.device
            Device to create tensor on

        Returns
        -------
        torch.Tensor
            Positional encoding of shape (1, num_patches, embed_dim)
        """
        pe = torch.zeros(1, num_patches, embed_dim, device=device)
        position = torch.arange(
            0,
            num_patches,
            dtype=torch.float,
            device=device
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float, device=device)
            * (-math.log(10000.0) / embed_dim)
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, input_channels, H, W)

        Returns
        -------
        torch.Tensor
            Output logits of shape (B, 2)
        """
        B, C, H, W = x.shape

        # Patch embedding
        patches = self.patch_embed(x)
        _, _, newH, newW = patches.shape
        patches = patches.flatten(2).transpose(1, 2)
        num_patches = newH * newW

        # Dynamic positional encoding
        pos_enc = self._pos_encoding(num_patches, self.embed_dim, patches.device)
        patches = patches + pos_enc

        # Transformer encoding
        encoded = self.transformer(patches)

        # Global average pooling and classification
        features = encoded.mean(dim=1)
        return self.classifier(features)



NAME_MAP = {
    'SimpleCNN': SimpleCNN,
    'BetterCNN': BetterCNN,
    'FusionCNN': FusionCNN,
    'AttentionCNN': AttentionCNN,
    'AdvancedSpatialClassifier': AdvancedSpatialClassifier,
    'TransformerSpatialClassifier': TransformerSpatialClassifier,
}


def build_model(net: str) -> nn.Module:
    """
    Build and return a classifier model by name.

    Parameters
    ----------
    net : str
        Name of the model architecture

    Returns
    -------
    nn.Module
        Instantiated classifier model

    Raises
    ------
    KeyError
        If model name not found in NAME_MAP
    """
    return NAME_MAP[net]()
