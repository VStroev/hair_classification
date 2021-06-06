import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torchvision import models as models
from pytorch_lightning import metrics


class HairDetectorModel(pl.LightningModule):
    def __init__(self):
        super(HairDetectorModel, self).__init__()
        self.backbone = models.shufflenet_v2_x0_5(pretrained=True)
        self.backbone.fc = torch.nn.Linear(1024, 1)
        self.f1 = metrics.classification.F1(2)

    def forward(self, x):
        return F.sigmoid(self.backbone(x))

    def training_step(self, batch, batch_nb):
        x, y = batch
        perdiction = self(x).squeeze(-1)
        loss = F.binary_cross_entropy(perdiction, y.float())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)