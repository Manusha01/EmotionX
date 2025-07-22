# emotionx_train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer
import torchaudio
import torchvision.models as models

# ——— Model Definitions ———

class TextEncoder(nn.Module):
    def __init__(self, pretrained='bert-base-uncased', hidden_size=768):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained)
        self.bert = BertModel.from_pretrained(pretrained)
    def forward(self, texts):
        enc = self.tokenizer(texts, return_tensors='pt',
                             padding=True, truncation=True, max_length=128)
        out = self.bert(**enc)
        return out.last_hidden_state[:,0]       # [CLS]

class AudioEncoder(nn.Module):
    def __init__(self, in_mels=40, hid=768):
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=in_mels)
        self.conv = nn.Sequential(
            nn.Conv1d(in_mels, 128, 3, padding=1), nn.ReLU(),
            nn.Conv1d(128, hid, 3, padding=1),   nn.ReLU(),
        )
        layer = nn.TransformerEncoderLayer(d_model=hid, nhead=8)
        self.trans = nn.TransformerEncoder(layer, num_layers=4)
    def forward(self, waveforms):
        x = self.melspec(waveforms)                  # (B, M, T)
        x = self.conv(x)                             # (B, H, T)
        x = x.permute(2,0,1)                         # (T, B, H)
        y = self.trans(x)                            # (T, B, H)
        return y.mean(0)                             # (B, H)

class VisualEncoder(nn.Module):
    def __init__(self, hid=768):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        feat = list(resnet.children())[:-2]
        self.backbone = nn.Sequential(*feat)
        self.pool     = nn.AdaptiveAvgPool2d((1,1))
        self.fc       = nn.Linear(resnet.fc.in_features, hid)
        layer = nn.TransformerEncoderLayer(d_model=hid, nhead=8)
        self.trans    = nn.TransformerEncoder(layer, num_layers=2)
    def forward(self, frames):
        f = self.backbone(frames)                    # (B, C, H', W')
        p = self.pool(f).view(frames.size(0), -1)    # (B, C)
        e = self.fc(p).unsqueeze(0)                  # (1, B, H)
        t = self.trans(e)                            # (1, B, H)
        return t.squeeze(0)                          # (B, H)

class FusionDecoder(nn.Module):
    def __init__(self, hid=768, num_classes=7):
        super().__init__()
        layer = nn.TransformerDecoderLayer(d_model=hid, nhead=8)
        self.dec = nn.TransformerDecoder(layer, num_layers=6)
        self.cls = nn.Linear(hid, num_classes)
    def forward(self, t, a, v):
        # t,a,v: (B, H)
        mem = torch.stack([t,a,v], dim=0).permute(1,0,2)  # (B,3,H)→(3,B,H)→(B,3,H)
        tgt = t.unsqueeze(0)                              # (1,B,H)
        out = self.dec(tgt, mem.permute(1,0,2))           # (1,B,H)
        return self.cls(out.squeeze(0))                   # (B, num_classes)

class EmotionX(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.text = TextEncoder()
        self.audio= AudioEncoder()
        self.visual=VisualEncoder()
        self.fusion=FusionDecoder(num_classes=num_classes)
    def forward(self, texts, waves, frames):
        t = self.text(texts)
        a = self.audio(waves)
        v = self.visual(frames)
        return self.fusion(t,a,v)

# ——— Training / Eval Loops ———

def train_epoch(model, loader, opt, crit, device):
    model.train()
    total_loss = 0
    for texts, waves, frames, labs in loader:
        waves, frames, labs = waves.to(device), frames.to(device), labs.to(device)
        opt.zero_grad()
        logits = model(texts, waves, frames)
        loss   = crit(logits, labs)
        loss.backward(); opt.step()
        total_loss += loss.item()
    return total_loss/len(loader)

def eval_model(model, loader, device):
    from sklearn.metrics import accuracy_score, f1_score
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for texts, waves, frames, labs in loader:
            logits = model(texts, waves.to(device), frames.to(device))
            preds  = logits.argmax(1).cpu().tolist()
            all_preds.extend(preds); all_labels.extend(labs.tolist())
    return accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average='macro')

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = EmotionX(num_classes=7).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    # Replace these with your actual dataset classes
    train_loader = DataLoader(IEMOCAPDataset(split="train"), batch_size=16, shuffle=True)
    val_loader   = DataLoader(IEMOCAPDataset(split="val"),   batch_size=16)
    test_loader  = DataLoader(MELDDataset(),                 batch_size=16)

    # Training
    for epoch in range(1, 4):  # demo 3 epochs
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_acc, val_f1 = eval_model(model, val_loader, device)
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.3f} | Val Acc: {val_acc:.3%} | Val F1: {val_f1:.3%}")

    # Final Test
    test_acc, test_f1 = eval_model(model, test_loader, device)
    print(f"▶︎ Test Accuracy: {test_acc:.3%} | Test Macro-F1: {test_f1:.3%}")
