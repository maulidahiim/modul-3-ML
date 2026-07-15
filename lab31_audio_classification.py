import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader, random_split
from torchaudio.datasets import SPEECHCOMMANDS
import torch.nn.functional as F
import time

# ============================================================
# HYPERPARAMETERS
# ============================================================

SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 400
HOP_LENGTH = 160
MAX_LEN = 101

BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASSES = [
    'yes',
    'no',
    'up',
    'down',
    'left',
    'right',
    'on',
    'off',
    'stop',
    'go'
]

# ============================================================
# MEL-SPECTROGRAM TRANSFORM
# ============================================================

mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS
)

to_db = T.AmplitudeToDB()

# ============================================================
# PREPROCESS FUNCTION
# ============================================================

def preprocess(waveform, sr):

    # Resample jika sample rate berbeda
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform,
            sr,
            SAMPLE_RATE
        )

    # Potong / tambah audio menjadi 1 detik
    target_len = SAMPLE_RATE

    if waveform.shape[-1] < target_len:
        waveform = F.pad(
            waveform,
            (0, target_len - waveform.shape[-1])
        )
    else:
        waveform = waveform[..., :target_len]

    # Mel Spectrogram
    mel = to_db(mel_transform(waveform))

    return mel

# ============================================================
# DATASET WRAPPER
# ============================================================

class SpeechCommandsSubset(torch.utils.data.Dataset):

    def __init__(self, subset):

        self.dataset = subset
        self.label_map = {
            c: i for i, c in enumerate(CLASSES)
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        waveform, sr, label, *_ = self.dataset[idx]

        mel = preprocess(waveform, sr)

        # Skip label selain class yang dipakai
        if label not in self.label_map:
            return self.__getitem__((idx + 1) % len(self.dataset))

        return mel, self.label_map[label]

# ============================================================
# DOWNLOAD DATASET
# ============================================================

print("Downloading dataset...")

dataset = SPEECHCOMMANDS(
    root="./data",
    download=True
)

print("Filtering dataset...")

filtered = [
    x for x in dataset
    if x[2] in CLASSES
]

subset = SpeechCommandsSubset(filtered)

train_size = int(0.8 * len(subset))
val_size = len(subset) - train_size

train_ds, val_ds = random_split(
    subset,
    [train_size, val_size]
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE
)

print("Dataset ready!")

# ============================================================
# MODEL A - SMALL CNN
# ============================================================

class SmallCNN(nn.Module):

    def __init__(self, num_classes=10):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Linear(
            128,
            num_classes
        )

    def forward(self, x):

        x = self.features(x)

        x = x.squeeze(-1).squeeze(-1)

        return self.classifier(x)

# ============================================================
# MODEL B - TINY TRANSFORMER
# ============================================================

class TinyAudioTransformer(nn.Module):

    def __init__(
        self,
        n_mels=80,
        d_model=128,
        nhead=4,
        num_layers=2,
        num_classes=10,
        max_len=101
    ):

        super().__init__()

        self.input_proj = nn.Linear(
            n_mels,
            d_model
        )

        self.pos_emb = nn.Embedding(
            max_len,
            d_model
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            enc_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Linear(
            d_model,
            num_classes
        )

    def forward(self, x):

        # (B,1,80,101) -> (B,101,80)
        x = x.squeeze(1).permute(0, 2, 1)

        B, T, _ = x.shape

        x = self.input_proj(x)

        pos = torch.arange(
            T,
            device=x.device
        )

        x = x + self.pos_emb(pos).unsqueeze(0)

        x = self.transformer(x)

        x = x.mean(dim=1)

        return self.classifier(x)

# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for mel, label in loader:

            mel = mel.to(DEVICE)
            label = label.to(DEVICE)

            output = model(mel)

            pred = output.argmax(1)

            correct += (pred == label).sum().item()

            total += label.size(0)

    return correct / total

# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_model(
    model,
    train_loader,
    val_loader,
    epochs=EPOCHS
):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    criterion = nn.CrossEntropyLoss()

    model.to(DEVICE)

    start = time.time()

    for epoch in range(1, epochs + 1):

        model.train()

        total_loss = 0
        correct = 0
        total = 0

        for mel, label in train_loader:

            mel = mel.to(DEVICE)
            label = label.to(DEVICE)

            optimizer.zero_grad()

            output = model(mel)

            loss = criterion(output, label)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            pred = output.argmax(1)

            correct += (pred == label).sum().item()

            total += label.size(0)

        val_acc = evaluate(
            model,
            val_loader
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Loss: {total_loss / len(train_loader):.4f} | "
            f"Train Acc: {correct / total:.3f} | "
            f"Val Acc: {val_acc:.3f}"
        )

    elapsed = time.time() - start

    print(f"Training Time: {elapsed:.1f} seconds")

    return elapsed

# ============================================================
# PARAMETER COUNTER
# ============================================================

def count_params(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n===================================")
    print(" AUDIO CLASSIFICATION LAB 3.1 ")
    print(" CNN vs TRANSFORMER ")
    print("===================================\n")

    # ========================================================
    # CNN
    # ========================================================

    print("\nTraining CNN Model...\n")

    cnn_model = SmallCNN()

    print(
        f"CNN Parameters: {count_params(cnn_model):,}"
    )

    cnn_time = train_model(
        cnn_model,
        train_loader,
        val_loader
    )

    cnn_acc = evaluate(
        cnn_model,
        val_loader
    )

    print(f"CNN Final Accuracy: {cnn_acc:.3f}")

    # ========================================================
    # TRANSFORMER
    # ========================================================

    print("\nTraining Transformer Model...\n")

    trans_model = TinyAudioTransformer()

    print(
        f"Transformer Parameters: {count_params(trans_model):,}"
    )

    trans_time = train_model(
        trans_model,
        train_loader,
        val_loader
    )

    trans_acc = evaluate(
        trans_model,
        val_loader
    )

    print(
        f"Transformer Final Accuracy: {trans_acc:.3f}"
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("\n===================================")
    print(" FINAL COMPARISON ")
    print("===================================\n")

    print(f"CNN Accuracy         : {cnn_acc:.3f}")
    print(f"Transformer Accuracy : {trans_acc:.3f}")

    print(f"\nCNN Parameters         : {count_params(cnn_model):,}")
    print(f"Transformer Parameters : {count_params(trans_model):,}")

    print(f"\nCNN Training Time         : {cnn_time:.1f} s")
    print(f"Transformer Training Time : {trans_time:.1f} s")