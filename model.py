import os

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

# Hyperparameters not stored in the checkpoint but fixed in the notebook.
NUM_LAYERS = 1
DROPOUT = 0.4
RESNET_FEATURE_SIZE = 2048

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class ImageEncoder(nn.Module):
    def __init__(self, input_size=2048, hidden_size=512, dropout=0.3):
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, image_features):
        features = self.fc(image_features)
        features = self.relu(features)
        features = self.dropout(features)
        return features


class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=256, num_layers=1, dropout=0.4, pad_idx=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.embedding_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.image_to_cell = nn.Linear(hidden_size, hidden_size)

    def forward(self, image_features, captions):
        embeddings = self.embedding(captions)
        embeddings = self.embedding_dropout(embeddings)

        hidden_state = image_features.unsqueeze(0)
        cell_state = self.image_to_cell(image_features).unsqueeze(0)

        lstm_output, _ = self.lstm(embeddings, (hidden_state, cell_state))
        lstm_output = self.dropout(lstm_output)

        return self.fc(lstm_output)


class ImageCaptioningModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, image_features, captions):
        encoded_features = self.encoder(image_features)
        return self.decoder(encoded_features, captions)


class CaptionModel:
    """Loads the checkpoint once and generates captions from raw images."""

    def __init__(self, checkpoint_path: str, device: str = None):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found at '{checkpoint_path}'. "
                "Train the model in the notebook first and place "
                "best_caption_model.pth in the checkpoints/ folder."
            )

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.word_to_idx = checkpoint["word_to_idx"]
        self.idx_to_word = checkpoint["idx_to_word"]
        self.max_length = checkpoint["max_length"]
        self.pad_idx = self.word_to_idx["<pad>"]
        self.start_idx = self.word_to_idx["<start>"]
        self.end_idx = self.word_to_idx["<end>"]

        encoder = ImageEncoder(
            input_size=RESNET_FEATURE_SIZE,
            hidden_size=checkpoint["hidden_size"],
            dropout=DROPOUT,
        )
        decoder = CaptionDecoder(
            vocab_size=checkpoint["vocab_size"],
            embed_size=checkpoint["embed_size"],
            hidden_size=checkpoint["hidden_size"],
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            pad_idx=self.pad_idx,
        )
        self.model = ImageCaptioningModel(encoder, decoder)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Pretrained, frozen ResNet50 for turning a raw image into the
        # 2048-d feature vector the decoder was trained on.
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        resnet.fc = nn.Identity()
        for param in resnet.parameters():
            param.requires_grad = False
        self.resnet = resnet.to(self.device).eval()

    @torch.no_grad()
    def predict(self, pil_image) -> str:
        """PIL image -> generated caption string."""
        image = pil_image.convert("RGB")
        image_tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(self.device)  # (1, 3, 224, 224)

        image_feature = self.resnet(image_tensor)  # (1, 2048)
        encoded = self.model.encoder(image_feature)  # (1, hidden_size)

        decoder = self.model.decoder
        hidden = encoded.unsqueeze(0)
        cell = decoder.image_to_cell(encoded).unsqueeze(0)

        input_token = torch.tensor([[self.start_idx]], device=self.device)
        generated_ids = []

        for _ in range(self.max_length):
            embedded = decoder.embedding_dropout(decoder.embedding(input_token))
            lstm_out, (hidden, cell) = decoder.lstm(embedded, (hidden, cell))
            logits = decoder.fc(lstm_out.squeeze(1))

            next_id = logits.argmax(dim=-1).item()
            if next_id == self.end_idx:
                break

            generated_ids.append(next_id)
            input_token = torch.tensor([[next_id]], device=self.device)

        return " ".join(self.idx_to_word[idx] for idx in generated_ids)
