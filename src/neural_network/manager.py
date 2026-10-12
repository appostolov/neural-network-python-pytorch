import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import config

from .dataset import EmailDataset
from .models import EmailValidator
from .vocab.util import text_to_tensor

def train()->None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training engine running on: {device}")

    dataset = EmailDataset(
        csv_path=config.DATA_PATH,
        max_len=config.MAX_EMAIL_LEN
    )

    train_size = int(0.80 * len(dataset))
    test_size = len(dataset) - train_size

    torch.manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,         
        pin_memory=True        
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,         
        pin_memory=True        
    )

    model = EmailValidator(
        vocab_size=config.VOCAB_JSON_LEN,
        embedding_dim=config.EMBEDDING_DIM
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    epochs = 5  
    print("\n--- Starting Training Optimization Loop ---")
    for epoch in range(epochs):
        
        model.train()
        running_loss = 0.0

        for batch_idx, (features, labels) in enumerate(train_loader):

            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            predictions = model(features)

            loss = criterion(predictions, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] complete. Average Loss: {epoch_loss:.4f}")

    model.eval()

    with torch.no_grad():

        correct = 0

        for batch_idx, (features, labels) in enumerate(test_loader):

            features = features.to(device)
            labels = labels.to(device)

            predictions = model(features)
            probability = torch.sigmoid(predictions)
            binary_predictions = (probability >= 0.5).float()
            matches_tensor = (binary_predictions == labels)

            batch_correct = matches_tensor.sum().item()
            correct += batch_correct

    print(f"\nAccuracy: {( correct / len(test_dataset) ) * 100} %")

    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print("\nSuccess! Weights exported to src/neural_network/model_weights.pth")

def run(email: str):

    model = EmailValidator(
        vocab_size=config.VOCAB_JSON_LEN,
        embedding_dim=config.EMBEDDING_DIM,
        num_filters=config.NUM_FILTERS
    )

    saved_weights = torch.load(config.MODEL_SAVE_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(saved_weights)

    model.eval()

    tokens = text_to_tensor(email, config.MAX_EMAIL_LEN)
    batch_input = tokens.unsqueeze(0)

    with torch.no_grad():
        raw_logit = model(batch_input)
        probability = torch.sigmoid(raw_logit).item()

    print(f"Is it valid? Validation confidence: {probability * 100:.2f}%")

