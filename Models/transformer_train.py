from tqdm import tqdm
import torch
from Models.transformer_model import GestureTransformer
from Models.transformer_dataset import JSONGestureDataset
from torch.utils.data import DataLoader, random_split
from Models.config import ROOT_DIR

def train_with_validation(model, train_loader, val_loader, epochs=50, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device -> {device}")
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")

        for seqs, labels in train_bar:
            seqs, labels = seqs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (pred == labels).sum().item()

            current_acc = 100 * train_correct / train_total
            train_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{current_acc:.2f}%")

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]")

        with torch.no_grad():
            for seqs, labels in val_bar:
                seqs, labels = seqs.to(device), labels.to(device)

                outputs = model(seqs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, pred = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (pred == labels).sum().item()

                v_acc = 100 * val_correct / val_total
                val_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{v_acc:.2f}%")

        val_acc = 100 * val_correct / val_total
        print(f"Summary -> Train Acc: {100 * train_correct / train_total:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_gesture_model.pth")
            print("⭐ New best model saved!")
        print("-" * 50)


if __name__ == "__main__":
    dataset = JSONGestureDataset(root_dir=str(ROOT_DIR / "dataset-keypoints"), target_frames=35)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    num_classes = len(dataset.class_to_idx)
    print(f"Detected classes: {dataset.class_to_idx}")

    model = GestureTransformer(
        input_dim=42,
        d_model=128,
        nhead=8,
        num_layers=3,
        num_classes=num_classes
    )

    train_with_validation(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10,
        lr=0.001
    )