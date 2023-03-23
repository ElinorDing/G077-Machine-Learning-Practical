import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class ViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=10, dim=768, depth=12, heads=12, mlp_dim=3072, dropout=0.1):
        super(ViT, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(dropout)
        self.transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout)
            for _ in range(depth)
        ])
        self.linear = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        for layer in self.transformer:
            x = layer(x)
        x = x.mean(dim=1)
        x = self.linear(x)
        return x

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument(
        "--dataset_train",
        type=str,
        default=None,
        help="The name of the train dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_test",
        type=str,
        default=None,
        help="The name of the test dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    
    args = parser.parse_args()

    return args

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    train_acc = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_acc += (predicted == labels).sum().item()
    train_loss /= len(train_loader)
    train_acc /= len(train_loader.dataset)
    return train_loss, train_acc

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_acc += (predicted == labels).sum().item()
    test_loss /= len(test_loader)
    test_acc /= len(test_loader.dataset)
    return test_loss, test_acc

def main():
    # set hyperparameters
    args = parse_args()
    image_size = 32
    patch_size = 4
    num_classes = 10
    dim = 256
    depth = 6
    heads = 8
    mlp_dim = 512
    dropout = 0.1
    batch_size = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # prepare data
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = load_dataset("imagefolder", data_dir= args.dataset_train)
    test_dataset = load_dataset("imagefolder", data_dir= args.dataset_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # initialize model
    model = ViT(image_size=image_size, patch_size=patch_size, num_classes=num_classes, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout).to(device)

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # training loop
    for epoch in range(1, args.num_train_epochs+1):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)
        print(f"Epoch {epoch}:\nTrain loss: {train_loss:.4f} | Train acc: {train_acc:.4f}\nTest loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")


if __name__ == '__main__':
    main()


# python3 training_imple.py --dataset_train ~/G077-Machine-Learning-Practical/Data/Clean_data/train/ --dataset_test ~/G077-Machine-Learning-Practical/Data/Clean_data/test/ --num_train_epochs 10 --learning_rate 1e-3