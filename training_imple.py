import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import datasets

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

batch_size = 32
num_epochs = 10
learning_rate = 3e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

class VIT(nn.Module):
    def __init__(self, img_size, patch_size, num_classes, dim, depth, heads, mlp_dim):
        super(VIT, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, dim, patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim), num_layers=depth)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed
        x = self.transformer(x)
        x = x[:, 0, :]
        x = self.fc(x)
        return x


(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

class VITTransformer(tf.keras.Model):
    def __init__(self, num_classes):
        super(VITTransformer, self).__init__()
        
        # Define the patch embedding layer
        self.patch_embedding = layers.Conv2D(64, kernel_size=3, strides=2, padding="valid")
        
        # Define the transformer encoder layers
        self.transformer_layers = [
            layers.TransformerEncoderLayer(d_model=64, num_heads=4, dff=128, dropout=0.1)
            for _ in range(4)
        ]
        self.transformer = layers.Transformer(self.transformer_layers)
        
        # Define the classification head
        self.classification_head = layers.Dense(num_classes, activation="softmax")
        
    def call(self, inputs):
        # Patch embedding layer
        x = self.patch_embedding(inputs)
        x = tf.reshape(x, shape=(-1, x.shape[1] * x.shape[2], x.shape[-1]))
        
        # Transformer layers
        x = self.transformer(x)
        
        # Classification head
        x = self.classification_head(x[:, 0, :])
        
        return x

model = VITTransformer(num_classes=10)
loss_fn = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
val_acc_metric = tf.keras.metrics.CategoricalAccuracy()

for epoch in range(num_epochs):
    # Training loop
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            # Forward pass
            logits = model(x_batch_train, training=True)
            # Compute loss
            loss_value = loss_fn(y_batch_train, logits)
        # Compute gradients
        grads = tape.gradient(loss_value, model.trainable_weights)
        # Update weights
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        # Update training accuracy metric
        train_acc_metric.update_state(y_batch_train, logits)

    # Validation loop
    for x_batch_val, y_batch_val in val_dataset:
        # Forward pass
        logits = model(x_batch_val, training=False)
        # Update validation accuracy metric
        val_acc_metric.update_state(y_batch_val, logits)

    # Get training and validation accuracy
    train_acc = train_acc_metric.result()
    val_acc = val_acc_metric.result()
    # Reset accuracy metrics
    train_acc_metric.reset_states()
    val_acc_metric.reset_states()

    # Print progress
    print(f"Epoch {epoch + 1}/{num_epochs}: train_acc={train_acc:.4f} val_acc={val_acc:.4f}")



import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

batch_size = 32
num_epochs = 10
learning_rate = 3e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

class VIT(nn.Module):
    def __init__(self, img_size, patch_size, num_classes, dim, depth, heads, mlp_dim):
        super(VIT, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, dim, patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim), num_layers=depth)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed
        x = self.transformer(x)
        x = x[:, 0, :]
        x = self.fc(x)
        return x


    
        

   

     
 
    




    


