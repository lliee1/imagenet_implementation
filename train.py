import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set hyperparameters
num_epochs = 90
batch_size = 256
learning_rate = 0.1

# Initialize transformations for data augmentation
transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the ImageNet Object Localization Challenge dataset
train_dataset = torchvision.datasets.ImageFolder(
    root='./data/train', 
    transform=transform
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Load the ResNet50 model
model = torchvision.models.resnet50(pretrained=True)

# Parallelize training across multiple GPUs
model = torch.nn.DataParallel(model)

# Set the model to run on the device
model = model.to(device)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

wandb.init(project='imagenet_implementation')
wandb.run.name = 'imagenet1k_resnet50_baseline'
wandb.run.save()

args = {
    # base
    "model": "resnet50",
    "batch_size": batch_size,
    "epochs": num_epochs,
    
    # optim
    "learning_rate": learning_rate,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    
    # scheduler
    "scheduler": "stepLR",
    "step_size":30,
    "gamma": 0.1
}
wandb.config.update(args)

# Train the model...
for epoch in range(num_epochs):
    for inputs, labels in tqdm(train_loader, desc="Epoch {0}".foramt(epoch), ascii=" =", leave=True):
        # Move input and label tensors to the device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero out the optimizer
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

    # Print the loss for every epoch
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
    wandb.log("train/loss": loss.item())

print(f'Finished Training, Loss: {loss.item():.4f}')