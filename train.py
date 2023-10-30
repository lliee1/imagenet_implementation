import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import wandb
import random
import os
import numpy as np

# seed
def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
setup_seed(20)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
multi = True

# Set hyperparameters
num_epochs = 90
batch_size = 256
learning_rate = 0.1

# Initialize transformations for data augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the ImageNet Object Localization Challenge dataset
train_dataset = torchvision.datasets.ImageFolder(
    root='./data/train', 
    transform=train_transform
)
val_dataset = torchvision.datasets.ImageFolder(
    root='./data/val', 
    transform=val_transform
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

# Load the ResNet50 model
model = torchvision.models.resnet50()

# Parallelize training across multiple GPUs
if multi:
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
    "gamma": 0.1,
    
    # implementation name
    "name": "imagenet1k_resnet50_baseline",
    "date": "20231030"
}
wandb.config.update(args)

# Train the model..

best_acc = 0
for epoch in range(num_epochs):
    train_losses = []
    val_losses = []
    for inputs, labels in tqdm(train_loader, desc="Epoch {0}".format(epoch), ascii=" =", leave=True):
        # Zero out the optimizer
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        train_losses.append(loss.item())

        # Backward pass
        loss.backward()
        optimizer.step()

    # Print the loss for every epoch
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {np.mean(train_losses)}')
    wandb.log({"train/loss": np.mean(train_losses)})
    
    
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Test".format(epoch), ascii=" =", leave=True):          
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_losses.append(loss.item())
            
            _, preds = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    acc = 100*(correct / total)
    wandb.log({"val/acc": acc, "val/loss": np.mean(val_losses)})
    if best_acc < acc:
        if multi:
            all_dict = {
                "epoch":epoch,
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            torch.save(all_dict, './output/'+args['name']+'/'+args['date']+'pt')
        
        else:
            all_dict = {
                "epoch":epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            torch.save(all_dict, './output/'+args['name']+'/'+args['date']+'pt')
        wandb.log({"best/acc": acc, "best/epoch": epoch})
    


