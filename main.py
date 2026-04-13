!pip install grad-cam
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import timm
import hashlib

# Grad-CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class SkinDiseaseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.images = []
        self.labels = []

        for c in self.classes:
            for img in (self.root_dir / c).glob("*"):
                if img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    self.images.append(str(img))
                    self.labels.append(self.class_to_idx[c])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2,0.2,0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])


def check_leakage(images):
    seen = {}
    for p in images:
        img = Image.open(p).resize((64,64)).tobytes()
        h = hashlib.md5(img).hexdigest()
        if h in seen:
            print("⚠️ DUPLICATE:", p, "==", seen[h])
        else:
            seen[h] = p


def train_epoch(model, loader, loss_fn, opt, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for x,y in tqdm(loader, desc="Training"):
        x,y = x.to(device), y.to(device)

        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out,y)
        loss.backward()
        opt.step()

        total_loss += loss.item()
        correct += (out.argmax(1)==y).sum().item()
        total += y.size(0)

    return total_loss/len(loader), 100*correct/total


def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, total = 0,0,0
    preds, labels = [], []

    with torch.no_grad():
        for x,y in tqdm(loader, desc="Validation"):
            x,y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out,y)

            total_loss += loss.item()
            correct += (out.argmax(1)==y).sum().item()
            total += y.size(0)

            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(y.cpu().numpy())

    return total_loss/len(loader), 100*correct/total, preds, labels



def gradcam(model, image, label, device):
    model.eval()

    target_layer = model.blocks[-1].norm1

    cam = GradCAM(
        model=model,
        target_layers=[target_layer],
        reshape_transform=lambda x: x[:,1:,:].reshape(x.size(0),14,14,x.size(2)).permute(0,3,1,2)
    )

    input_tensor = image.unsqueeze(0).to(device)

    rgb = image.permute(1,2,0).cpu().numpy()
    rgb = rgb * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])
    rgb = np.clip(rgb,0,1)

    mask = cam(input_tensor=input_tensor,
               targets=[ClassifierOutputTarget(label)])[0]

    return show_cam_on_image(rgb, mask, use_rgb=True)


def plot_history(train_l, val_l, train_a, val_a):
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot(train_l,label="train")
    plt.plot(val_l,label="val")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(train_a,label="train")
    plt.plot(val_a,label="val")
    plt.title("Accuracy")
    plt.legend()

    plt.savefig("training_history.png")
    plt.close()


def confusion(y_true,y_pred,classes):
    cm = confusion_matrix(y_true,y_pred)
    plt.figure(figsize=(7,6))
    sns.heatmap(cm,annot=True,fmt="d",
                xticklabels=classes,
                yticklabels=classes)
    plt.savefig("confusion_matrix.png")
    plt.close()


def visualize(model, dataset, device, classes):
    fig,ax = plt.subplots(5,2,figsize=(8,20))

    idxs = np.random.choice(len(dataset),5,replace=False)

    for i,idx in enumerate(idxs):
        img,label = dataset[idx]

        cam = gradcam(model,img,label,device)

        ax[i,0].imshow(img.permute(1,2,0))
        ax[i,0].set_title(classes[label])
        ax[i,0].axis("off")

        ax[i,1].imshow(cam)
        ax[i,1].set_title("Grad-CAM")
        ax[i,1].axis("off")

    plt.tight_layout()
    plt.savefig("attention_visualization.png")
    plt.close()



def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    path = "/kaggle/input/datasets/chibuezedev/skin-disease/train"

    full = SkinDiseaseDataset(path)

    print("Classes:", full.classes)

    check_leakage(full.images)

    idxs = np.random.permutation(len(full))
    split = int(0.8*len(full))

    train_idx, val_idx = idxs[:split], idxs[split:]

    train_set = Subset(SkinDiseaseDataset(path,get_transforms(True)),train_idx)
    val_set = Subset(SkinDiseaseDataset(path,get_transforms(False)),val_idx)

    train_loader = DataLoader(train_set,batch_size=16,shuffle=True)
    val_loader = DataLoader(val_set,batch_size=16)

    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=True,
        num_classes=len(full.classes)
    ).to(device)

    # freeze first
    for p in model.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(),lr=1e-4)

    train_l, val_l = [], []
    train_a, val_a = [], []

    best = 0

    for epoch in range(20):
        print("\nEpoch", epoch+1)

        if epoch == 5:
            print("Unfreezing...")
            for p in model.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW(model.parameters(),lr=3e-5)

        tl,ta = train_epoch(model,train_loader,loss_fn,opt,device)
        vl,va,preds,labels = validate(model,val_loader,loss_fn,device)

        train_l.append(tl)
        val_l.append(vl)
        train_a.append(ta)
        val_a.append(va)

        print(f"Train Acc {ta:.2f} | Val Acc {va:.2f}")

        if va > best:
            best = va
            torch.save(model.state_dict(),"best_model.pth")

    model.load_state_dict(torch.load("best_model.pth"))

    _,_,preds,labels = validate(model,val_loader,loss_fn,device)

    print(classification_report(labels,preds,target_names=full.classes))

    confusion(labels,preds,full.classes)
    plot_history(train_l,val_l,train_a,val_a)
    visualize(model,val_set.dataset,device,full.classes)

    print("\nSaved:")
    print("- best_model.pth")
    print("- confusion_matrix.png")
    print("- training_history.png")
    print("- attention_visualization.png")


if __name__ == "__main__":
    main()
