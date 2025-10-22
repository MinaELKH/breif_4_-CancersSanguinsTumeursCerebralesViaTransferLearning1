import os
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shutil
import random
from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF


dataset_path = "../data/Blood_Cells_Cancer"
valid_extensions = ('.jpeg', '.jpg', '.bmp', '.png')

## valider les images et supprimer les fichiers parasites
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        file_path = os.path.join(root, file)
        if not file.lower().endswith(valid_extensions):
            os.remove(file_path)
        else:
            try:
                img = Image.open(file_path)
                img.verify()
            except (IOError, SyntaxError):
                os.remove(file_path)

## resize les images                
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

classes = dataset.classes
print("Classes :", classes)



##  histogramme de nombre d images par classes
class_counts = [0]*len(classes)
for _, label in dataset:
    class_counts[label] += 1

sns.barplot(x=classes, y=class_counts)
plt.xlabel("Classes")
plt.ylabel("Nombre d'échantillons")
plt.title("Répartition des échantillons par classe")
plt.show()


##### echantillon
fig, axes = plt.subplots(1, len(classes), figsize=(5*len(classes), 5))
if len(classes) == 1:
    axes = [axes]

for idx, class_name in enumerate(classes):
    class_idx = dataset.class_to_idx[class_name]
    for img, label in dataset:
        if label == class_idx:
            image = img.permute(1, 2, 0).numpy()
            axes[idx].imshow(image)
            axes[idx].set_title(class_name)
            axes[idx].axis("off")
            break

plt.show()





## creation des sous dossiers 
output_path  = r"../data/Blood_Cells_Cancer_Split"

os.makedirs(output_path, exist_ok=True)
splits = ['Train', 'Validation', 'Test']

for split in splits:
    os.makedirs(os.path.join(output_path, split), exist_ok=True)

for class_name in os.listdir(dataset_path):
    class_folder = os.path.join(dataset_path, class_name)
    if not os.path.isdir(class_folder):
        continue

    images = [f for f in os.listdir(class_folder) if f.lower().endswith(('.jpeg', '.jpg', '.bmp', '.png'))]
    random.shuffle(images)

    n_total = len(images)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]

    for split_name, split_imgs in zip(splits, [train_imgs, val_imgs, test_imgs]):
        split_class_folder = os.path.join(output_path, split_name, class_name)
        os.makedirs(split_class_folder, exist_ok=True)
        for img in split_imgs:
            src = os.path.join(class_folder, img)
            dst = os.path.join(split_class_folder, img)
            shutil.copy2(src, dst)

## compte les images par dossier
for split in splits:
    print(f"\n{split} :")
    split_path = os.path.join(output_path, split)
    for class_name in os.listdir(split_path):
        class_folder = os.path.join(split_path, class_name)
        if os.path.isdir(class_folder):
            count = len([f for f in os.listdir(class_folder) if f.lower().endswith(('.jpeg', '.jpg', '.bmp', '.png'))])
            print(f"  {class_name}: {count} images") 

#"""""""""""" resultat : 
            # Train :
            #   Benign: 358 images
            #   early Pre-B: 685 images
            #   Pre-B: 668 images
            #   Pro-B: 557 images

            # Validation :
            #   Benign: 76 images
            #   early Pre-B: 146 images
            #   Pre-B: 143 images
            #   Pro-B: 119 images

            # Test :
            #   Benign: 78 images
            #   early Pre-B: 148 images
            #   Pre-B: 144 images
            #   Pro-B: 120 images 
            # 
         
## equilbre  les nombre d images  par application des filtre 
train_dir = "../data/Blood_Cells_Cancer_Split/Train"
valid_extensions= ('.jpg', '.jpeg', '.png', '.bmp')

classes = {cls: [os.path.join(train_dir, cls, f) for f in os.listdir(os.path.join(train_dir, cls)) if f.lower().endswith(valid_extensions)] 
           for cls in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, cls))}

max_count = max(len(imgs) for imgs in classes.values())

for cls, img_paths in classes.items():
    n_current = len(img_paths)
    n_to_add = max_count - n_current
    if n_to_add <= 0:
        continue
    print(f"Classe {cls}: ajout de {n_to_add} images")
    for i in range(n_to_add):
        img_path = random.choice(img_paths)
        img = Image.open(img_path).convert("RGB")
        t = random.choice(["flip", "blur", "noise"])
        if t == "flip":
            img = TF.hflip(img)
        elif t == "blur":
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1, 2)))
        elif t == "noise":
            arr = np.array(img)
            noise = np.clip(arr + np.random.normal(0, 25, arr.shape), 0, 255).astype(np.uint8)
            img = Image.fromarray(noise)
        new_name = f"aug_{t}_{i}_{os.path.basename(img_path)}"
        img.save(os.path.join(train_dir, cls, new_name))