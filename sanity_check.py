from src.dataset import CasiaDataset
import torchvision.transforms as T

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

ds = CasiaDataset(root_dir="data/raw/casia_v2/CASIA2", transform=transform)
img, label = ds[0]
print(img.shape, label)