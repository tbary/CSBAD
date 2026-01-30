import os
import argparse
import torch
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from ultralytics import YOLO

# Argument parser
parser = argparse.ArgumentParser(description="Extract embeddings using YOLO11nano.")
parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset root.")
parser.add_argument("--cams", type=str, nargs="*", help="Optional list of cameras to process.")
parser.add_argument("--week", type=str, help="Optional specific week to process.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
args = parser.parse_args()

# Load YOLOv11 model
def load_yolo_model():
    model = YOLO('yolo11n.pt')  # Load YOLOv11 nano model
    preprocessor = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.]),
    ])
    model.eval()
    return model, preprocessor

# Dataset class
class ImageDataset(Dataset):
    def __init__(self, image_paths, preprocessor, embeddings_dir):
        self.image_paths = image_paths
        self.preprocessor = preprocessor
        self.embeddings_dir = embeddings_dir

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        embedding_file = os.path.join(self.embeddings_dir, f"{os.path.splitext(os.path.basename(path))[0]}_embedding.pt")
        if os.path.exists(embedding_file):
            return None
        try:
            image = Image.open(path).convert("RGB")
            tensor = self.preprocessor(image)
            return tensor, path
        except (UnidentifiedImageError, OSError):
            return None

# Extract embeddings from YOLOv11
def extract_embeddings(model, dataloader, device):
    embeddings_dict = {}
    embeddings = []

    def hook_fn(module, input, output):
        if output.dim() == 4:
            pooled = output.mean(dim=[2, 3]).cpu()
        elif output.dim() == 3:
            pooled = output.mean(dim=[1]).cpu()
        elif output.dim() == 2:
            pooled = output.cpu()
        else:
            raise RuntimeError(f"Unexpected output shape: {output.shape}")

        embeddings.append(pooled)

    backbone = list(model.model.modules())[-2]
    hook = backbone.register_forward_hook(hook_fn)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            batch = [b for b in batch if b is not None]
            if not batch:
                continue

            images, paths = zip(*batch)
            images = torch.stack(images).to(device)

            # Use backbone directly to avoid triggering detection
            _ = model.model(images)

            for path, emb in zip(paths, embeddings):
                embeddings_dict[path] = emb

            embeddings.clear()

    hook.remove()

    if len(embeddings_dict) == 0:
        raise RuntimeError("No embeddings generated — check model hook or layer output.")

    return embeddings_dict

# Main script
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocessor = load_yolo_model()
    model.to(device)

    cameras_to_process = args.cams if args.cams else sorted(os.listdir(args.dataset_path))

    for cam_dir in cameras_to_process:
        cam_path = os.path.join(args.dataset_path, cam_dir)
        if os.path.isdir(cam_path):
            week_dirs = [args.week] if args.week else sorted(os.listdir(cam_path))
            for week_dir in week_dirs:
                week_path = os.path.join(cam_path, week_dir)
                images_path = os.path.join(week_path, "bank", "images")
                if os.path.exists(images_path):
                    embeddings_dir = os.path.join(week_path, "bank", "yolo11_embeddings")
                    os.makedirs(embeddings_dir, exist_ok=True)

                    image_files = [os.path.join(images_path, f) for f in os.listdir(images_path)
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

                    dataset = ImageDataset(image_files, preprocessor, embeddings_dir)
                    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                            collate_fn=lambda x: [i for i in x if i is not None])

                    embeddings_dict = extract_embeddings(model, dataloader, device)

                    print(f"Saving {len(embeddings_dict)} embeddings to disk...")
                    for path, embedding in embeddings_dict.items():
                        output_file = os.path.join(embeddings_dir, f"{os.path.splitext(os.path.basename(path))[0]}_embedding.pt")
                        print(output_file)
                        torch.save(embedding, output_file)

if __name__ == "__main__":
    main()
