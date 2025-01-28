import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
import open_clip
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# pip install open_clip_torch

# Set up argument parser
parser = argparse.ArgumentParser(description="Infer embeddings using ViT models.")
parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset root.")
parser.add_argument("--model", type=str, required=True, choices=["dinov2", "openclip"], help="Model to use for inference.")
parser.add_argument("--cams", type=str, nargs="*", help="Optional list of cameras to process (e.g., cam1 cam2 cam3).")
parser.add_argument("--week", type=str, help="Optional specific week to process (e.g., week1, week2).")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
args = parser.parse_args()

# Load models and preprocessors
def load_model_and_preprocessor(model_name):
    if model_name == "dinov2":
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        preprocessor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif model_name == "openclip":
        model, _, preprocessor = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    else:
        raise ValueError("Invalid model name.")

    model.eval()
    return model, preprocessor

# Custom Dataset class
class ImageDataset(Dataset):
    def __init__(self, image_paths, preprocessor, embeddings_dir, model_name):
        self.image_paths = image_paths
        self.preprocessor = preprocessor
        self.embeddings_dir = embeddings_dir
        self.model_name = model_name

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        embedding_file = os.path.join(self.embeddings_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_embedding.pt")
        if os.path.exists(embedding_file):
            return None
        image = Image.open(image_path).convert("RGB")
        tensor = self.preprocessor(image)
        return tensor, image_path

# Batch inference function
def batch_infer_embeddings(model, dataloader, device, model_name):
    embeddings_dict = {}
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            # Remove skipped items (None values)
            batch = [b for b in batch if b is not None]
            if not batch:
                continue

            images, image_paths = zip(*batch)
            images = torch.stack(images).to(device)
            if model_name == 'dinov2':
                embeddings = model(images)
            else:  # For OpenCLIP
                embeddings = model.encode_image(images)
            embeddings = (embeddings / embeddings.norm(dim=-1, keepdim=True)).cpu()

            for embedding, path in zip(embeddings, image_paths):
                embeddings_dict[path] = embedding
    return embeddings_dict

# Main script
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model and preprocessor for {args.model}...")
    model, preprocessor = load_model_and_preprocessor(args.model)
    model.to(device)
    print(f"Model and preprocessor loaded successfully.")

    # Get the list of cameras to process
    cameras_to_process = args.cams if args.cams else sorted(os.listdir(args.dataset_path))
    print(f"Processing cameras: {', '.join(cameras_to_process)}")

    # Iterate through dataset structure
    for cam_dir in cameras_to_process:
        cam_path = os.path.join(args.dataset_path, cam_dir)
        if os.path.isdir(cam_path):
            print(f"Processing camera folder: {cam_dir}")
            week_dirs = [args.week] if args.week else sorted(os.listdir(cam_path))
            for week_dir in week_dirs:
                week_path = os.path.join(cam_path, week_dir)
                images_path = os.path.join(week_path, "bank", "images")
                if os.path.exists(images_path):
                    print(f"  Processing week folder: {week_dir}")
                    embeddings_dir = os.path.join(week_path, "bank", f"{args.model}_embeddings")
                    os.makedirs(embeddings_dir, exist_ok=True)

                    # Get list of image files
                    image_files = [os.path.join(images_path, f) for f in sorted(os.listdir(images_path))
                                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]

                    if not image_files:
                        print(f"    No images found for week {week_dir}.")
                        continue

                    # Create DataLoader
                    dataset = ImageDataset(image_files, preprocessor, embeddings_dir, args.model)
                    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=lambda x: [i for i in x if i is not None])

                    # Perform batch inference
                    embeddings_dict = batch_infer_embeddings(model, dataloader, device, args.model)

                    # Save embeddings
                    for path, embedding in embeddings_dict.items():
                        output_file = os.path.join(embeddings_dir, f"{os.path.splitext(os.path.basename(path))[0]}_embedding.pt")
                        torch.save(embedding, output_file)
                        print(f"      Saved embeddings to {output_file}")

if __name__ == "__main__":
    main()
