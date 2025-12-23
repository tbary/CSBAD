import os
import argparse
from PIL import Image, UnidentifiedImageError
#import open_clip
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch

# pip install open_clip_torch

# Set up argument parser
parser = argparse.ArgumentParser(description="Infer embeddings using ViT models.")
parser.add_argument("--dataset_path", type=str, required=True,
                    help="Path to the input folder containing augmented images (e.g., .../augmented/epoch0).")
parser.add_argument("--model", type=str, required=True, choices=["dinov2", "dinov3", "openclip"], help="Model to use for inference.")
parser.add_argument("--arch", type=str, required=True, choices=["S", "B", "L"], default="L", help="Model architecture.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
args = parser.parse_args()

# Load models and preprocessors
def load_model_and_preprocessor(args):
    if args.model in ["dinov2","dinov3"]:
        from emb_models import load_dino
        return load_dino(args)
    elif args.model == "openclip":
        from emb_models import load_openclip
        return load_openclip(args)
    else:
        raise ValueError("Invalid model name.")

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
        embedding_file = os.path.join(self.embeddings_dir,
                                      f"{os.path.splitext(os.path.basename(image_path))[0]}_embedding.pt")
        if os.path.exists(embedding_file):
            return None
        try:
            image = Image.open(image_path).convert("RGB")
            tensor = self.preprocessor(image)
            return tensor, image_path
        except (UnidentifiedImageError, OSError) as e:
            print(f"Error loading image {image_path}: {e}")
            return None

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
            if model_name in ['dinov2','dinov3']:
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
    model, preprocessor = load_model_and_preprocessor(args)
    model.eval()
    model.to(device)
    print(f"Model and preprocessor loaded successfully.")

    # Input images are directly in args.dataset_path (e.g., .../augmented/epoch0)
    images_path = args.dataset_path
    if not os.path.isdir(images_path):
        print(f"Input path does not exist or is not a directory: {images_path}")
        return

    # Build output embeddings dir as sibling: .../augmented_embeddings/epochX
    dst_dir = os.path.dirname(images_path)            
    root_dir = os.path.dirname(dst_dir)
    sub_name = os.path.basename(images_path)          #it can be the class, epoch and so on.     
    embeddings_dir = os.path.join(root_dir, f"{args.model}_{args.arch}_embs", sub_name)
    os.makedirs(embeddings_dir, exist_ok=True)
    print(f"Saving embeddings to: {embeddings_dir}")

    # Get list of image files
    image_files = [os.path.join(images_path, f) for f in sorted(os.listdir(images_path))
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not image_files:
        print(f"No images found in {images_path}.")
        return

    # Create DataLoader
    dataset = ImageDataset(image_files, preprocessor, embeddings_dir, args.model)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                            collate_fn=lambda x: [i for i in x if i is not None])

    # Perform batch inference
    embeddings_dict = batch_infer_embeddings(model, dataloader, device, args.model)

    # Save embeddings
    for path, embedding in embeddings_dict.items():
        output_file = os.path.join(embeddings_dir,
                                   f"{os.path.splitext(os.path.basename(path))[0]}_embedding.pt")
        torch.save(embedding, output_file)

if __name__ == "__main__":
    main()
