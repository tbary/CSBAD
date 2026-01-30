
import torch
from torchvision import transforms
import open_clip
import os 
DINO_SPECS = {
    "dinov2": {
        "S": "dinov2_vits14",
        "B": "dinov2_vitb14",
        "L": "dinov2_vitl14",
    },
    "dinov3": {
        "S": "dinov3_vits16",   # #RIGHT NOW EMB HAVE TO BE DOWNLOADED IN A CHECKPOINT AND CANNOT BE DOWN. HAVE TO REGISTER AT META
        "B": "dinov3_vitb16",
        "L": "dinov3_vitl16",
    },
}

def load_dino(args):
    try:
        model_name = DINO_SPECS[args.model][args.arch]
    except KeyError:
        raise ValueError(
            f"Invalid DINO config: model={args.model}, arch={args.arch}"
        )

    if args.model == "dinov2":
        model = torch.hub.load(
        f"facebookresearch/{args.model}",
        model_name,
    )
    else:
        weights_path = os.path.join(
            "checkpoints", f"{model_name}_pretrain_lvd1689m-8aa4cbdd.pth"
        )
        print(weights_path)
        model = torch.hub.load("facebookresearch/dinov3", model_name, weights=weights_path)


    preprocessor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return model, preprocessor


def load_openclip(args):
    model, _, preprocessor = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    return model, preprocessor