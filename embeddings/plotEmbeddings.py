import torch
import os
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from tqdm import tqdm  # Progress bar for loading

# ==========================
# 1. DEFINE CAMERA FOLDERS
# ==========================
camera_folders = {
    "Cam1": "/home/dani/data/WALT-challenge/cam1/week12345/bank/dinov2L/",
    "Cam2": "/home/dani/data/WALT-challenge/cam2/week12345/bank/dinov2L/",
    "Cam3": "/home/dani/data/WALT-challenge/cam3/week5/bank/dinov2L/",
    "Cam4": "/home/dani/data/WALT-challenge/cam4/week23/bank/dinov2L/",
    "Cam5": "/home/dani/data/WALT-challenge/cam5/week1235/bank/dinov2L/",
    "Cam6": "/home/dani/data/WALT-challenge/cam6/week134/bank/dinov2L/",
    "Cam7": "/home/dani/data/WALT-challenge/cam7/week4/bank/dinov2L/",
    "Cam8": "/home/dani/data/WALT-challenge/cam8/week3/bank/dinov2L/",
    "Cam9": "/home/dani/data/WALT-challenge/cam9/week12/bank/dinov2L/",
}

TARGET_SAMPLES = 8000  # Fixed number of samples per camera

# ==========================
# 2. LOAD AND UNIFORMLY SAMPLE FEATURES FROM EACH CAMERA
# ==========================
all_features = []
all_labels = []  # To store camera labels

for cam_name, folder_path in camera_folders.items():
    feature_list = []
    
    print(f"Processing {cam_name} from {folder_path}...")
    
    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} does not exist. Skipping...")
        continue

    for file_name in tqdm(os.listdir(folder_path), desc=f"Loading {cam_name}"):
        if file_name.endswith(".pt"):
            file_path = os.path.join(folder_path, file_name)
            data = torch.load(file_path)  # Load .pt file
            
            # If file contains a dictionary, extract features
            if isinstance(data, dict):
                features = data.get("features", None)
            else:
                features = data  # Assume it's a raw tensor
            
            # Convert to NumPy and store
            feature_list.append(features.cpu().numpy())
    
    # Stack all features from this camera
    if feature_list:
        features_np = np.vstack(feature_list)
        N = features_np.shape[0]  # Number of available representations

        # Compute sampling indices using uniform interval sampling
        sample_indices = np.linspace(0, N-1, TARGET_SAMPLES, dtype=int)

        # Subset the features to get exactly 8000 samples
        sampled_features = features_np[sample_indices]

        all_features.append(sampled_features)
        all_labels.append(np.full(sampled_features.shape[0], cam_name))  # Assign camera name as label

# Convert lists to arrays
all_features = np.vstack(all_features)
all_labels = np.concatenate(all_labels)

print(f"Total dataset size after sampling: {all_features.shape}")

# ==========================
# 3. DIMENSIONALITY REDUCTION (PCA)
# ==========================
print("Applying PCA...")
pca = PCA(n_components=3)
features_3d = pca.fit_transform(all_features)
print("Explained variance:", pca.explained_variance_ratio_)

# ==========================
# 4. APPLY CUSTOM COLOR PALETTE FOR HIGH-VISIBILITY 3D VISUALIZATION
# ==========================
import plotly.express as px

# **Adjusted high-contrast palette (NO BLACK, CAM3 & CAM9 DISTINCT)**
custom_palette = [
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#009E73",  # Green
    "#F0E442",  # Yellow
    "#0072B2",  # Deep Blue
    "#D55E00",  # Red-Orange
    "#CC79A7",  # Pink-Purple
    "#F4A261",  # Warm Peach
    "#9400D3",  # **Dark Violet (Replaced Cam3)**
    "#FFFFFF",  # **White (Kept for Cam9)**
]

# Assign **one unique color per camera model**
unique_cameras = np.unique(all_labels)
camera_to_color = {cam: custom_palette[idx % len(custom_palette)] for idx, cam in enumerate(unique_cameras)}

fig = go.Figure()

# Add scatter plot for each camera model with the **custom colors**
for cam in unique_cameras:
    indices = all_labels == cam
    fig.add_trace(go.Scatter3d(
        x=features_3d[indices, 0], 
        y=features_3d[indices, 1], 
        z=features_3d[indices, 2],
        mode='markers',
        marker=dict(
            size=4,  # Small point size for dense visualization
            opacity=0.85,  # Transparency to create a depth effect
            color=camera_to_color[cam],  # Custom color mapping
            symbol='circle'  # Maintain circular markers
        ),
        text=[f"Camera: {cam}" for _ in range(len(indices))],  # Hover text
        hoverinfo="text",
        name=cam  # Legend entry
    ))

# **Dark-themed aesthetics**
fig.update_layout(
    title=dict(
        text="📷 3D Projection of DINOv2 Representations Per Camera Model (Uniform Sampling)",
        x=0.5,
        font=dict(size=22, color="white", family="Arial Black")
    ),
    scene=dict(
        xaxis=dict(
            title="PC1", 
            backgroundcolor="rgba(20, 20, 20, 0.8)", 
            gridcolor="gray", 
            tickfont=dict(color="white")
        ),
        yaxis=dict(
            title="PC2", 
            backgroundcolor="rgba(20, 20, 20, 0.8)", 
            gridcolor="gray", 
            tickfont=dict(color="white")
        ),
        zaxis=dict(
            title="PC3", 
            backgroundcolor="rgba(20, 20, 20, 0.8)", 
            gridcolor="gray", 
            tickfont=dict(color="white")
        ),
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    legend=dict(
        title="Cameras",
        x=1,  # Move legend to the right
        font=dict(size=22, color="white", family="Arial Bold", weight='bold'),
        itemsizing="constant",
    ),
    template="plotly_dark",  # **Dark mode theme**
)

# Improve viewing angle for **cinematic effect**
fig.update_layout(
    scene_camera=dict(
        eye=dict(x=1.8, y=1.8, z=1.2)
    )
)

# **Save as interactive HTML for sharing**
fig.write_html("3D_uniform_sampling_projection.html")

# Show the refined visualization
fig.show()
