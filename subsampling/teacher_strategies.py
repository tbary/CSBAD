from .utils import *
import numpy as np
import os

DEFAULT_SUB_SAMPLE = 300

def teacher_random(
    client_subsample_names:list,
    n: int = DEFAULT_SUB_SAMPLE,
    seed: int = 42,
    **kwargs,
) -> list:
    """
    :param image_folder_path: path to the bank image folder
    :param n: number of frames to select
    :return output_list: a list containing the selected images path
    """
 
    if n <= 0:
        raise SamplingException(
            f"You must select a strictly positive number of frames to select"
        )
    if n > len(client_subsample_names):
        raise SamplingException(
            f"Image bank contains {len(client_subsample_names)} frames, but {n} frames where required for the "
            f"random strategy !"
        )
    client_subsample_names.sort()
    rng = np.random.default_rng(seed)
    output_list = rng.choice(client_subsample_names, n, replace=False)
    output_list.sort()
    return output_list
 
def teacher_thresholding_top_confidence(
    client_subsample_names:list,
    teacher_image_label_path:str,
    n: int = DEFAULT_SUB_SAMPLE,
    aggregation_function: str = "max",
    **kwargs,
) -> list:
    if len(client_subsample_names) < n:
        raise SamplingException("The teacher can only select at most all the images sent by the student.")
    elif len(client_subsample_names) == n:
        return client_subsample_names

    confidences = np.zeros(len(client_subsample_names))
    for idx, subsample_name in enumerate(client_subsample_names):
        with open(os.path.join(teacher_image_label_path, subsample_name + '.txt'), "r") as f:
            lines = f.readlines()
            if lines:
                # If the file is not empty, compute the image confidence score
                if aggregation_function == "max":
                    image_confidence = max(
                        [float(line.strip().split()[5]) for line in lines]
                    )
                elif aggregation_function == "min":
                    image_confidence = min(
                        [float(line.strip().split()[5]) for line in lines]
                    )
                elif aggregation_function == "mean":
                    object_confidences_scores = [
                        float(line.strip().split()[5]) for line in lines
                    ]
                    image_confidence = sum(object_confidences_scores) / len(
                        object_confidences_scores
                    )
                elif aggregation_function == "sum":
                    object_confidences_scores = [
                        float(line.strip().split()[5]) for line in lines
                    ]
                    image_confidence = sum(object_confidences_scores)
                else:
                    raise SamplingException(
                        f"You must select a valid aggregation function"
                    )
                confidences[idx] = image_confidence
    
    idx_to_keep = np.sort(np.argsort(-confidences)[:n])
    filtered_subsample_names = [client_subsample_names[int(i)] for i in idx_to_keep]

    return filtered_subsample_names

def teacher_thresholding_least_confidence(
    client_subsample_names:list,
    teacher_image_label_path:str,
    n: int = DEFAULT_SUB_SAMPLE,
    aggregation_function: str = "max",
    **kwargs,
) -> list:
    if len(client_subsample_names) < n:
        raise SamplingException("The teacher can only select at most all the images sent by the student.")
    elif len(client_subsample_names) == n:
        return client_subsample_names
    
    confidences = np.zeros(len(client_subsample_names))
    for idx, subsample_name in enumerate(client_subsample_names):
        with open(os.path.join(teacher_image_label_path, subsample_name + '.txt'), "r") as f:
            lines = f.readlines()
            if lines:
                # If the file is not empty, compute the image confidence score
                if aggregation_function == "max":
                    image_confidence = max(
                        [float(line.strip().split()[5]) for line in lines]
                    )
                elif aggregation_function == "min":
                    image_confidence = min(
                        [float(line.strip().split()[5]) for line in lines]
                    )
                elif aggregation_function == "mean":
                    object_confidences_scores = [
                        float(line.strip().split()[5]) for line in lines
                    ]
                    image_confidence = sum(object_confidences_scores) / len(
                        object_confidences_scores
                    )
                elif aggregation_function == "sum":
                    object_confidences_scores = [
                        float(line.strip().split()[5]) for line in lines
                    ]
                    image_confidence = sum(object_confidences_scores)
                else:
                    raise SamplingException(
                        f"You must select a valid aggregation function"
                    )
                confidences[idx] = image_confidence
    
    idx_to_keep = np.sort(np.argsort(confidences)[:n])
    filtered_subsample_names = [client_subsample_names[int(i)] for i in idx_to_keep]
    return filtered_subsample_names

def teacher_top_mAP50_95_oracle(
    client_subsample_names:list,
    teacher_image_label_path:str,
    student_image_label_path:str,
    n: int = DEFAULT_SUB_SAMPLE,
    **kwargs,
) -> list:
    if len(client_subsample_names) < n:
        raise SamplingException("The teacher can only select at most all the images sent by the student.")
    elif len(client_subsample_names) == n:
        return client_subsample_names
    
    student_files = [os.path.join(student_image_label_path, subsample_name + ".txt") for subsample_name in client_subsample_names]
    teacher_files = [os.path.join(teacher_image_label_path, subsample_name + ".txt") for subsample_name in client_subsample_names]
    imgs_mAP_50_95 = np.array(compute_map50_95(student_files, teacher_files))
    
    idx_to_keep = np.sort(np.argsort(-imgs_mAP_50_95)[:n])
    filtered_subsample_names = [client_subsample_names[int(i)] for i in idx_to_keep]

    return filtered_subsample_names

def teacher_least_mAP50_95_oracle(
    client_subsample_names:list,
    teacher_image_label_path:str,
    student_image_label_path:str,
    n: int = DEFAULT_SUB_SAMPLE,
    **kwargs,
) -> list:
    if len(client_subsample_names) < n:
        raise SamplingException("The teacher can only select at most all the images sent by the student.")
    elif len(client_subsample_names) == n:
        return client_subsample_names

    student_files = [os.path.join(student_image_label_path, subsample_name + ".txt") for subsample_name in client_subsample_names]
    teacher_files = [os.path.join(teacher_image_label_path, subsample_name + ".txt") for subsample_name in client_subsample_names]
    imgs_mAP_50_95 = np.array(compute_map50_95(student_files, teacher_files))

    idx_to_keep = np.sort(np.argsort(imgs_mAP_50_95)[:n])
    filtered_subsample_names = [client_subsample_names[int(i)] for i in idx_to_keep]

    return filtered_subsample_names

def teacher_diversity_from_embeddings(
    client_subsample_names:list,
    embeddings_paths:str,
    n:int = DEFAULT_SUB_SAMPLE,
    **kwargs
):
    if len(client_subsample_names) < n:
        raise SamplingException("The teacher can only select at most all the images sent by the student.")
    elif len(client_subsample_names) == n:
        return client_subsample_names

    embeddings = torch.stack([torch.load(os.path.join(embeddings_paths, embedding_file + '_embedding.pt'), map_location='cpu') for embedding_file in client_subsample_names])
    embeddings_kept_mask = np.zeros(len(embeddings), dtype=bool)

    start_idx = select_start_embedding_idx(embeddings)
    embeddings_kept_mask[start_idx] = True

    for _ in range(n-1):
        next_embedding = min_max_cosine_similarity(embeddings[~embeddings_kept_mask], embeddings[embeddings_kept_mask])
        embeddings_kept_mask[torch.nonzero(torch.all(embeddings == next_embedding, dim=1))[0]] = True

    filtered_subsample_names = list(np.array(client_subsample_names)[embeddings_kept_mask])

    return filtered_subsample_names

def teacher_moderate_coreset(
    client_subsample_names:list,
    embeddings_paths:str,
    n:int = DEFAULT_SUB_SAMPLE,
    **kwargs
):  
    if len(client_subsample_names) < n:
        raise SamplingException("The teacher can only select at most all the images sent by the student.")
    elif len(client_subsample_names) == n:
        return client_subsample_names

    embeddings = torch.stack([torch.load(os.path.join(embeddings_paths, embedding_file + '_embedding.pt'), map_location='cpu') for embedding_file in client_subsample_names])
    embeddings_center = embeddings.mean(dim=0)

    distances_to_center = torch.norm(embeddings - embeddings_center, dim=1)

    median_distance = torch.median(distances_to_center)

    diffs_to_median = torch.abs(distances_to_center - median_distance)
    
    # Get indices of the n smallest absolute differences
    _, min_indices = torch.topk(-diffs_to_median, n)  # Negative for smallest values

    # Create binary mask
    mask = torch.zeros_like(distances_to_center, dtype=torch.bool)
    mask[min_indices] = True

    filtered_subsample_names = list(np.array(client_subsample_names)[mask])

    return filtered_subsample_names

def teacher_TFDP(
    client_subsample_names:list,
    student_image_label_path:str,
    n: int = DEFAULT_SUB_SAMPLE,
    **kwargs,
) -> list:
    if len(client_subsample_names) < n:
        raise SamplingException("The teacher can only select at most all the images sent by the student.")
    elif len(client_subsample_names) == n:
        return client_subsample_names

    scores = np.zeros(len(client_subsample_names))
    for idx, subsample_name in enumerate(client_subsample_names):
        with open(os.path.join(student_image_label_path, subsample_name + '.txt'), "r") as f:
            lines = f.readlines()

        if lines:
            # Vectorized calculation for boxes_shapes, boxes_perimeters, and boxes_areas
            boxes_shapes = np.array([list(map(float, line.strip().split()[3:5])) for line in lines])
            widths, heights = boxes_shapes[:, 0], boxes_shapes[:, 1]
            boxes_areas = widths * heights
            boxes_perimeters = 2 * (widths + heights)
            
            # Using vectorized operations to calculate the score
            score = 2 * np.sqrt(np.pi) * np.sum(boxes_perimeters / np.sqrt(boxes_areas))
            scores[idx] = score
    
    idx_to_keep = np.sort(np.argsort(-scores)[:n])
    filtered_subsample_names = [client_subsample_names[int(i)] for i in idx_to_keep]

    return filtered_subsample_names

def teacher_maximum_entropy(
    client_subsample_names:list,
    teacher_image_label_path:str,
    n: int = DEFAULT_SUB_SAMPLE,
    aggregation_function: str = "sum",
    **kwargs,     
)->list:
    if len(client_subsample_names) < n:
        raise SamplingException("The teacher can only select at most all the images sent by the student.")
    elif len(client_subsample_names) == n:
        return client_subsample_names

    entropies = np.zeros(len(client_subsample_names))
    for idx, subsample_name in enumerate(client_subsample_names):
        with open(os.path.join(teacher_image_label_path, subsample_name + '.txt'), "r") as f:
            lines = f.readlines()
            if lines:
                image_confidences = np.array([float(line.strip().split()[5]) for line in lines])
                image_entropies = -image_confidences*np.log(image_confidences)
                # If the file is not empty, compute the image confidence score
                if aggregation_function == "max":
                    img_entropy = np.max(image_entropies)
                elif aggregation_function == "min":
                    img_entropy = np.min(image_entropies)
                elif aggregation_function == "mean":
                    img_entropy = np.mean(image_entropies)
                elif aggregation_function == "sum":
                    img_entropy = np.sum(image_entropies)
                else:
                    raise SamplingException(
                        f"You must select a valid aggregation function"
                    )
                entropies[idx] = img_entropy
    
    idx_to_keep = np.sort(np.argsort(-entropies)[:n])

    filtered_subsample_names = [client_subsample_names[int(i)] for i in idx_to_keep]

    return filtered_subsample_names