from .utils import *
import numpy as np
import os

DEFAULT_SUB_SAMPLE = 300

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

    confidences = np.empty(len(client_subsample_names))
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
    
    confidences = np.empty(len(client_subsample_names))
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

    embeddings = [torch.load(os.path.join(embeddings_paths, embedding_file + '_embedding.pt'), map_location='cpu') for embedding_file in client_subsample_names]
    start_idx = select_start_embedding_idx(torch.stack(embeddings))

    idx_to_keep = [start_idx]
    embeddings_kept = [embeddings.pop(start_idx)]

    for _ in range(n-1):
        next_index = max_min_cosine_similarity(embeddings, torch.stack(embeddings_kept))
        idx_to_keep.append(next_index)
        embeddings_kept.append(embeddings.pop(next_index))

    idx_to_keep = np.sort(idx_to_keep)
    filtered_subsample_names = [client_subsample_names[int(i)] for i in idx_to_keep]

    return filtered_subsample_names