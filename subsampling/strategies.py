from .utils import *
import numpy as np
import os
import warnings

DEFAULT_SUB_SAMPLE = 300


def uniform_stream_based(
    image_labels_path: str,
    n: int = 10,
    sampling_rate: float = 0.005,
    seed: int = 0,
    **kwargs,
) -> list:
    """
    Simulating a scenario where at each time-step we draw a number from a uniform distribution bounded in [0, 1].
    Should be less spread out than randomly drawing from a pool.
    :param image_folder_path: path to the bank image folder
    :param n: number of frames to select
    :param sampling_rate: probability of selecting each frame
    :param seed: seed for the random number generator
    :return output_list: a list containing the selected images path
    """
    flag=0 #Indicate whether the strategy operated normally (0 = Yes, any other value = No).
    requested_sampling_rate=sampling_rate
    if n <= 0:
        raise SamplingException(
            f"You must select a strictly positive number of frames to select"
        )

    path_list = [
        os.path.splitext(filename)[0] for filename in os.listdir(image_labels_path)
    ]
    if n > len(path_list):
        raise SamplingException(
            f"Image bank contains {len(path_list)} frames, but {n} frames where required for the "
            f"random strategy !"
        )
    path_list.sort()
    np.random.seed(seed)
    
    rand_array = np.random.uniform(0, 1, len(path_list))
    while True:
        assert sampling_rate <=1.00, "Sampling rate must not exceed 1.00"
        output_list = [
            path_list[i]
            for i in range(len(path_list))
            if rand_array[i] >= (1 - sampling_rate)
        ]
        if len(output_list[:n]) == n:
            if requested_sampling_rate != sampling_rate:
                warnings.warn(f"Budget {n} was not met with initial sampling_rate = {requested_sampling_rate:.2f}. It was adjusted to {sampling_rate:.2f}")
            break
        else:
            flag=-1
            print(f"Budget {int(n)} is not met with sampling_rate = {sampling_rate:.2f}. Only {len(output_list)} images selected.")
            sampling_rate = sampling_rate + 0.05
    return output_list[:n], flag

def interval_sampling(
    image_labels_path: str,
    n: int = DEFAULT_SUB_SAMPLE,
    **kwargs,
)->list:
    if n <= 0:
        raise SamplingException(
            f"You must select a strictly positive number of frames to select"
        )

    path_list = [
        os.path.splitext(filename)[0] for filename in os.listdir(image_labels_path)
    ]
    if n > len(path_list):
        raise SamplingException(
            f"Image bank contains {len(path_list)} frames, but {n} frames where required for the "
            f"interval sampling strategy !"
        )

    step = (len(path_list) - 1) / (n - 1)
    indices = [int(i * step) for i in range(n - 1)]
    indices.append(len(path_list) - 1)  # Include the last element

    return [path_list[i] for i in indices], 0

def thresholding_least_confidence(
    image_labels_path: str,
    n: int = DEFAULT_SUB_SAMPLE,
    aggregation_function: str = "max",
    warmup_length=720,
    sampling_rate=0.1,
    **kwargs,
) -> list:
    """
    Performs active learning for object detection using the confidence scores.

    Parameters:
    - image_labels_path: paths to the .txt files with the object detections (last element of each line = confidence score).
    - n: number of images to label.
    - aggregation_function: how to compute the confidence of an image based on the confidence of the single objects:
        a) "max": minmax approach, where the confidence of an image is given by the most confidently detected object.
        b) "min": confidence of the whole image is given by the most difficult object detected.
        c) "mean": average of all the confidence scores, it is not sensible to the number of objects detected.
        d) "sum": sensible to the number of objects detected.

    Returns:
    - images_to_label: list of strings, paths to the .txt files with the images to be labeled
    """
    flag=0 #Indicate whether the strategy operated normally (0 = Yes, any other value = No).
    requested_sampling_rate=sampling_rate
    txt_files = [filename for filename in os.listdir(image_labels_path)]
    if n <= 0:
        raise SamplingException(
            f"You must select a strictly positive number of frames to select"
        )
    if n > len(txt_files):
        raise SamplingException(
            f"Image bank contains {len(txt_files)} frames, but {n} frames where required for the "
            f"least confidence strategy !"
        )
    confidences = []
    for txt_file in txt_files:
        with open(os.path.join(image_labels_path, txt_file), "r") as f:
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
                confidences.append((txt_file, image_confidence))

    # Get the warm-up set
    warmup_set = confidences[:warmup_length]

    while True:
        # Compute the threshold
        threshold = np.percentile([conf for _, conf in warmup_set], 100 * sampling_rate)

        # Filtering images based on the confidence scores
        low_confidence_images = [
            (img, conf) for img, conf in confidences[warmup_length:] if conf < threshold
        ]

        # Get N-first images with a confidence lower than the threshold
        images_to_label = [os.path.splitext(img)[0] for img, _ in low_confidence_images[:n]]
        if len(images_to_label) == n:
            if requested_sampling_rate != sampling_rate:
                warnings.warn(f"Budget {n} was not met with initial sampling_rate = {requested_sampling_rate:.2f}. It was adjusted to {sampling_rate:.2f}")
            break
        else:
            flag=-1
            print(f"Budget {int(n)} is not met with sampling_rate = {sampling_rate:.2f}. Only {len(images_to_label)} images selected.")
        sampling_rate = sampling_rate + 0.05   

    return images_to_label, flag




def teacher_maximum_entropy(
    image_folder_path:str,
    student_image_label_path:str,
    n: int = DEFAULT_SUB_SAMPLE,
    aggregation_function: str = "sum",
    **kwargs,     
)->list:
    
    path_list, _ = list_files_without_extensions(image_folder_path)
    client_subsample_names = path_list

    if len(client_subsample_names) < n:
        raise SamplingException("The teacher can only select at most all the images sent by the student.")
    elif len(client_subsample_names) == n:
        return client_subsample_names

    entropies = np.zeros(len(client_subsample_names))
    for idx, subsample_name in enumerate(client_subsample_names):
        with open(os.path.join(student_image_label_path, subsample_name + '.txt'), "r") as f:
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




def student_TFDP(
    image_folder_path:str,
    student_image_label_path:str,
    n: int = DEFAULT_SUB_SAMPLE,
    **kwargs,
) -> list:
    
    flag = 0

    path_list, _ = list_files_without_extensions(image_folder_path)
    client_subsample_names = path_list
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
            score = np.sum(boxes_perimeters / np.sqrt(boxes_areas)) / (2 * np.sqrt(np.pi))
            scores[idx] = score
    
    idx_to_keep = np.sort(np.argsort(-scores)[:n])
    filtered_subsample_names = [client_subsample_names[int(i)] for i in idx_to_keep]

    return filtered_subsample_names, flag




def student_moderate_coreset(
    image_folder_path:str,
    embeddings_paths:str,
    n:int = DEFAULT_SUB_SAMPLE,
    **kwargs
):  
    flag = 0
    client_subsample_names, _ = list_files_without_extensions(image_folder_path)

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

    return filtered_subsample_names, flag




def thresholding_top_confidence(
    image_labels_path: str,
    n: int = DEFAULT_SUB_SAMPLE,
    aggregation_function: str = "max",
    warmup_length=720,
    sampling_rate=0.10,
    **kwargs,
) -> list:
    """
    Performs active learning for object detection using the confidence scores.

    Parameters:
    - image_labels_path: paths to the .txt files with the object detections (last element of each line = confidence score).
    - n: number of images to label.
    - aggregation_function: how to compute the confidence of an image based on the confidence of the single objects:
        a) "max": minmax approach, where the confidence of an image is given by the most confidently detected object.
        b) "min": confidence of the whole image is given by the most difficult object detected.
        c) "mean": average of all the confidence scores, it is not sensible to the number of objects detected.
        d) "sum": sensible to the number of objects detected.

    Returns:
    - images_to_label: list of strings, paths to the .txt files with the images to be labeled
    """
    flag=0 #Indicate whether the strategy operated normally (0 = Yes, any other value = No).
    requested_sampling_rate=sampling_rate
    txt_files = [filename for filename in os.listdir(image_labels_path)]
    if n <= 0:
        raise SamplingException(
            f"You must select a strictly positive number of frames to select"
        )
    if n > len(txt_files):
        raise SamplingException(
            f"Image bank contains {len(txt_files)} frames, but {n} frames where required for the "
            f"least confidence strategy !"
        )
    confidences = []
    for txt_file in txt_files:
        with open(os.path.join(image_labels_path, txt_file), "r") as f:
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
                confidences.append((txt_file, image_confidence))

    # Get the warm-up set
    warmup_set = confidences[:warmup_length]

    while True:
        # Compute the threshold
        assert sampling_rate <=1.00, "Sampling rate must not exceed 1.00"
        threshold = np.percentile(
            [conf for _, conf in warmup_set], 100 * (1 - sampling_rate)
        )

        # Filtering images based on the confidence scores
        top_confidence_images = [
            (img, conf) for img, conf in confidences[warmup_length:] if conf > threshold
        ]
        # Get N-first images with a confidence lower than the threshold
        images_to_label = [os.path.splitext(img)[0] for img, _ in top_confidence_images[:n]]

        if len(images_to_label) == n:
            if requested_sampling_rate != sampling_rate:
                warnings.warn(f"Budget {n} was not met with initial sampling_rate = {requested_sampling_rate:.2f}. It was adjusted to {sampling_rate:.2f}")
            break
        else:
            flag=-1
            print(f"Budget {int(n)} is not met with sampling_rate = {sampling_rate:.2f}. Only {len(images_to_label)} images selected.")
        sampling_rate = sampling_rate + 0.05
    return images_to_label, flag

def strategy_n_first(
    image_folder_path: str,
    n: int = DEFAULT_SUB_SAMPLE,
    **kwargs,
) -> list:
    """
    :param image_folder_path: path to the bank image folder
    :param n: number of frames to select
    :return output_list: a list containing the selected images path
    """
    flag = 0 #Indicate whether the strategy operated normally (0 = Yes, any other value = No).
    path_list, _ = list_files_without_extensions(image_folder_path)

    if n <= 0:
        raise SamplingException(
            f"You must select a strictly positive number of frames to select"
        )
    if n > len(path_list):
        raise SamplingException(
            f"Image bank contains {len(path_list)} frames, but {n} frames where required for the "
            f"N first strategy !"
        )
    path_list.sort()
    output_list = path_list[:n]
    return output_list,flag

def diversity_from_embeddings(
    image_labels_path: str,
    embeddings_paths:str,
    n: int = DEFAULT_SUB_SAMPLE,
    **kwargs,
):
    if n <= 0:
        raise SamplingException(f"You must select a strictly positive number of frames to select")
    
    txt_files = [filename for filename in os.listdir(image_labels_path)]
    embeddings = torch.stack([torch.load(os.path.join(embeddings_paths, os.path.splitext(txt_file)[0] + '_embedding.pt'), map_location='cpu') for txt_file in txt_files])
    kept_embedding_mask = np.ones(len(txt_files), dtype=bool)

    for idx, txt_file in enumerate(txt_files):
        with open(os.path.join(image_labels_path, txt_file), "r") as f:
            lines = f.readlines()
            if not lines:
                kept_embedding_mask[idx] = False

    if n > np.sum(kept_embedding_mask):
        raise SamplingException(
            f"Image bank contains {np.sum(kept_embedding_mask)} valid frames, but {n} frames where required for the "
            f"diversity strategy !"
        )
    
    embeddings = embeddings[kept_embedding_mask][:8052]
    m = len(embeddings)
    batch_size = m // n
    batched_embeddings = torch.split(embeddings, batch_size)

    first_batch = batched_embeddings[0]
    selected_idx = [select_start_embedding_idx(first_batch)]
    selected_embeddings = [first_batch[selected_idx[0]]]

    for batch_idx in range(1, n):
        batch = batched_embeddings[batch_idx]
        sel_embedding = min_max_cosine_similarity(batch, torch.stack(selected_embeddings))
        sel_embedding_batch_idx = np.arange(len(batch))[torch.nonzero(torch.all(batch == sel_embedding, dim=1))[0]]
        selected_idx.append(sel_embedding_batch_idx + batch_idx * len(batch))
        selected_embeddings.append(sel_embedding)
    
    selected_images = [os.path.splitext(txt_files[int(i)])[0] for i in selected_idx]

    return selected_images, 0
