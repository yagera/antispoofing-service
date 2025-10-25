import torch
import random


def apply_random_segment_extraction(
        input_tensor: torch.Tensor,
        target_length: int = 64600
) -> torch.Tensor:
    processed_tensor = input_tensor.clone()
    while processed_tensor.ndim > 1:
        processed_tensor = processed_tensor.squeeze(0)

    current_length = processed_tensor.shape[0]

    if current_length > target_length:
        start_position = random.randint(0, current_length - target_length)
        return processed_tensor[start_position:start_position + target_length]

    repetition_factor = (target_length // current_length) + 1
    extended_tensor = processed_tensor.repeat(repetition_factor)

    return extended_tensor[:target_length]
