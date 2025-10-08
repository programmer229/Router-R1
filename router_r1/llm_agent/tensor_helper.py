import torch
from typing import Dict, Tuple, List
from dataclasses import dataclass

@dataclass
class TensorConfig:
    pad_token_id: int
    max_prompt_length: int
    max_obs_length: int
    max_start_length: int

class TensorHelper:
    def __init__(self, config: TensorConfig):
        self.config = config

    def cut_to_effective_len(self, tensor_dict: Dict[str, torch.Tensor], 
                            keys: List[str], cut_left: bool = True) -> Dict[str, torch.Tensor]:
        """Cut tensors to their effective length based on attention mask."""
        effective_len = tensor_dict['attention_mask'].sum(dim=1).max()
        result = tensor_dict.copy()
        
        for key in keys:
            if cut_left:
                result[key] = tensor_dict[key][:, -effective_len:]
            else:
                result[key] = tensor_dict[key][:, :effective_len]
        return result

    def convert_pad_structure(self, tensor: torch.Tensor, pad_to_left: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert padding structure and return sorted tensor with indices."""
        mask = tensor != self.config.pad_token_id if pad_to_left else tensor == self.config.pad_token_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        return tensor.gather(1, sorted_indices), sorted_indices

    def create_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create attention mask from input ids."""
        return torch.where(input_ids != self.config.pad_token_id, 1, 0)

    def create_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Create position ids from attention mask."""
        return (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

    def concatenate_with_padding(self, tensors: List[torch.Tensor], 
                               pad_to_left: bool = True) -> torch.Tensor:
        """Concatenate tensors and handle padding."""
        concatenated = torch.cat(tensors, dim=1)
        padded_tensor, _ = self.convert_pad_structure(concatenated, pad_to_left)
        return padded_tensor

    def _example_level_pad(self, responses: torch.Tensor,
                          responses_str: List[str],
                          active_mask: torch.Tensor) -> Tuple[torch.Tensor, List[List[str]], int]:
        """
        Pad responses for non-active examples with pad tokens.

        When multiple samples-per-environment are generated (n > 1),
        `responses` will contain `active_envs * samples_per_env` rows.
        This method reshapes them back into per-environment slots while
        keeping inactive examples padded out.
        """
        num_active = int(active_mask.sum().item())

        if responses.numel() == 0 or num_active == 0:
            samples_per_env = 0
            batch_size = active_mask.shape[0]
            seq_len = responses.shape[1] if responses.dim() > 1 else 0
            padded = torch.full(
                (batch_size, samples_per_env, seq_len),
                self.config.pad_token_id,
                dtype=responses.dtype if responses.numel() else torch.long,
                device=responses.device if responses.numel() else active_mask.device,
            )
            padded_responses_str: List[List[str]] = [[] for _ in range(batch_size)]
            return padded, padded_responses_str, samples_per_env

        total_samples = responses.shape[0]
        assert total_samples % num_active == 0, (
            f"Mismatch between active envs ({num_active}) and responses ({total_samples})"
        )
        samples_per_env = total_samples // num_active
        seq_len = responses.shape[1]
        batch_size = active_mask.shape[0]

        padded_responses = torch.full(
            (batch_size, samples_per_env, seq_len),
            self.config.pad_token_id,
            dtype=responses.dtype,
            device=responses.device,
        )
        padded_responses_str = [[""] * samples_per_env for _ in range(batch_size)]

        active_indices = torch.nonzero(active_mask, as_tuple=False).flatten().tolist()
        cursor = 0

        for idx in active_indices:
            span_slice = slice(cursor, cursor + samples_per_env)
            padded_responses[idx] = responses[span_slice]
            padded_responses_str[idx] = responses_str[span_slice]
            cursor += samples_per_env

        return padded_responses, padded_responses_str, samples_per_env
