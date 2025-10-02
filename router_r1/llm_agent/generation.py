import torch
import re
from collections import deque
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
from .route_service import access_routing_pool


@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    exp_name: str = None
    api_base: str = None
    api_key: str = None

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        processed_responses = []
        for resp in responses_str:
            if '</answer>' in resp:
                cutoff = resp.index('</answer>') + len('</answer>')
                processed_responses.append(resp[:cutoff])
            elif '</search>' in resp:
                cutoff = resp.rindex('</search>') + len('</search>')
                processed_responses.append(resp[:cutoff])
            else:
                processed_responses.append(resp)

        responses_str = processed_responses


        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
        batch_completion_tokens = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.float32)
        
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_route_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum().item() > 0:
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            try:
                gen_output = self._generate_with_gpu_padding(rollings_active)
            except Exception as e:
                print(e)
                break

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # Execute in environment and process observations
            next_obs, dones, valid_action, is_route, cur_completion_tokens = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask
            )
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_route_stats += torch.tensor(is_route, dtype=torch.int)
            batch_completion_tokens += torch.tensor(cur_completion_tokens, dtype=torch.float32)

            next_obs_ids = self._process_next_obs(next_obs)
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )

        # final LLM rollout
        if active_mask.sum().item() > 0:
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # # Execute in environment and process observations
            _, dones, valid_action, is_route, cur_completion_tokens = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_route=False
            )

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_route_stats += torch.tensor(is_route, dtype=torch.int)
            batch_completion_tokens += torch.tensor(cur_completion_tokens, dtype=torch.float32)

            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
            )
        
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_route_stats'] = valid_route_stats.tolist()
        meta_info['batch_completion_tokens'] = batch_completion_tokens.tolist()
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_route=True) -> List[str]:
        """
        Execute a batch of LLM predictions against the environment.

        Supports multiple `<search>` blocks per prediction by sequentially
        routing each query and accumulating the associated observations.
        """
        action_sequences = self.postprocess_predictions(predictions)
        batch_size = len(action_sequences)

        if active_mask is None:
            active_mask_list = [True] * batch_size
        elif isinstance(active_mask, torch.Tensor):
            active_mask_list = [bool(x) for x in active_mask.tolist()]
        else:
            active_mask_list = [bool(x) for x in active_mask]

        next_obs_segments = [[] for _ in range(batch_size)]
        dones = [0] * batch_size
        valid_action_counts = [0] * batch_size
        route_counts = [0] * batch_size
        completion_tokens_totals = [0.0] * batch_size

        action_queues = [deque(seq) for seq in action_sequences]

        # Immediate handling for inactive trajectories
        for idx, active in enumerate(active_mask_list):
            if not active:
                dones[idx] = 1
                action_queues[idx].clear()

        # Iteratively process search actions while available
        while True:
            search_indices = []
            search_queries = []
            for idx, queue in enumerate(action_queues):
                if not active_mask_list[idx] or not queue:
                    continue
                action, content = queue[0]
                if action == 'search':
                    queue.popleft()
                    search_indices.append(idx)
                    search_queries.append(content)

            if not search_indices:
                break

            if do_route:
                route_results, completion_tokens_list = self.batch_route(search_queries)
            else:
                route_results = [''] * len(search_queries)
                completion_tokens_list = [0.0] * len(search_queries)

            for offset, idx in enumerate(search_indices):
                result = route_results[offset]
                tokens = completion_tokens_list[offset]

                result_str = '' if result is None else str(result)
                tokens = float(tokens)

                route_counts[idx] += 1
                completion_tokens_totals[idx] += tokens

                lowered = result_str.strip().lower()
                if lowered in {"llm name error", "api request error"}:
                    next_obs_segments[idx].append('\n\n<information>None</information>\n\n')
                    # invalid route attempt does not increase valid_action_counts
                else:
                    next_obs_segments[idx].append(f'\n\n<information>{result_str.strip()}</information>\n\n')
                    valid_action_counts[idx] += 1

        # Process remaining non-search actions (answers / invalid attempts)
        for idx, queue in enumerate(action_queues):
            if not active_mask_list[idx]:
                continue

            while queue:
                action, content = queue.popleft()

                if action == 'answer':
                    dones[idx] = 1
                    valid_action_counts[idx] += 1
                    break
                if action.startswith('route invalid'):
                    # Invalid route specification: mark trajectory as still active but do not add information
                    break
                if action == 'noop':
                    break
                # Unknown action: ignore and continue checking remaining items

        next_obs = [''.join(segments) for segments in next_obs_segments]

        return next_obs, dones, valid_action_counts, route_counts, completion_tokens_totals

    def postprocess_predictions(self, predictions: List[Any]) -> List[List[Tuple[str, str]]]:
        """
        Process text predictions into an ordered list of (action, content) tuples for each sample.

        Args:
            predictions: raw LLM outputs.

        Returns:
            For each prediction, an ordered list containing one entry per
            `<search>`/`<answer>` block (after basic validation).
        """
        pattern = re.compile(r'<(search|answer)>(.*?)</\1>', re.DOTALL)
        sequences: List[List[Tuple[str, str]]] = []

        for prediction in predictions:
            if not isinstance(prediction, str):
                raise ValueError(f"Invalid prediction type: {type(prediction)}")

            actions: List[Tuple[str, str]] = []
            for match in pattern.finditer(prediction):
                action = match.group(1)
                content = match.group(2).strip()

                if action == 'search':
                    lowered = content.strip().lower()
                    if 'llm-name' in lowered or 'your-query' in lowered:
                        actions.append(('route invalid-1', content))
                        continue
                    if ':' not in content:
                        actions.append(('route invalid-2', content))
                        continue
                    if lowered.split(':')[-1].strip() == '':
                        actions.append(('route invalid-3', content))
                        continue

                actions.append((action, content))

            if not actions:
                actions.append(('noop', ''))

            sequences.append(actions)

        return sequences

    def batch_route(self, queries: List[str] = None) -> str:
        ret = access_routing_pool(queries=queries, api_base=self.config.api_base, api_key=self.config.api_key)
        
        return ret['result'], ret["completion_tokens_list"]
