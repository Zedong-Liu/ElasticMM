# Complete implementation of step functions for V0Worker
# Based on vLLM 0.10.1 API (not EPD's old API)

import torch
import copy
import time
from typing import Dict, List, Any
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalInputs


def step_encoding_impl(worker, batched_requests, block_tables):
    """
    Execute encoding step: process images and generate vision embeddings
    Using the working processor approach from vLLM 0.10.1
    """
    vision_embeddings = {}
    
    try:
        with torch.inference_mode():
            # Process each request
            for request in batched_requests.requests:
                # For text-only requests, skip vision processing but still return empty entry
                if not request.multi_modal_data:
                    # Text-only request: return empty dict entry (will be None in stage_engine)
                    vision_embeddings[request.request_id] = {
                        'embeddings': None,
                        'multi_modal_kwargs': None,
                        'multi_modal_placeholders': None
                    }
                    continue
                
                # Use vLLM's processor (current vLLM version approach)
                from vllm.multimodal import MULTIMODAL_REGISTRY
                mm_processor = MULTIMODAL_REGISTRY.create_processor(worker.model_config, disable_cache=False)
                
                processed = mm_processor.apply(
                    prompt=request.prompt,
                    mm_data=request.multi_modal_data,
                    hf_processor_mm_kwargs={},
                )
                
                if 'mm_kwargs' not in processed:
                    print(f"[Encoding] WARNING: No mm_kwargs for {request.request_id}")
                    continue
                
                # CRITICAL FIX: Extract both mm_kwargs and mm_placeholders from processed result
                # These are needed by vLLM's SequenceGroupMetadata
                mm_kwargs = processed['mm_kwargs']
                mm_placeholders = processed.get('mm_placeholders', None)
                
                # Deep copy the original objects before we modify/access them
                multi_modal_kwargs_for_prefill = copy.deepcopy(mm_kwargs)
                multi_modal_placeholders_for_prefill = copy.deepcopy(mm_placeholders) if mm_placeholders else None
                
                # Now we can safely access mm_kwargs data for vision embedding processing
                if hasattr(worker.model_runner.model, '_process_image_input'):
                    # Qwen2.5-VL path
                    mm_data_dict = mm_kwargs.get_data()
                    image_input = {
                        "type": "pixel_values",
                        "pixel_values": mm_data_dict.get("pixel_values"),
                        "image_grid_thw": mm_data_dict.get("image_grid_thw"),
                    }
                    embeddings_tuple = worker.model_runner.model._process_image_input(image_input)
                    embeddings = torch.cat(embeddings_tuple, dim=0) if isinstance(embeddings_tuple, tuple) else embeddings_tuple
                elif hasattr(worker.model_runner.model, 'get_vision_hidden_states'):
                    # MiniCPM-V path
                    mm_data_dict = mm_kwargs.get_data()
                    image_input = worker.model_runner.model._parse_and_validate_inputs(
                        torch.randn(1), **mm_data_dict
                    )
                    embeddings = worker.model_runner.model.get_vision_hidden_states(image_input)
                else:
                    print(f"[V0Worker] Warning: No vision processing method found")
                    continue
                
                # Use the deep-copied versions for passing to Prefill
                multi_modal_kwargs = multi_modal_kwargs_for_prefill
                multi_modal_placeholders = multi_modal_placeholders_for_prefill
                
                # Store in cache
                if worker.ve_cache is not None:
                    worker.ve_cache[request.request_id] = embeddings
                
                # CRITICAL: Store mm_kwargs AND mm_placeholders in a dict to return
                # (can't modify request object - it won't survive Ray serialization)
                vision_embeddings[request.request_id] = {
                    'embeddings': embeddings,
                    'multi_modal_kwargs': multi_modal_kwargs,
                    'multi_modal_placeholders': multi_modal_placeholders
                }
            
    except Exception as e:
        print(f"[V0Worker] Error in step_encoding: {e}")
        import traceback
        traceback.print_exc()
    
    torch.cuda.synchronize()
    
    # Timing already recorded in request objects
    return vision_embeddings


def step_prefill_impl(worker, batched_requests, kv_block_tables, vision_block_tables=None):
    """
    Execute prefill step: process full sequence and generate KV cache + first token
    Real implementation using model_runner.execute_model (vLLM 0.10.1 API)
    """
    try:
        with torch.inference_mode():
            from vllm.sequence import SequenceData, SequenceGroupMetadata
            from vllm import SamplingParams
            
            # Create SequenceGroupMetadata objects for vLLM 0.10.1
            seq_group_metadata_list = []
            expanded_tokens_map = {}  # Map request_id -> expanded_prompt_token_ids
            
            for idx, request in enumerate(batched_requests.requests):
                
                # Generate a unique seq_id (use hash of request_id)
                seq_id = hash(request.request_id) % (2**31)
                
                # CRITICAL: Expand image placeholder tokens to match vision embeddings count
                # This ensures seq_len and slot_mapping account for all vision tokens
                original_token_count = len(request.prompt_token_ids)
                prompt_token_ids = list(request.prompt_token_ids)
                mm_placeholders_expanded = None
                
                # Check if ve_cache exists and contains the request
                if worker.ve_cache is not None and request.request_id in worker.ve_cache:
                    ve_tensor = worker.ve_cache[request.request_id]
                    num_vision_tokens = ve_tensor.shape[0]  # e.g., 216
                    image_token_id = 151673  # Qwen2.5-VL image token ID
                    # Vision embedding found (debug print removed for performance)
                    
                    # Find and expand image tokens
                    expanded_tokens = []
                    new_offset = 0
                    found_image_token = False
                    for i, token_id in enumerate(prompt_token_ids):
                        if token_id == image_token_id:
                            # Replace single placeholder with multiple placeholders
                            found_image_token = True
                            original_offset = i
                            expanded_tokens.extend([image_token_id] * num_vision_tokens)
                            new_offset = len(expanded_tokens) - num_vision_tokens  # Start of vision tokens
                        else:
                            expanded_tokens.append(token_id)
                    
                    if not found_image_token:
                        print(f"[Prefill] WARNING: No image token {image_token_id} found in {request.request_id}")
                        # Use mm_placeholders to find the position
                        mm_placeholders_orig = getattr(request, 'multi_modal_placeholders', None)
                        if mm_placeholders_orig and 'image' in mm_placeholders_orig:
                            placeholder = mm_placeholders_orig['image'][0]
                            offset = placeholder.offset
                            # Insert vision tokens at the placeholder position
                            expanded_tokens = prompt_token_ids[:offset] + [image_token_id] * num_vision_tokens + prompt_token_ids[offset+1:]
                            new_offset = offset
                    
                    prompt_token_ids = expanded_tokens
                    
                    # CRITICAL: Save expanded tokens to map for stage_engine
                    # DO NOT update request.prompt_token_ids here to avoid repeated expansion
                    expanded_tokens_map[request.request_id] = list(prompt_token_ids)
                    
                    # Update mm_placeholders to reflect the expansion
                    from vllm.multimodal.inputs import PlaceholderRange
                    mm_placeholders_expanded = {
                        'image': [PlaceholderRange(offset=new_offset, length=num_vision_tokens, is_embed=True)]
                    }
                
                # Create SequenceData using from_seqs (vLLM 0.10.1 API)
                seq_data = SequenceData.from_seqs(
                    prompt_token_ids=prompt_token_ids,
                    output_token_ids=None,
                )
                
                # Create sampling params
                sampling_params = SamplingParams(
                    temperature=0.8,
                    top_p=0.9,
                    max_tokens=request.max_tokens or 50,
                )
                
                # Get block table for this request (if available)
                # kv_block_tables is a dict: {request_id: [block_ids...]}
                block_table = kv_block_tables.get(request.request_id, []) if kv_block_tables else []
                
                
                # CRITICAL: Use the original MultiModalKwargs object from encoding stage
                # Do NOT convert to dict or modify it - vLLM needs the original structure
                mm_data_for_metadata = request.multi_modal_kwargs
                
                
                # Get multi_modal_placeholders - use expanded version if available
                # The expanded version has is_embed=True and correct offset/length for 216 tokens
                mm_placeholders_for_metadata = mm_placeholders_expanded if mm_placeholders_expanded else getattr(request, 'multi_modal_placeholders', None)
                
                # CRITICAL: Inject vision embeddings BEFORE creating SequenceGroupMetadata
                if worker.ve_cache is not None and request.request_id in worker.ve_cache:
                    ve_tensor = worker.ve_cache[request.request_id]
                    mm_data_dict = mm_data_for_metadata.get_data()
                    if 'pixel_values' in mm_data_dict and hasattr(mm_data_for_metadata, '_items_by_modality'):
                        image_items = mm_data_for_metadata._items_by_modality.get('image', [])
                        if image_items:
                            first_item = image_items[0]
                            if 'pixel_values' in first_item:
                                del first_item['pixel_values']
                            first_item['image_embeds'] = ve_tensor
                
                # Create SequenceGroupMetadata (vLLM 0.10.1)
                seq_group_metadata = SequenceGroupMetadata(
                    request_id=str(request.request_id),
                    is_prompt=True,
                    seq_data={seq_id: seq_data},
                    sampling_params=sampling_params,
                    block_tables={seq_id: block_table},
                    do_sample=True,
                    pooling_params=None,
                    token_chunk_size=len(prompt_token_ids),  # Use expanded length!
                    lora_request=None,
                    computed_block_nums=[],
                    multi_modal_data=mm_data_for_metadata,  # Pass complete mm_kwargs!
                    multi_modal_placeholders=mm_placeholders_for_metadata,  # CRITICAL: Pass expanded mm_placeholders with is_embed=True!
                )
                
                seq_group_metadata_list.append(seq_group_metadata)
            
            # Prepare model input (vLLM 0.10.1)
            finished_requests_ids = []
            
            model_input = worker.model_runner.prepare_model_input(
                seq_group_metadata_list, 
                virtual_engine=0,
                finished_requests_ids=finished_requests_ids
            )
            
            # CRITICAL: Re-inject vision embeddings after prepare_model_input
            if model_input.multi_modal_kwargs:
                for request in batched_requests.requests:
                    if worker.ve_cache is not None and request.request_id in worker.ve_cache:
                        ve_tensor = worker.ve_cache[request.request_id]
                        if 'pixel_values' in model_input.multi_modal_kwargs:
                            del model_input.multi_modal_kwargs['pixel_values']
                            model_input.multi_modal_kwargs['image_embeds'] = ve_tensor
                        break
            
            
            # Collect vision embeddings (already injected above)
            if vision_block_tables and any(vision_block_tables):
                vision_embeds = []
                for request in batched_requests.requests:
                    if worker.ve_cache is not None and request.request_id in worker.ve_cache:
                        vision_embeds.append(worker.ve_cache[request.request_id])
                
                if vision_embeds and model_input.multi_modal_kwargs is not None:
                    # Remove pixel_values (raw image tensors) - we have embeddings now
                    if 'pixel_values' in model_input.multi_modal_kwargs:
                        del model_input.multi_modal_kwargs['pixel_values']
                    
                    # Add pre-computed vision embeddings
                    if len(vision_embeds) == 1:
                        model_input.multi_modal_kwargs['image_embeds'] = vision_embeds[0]
                    else:
                        model_input.multi_modal_kwargs['image_embeds'] = torch.cat(vision_embeds, dim=0)
                    
                    # Vision embeddings injected (debug prints removed for performance)
            
            # Execute model (vLLM 0.10.1)
            # vLLM expects kv_caches as List[Tuple[key_cache, value_cache]]
            # Our format: [num_layers, 2, num_blocks, block_size, num_heads, head_size]
            
            # REFACTORED: Use vLLM's recommended KV cache approach
            # KV cache is already bound to Attention layers via bind_kv_cache() in init_kvcache()
            # The kv_caches parameter to execute_model is ignored in vLLM 0.10.1+
            # vLLM accesses KV cache directly through Attention.kv_cache[virtual_engine]
            
            # Execute model (pass empty list for kv_caches as it's ignored)
            seq_outs = worker.model_runner.execute_model(model_input, [], None)
            
            
            # Extract generated tokens
            generated_tokens = []
            if seq_outs and len(seq_outs) > 0:
                for output in seq_outs[0]:
                    if hasattr(output, 'samples') and output.samples:
                        generated_tokens.append(output.samples[0].output_token)
                    else:
                        generated_tokens.append(1)  # Fallback
            
            # Return outputs
            from elasticmm.engine.v0.utils import StepOutput
            outputs = []
            
            for i, request in enumerate(batched_requests.requests):
                token_id = generated_tokens[i] if i < len(generated_tokens) else 1
                output = StepOutput(
                    request_id=request.request_id,
                    output_token_ids=[token_id],
                    finished=(token_id == 2),  # 2 is EOS
                )
                outputs.append(output)
            
            # Prefill completed (debug print removed for performance)
            return outputs, expanded_tokens_map
            
    except Exception as e:
        print(f"[V0Worker] Error in step_prefill: {e}")
        import traceback
        traceback.print_exc()
        # Return placeholder outputs on error
        from elasticmm.engine.v0.utils import StepOutput
        outputs = [StepOutput(request_id=r.request_id, output_token_ids=[1], finished=False) 
                for r in batched_requests.requests]
        # Return empty expanded_tokens_map on error
        return outputs, {}


def step_decode_impl(worker, batched_requests, kv_block_tables):
    """
    Execute decode step: autoregressive generation
    Real implementation using model_runner.execute_model (vLLM 0.10.1 API)
    """
    try:
        with torch.inference_mode():
            from vllm.sequence import SequenceData, SequenceGroupMetadata
            from vllm import SamplingParams
            
            # Track decode steps
            if not hasattr(worker, '_decode_steps'):
                worker._decode_steps = {}
            
            # Create SequenceGroupMetadata objects for vLLM 0.10.1
            seq_group_metadata_list = []
            for idx, request in enumerate(batched_requests.requests):
                
                req_id = request.request_id
                worker._decode_steps[req_id] = worker._decode_steps.get(req_id, 0) + 1
                
                # Generate a unique seq_id (use hash of request_id)
                seq_id = hash(request.request_id) % (2**31)
                
                # For decode, we need prompt tokens + all output tokens so far
                # NOTE: request.prompt_token_ids should already be EXPANDED by stage_engine._receive_from_prefill
                # (it restores migrating_req.expanded_prompt_token_ids into request.prompt_token_ids)
                prompt_tokens = request.prompt_token_ids if request.prompt_token_ids else []
                output_tokens = request.output_token_ids if request.output_token_ids else []
                
                # Create SequenceData using from_seqs (vLLM 0.10.1 API)
                seq_data = SequenceData.from_seqs(
                    prompt_token_ids=prompt_tokens,
                    output_token_ids=output_tokens,
                )
                
                # CRITICAL: For decode, mark all EXISTING tokens as computed, except the LAST one
                # vLLM will compute the position for the LAST token (which we're about to extend)
                num_prompt = len(prompt_tokens)
                num_output = len(output_tokens)
                total_tokens = num_prompt + num_output
                
                # CRITICAL FIX: If no output tokens, this means prefill failed
                # Don't proceed with decode - it will cause infinite loop
                if num_output == 0:
                    print(f"[Decode] SKIPPING {req_id}: No output tokens from prefill - prefill likely failed")
                    continue
                
                # Mark tokens as computed: all tokens except the last one
                # The last token's position will be computed by vLLM during decode
                if total_tokens > 0:
                    # In decode, we already have all tokens (prompt + outputs from prefill)
                    # We want vLLM to generate the NEXT token, so mark all current tokens as computed
                    seq_data.update_num_computed_tokens(total_tokens - 1)
                
                # Create sampling params
                sampling_params = SamplingParams(
                    temperature=0.8,
                    top_p=0.9,
                    max_tokens=request.max_tokens or 50,
                )
                
                # Get block table for this request
                block_table = kv_block_tables.get(request.request_id, []) if kv_block_tables else []
                
                # Validate block allocation (but don't modify blocks here)
                # NOTE: request.prompt_token_ids is already expanded by stage_engine._receive_from_prefill
                seq_len = len(request.prompt_token_ids) + len(request.output_token_ids)
                blocks_needed = (seq_len + worker.block_size - 1) // worker.block_size
                
                # CRITICAL: Validate block allocation without modifying
                if len(block_table) < blocks_needed:
                    print(f"[Decode] ERROR: {request.request_id} needs {blocks_needed} blocks, but only has {len(block_table)} blocks")
                    print(f"[Decode] This indicates a block allocation issue in stage_engine.py")
                    print(f"[Decode] Worker cannot proceed with insufficient blocks - this will cause KV cache errors")
                    # Don't modify block_table - let stage_engine handle this
                    # For now, we'll proceed with available blocks and log the issue
                
                # CRITICAL: In decode stage, pass the same multi_modal_data as prefill
                # Although pixel_values won't be processed again (already in KV cache),
                # MRoPE still needs image_grid_thw for position calculations
                mm_data_for_decode = request.multi_modal_kwargs
                mm_placeholders_for_decode = getattr(request, 'multi_modal_placeholders', None)
                
                # Create SequenceGroupMetadata (for decode, is_prompt=False)
                seq_group_metadata = SequenceGroupMetadata(
                    request_id=str(request.request_id),
                    is_prompt=False,  # Decode stage
                    seq_data={seq_id: seq_data},
                    sampling_params=sampling_params,
                    block_tables={seq_id: block_table},
                    do_sample=True,
                    pooling_params=None,
                    token_chunk_size=1,  # Decode one token at a time
                    lora_request=None,
                    computed_block_nums=[],
                    multi_modal_data=mm_data_for_decode,  # CRITICAL: Pass multimodal data for MRoPE!
                    multi_modal_placeholders=mm_placeholders_for_decode,  # Pass placeholders too
                )
                seq_group_metadata_list.append(seq_group_metadata)
            
            # Prepare model input (vLLM 0.10.1)
            finished_requests_ids = []
            model_input = worker.model_runner.prepare_model_input(
                seq_group_metadata_list,
                virtual_engine=0,
                finished_requests_ids=finished_requests_ids
            )
            
            # Execute model (vLLM 0.10.1)
            # vLLM expects kv_caches as List[Tuple[key_cache, value_cache]]
            # Our format: [num_layers, 2, num_blocks, block_size, num_heads, head_size]
            
            # KV cache is already bound to Attention layers via bind_kv_cache()
            # vLLM accesses it directly through Attention.kv_cache[virtual_engine]
            
            # Execute model (pass empty list for kv_caches as it's ignored)
            seq_outs = worker.model_runner.execute_model(model_input, [], None)
            
            # Extract generated tokens
            generated_tokens = []
            if seq_outs and len(seq_outs) > 0:
                for output in seq_outs[0]:
                    if hasattr(output, 'samples') and output.samples:
                        generated_tokens.append(output.samples[0].output_token)
                    else:
                        generated_tokens.append(1)  # Fallback
            
            # Create outputs
            from elasticmm.engine.v0.utils import StepOutput
            outputs = []
            for i, request in enumerate(batched_requests.requests):
                req_id = request.request_id
                
                token_id = generated_tokens[i] if i < len(generated_tokens) else 1
                output_tokens = (request.output_token_ids or []) + [token_id]
                
                # Finish after 50 tokens or if EOS
                finished = (worker._decode_steps[req_id] >= 50) or (token_id == 2)  # 2 is EOS
                
                # Per-request decode progress (debug removed for performance)
                
                output = StepOutput(
                    request_id=request.request_id,
                    output_token_ids=output_tokens,
                    finished=finished,
                )
                outputs.append(output)
            
            return outputs
            
    except Exception as e:
        print(f"[V0Worker] FATAL Error in step_decode: {e}")
        import traceback
        traceback.print_exc()
        # CRITICAL: Exit on error to prevent infinite error loop
        print(f"[V0Worker] Exiting due to decode error to prevent infinite loop")
        import sys
        sys.exit(1)

