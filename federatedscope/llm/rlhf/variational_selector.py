"""
Variational Selector for RLHF
Uses VPL-trained variational encoder to sample client-specific z and make conditional choices.
"""
import torch
import torch.nn.functional as F
import numpy as np
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader

from federatedscope.llm.dataset.llm_dataset import LLMComparisonDataset, DefaultToken
from federatedscope.llm.model.variational_encoder import VariationalEncoder
from federatedscope.llm.model.variational_encoder_gp import VariationalEncoderGP
from federatedscope.llm.dataloader import LLMRewardCollator

logger = logging.getLogger(__name__)


def cal_acc(logits, labels, choices):
    """Calculate accuracy for binary choice task."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    new_labels = torch.full_like(shift_labels, DefaultToken.IGNORE_INDEX.value)
    for idx, choice in enumerate(choices):
        new_labels[shift_labels == choice] = idx
    
    new_labels = new_labels.view(-1)
    new_logits = shift_logits[..., choices].view(-1, len(choices))
    new_logits = new_logits[(new_labels != DefaultToken.IGNORE_INDEX.value), :]
    new_labels = new_labels[(new_labels != DefaultToken.IGNORE_INDEX.value)]
    _, predicted = new_logits.max(1)
    
    return new_labels, new_logits, predicted, predicted.eq(new_labels).sum().item()


def extract_preference_features_for_variational(model, tokenizer, list_data_dict, prompt_template, 
                                                choices, device='cuda:0', use_feature_difference=True):
    """
    Extract preference features from data for variational encoder.
    
    Args:
        model: LLM model (selector model)
        tokenizer: Tokenizer
        list_data_dict: List of data samples with output_A and output_B
        prompt_template: Prompt template for comparison
        choices: Choice token indices [A_token, B_token]
        device: Device to run on
        use_feature_difference: Whether to use embedding difference
        
    Returns:
        features: Extracted preference features (batch, feature_dim)
        hidden_states_list: List of hidden states for each sample
    """
    # Ensure model is on the correct device
    model_device = device
    if hasattr(model, 'device'):
        model_device = model.device
    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'device'):
        model_device = model.base_model.device
    else:
        try:
            model_device = next(model.parameters()).device
        except:
            model_device = device
    
    # Create dataset
    dataset = LLMComparisonDataset(
        list_data_dict,
        tokenizer,
        prompt_input=prompt_template,
        prompt_no_input=prompt_template,
        output_A="output_A",
        output_B="output_B",
        choice="fake_choice",  # Temporary choice for dataset creation
    )
    
    collator = LLMRewardCollator(tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collator)
    
    features_list = []
    hidden_states_list = []
    
    model.eval()
    with torch.no_grad():
        for data_batch in tqdm(dataloader, desc="Extracting preference features"):
            # Get both win and lose inputs - move to model device
            win_input_ids = data_batch["win_input_ids"].to(model_device)
            win_attention_mask = data_batch["win_attention_mask"].to(model_device)
            lose_input_ids = data_batch["lose_input_ids"].to(model_device)
            lose_attention_mask = data_batch["lose_attention_mask"].to(model_device)
            
            # Get hidden states
            win_outputs = model(input_ids=win_input_ids, attention_mask=win_attention_mask, output_hidden_states=True)
            lose_outputs = model(input_ids=lose_input_ids, attention_mask=lose_attention_mask, output_hidden_states=True)
            
            win_hidden = win_outputs.hidden_states[-1]  # Last layer hidden states
            lose_hidden = lose_outputs.hidden_states[-1]
            
            batch_size = win_hidden.shape[0]
            hidden_dim = win_hidden.shape[-1]
            
            # Extract embeddings at choice positions
            A_token, B_token = choices[0], choices[1]
            
            for b in range(batch_size):
                # For win (output_A) - find A token
                win_labels = win_input_ids[b]
                win_h = win_hidden[b]  # (seq_len, hidden_dim)
                
                # For lose (output_B) - find B token  
                lose_labels = lose_input_ids[b]
                lose_h = lose_hidden[b]  # (seq_len, hidden_dim)
                
                # Find choice token positions
                win_A_pos = (win_labels[1:] == A_token)  # Shift by 1 for alignment
                lose_B_pos = (lose_labels[1:] == B_token)
                
                if win_A_pos.any():
                    chosen_emb = win_h[:-1][win_A_pos].mean(dim=0)  # Mean over A positions
                else:
                    chosen_emb = win_h.mean(dim=0)  # Fallback to mean
                
                if lose_B_pos.any():
                    rejected_emb = lose_h[:-1][lose_B_pos].mean(dim=0)  # Mean over B positions
                else:
                    rejected_emb = lose_h.mean(dim=0)  # Fallback to mean
                
                # Compute difference: chosen - rejected
                feature_diff = chosen_emb - rejected_emb
                
                if use_feature_difference:
                    # Use [chosen, rejected, difference] for richer representation
                    feature = torch.cat([chosen_emb, rejected_emb, feature_diff], dim=0)  # (hidden_dim * 3,)
                else:
                    # Use only difference
                    feature = feature_diff  # (hidden_dim,)
                
                features_list.append(feature)
                hidden_states_list.append({
                    'chosen_emb': chosen_emb,
                    'rejected_emb': rejected_emb,
                    'difference': feature_diff
                })
    
    features = torch.stack(features_list, dim=0)  # (batch, feature_dim)
    return features, hidden_states_list


def variational_better_response(list_data_dict, selector_model, selector_tokenizer, 
                                variational_encoder, feature_extractor, prompt_template,
                                choices, device='cuda:0', use_feature_difference=True,
                                num_samples=1, latent_projection=None, use_provided_z=False,
                                z_to_embedding=None, client_average_z_dict=None):
    """
    Use variational encoder to sample z and make conditional choices.
    
    Args:
        list_data_dict: List of data samples with output_A and output_B
            If use_provided_z=True, samples may contain "z" field to use directly
        selector_model: Trained selector model (binary choice model)
        selector_tokenizer: Tokenizer
        variational_encoder: Trained VariationalEncoder or VariationalEncoderGP
        feature_extractor: Feature extractor network (from VPLRewardChoiceTrainer)
        prompt_template: Prompt template for comparison
        choices: Choice token indices [A_token, B_token]
        device: Device to run on
        use_feature_difference: Whether to use embedding difference
        num_samples: Number of z samples to draw per sample (for averaging)
        latent_projection: Projection layer from z to choice logits (from VPLRewardChoiceTrainer)
        use_provided_z: If True, use z from data samples if available; otherwise infer from data
        
    Returns:
        list_data_dict: Updated with "choice" field (0 for A, 1 for B) and z values
    """
    # Check if z values are provided in data or if we should use client-specific z
    provided_z_list = []
    provided_z_tensors = []
    use_client_z = False
    num_provided = 0
    if client_average_z_dict is not None and len(client_average_z_dict) > 0:
        # Check if samples have client_id and we can use client-specific z
        samples_with_client_id = [sample.get('client_id', None) for sample in list_data_dict]
        num_with_client_id = sum(1 for cid in samples_with_client_id if cid is not None and cid in client_average_z_dict)
        if num_with_client_id > 0:
            use_client_z = True
            logger.info(f"Using client-specific z from client_average_z_dict for {num_with_client_id}/{len(list_data_dict)} samples")
            # Get client-specific z for each sample
            provided_z_tensors = []
            for sample in list_data_dict:
                client_id = sample.get('client_id', None)
                if client_id is not None and client_id in client_average_z_dict:
                    z_val = client_average_z_dict[client_id]
                    if isinstance(z_val, torch.Tensor):
                        z_tensor = z_val.to(device)
                    else:
                        z_tensor = torch.tensor(z_val, dtype=torch.float32).to(device)
                    provided_z_tensors.append(z_tensor)
                else:
                    provided_z_tensors.append(None)
            provided_z_list = [z.cpu().numpy() if z is not None else None for z in provided_z_tensors]
            num_provided = len([z for z in provided_z_tensors if z is not None])
            use_provided_z = True  # Treat client z as provided z
        else:
            logger.info("client_average_z_dict available but samples don't have matching client_id, will infer from data")
    
    if not use_client_z and use_provided_z:
        provided_z_list = [sample.get("z", None) for sample in list_data_dict]
        num_provided = sum(1 for z in provided_z_list if z is not None)
        if num_provided > 0:
            logger.info(f"Using provided z values for {num_provided}/{len(list_data_dict)} samples")
            # Convert provided z to tensors
            provided_z_tensors = []
            for z in provided_z_list:
                if z is not None:
                    if isinstance(z, list):
                        z = torch.tensor(z, dtype=torch.float32).to(device)
                    else:
                        z = torch.tensor(z, dtype=torch.float32).to(device)
                    provided_z_tensors.append(z)
                else:
                    provided_z_tensors.append(None)
        else:
            logger.info("No provided z values found, will infer from data")
            use_provided_z = False
    
    if not use_provided_z or num_provided < len(list_data_dict):
        # Need to extract features and infer z for samples without provided z
        logger.info("Extracting preference features for variational selection...")
        
        # Extract preference features
        features, hidden_states_list = extract_preference_features_for_variational(
            selector_model, selector_tokenizer, list_data_dict, prompt_template,
            choices, device, use_feature_difference
        )
        
        # Process features through feature extractor if provided
        if feature_extractor is not None:
            # Ensure feature_extractor is on the same device as features
            feature_extractor_device = device
            if hasattr(feature_extractor, 'parameters'):
                try:
                    feature_extractor_device = next(feature_extractor.parameters()).device
                except:
                    feature_extractor_device = device
            features = features.to(feature_extractor_device)
            features = feature_extractor(features)
        
        logger.info(f"Extracted features shape: {features.shape}")
    
    # Sample z from posterior q(z|x) using variational encoder
    variational_encoder.eval()
    selector_model.eval()
    
    predicted_indices = []
    
    with torch.no_grad():
        if use_provided_z and num_provided == len(list_data_dict):
            # Use provided z values directly (skip inference, but still need to perform selection)
            logger.info("Using all provided z values, skipping inference")
            mu_cpu = None
            logvar_cpu = None
            all_z_samples = []
            for z_tensor in provided_z_tensors:
                if z_tensor is not None:
                    all_z_samples.append([z_tensor.cpu().numpy()])
                else:
                    # Fallback: need to infer
                    logger.warning("Some z values are None, falling back to inference")
                    use_provided_z = False
                    break
            
            if use_provided_z:
                # All z provided, use them directly
                z_mean = np.array([z[0] for z in all_z_samples])  # (batch, latent_dim)
                # Still need to perform selection using provided z values
                # Convert z_mean to tensor for selection
                z_tensor = torch.tensor(z_mean, dtype=torch.float32).to(device)
                
                # Add fake_choice to all samples so LLMComparisonDataset doesn't filter them out
                # We'll compute the actual choice during selection
                for sample in list_data_dict:
                    if "fake_choice" not in sample:
                        sample["fake_choice"] = 0  # Temporary placeholder, will be replaced
                
                # Create dataset for binary choice
                dataset = LLMComparisonDataset(
                    list_data_dict,
                    selector_tokenizer,
                    prompt_input=prompt_template,
                    prompt_no_input=prompt_template,
                    output_A="output_A",
                    output_B="output_B",
                    choice="fake_choice",
                )
                collator = LLMRewardCollator(tokenizer=selector_tokenizer)
                dataloader = DataLoader(dataset, batch_size=4, collate_fn=collator)
                
                # Ensure selector_model is on the correct device
                selector_model_device = device
                if hasattr(selector_model, 'device'):
                    selector_model_device = selector_model.device
                elif hasattr(selector_model, 'base_model') and hasattr(selector_model.base_model, 'device'):
                    selector_model_device = selector_model.base_model.device
                else:
                    try:
                        selector_model_device = next(selector_model.parameters()).device
                    except:
                        selector_model_device = device
                
                # Project z to choice logits
                latent_dim = z_tensor.shape[-1]
                num_choices = len(choices)
                
                batch_predictions = []
                z_idx = 0
                
                for batch_idx, data_batch in enumerate(tqdm(dataloader, desc="Variational selection (using provided z)")):
                    win_input_ids = data_batch["win_input_ids"].to(selector_model_device)
                    win_labels = data_batch["win_labels"].to(selector_model_device)
                    win_attention_mask = data_batch["win_attention_mask"].to(selector_model_device)
                    lose_input_ids = data_batch["lose_input_ids"].to(selector_model_device)
                    lose_labels = data_batch["lose_labels"].to(selector_model_device)
                    lose_attention_mask = data_batch["lose_attention_mask"].to(selector_model_device)
                    
                    batch_size = win_input_ids.shape[0]
                    z_batch = z_tensor[z_idx:z_idx + batch_size].to(selector_model_device)
                    z_idx += batch_size
                
                    # Get logits for both responses
                    win_outputs = selector_model(input_ids=win_input_ids, attention_mask=win_attention_mask)
                    lose_outputs = selector_model(input_ids=lose_input_ids, attention_mask=lose_attention_mask)
                
                    win_logits = win_outputs.logits
                    lose_logits = lose_outputs.logits
                
                    # Use provided latent_projection if available
                    if latent_projection is not None:
                        if next(latent_projection.parameters()).device != z_batch.device:
                            latent_projection = latent_projection.to(z_batch.device)
                        z_projection = latent_projection(z_batch)  # (batch, num_choices)
                    else:
                        # Create temporary projection
                        temp_projection = torch.nn.Linear(latent_dim, num_choices).to(z_batch.device)
                        torch.nn.init.normal_(temp_projection.weight, mean=0.0, std=0.01)
                        torch.nn.init.zeros_(temp_projection.bias)
                        z_projection = temp_projection(z_batch)
                
                    # Get logits at choice positions
                    shift_win_logits = win_logits[..., :-1, :].contiguous()
                    shift_lose_logits = lose_logits[..., :-1, :].contiguous()
                    shift_win_labels = win_labels[..., 1:].contiguous()
                    shift_lose_labels = lose_labels[..., 1:].contiguous()
                
                    # Extract choice logits
                    win_choice_logits = shift_win_logits[..., choices]  # (batch, seq_len-1, num_choices)
                    lose_choice_logits = shift_lose_logits[..., choices]
                
                    # Find choice token positions
                    A_token, B_token = choices[0], choices[1]
                
                    batch_choices = []
                    for b in range(batch_size):
                        # Find A and B positions in win (output_A)
                        win_A_pos = (shift_win_labels[b] == A_token)
                        win_B_pos = (shift_win_labels[b] == B_token)
                    
                        # Find A and B positions in lose (output_B)
                        lose_A_pos = (shift_lose_labels[b] == A_token)
                        lose_B_pos = (shift_lose_labels[b] == B_token)
                    
                        # Get logits at choice positions
                        if win_A_pos.any():
                            win_A_logit = win_choice_logits[b, win_A_pos, 0].mean()  # Choice 0 = A
                        else:
                            win_A_logit = win_choice_logits[b, :, 0].mean()
                    
                        if lose_B_pos.any():
                            lose_B_logit = lose_choice_logits[b, lose_B_pos, 1].mean()  # Choice 1 = B
                        else:
                            lose_B_logit = lose_choice_logits[b, :, 1].mean()
                    
                        # Add z-conditioned bias
                        z_bias_A = z_projection[b, 0]  # Bias for choice A
                        z_bias_B = z_projection[b, 1]  # Bias for choice B
                    
                        # Compare: A (win) vs B (lose) with z conditioning
                        score_A = win_A_logit + z_bias_A
                        score_B = lose_B_logit + z_bias_B
                    
                        # Choose based on scores
                        if score_A > score_B:
                            choice = 0  # A is better
                        else:
                            choice = 1  # B is better
                    
                        batch_choices.append(choice)
                
                    batch_predictions.extend(batch_choices)
                
                predicted_indices = batch_predictions
                mu_cpu = None
                logvar_cpu = None
                all_z_samples = [z_mean]  # Store z_mean for consistency
        else:
            # Encode to get posterior parameters
            # Ensure variational_encoder is on the same device as features
            variational_encoder_device = device
            if hasattr(variational_encoder, 'parameters'):
                try:
                    variational_encoder_device = next(variational_encoder.parameters()).device
                except:
                    variational_encoder_device = device
            features = features.to(variational_encoder_device)
            mu, logvar = variational_encoder.encode(features)
            
            # Store z distribution parameters (mu, logvar) for each sample
            # Convert to CPU and numpy for JSON serialization
            mu_cpu = mu.cpu().numpy()  # (batch, latent_dim)
            logvar_cpu = logvar.cpu().numpy()  # (batch, latent_dim)
        
            # Sample z multiple times and average predictions
            all_predictions = []
            all_z_samples = []  # Store z samples for each data point
            
            # Ensure selector_model is on the correct device (check once before loop)
            selector_model_device = device
            if hasattr(selector_model, 'device'):
                selector_model_device = selector_model.device
            elif hasattr(selector_model, 'base_model') and hasattr(selector_model.base_model, 'device'):
                selector_model_device = selector_model.base_model.device
            else:
                try:
                    selector_model_device = next(selector_model.parameters()).device
                except:
                    selector_model_device = device
            
            for sample_idx in range(num_samples):
                # Sample z from posterior
                z = variational_encoder.reparameterize(mu, logvar)  # (batch, latent_dim)
                # Ensure z is on selector_model_device
                z = z.to(selector_model_device)
                all_z_samples.append(z.cpu().numpy())  # Store z sample
            
                # Create dataset for binary choice
                dataset = LLMComparisonDataset(
                    list_data_dict,
                    selector_tokenizer,
                    prompt_input=prompt_template,
                    prompt_no_input=prompt_template,
                    output_A="output_A",
                    output_B="output_B",
                    choice="fake_choice",
                )
                collator = LLMRewardCollator(tokenizer=selector_tokenizer)
                dataloader = DataLoader(dataset, batch_size=4, collate_fn=collator)
                
                batch_predictions = []
                z_idx = 0  # Track z index across batches
                
                # Project z to choice logits (similar to VPLRewardChoiceTrainer)
                # latent_projection should be provided as a separate parameter
                latent_dim = z.shape[-1]
                num_choices = len(choices)
                
                for batch_idx, data_batch in enumerate(tqdm(dataloader, desc=f"Variational selection (sample {sample_idx+1}/{num_samples})")):
                    win_input_ids = data_batch["win_input_ids"].to(selector_model_device)
                    win_labels = data_batch["win_labels"].to(selector_model_device)
                    win_attention_mask = data_batch["win_attention_mask"].to(selector_model_device)
                    lose_input_ids = data_batch["lose_input_ids"].to(selector_model_device)
                    lose_labels = data_batch["lose_labels"].to(selector_model_device)
                    lose_attention_mask = data_batch["lose_attention_mask"].to(selector_model_device)
                    
                    batch_size = win_input_ids.shape[0]
                    z_batch = z[z_idx:z_idx + batch_size]  # Get z for this batch (already on selector_model_device)
                    z_idx += batch_size
                
                    # Get logits for both responses
                    win_outputs = selector_model(input_ids=win_input_ids, attention_mask=win_attention_mask)
                    lose_outputs = selector_model(input_ids=lose_input_ids, attention_mask=lose_attention_mask)
                
                    win_logits = win_outputs.logits
                    lose_logits = lose_outputs.logits
                
                    # Condition on z: project z to choice logits and add to model logits
                    # Option 1: Add z-conditioned bias to choice token logits
                    # Option 2: Scale logits based on z
                    # We'll use Option 1: project z to choice space and add as bias
                
                    # Use provided latent_projection if available, otherwise create temporary one
                    if latent_projection is not None:
                        # Ensure latent_projection is on the same device as z_batch
                        if next(latent_projection.parameters()).device != z_batch.device:
                            latent_projection = latent_projection.to(z_batch.device)
                        z_projection = latent_projection(z_batch)  # (batch, num_choices)
                    else:
                        # Create temporary projection (should be loaded from trainer)
                        temp_projection = torch.nn.Linear(latent_dim, num_choices).to(z_batch.device)
                        torch.nn.init.normal_(temp_projection.weight, mean=0.0, std=0.01)
                        torch.nn.init.zeros_(temp_projection.bias)
                        z_projection = temp_projection(z_batch)
                
                    # Get logits at choice positions
                    shift_win_logits = win_logits[..., :-1, :].contiguous()
                    shift_lose_logits = lose_logits[..., :-1, :].contiguous()
                    shift_win_labels = win_labels[..., 1:].contiguous()
                    shift_lose_labels = lose_labels[..., 1:].contiguous()
                
                    # Extract choice logits
                    win_choice_logits = shift_win_logits[..., choices]  # (batch, seq_len-1, num_choices)
                    lose_choice_logits = shift_lose_logits[..., choices]
                
                    # Find choice token positions
                    A_token, B_token = choices[0], choices[1]
                
                    batch_choices = []
                    for b in range(batch_size):
                        # Find A and B positions in win (output_A)
                        win_A_pos = (shift_win_labels[b] == A_token)
                        win_B_pos = (shift_win_labels[b] == B_token)
                    
                        # Find A and B positions in lose (output_B)
                        lose_A_pos = (shift_lose_labels[b] == A_token)
                        lose_B_pos = (shift_lose_labels[b] == B_token)
                    
                        # Get logits at choice positions
                        if win_A_pos.any():
                            win_A_logit = win_choice_logits[b, win_A_pos, 0].mean()  # Choice 0 = A
                        else:
                            win_A_logit = win_choice_logits[b, :, 0].mean()
                    
                        if lose_B_pos.any():
                            lose_B_logit = lose_choice_logits[b, lose_B_pos, 1].mean()  # Choice 1 = B
                        else:
                            lose_B_logit = lose_choice_logits[b, :, 1].mean()
                    
                        # Add z-conditioned bias
                        z_bias_A = z_projection[b, 0]  # Bias for choice A
                        z_bias_B = z_projection[b, 1]  # Bias for choice B
                    
                        # Compare: A (win) vs B (lose) with z conditioning
                        score_A = win_A_logit + z_bias_A
                        score_B = lose_B_logit + z_bias_B
                    
                        # Choose based on scores
                        if score_A > score_B:
                            choice = 0  # A is better
                        else:
                            choice = 1  # B is better
                    
                        batch_choices.append(choice)
                
                    batch_predictions.extend(batch_choices)
                
                all_predictions.append(batch_predictions)
            
            # Average predictions across multiple z samples
            if num_samples > 1:
                all_predictions = torch.tensor(all_predictions)  # (num_samples, batch)
                predicted_indices = all_predictions.mode(dim=0)[0].tolist()  # Majority vote
                # Average z samples across multiple samples
                all_z_samples = np.array(all_z_samples)  # (num_samples, batch, latent_dim)
                z_mean = all_z_samples.mean(axis=0)  # (batch, latent_dim) - average z across samples
            else:
                predicted_indices = all_predictions[0]
                z_mean = all_z_samples[0]  # (batch, latent_dim)
    
    # Update data with choices and z values
    for idx, (choice, sample) in enumerate(zip(predicted_indices, list_data_dict)):
        sample["choice"] = choice
        sample.pop("fake_choice", None)
        
        # Add chosen and rejected based on choice
        if choice == 0:
            # Chose A
            sample["chosen"] = sample.get("output_A", "")
            sample["rejected"] = sample.get("output_B", "")
        else:
            # Chose B
            sample["chosen"] = sample.get("output_B", "")
            sample["rejected"] = sample.get("output_A", "")
        
        # Store z distribution parameters and sampled z (if inferred)
        if mu_cpu is not None and logvar_cpu is not None:
            sample["z_mu"] = mu_cpu[idx].tolist()  # Posterior mean
            sample["z_logvar"] = logvar_cpu[idx].tolist()  # Posterior log variance
            sample["z"] = z_mean[idx].tolist()  # Sampled z (or averaged if num_samples > 1)
        elif "z" not in sample and len(all_z_samples) > 0:
            # If z was provided and not inferred, store the provided z
            if isinstance(z_mean, np.ndarray):
                sample["z"] = z_mean[idx].tolist()
            elif isinstance(z_mean, torch.Tensor):
                sample["z"] = z_mean[idx].cpu().numpy().tolist()
            else:
                logger.warning(f"Sample {idx} has no z value (neither provided nor inferred)")
    
    logger.info(f"Variational selection completed. Choices: {sum(predicted_indices)}/{len(predicted_indices)} chose B")
    logger.info(f"Stored z values (mu, logvar, z) for {len(list_data_dict)} samples")
    
    return list_data_dict
