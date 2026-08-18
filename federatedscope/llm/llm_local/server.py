import logging
import torch
import random
import math
import numpy as np
from collections import defaultdict
from federatedscope.core.message import Message

from federatedscope.core.workers.server import Server
from federatedscope.core.auxiliaries.utils import merge_param_dict

logger = logging.getLogger(__name__)


class LLMMultiLoRAServer(Server):
    """
    Server implementation
    We broadcast the model to each client and ask them to train locally
    Afterward, we collect the model back and save it as checkpoints
    """
    def __init__(self,
                 ID=-1,
                 state=0,
                 config=None,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 **kwargs):
        super(LLMMultiLoRAServer,
              self).__init__(ID, state, config, data, model, client_num,
                             total_round_num, device, strategy, **kwargs)
        if self._cfg.llm.adapter.count > 1:
            self.aggregator.total_train_size = len(data.train_data)

            self.aggregator.num_clients = client_num

        if self._cfg.llm.adapter.local_only:
            logger.warning("In local training mode, we will use all clients. "
                           "And we set the total round to 0 for one training "
                           "round only. ")

            self.sampler = None
            self.sample_client_num = client_num

        if self._cfg.llm.adapter.grouping.use:
            self.msg_buffer['adapter_eval'] = dict()
        
        # VPL-GP related attributes
        self.vpl_gp_prior_mus = None
        self.vpl_gp_prior_logvars = None
        self.vpl_gp_prior_weights = None
        self.vpl_orthogonal_client_labels = None
        self.client_z_values_dict = defaultdict(list)  # {client_id: [z_values]}
        self.client_orthogonal_labels_dict = {}  # {client_id: orthogonal_label} - stores label for each client
        self.client_orthogonal_prototypes_dict = {}  # {client_id: prototypes}
        self.client_average_z_dict = {}  # {client_id: average_z_tensor} - computed from z_values_dict for checkpoint saving

        # Initialize canonical prototypes on server (shared across all
        # clients).  Using a fixed seed ensures every client receives
        # identical prototypes, preventing FedAvg from corrupting them.
        self._canonical_prototypes = None
        if (hasattr(self._cfg.llm, 'vpl_orthogonal_weight')
                and self._cfg.llm.vpl_orthogonal_weight > 0):
            num_proto = getattr(self._cfg.llm, 'vpl_num_prototypes', 2)
            latent_dim = getattr(self._cfg.llm, 'vpl_latent_dim', 32)
            proto_scale = getattr(
                self._cfg.llm, 'vpl_prototype_scale', 5.0)
            gen = torch.Generator().manual_seed(42)
            rand_mat = torch.randn(
                num_proto, latent_dim, generator=gen)
            q, _ = torch.linalg.qr(rand_mat.T)
            self._canonical_prototypes = (
                q.T[:num_proto] * proto_scale)
            logger.info(
                f"Server initialized canonical prototypes: "
                f"shape={self._canonical_prototypes.shape}, "
                f"scale={proto_scale}")

    def _register_default_handlers(self):
        super()._register_default_handlers()
        self.register_handlers('grouping', self.callback_funcs_for_grouping,
                               ['set_active_adapter_idx'])

    def _start_new_training_round(self, aggregated_num=0, skip_grouping=False):
        if self._cfg.llm.adapter.grouping.use and not skip_grouping:
            total_warmup_round = 0
            if self._cfg.llm.adapter.warmup.use:
                warmup_round = self._cfg.llm.adapter.warmup.round
                total_warmup_round = \
                    warmup_round * self._cfg.llm.adapter.count

            r = self._cfg.llm.adapter.grouping.round
            if self.state >= total_warmup_round and \
                    (self.state - total_warmup_round) % r == 0:
                logger.info('Server: Performing a grouping step...')
                self.broadcast_model_para(msg_type='adapter_eval',
                                          filter_unseen_clients=False)
                return

        super()._start_new_training_round(aggregated_num)

    def _perform_federated_aggregation(self):
        """
        Perform federated aggregation and update the global model
        """
        train_msg_buffer = self.msg_buffer['train'][self.state]
        for model_idx in range(self.model_num):
            model = self.models[model_idx]
            aggregator = self.aggregators[model_idx]
            msg_list = list()
            # merged_adapter = dict()

            # for client_id in train_msg_buffer.keys():
            #     if self.model_num == 1:
            #         _, model_param = train_msg_buffer[client_id]
            #         # merged_adapter.update(model_param)
            #         for key, value in model_param.items():
            #             if key not in merged_adapter:
            #                 merged_adapter[key] = [value]
            #             else:
            #                 merged_adapter[key].append(value)
            #     else:
            #         train_data_size, model_para_multiple = \
            #             train_msg_buffer[client_id]
            #         # merged_adapter.update(model_para_multiple[model_idx])
            #         for key, value in model_para_multiple[model_idx].items():
            #             if key not in merged_adapter:
            #                 merged_adapter[key] = [value]
            #             else:
            #                 merged_adapter[key].append(value)

            # # calculate the mean
            # for key in merged_adapter.keys():
            #     # logger.info(f'{key}: {len(merged_adapter[key])}')
            #     avg_tensor = torch.zeros_like(merged_adapter[key][0])
            #     for value in merged_adapter[key]:
            #         avg_tensor += (value / len(merged_adapter[key]))
            #     merged_adapter[key] = avg_tensor

            # msg_list = [(1, merged_adapter)]

            # Collect VPL components separately for aggregation
            vpl_components_dict = {}  # {component_name: {param_name: [values from clients]}}
            
            for client_id in train_msg_buffer.keys():
                if self.model_num == 1:
                    sample_size, model_para = train_msg_buffer[client_id]
                    # Extract VPL components (variational_encoder,
                    # feature_extractor, latent_projection,
                    # z_to_embedding, orthogonal_prototypes)
                    # for FedAvg aggregation.
                    vpl_comp_prefixes = [
                        'variational_encoder.',
                        'feature_extractor.',
                        'latent_projection.',
                        'z_to_embedding.',
                        'orthogonal_prototypes.',
                    ]
                    vpl_component_keys = []
                    for key in model_para.keys():
                        if any(key.startswith(p)
                               for p in vpl_comp_prefixes):
                            vpl_component_keys.append(key)
                    
                    # Collect VPL components
                    for key in vpl_component_keys:
                        if key not in vpl_components_dict:
                            vpl_components_dict[key] = []
                        vpl_components_dict[key].append((sample_size, model_para[key]))
                    
                    # Remove VPL-related keys that are not model parameters
                    # These should be handled separately, not by the aggregator
                    vpl_keys_to_remove = ['client_z_values', 'client_z_mu', 'client_z_logvar', 
                                          'client_orthogonal_prototypes', 'sample_size'] + vpl_component_keys
                    model_para_clean = {k: v for k, v in model_para.items() 
                                       if k not in vpl_keys_to_remove}
                    msg_list.append((sample_size, model_para_clean))
                else:
                    train_data_size, model_para_multiple = \
                        train_msg_buffer[client_id]
                    # Remove VPL-related keys from model_para_multiple[model_idx]
                    vpl_keys_to_remove = ['client_z_values', 'client_z_mu', 'client_z_logvar', 
                                          'client_orthogonal_prototypes']
                    model_para_clean = {k: v for k, v in model_para_multiple[model_idx].items() 
                                      if k not in vpl_keys_to_remove}
                    msg_list.append((train_data_size, model_para_clean))

            for staled_message in self.staled_msg_buffer:
                state, client_id, content = staled_message
                if self.model_num == 1:
                    sample_size, model_para = content
                    # Remove VPL-related keys that are not model parameters
                    vpl_keys_to_remove = ['client_z_values', 'client_z_mu', 'client_z_logvar', 
                                          'client_orthogonal_prototypes']
                    model_para_clean = {k: v for k, v in model_para.items() 
                                       if k not in vpl_keys_to_remove}
                    msg_list.append((sample_size, model_para_clean))
                else:
                    train_data_size, model_para_multiple = content
                    # Remove VPL-related keys from model_para_multiple[model_idx]
                    vpl_keys_to_remove = ['client_z_values', 'client_z_mu', 'client_z_logvar', 
                                          'client_orthogonal_prototypes']
                    model_para_clean = {k: v for k, v in model_para_multiple[model_idx].items() 
                                      if k not in vpl_keys_to_remove}
                    msg_list.append((train_data_size, model_para_clean))

            # Trigger the monitor here (for training)
            self._monitor.calc_model_metric(self.models[0].state_dict(),
                                            msg_list,
                                            rnd=self.state)

            # Aggregate
            aggregated_num = len(msg_list)
            agg_info = {
                'client_feedback': msg_list,
                'recover_fun': self.recover_fun,
            }
            # logger.info(f'The staleness is {staleness}')

            warmup_round = self._cfg.llm.adapter.warmup.round
            total_warmup_round = \
                warmup_round * self._cfg.llm.adapter.count
            if self._cfg.llm.adapter.warmup.use and \
                    self.state < total_warmup_round:
                result = aggregator.aggregate(agg_info)
            else:
                result = aggregator.aggregate_on_model(agg_info)

            # Due to lazy load, we merge two state dict
            merged_param = merge_param_dict(model.state_dict().copy(), result)
            model.load_state_dict(merged_param, strict=False)
            
            # Aggregate VPL components (weighted average by sample size)
            if len(vpl_components_dict) > 0:
                aggregated_vpl_components = {}
                for key, client_values in vpl_components_dict.items():
                    if len(client_values) == 0:
                        continue
                    # Weighted average
                    total_weight = sum(sample_size for sample_size, _ in client_values)
                    if total_weight > 0:
                        avg_value = None
                        for sample_size, value in client_values:
                            weight = sample_size / total_weight
                            if isinstance(value, torch.Tensor):
                                weighted_value = value * weight
                                if avg_value is None:
                                    avg_value = weighted_value.clone()
                                else:
                                    avg_value = avg_value + weighted_value
                            else:
                                # For non-tensor values, use first client's value
                                if avg_value is None:
                                    avg_value = value
                        if avg_value is not None:
                            aggregated_vpl_components[key] = avg_value
                
                # Store aggregated VPL components in aggregator for checkpoint saving
                if not hasattr(aggregator, 'vpl_components'):
                    aggregator.vpl_components = {}
                aggregator.vpl_components.update(aggregated_vpl_components)
                logger.info(f"Aggregated {len(aggregated_vpl_components)} VPL component parameters")
        
        # VPL-GP: Collect z distributions from clients (only if GP prior is enabled)
        if hasattr(self._cfg.llm, 'vpl_use_gp_prior') and self._cfg.llm.vpl_use_gp_prior:
            self._collect_vpl_gp_prior_distributions()
            # Compute orthogonal labels only if orthogonal loss is enabled
            if (hasattr(self._cfg.llm, 'vpl_orthogonal_weight') and 
                self._cfg.llm.vpl_orthogonal_weight > 0):
                self._compute_balanced_orthogonal_labels()
        
        # Always collect z values from this round so the persistent
        # `client_z_values_dict` and `client_average_z_dict` are up to
        # date for every client (needed for the Stage-2 RL
        # checkpoint).  The t-SNE plot is rendered separately inside
        # `_collect_z_values_for_visualization`, gated by
        # `vpl_tsne_visualize_freq`.
        if hasattr(self._cfg.llm, 'vpl_latent_dim'):  # VPL is enabled
            self._collect_z_values_for_visualization()
        
        return aggregated_num
    
    def merge_eval_results_from_all_clients(self):
        """
        Override to add VPL-specific wandb logging for aggregated results.
        Also handles non-VPL FedBiscuit logging to avoid logline_2_wandb_dict removing Results_raw.
        """
        # Call parent method to get aggregated results
        formatted_logs_all_set = super().merge_eval_results_from_all_clients()
        
        # Log metrics to wandb if enabled (for both VPL and non-VPL)
        # This avoids the issue where logline_2_wandb_dict removes Results_raw for Server role
        if (self._cfg.wandb.use and self._cfg.wandb.online_track):
            is_vpl = hasattr(self._cfg.llm, 'vpl_latent_dim')  # VPL is enabled
            try:
                import wandb
                
                round = max(self.msg_buffer['eval'].keys())
                eval_msg_buffer = self.msg_buffer['eval'][round]
                
                # Check if this is HRL dataset for reward model metrics
                dataset_type = getattr(self._cfg.data, 'type', '').lower()
                is_hrl = 'hh-rlhf' in dataset_type or 'hrl' in dataset_type
                
                if is_vpl:
                    # VPL-specific metrics
                    # Collect VPL metrics from all clients
                    vpl_metrics_all_clients = {
                        'vpl_total_loss': [],
                        'vpl_reconstruction_loss': [],
                        'vpl_kl_loss': [],
                        'vpl_orthogonal_loss': []
                    }
                    client_ids = []
                    
                    for client_id in eval_msg_buffer:
                        if eval_msg_buffer[client_id] is None:
                            continue
                        if client_id in self.unseen_clients_id:
                            continue  # Skip unseen clients for aggregated metrics
                        
                        client_results = eval_msg_buffer[client_id]
                        client_ids.append(client_id)
                        
                        if 'loss' in client_results:
                            vpl_metrics_all_clients['vpl_total_loss'].append(float(client_results['loss']))
                        if 'vpl_reconstruction_loss' in client_results:
                            vpl_metrics_all_clients['vpl_reconstruction_loss'].append(float(client_results['vpl_reconstruction_loss']))
                        if 'vpl_kl_loss' in client_results:
                            vpl_metrics_all_clients['vpl_kl_loss'].append(float(client_results['vpl_kl_loss']))
                        if 'vpl_orthogonal_loss' in client_results:
                            vpl_metrics_all_clients['vpl_orthogonal_loss'].append(float(client_results['vpl_orthogonal_loss']))
                    
                    # Collect reward model metrics for HRL
                    reward_metrics_all_clients = {
                        'avg_harmlessness': [],
                        'avg_helpfulness': [],
                        'helpfulness_winrate': [],
                        'harmlessness_winrate': []
                    }
                    if is_hrl:
                        for client_id in eval_msg_buffer:
                            if eval_msg_buffer[client_id] is None:
                                continue
                            if client_id in self.unseen_clients_id:
                                continue
                            
                            client_results = eval_msg_buffer[client_id]
                            if 'avg_harmlessness' in client_results:
                                reward_metrics_all_clients['avg_harmlessness'].append(float(client_results['avg_harmlessness']))
                            if 'avg_helpfulness' in client_results:
                                reward_metrics_all_clients['avg_helpfulness'].append(float(client_results['avg_helpfulness']))
                            if 'helpfulness_winrate' in client_results:
                                reward_metrics_all_clients['helpfulness_winrate'].append(float(client_results['helpfulness_winrate']))
                            if 'harmlessness_winrate' in client_results:
                                reward_metrics_all_clients['harmlessness_winrate'].append(float(client_results['harmlessness_winrate']))
                    
                    # Log aggregated metrics (averaged over all clients)
                    wandb_metrics = {}
                    if vpl_metrics_all_clients['vpl_total_loss']:
                        wandb_metrics['server/train/vpl_total_loss_avg'] = np.mean(vpl_metrics_all_clients['vpl_total_loss'])
                    if vpl_metrics_all_clients['vpl_reconstruction_loss']:
                        wandb_metrics['server/train/vpl_reconstruction_loss_avg'] = np.mean(vpl_metrics_all_clients['vpl_reconstruction_loss'])
                    if vpl_metrics_all_clients['vpl_kl_loss']:
                        wandb_metrics['server/train/vpl_kl_loss_avg'] = np.mean(vpl_metrics_all_clients['vpl_kl_loss'])
                    if vpl_metrics_all_clients['vpl_orthogonal_loss']:
                        wandb_metrics['server/train/vpl_orthogonal_loss_avg'] = np.mean(vpl_metrics_all_clients['vpl_orthogonal_loss'])
                    
                    # Log reward model metrics for HRL (averaged)
                    if is_hrl:
                        if reward_metrics_all_clients['avg_harmlessness']:
                            wandb_metrics['server/train/avg_harmlessness_avg'] = np.mean(reward_metrics_all_clients['avg_harmlessness'])
                        if reward_metrics_all_clients['avg_helpfulness']:
                            wandb_metrics['server/train/avg_helpfulness_avg'] = np.mean(reward_metrics_all_clients['avg_helpfulness'])
                        if reward_metrics_all_clients['helpfulness_winrate']:
                            wandb_metrics['server/train/helpfulness_winrate_avg'] = np.mean(reward_metrics_all_clients['helpfulness_winrate'])
                        if reward_metrics_all_clients['harmlessness_winrate']:
                            wandb_metrics['server/train/harmlessness_winrate_avg'] = np.mean(reward_metrics_all_clients['harmlessness_winrate'])
                
                    # Log individual client metrics (for designated clients)
                    # Log first 3 clients as designated clients (or all if less than 3)
                    designated_clients = client_ids[:min(3, len(client_ids))]
                    for client_id in designated_clients:
                        if eval_msg_buffer[client_id] is None:
                            continue
                        client_results = eval_msg_buffer[client_id]
                        
                        if 'loss' in client_results:
                            wandb_metrics[f'client_{client_id}/train/vpl_total_loss'] = float(client_results['loss'])
                        if 'vpl_reconstruction_loss' in client_results:
                            wandb_metrics[f'client_{client_id}/train/vpl_reconstruction_loss'] = float(client_results['vpl_reconstruction_loss'])
                        if 'vpl_kl_loss' in client_results:
                            wandb_metrics[f'client_{client_id}/train/vpl_kl_loss'] = float(client_results['vpl_kl_loss'])
                        if 'vpl_orthogonal_loss' in client_results:
                            wandb_metrics[f'client_{client_id}/train/vpl_orthogonal_loss'] = float(client_results['vpl_orthogonal_loss'])
                        
                        # Log reward model metrics for HRL (individual clients)
                        if is_hrl:
                            if 'avg_harmlessness' in client_results:
                                wandb_metrics[f'client_{client_id}/train/avg_harmlessness'] = float(client_results['avg_harmlessness'])
                            if 'avg_helpfulness' in client_results:
                                wandb_metrics[f'client_{client_id}/train/avg_helpfulness'] = float(client_results['avg_helpfulness'])
                            if 'helpfulness_winrate' in client_results:
                                wandb_metrics[f'client_{client_id}/train/helpfulness_winrate'] = float(client_results['helpfulness_winrate'])
                            if 'harmlessness_winrate' in client_results:
                                wandb_metrics[f'client_{client_id}/train/harmlessness_winrate'] = float(client_results['harmlessness_winrate'])
                    
                    if wandb_metrics:
                        wandb.log(wandb_metrics, step=round)
                        logger.info(f"Logged VPL metrics to wandb for round {round}: {len(wandb_metrics)} metrics")
                else:
                    # Non-VPL FedBiscuit: log standard metrics (loss, acc, etc.)
                    # Collect standard metrics from all clients
                    standard_metrics_all_clients = {
                        'loss': [],
                        'avg_loss': [],
                        'acc': [],
                        'total': []
                    }
                    
                    # Collect reward model metrics for HRL
                    reward_metrics_all_clients = {
                        'avg_harmlessness': [],
                        'avg_helpfulness': [],
                        'helpfulness_winrate': [],
                        'harmlessness_winrate': []
                    }
                    
                    client_ids = []
                    for client_id in eval_msg_buffer:
                        if eval_msg_buffer[client_id] is None:
                            continue
                        if client_id in self.unseen_clients_id:
                            continue
                        
                        client_results = eval_msg_buffer[client_id]
                        client_ids.append(client_id)
                        
                        # Collect standard metrics
                        if 'loss' in client_results:
                            standard_metrics_all_clients['loss'].append(float(client_results['loss']))
                        if 'avg_loss' in client_results:
                            standard_metrics_all_clients['avg_loss'].append(float(client_results['avg_loss']))
                        if 'acc' in client_results:
                            standard_metrics_all_clients['acc'].append(float(client_results['acc']))
                        if 'total' in client_results:
                            standard_metrics_all_clients['total'].append(float(client_results['total']))
                        
                        # Collect reward model metrics for HRL
                        if is_hrl:
                            if 'avg_harmlessness' in client_results:
                                reward_metrics_all_clients['avg_harmlessness'].append(float(client_results['avg_harmlessness']))
                            if 'avg_helpfulness' in client_results:
                                reward_metrics_all_clients['avg_helpfulness'].append(float(client_results['avg_helpfulness']))
                            if 'helpfulness_winrate' in client_results:
                                reward_metrics_all_clients['helpfulness_winrate'].append(float(client_results['helpfulness_winrate']))
                            if 'harmlessness_winrate' in client_results:
                                reward_metrics_all_clients['harmlessness_winrate'].append(float(client_results['harmlessness_winrate']))
                    
                    # Log aggregated metrics (averaged over all clients)
                    wandb_metrics = {}
                    if standard_metrics_all_clients['loss']:
                        wandb_metrics['server/train/loss_avg'] = np.mean(standard_metrics_all_clients['loss'])
                    if standard_metrics_all_clients['avg_loss']:
                        wandb_metrics['server/train/avg_loss_avg'] = np.mean(standard_metrics_all_clients['avg_loss'])
                    if standard_metrics_all_clients['acc']:
                        wandb_metrics['server/train/acc_avg'] = np.mean(standard_metrics_all_clients['acc'])
                    if standard_metrics_all_clients['total']:
                        wandb_metrics['server/train/total_avg'] = np.mean(standard_metrics_all_clients['total'])
                    
                    # Log reward model metrics for HRL (averaged)
                    if is_hrl:
                        if reward_metrics_all_clients['avg_harmlessness']:
                            wandb_metrics['server/train/avg_harmlessness_avg'] = np.mean(reward_metrics_all_clients['avg_harmlessness'])
                        if reward_metrics_all_clients['avg_helpfulness']:
                            wandb_metrics['server/train/avg_helpfulness_avg'] = np.mean(reward_metrics_all_clients['avg_helpfulness'])
                        if reward_metrics_all_clients['helpfulness_winrate']:
                            wandb_metrics['server/train/helpfulness_winrate_avg'] = np.mean(reward_metrics_all_clients['helpfulness_winrate'])
                        if reward_metrics_all_clients['harmlessness_winrate']:
                            wandb_metrics['server/train/harmlessness_winrate_avg'] = np.mean(reward_metrics_all_clients['harmlessness_winrate'])
                    
                    # Log individual client metrics (for designated clients)
                    # Log first 3 clients as designated clients (or all if less than 3)
                    designated_clients = client_ids[:min(3, len(client_ids))]
                    for client_id in designated_clients:
                        if eval_msg_buffer[client_id] is None:
                            continue
                        client_results = eval_msg_buffer[client_id]
                        
                        if 'loss' in client_results:
                            wandb_metrics[f'client_{client_id}/train/loss'] = float(client_results['loss'])
                        if 'avg_loss' in client_results:
                            wandb_metrics[f'client_{client_id}/train/avg_loss'] = float(client_results['avg_loss'])
                        if 'acc' in client_results:
                            wandb_metrics[f'client_{client_id}/train/acc'] = float(client_results['acc'])
                        
                        # Log reward model metrics for HRL (individual clients)
                        if is_hrl:
                            if 'avg_harmlessness' in client_results:
                                wandb_metrics[f'client_{client_id}/train/avg_harmlessness'] = float(client_results['avg_harmlessness'])
                            if 'avg_helpfulness' in client_results:
                                wandb_metrics[f'client_{client_id}/train/avg_helpfulness'] = float(client_results['avg_helpfulness'])
                            if 'helpfulness_winrate' in client_results:
                                wandb_metrics[f'client_{client_id}/train/helpfulness_winrate'] = float(client_results['helpfulness_winrate'])
                            if 'harmlessness_winrate' in client_results:
                                wandb_metrics[f'client_{client_id}/train/harmlessness_winrate'] = float(client_results['harmlessness_winrate'])
                    
                    if wandb_metrics:
                        wandb.log(wandb_metrics, step=round)
                        logger.info(f"Logged FedBiscuit metrics to wandb for round {round}: {len(wandb_metrics)} metrics")
                        
            except ImportError:
                logger.warning("wandb not installed, skipping metrics logging")
            except Exception as e:
                logger.warning(f"Failed to log metrics to wandb: {e}")

        return formatted_logs_all_set

    def trigger_for_start(self):
        # start feature engineering (This part is for hard code)
        if self.check_client_join_in():
            logger.info('Waited all clients join, start now...')

            # Broadcast canonical prototypes to ALL clients so they
            # start with identical orthogonal prototypes.
            if self._canonical_prototypes is not None:
                all_clients = list(
                    self.comm_manager.neighbors.keys())
                for receiver in all_clients:
                    self.comm_manager.send(
                        Message(
                            msg_type='vpl_orthogonal_labels',
                            sender=self.ID,
                            receiver=[receiver],
                            state=self.state,
                            timestamp=self.cur_timestamp,
                            content={
                                'labels': None,
                                'prototypes':
                                    self._canonical_prototypes.cpu(),
                            }))
                logger.info(
                    f"Broadcast canonical prototypes to "
                    f"{len(all_clients)} clients")

            # Only send adapter_eval message if grouping is enabled
            if self._cfg.llm.adapter.grouping.use:
                self.trigger_for_feat_engr(self.broadcast_model_para, {
                    'msg_type': 'adapter_eval',
                    'filter_unseen_clients': False,
                })
                logger.info('Server: Performing a grouping step...')
            else:
                # If grouping is not enabled, start training round directly
                logger.info(
                    '----------- Starting training (Round #{:d}) -------------'.
                    format(self.state))
                self._start_new_training_round()

    def callback_funcs_for_grouping(self, message: Message):
        rnd = message.state
        sender = message.sender
        content = message.content

        # Only process adapter_eval if grouping is enabled
        if not self._cfg.llm.adapter.grouping.use:
            return False
        
        # Initialize adapter_eval if not already initialized
        if 'adapter_eval' not in self.msg_buffer:
            self.msg_buffer['adapter_eval'] = dict()
        
        if rnd not in self.msg_buffer['adapter_eval'].keys():
            self.msg_buffer['adapter_eval'][rnd] = dict()

        self.msg_buffer['adapter_eval'][rnd][sender] = \
            [(i, content[f'adapter_{i}_avg_loss'])
             for i in range(self._cfg.llm.adapter.count)]
        self.msg_buffer['adapter_eval'][rnd][sender] = \
            sorted(self.msg_buffer['adapter_eval'][rnd][sender],
                   key=lambda x: x[1])

        return self.check_and_grouping()

    def check_and_grouping(self):
        if 'adapter_eval' not in self.msg_buffer.keys() or \
                len(self.msg_buffer['adapter_eval'].keys()) == 0:
            return False

        buffer = self.msg_buffer['adapter_eval']
        cur_round = max(buffer.keys())
        cur_buffer = buffer[cur_round]
        if len(cur_buffer) < self.client_num:
            return False

        # convert the list to the iterator
        for sender in cur_buffer.keys():
            cur_buffer[sender] = iter(cur_buffer[sender])

        num_adap = self._cfg.llm.adapter.count
        self.adapter_grouping = dict()
        adapter_grouping = {i: [] for i in range(num_adap)}
        senders = [sender for sender in cur_buffer.keys()]
        random.shuffle(senders)
        unassigned_client_num = len(senders)
        while unassigned_client_num > 0:
            num_finished = len(self.adapter_grouping)
            max_size = math.ceil(unassigned_client_num /
                                 (num_adap - num_finished))

            # step 1: Assign to the adapter where the clients
            # well performs
            for sender in senders:
                adap_idx, loss = next(cur_buffer[sender])
                while adap_idx not in adapter_grouping:
                    adap_idx, loss = next(cur_buffer[sender])
                adapter_grouping[adap_idx].append(sender)

            # step 2: Find the adapter with the most clients
            max_adap_idx_size = [0, 0]
            for adap_idx, candidates in adapter_grouping.items():
                if len(candidates) > max_adap_idx_size[1]:
                    max_adap_idx_size = [adap_idx, len(candidates)]

            # step 3: If the number of candidates is greater than
            # max_size, preserve the first max_size
            adap_idx = max_adap_idx_size[0]
            candidates = adapter_grouping[adap_idx][:max_size]

            # step 4: update the senders list, remove the selected
            # adapter from adapter_grouping
            senders = adapter_grouping[adap_idx][max_size:]
            self.adapter_grouping[adap_idx] = candidates
            adapter_grouping.pop(adap_idx)
            unassigned_client_num -= len(self.adapter_grouping[adap_idx])
            logger.info(f'Adapter {adap_idx} is done with the clients '
                        f'{self.adapter_grouping[adap_idx]}')

        # broadcast the new grouping info to all clients
        for adap_idx, receiver in self.adapter_grouping.items():
            self.comm_manager.send(
                Message(msg_type='set_active_adapter_idx',
                        sender=self.ID,
                        receiver=receiver,
                        state=self.state,
                        timestamp=self.cur_timestamp,
                        content=adap_idx))

        # resume the training based on the new group...
        self._start_new_training_round(skip_grouping=True)

        return True  # move_on_flag
    
    @staticmethod
    def _get_dataset_num_categories(cfg):
        """Return the number of natural categories for a dataset."""
        dataset_type = getattr(cfg.data, 'type', '').lower()
        if 'ultrafeedback' in dataset_type:
            return 4  # helpfulness, honesty, instruction_following, truthfulness
        # hh-rlhf or default
        return 2  # harmless, helpful

    @staticmethod
    def _build_client_category_map(client_num, num_categories,
                                   clients_per_cat_override=None):
        """Build client_id -> category_label map matching MetaSplitter.

        Args:
            client_num: Total number of clients.
            num_categories: Number of preference categories.
            clients_per_cat_override: Optional list of per-category
                client counts (e.g. [3, 7]).  Must sum to
                ``client_num``.  When ``None``, falls back to the
                equal-split default.
        """
        if (clients_per_cat_override is not None
                and len(clients_per_cat_override) == num_categories
                and sum(clients_per_cat_override) == client_num):
            cat_counts = list(clients_per_cat_override)
        else:
            base = client_num // num_categories
            rem = client_num % num_categories
            cat_counts = [
                base + (1 if i < rem else 0)
                for i in range(num_categories)]

        labels = {}
        cid = 1
        for cat, n in enumerate(cat_counts):
            for _ in range(n):
                labels[cid] = cat
                cid += 1
        return labels

    def _compute_manual_orthogonal_labels(self, train_msg_buffer):
        """
        Assign manual orthogonal labels based on dataset category
        structure.  Works for any dataset by matching the MetaSplitter
        client-to-category assignment.
        """
        total_client_num = self._cfg.federate.client_num
        num_cats = self._get_dataset_num_categories(self._cfg)
        ratio = getattr(self._cfg.data, 'meta_split_clients_per_cat',
                        None)
        labels = self._build_client_category_map(
            total_client_num, num_cats, ratio)

        # Log label distribution
        from collections import Counter
        dist = Counter(labels.values())
        logger.info(
            f"Manual orthogonal labels ({num_cats} categories, "
            f"{total_client_num} clients):")
        for cat in sorted(dist):
            cids = [k for k, v in labels.items() if v == cat]
            logger.info(f"  Label {cat}: {len(cids)} clients {cids}")
        return labels
    
    def _collect_vpl_gp_prior_distributions(self):
        """
        Collect z distributions (mu, logvar) from clients for VPL-GP mixture prior.
        Called after _perform_federated_aggregation.
        """
        train_msg_buffer = self.msg_buffer['train'][self.state]
        
        client_mus = []
        client_logvars = []
        client_weights = []
        client_ids = []
        
        for client_id in train_msg_buffer.keys():
            if self.model_num == 1:
                _, model_para = train_msg_buffer[client_id]
            else:
                _, model_para_multiple = train_msg_buffer[client_id]
                model_para = model_para_multiple[0]  # Use first model
            
            # Extract z distribution from model parameters
            if 'client_z_mu' in model_para and 'client_z_logvar' in model_para:
                mu = model_para['client_z_mu']
                logvar = model_para['client_z_logvar']
                
                # Convert to tensors if needed
                if not isinstance(mu, torch.Tensor):
                    mu = torch.tensor(mu, dtype=torch.float32)
                if not isinstance(logvar, torch.Tensor):
                    logvar = torch.tensor(logvar, dtype=torch.float32)
                
                client_mus.append(mu)
                client_logvars.append(logvar)
                client_ids.append(client_id)
                
                # Use sample size as weight
                sample_size = model_para.get('sample_size', 1)
                client_weights.append(sample_size)
        
        if len(client_mus) == 0:
            return
        
        # Normalize weights
        total_weight = sum(client_weights)
        if total_weight > 0:
            client_weights = [w / total_weight for w in client_weights]
        
        # Stack tensors
        client_mus = torch.stack(client_mus)  # (num_clients, latent_dim)
        client_logvars = torch.stack(client_logvars)  # (num_clients, latent_dim)
        client_weights = torch.tensor(client_weights, dtype=torch.float32)
        
        # Update prior (accumulate across rounds)
        if self.vpl_gp_prior_mus is None:
            # First round: initialize
            self.vpl_gp_prior_mus = client_mus
            self.vpl_gp_prior_logvars = client_logvars
            self.vpl_gp_prior_weights = client_weights
            self._vpl_gp_client_ids = client_ids  # Initialize client IDs
            updated_count = len(client_mus)
            from_previous = 0
        else:
            # Update existing clients and add new ones
            existing_client_ids = set(getattr(self, '_vpl_gp_client_ids', []))
            current_client_ids = set(client_ids)
            
            # Update existing
            updated_indices = []
            new_mus = []
            new_logvars = []
            new_weights = []
            new_ids = []
            
            for idx, cid in enumerate(client_ids):
                if cid in existing_client_ids:
                    # Update existing
                    old_idx = self._vpl_gp_client_ids.index(cid)
                    self.vpl_gp_prior_mus[old_idx] = client_mus[idx]
                    self.vpl_gp_prior_logvars[old_idx] = client_logvars[idx]
                    self.vpl_gp_prior_weights[old_idx] = client_weights[idx]
                    updated_indices.append(old_idx)
                else:
                    # New client
                    new_mus.append(client_mus[idx])
                    new_logvars.append(client_logvars[idx])
                    new_weights.append(client_weights[idx])
                    new_ids.append(cid)
            
            # Append new clients
            if len(new_mus) > 0:
                new_mus = torch.stack(new_mus)
                new_logvars = torch.stack(new_logvars)
                new_weights = torch.stack(new_weights)
                
                self.vpl_gp_prior_mus = torch.cat([self.vpl_gp_prior_mus, new_mus], dim=0)
                self.vpl_gp_prior_logvars = torch.cat([self.vpl_gp_prior_logvars, new_logvars], dim=0)
                self.vpl_gp_prior_weights = torch.cat([self.vpl_gp_prior_weights, new_weights], dim=0)
                
                # Update client ID list
                if not hasattr(self, '_vpl_gp_client_ids'):
                    self._vpl_gp_client_ids = []
                self._vpl_gp_client_ids.extend(new_ids)
            
            # Renormalize weights
            total_weight = self.vpl_gp_prior_weights.sum()
            if total_weight > 0:
                self.vpl_gp_prior_weights = self.vpl_gp_prior_weights / total_weight
            
            updated_count = len(updated_indices) + len(new_ids)
            from_previous = len(existing_client_ids) - len(updated_indices)
        
        # Store client IDs for tracking
        if not hasattr(self, '_vpl_gp_client_ids'):
            self._vpl_gp_client_ids = client_ids
        
        logger.info(f"Collected {len(client_mus)} client z distributions for VPL-GP prior. "
                   f"Total clients in prior: {len(self.vpl_gp_prior_mus)} "
                   f"(updated: {updated_count}, from previous rounds: {from_previous})")
    
    def _compute_balanced_orthogonal_labels(self):
        """
        Compute balanced orthogonal labels using k-means on z means.
        Uses manual labels if configured, otherwise uses k-means.
        For hh-rlhf dataset, k is fixed to 2.
        """
        train_msg_buffer = self.msg_buffer['train'][self.state]
        
        # Check if manual labels are explicitly requested
        use_manual = (hasattr(self._cfg.llm, 'vpl_use_manual_orthogonal_labels') and 
                     self._cfg.llm.vpl_use_manual_orthogonal_labels)
        
        if use_manual:
            # Use manual labels if explicitly configured
            manual_labels = self._compute_manual_orthogonal_labels(train_msg_buffer)
            if manual_labels is not None:
                self.vpl_orthogonal_client_labels = manual_labels
                # Count labels for logging (label -> list of client IDs)
                label_to_clients = defaultdict(list)
                for client_id, label in manual_labels.items():
                    label_to_clients[label].append(client_id)
                
                # Log detailed assignment information
                label_counts = {label: len(clients) for label, clients in label_to_clients.items()}
                logger.info(f"Computed manual orthogonal labels for {len(manual_labels)} clients "
                          f"at round {self.state}")
                for label, count in sorted(label_counts.items()):
                    client_ids = sorted(label_to_clients[label])
                    logger.info(f"  Label {label}: {count} clients -> {client_ids}")
                return
        
        # Otherwise, use k-means on z means
        if self.vpl_gp_prior_mus is None or len(self.vpl_gp_prior_mus) == 0:
            return
        
        try:
            from sklearn.cluster import KMeans
            
            # Get z means for participating clients
            participating_client_ids = sorted(train_msg_buffer.keys())
            if len(participating_client_ids) < 2:
                return
            
            # Map client IDs to indices in prior
            client_id_to_idx = {cid: idx for idx, cid in enumerate(self._vpl_gp_client_ids)}
            z_means = []
            valid_client_ids = []
            
            for client_id in participating_client_ids:
                if client_id in client_id_to_idx:
                    idx = client_id_to_idx[client_id]
                    z_means.append(self.vpl_gp_prior_mus[idx].cpu().numpy())
                    valid_client_ids.append(client_id)
            
            if len(z_means) < 2:
                return
            
            z_means = np.array(z_means)
            
            # Determine number of clusters from dataset categories
            n_clusters = self._get_dataset_num_categories(self._cfg)
            # Allow config override
            cfg_prototypes = getattr(
                self._cfg.llm, 'vpl_num_prototypes', None)
            if cfg_prototypes is not None and cfg_prototypes > 0:
                n_clusters = cfg_prototypes
            logger.info(f"Using k={n_clusters} for k-means clustering")
            
            # Ensure n_clusters doesn't exceed number of clients
            n_clusters = min(n_clusters, len(z_means))
            
            # K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(z_means)
            
            # Create label dict
            self.vpl_orthogonal_client_labels = {
                client_id: int(label) for client_id, label in zip(valid_client_ids, labels)
            }
            
            # Count labels for logging (label -> list of client IDs)
            label_to_clients = defaultdict(list)
            for client_id, label in zip(valid_client_ids, labels):
                label_to_clients[int(label)].append(client_id)
            
            # Log detailed assignment information
            label_counts = {label: len(clients) for label, clients in label_to_clients.items()}
            logger.info(f"Computed k-means orthogonal labels (k={n_clusters}) for {len(valid_client_ids)} clients "
                      f"at round {self.state}")
            for label, count in sorted(label_counts.items()):
                client_ids = sorted(label_to_clients[label])
                logger.info(f"  Label {label}: {count} clients -> {client_ids}")
        except Exception as e:
            logger.warning(f"Failed to compute balanced orthogonal labels: {e}")
    
    def _collect_z_values_for_visualization(self):
        """
        Collect z values and orthogonal prototypes from clients for t-SNE visualization.
        Keeps the latest available z per client so that non-participating clients
        still appear in plots.
        Also stores orthogonal labels for each client (including non-participating ones).
        """
        train_msg_buffer = self.msg_buffer['train'][self.state]
        
        # Do NOT reset; keep last known z/prototypes for non-participating clients
        has_new_z = False
        z_values_list = []
        client_ids_list = []
        participating_clients = set()
        
        # Collect z values from participating clients
        for client_id in train_msg_buffer.keys():
            participating_clients.add(client_id)
            
            if self.model_num == 1:
                _, model_para = train_msg_buffer[client_id]
            else:
                _, model_para_multiple = train_msg_buffer[client_id]
                model_para = model_para_multiple[0]
            
            # Extract z values (keep only latest round)
            if 'client_z_values' in model_para:
                z_values = model_para['client_z_values']
                if isinstance(z_values, torch.Tensor):
                    z_values = z_values.detach().cpu().numpy()
                elif isinstance(z_values, list):
                    z_values = np.array(z_values)
                
                if len(z_values.shape) == 1:
                    z_values = z_values.reshape(1, -1)
                
                z_values_list.append(z_values)
                client_ids_list.extend([client_id] * len(z_values))
                has_new_z = True
                
                # Log if this is new or reused z
                if client_id in self.client_z_values_dict and len(self.client_z_values_dict[client_id]) > 0:
                    logger.debug(f"Client {client_id}: Updating z values (new round {self.state})")
                else:
                    logger.debug(f"Client {client_id}: First time storing z values")
            
            # Collect orthogonal prototypes (if orthogonal loss is enabled)
            if (hasattr(self._cfg.llm, 'vpl_orthogonal_weight') and 
                self._cfg.llm.vpl_orthogonal_weight > 0 and
                'client_orthogonal_prototypes' in model_para):
                prototypes = model_para['client_orthogonal_prototypes']
                if isinstance(prototypes, torch.Tensor):
                    prototypes = prototypes.detach().cpu().numpy()  # Fix: detach() before numpy()
                elif isinstance(prototypes, list):
                    prototypes = np.array(prototypes)
                
                # Store only latest round prototypes per client
                self.client_orthogonal_prototypes_dict[client_id] = prototypes
        
        # Store/update orthogonal labels for all clients (including non-participating)
        # This ensures all clients have labels even if they didn't participate this round
        # Always compute labels based on client data type (harmlessness/helpfulness)
        # regardless of whether orthogonal loss is enabled
        manual_labels = self._compute_manual_orthogonal_labels(train_msg_buffer)
        if manual_labels is not None:
            for client_id in range(1, self.client_num + 1):
                if client_id in manual_labels:
                    self.client_orthogonal_labels_dict[client_id] = manual_labels[client_id]
                # Also use vpl_orthogonal_client_labels if available (from k-means)
                elif self.vpl_orthogonal_client_labels is not None and client_id in self.vpl_orthogonal_client_labels:
                    self.client_orthogonal_labels_dict[client_id] = self.vpl_orthogonal_client_labels[client_id]
                if client_id not in participating_clients:
                    logger.debug(f"Client {client_id}: Storing orthogonal label {self.client_orthogonal_labels_dict.get(client_id, 'N/A')} (non-participating)")
        elif self.vpl_orthogonal_client_labels is not None:
            # Fallback to vpl_orthogonal_client_labels if manual_labels not available
            for client_id in range(1, self.client_num + 1):
                if client_id in self.vpl_orthogonal_client_labels:
                    self.client_orthogonal_labels_dict[client_id] = self.vpl_orthogonal_client_labels[client_id]
                    if client_id not in participating_clients:
                        logger.debug(f"Client {client_id}: Storing orthogonal label {self.vpl_orthogonal_client_labels[client_id]} (non-participating)")
        
        # If no new z this round but we have previously stored z, keep using them
        if len(z_values_list) == 0:
            if len(self.client_z_values_dict) == 0:
                return
        else:
            # Concatenate all new z values and update per-client latest
            all_z_values = np.concatenate(z_values_list, axis=0)
            # Store only the latest round's z values per client (replace old z, don't accumulate)
            # This ensures t-SNE visualization shows only the current round's z distribution
            
            for client_id in set(client_ids_list):
                client_z_mask = np.array(client_ids_list) == client_id
                client_z = all_z_values[client_z_mask]
                
                # Get orthogonal label for this client
                client_label = self.client_orthogonal_labels_dict.get(client_id, None)
                
                # Replace old z with new z (only keep latest round's z values)
                # This prevents accumulation across rounds and keeps t-SNE visualization clean
                self.client_z_values_dict[client_id] = client_z.tolist()
            
            # After updating all clients, balance across harmlessness and helpfulness
            # Group clients by label and ensure balanced representation
            harmless_clients = [cid for cid, label in self.client_orthogonal_labels_dict.items() if label == 0]
            helpful_clients = [cid for cid, label in self.client_orthogonal_labels_dict.items() if label == 1]
            
            # Store only latest round's z values per client (no accumulation across rounds)
        
        # Check all clients (1 to client_num) to ensure we have z for all
        all_client_ids = set(range(1, self.client_num + 1))
        missing_clients = all_client_ids - set(self.client_z_values_dict.keys())
        non_participating_with_z = set(self.client_z_values_dict.keys()) - participating_clients
        if missing_clients:
            logger.debug(f"Round {self.state}: Clients without any z values (never participated): {sorted(missing_clients)} (will not appear in t-SNE)")
        if non_participating_with_z:
            logger.debug(f"Round {self.state}: Non-participating clients using previous round z values: {sorted(non_participating_with_z)} (will appear in t-SNE)")
        
        # Visualize every 10 rounds (or every round if configured)
        visualize_freq = getattr(self._cfg.llm, 'vpl_tsne_visualize_freq', 10)  # Default: every 10 rounds
        if self.state % visualize_freq == 0:
            self._visualize_cross_client_z()
        
        total_points = sum(len(v) for v in self.client_z_values_dict.values())
        unique_clients = len(self.client_z_values_dict)
        non_participating_count = len(set(self.client_z_values_dict.keys()) - participating_clients)
        logger.info(f"Round {self.state}: Collected z values from {len(participating_clients)} participating clients "
                   f"(new_z={has_new_z}). Total stored: {total_points} points across {unique_clients} clients "
                   f"({non_participating_count} non-participating using previous z). "
                   f"Orthogonal labels stored for {len(self.client_orthogonal_labels_dict)} clients.")
    
    def _visualize_cross_client_z(self):
        """
        Visualize cross-client z values using t-SNE.
        Only plots z from clients that participated in the current
        round (not stale values from previous rounds).
        """
        try:
            from federatedscope.llm.llm_local.z_visualization import visualize_cross_client_z

            # Determine which clients participated this round
            train_msg_buffer = self.msg_buffer['train'].get(
                self.state, {})
            current_round_clients = set(train_msg_buffer.keys())

            # Prepare z values and labels — only current round
            z_values_list = []
            client_labels_list = []
            orthogonal_labels_list = []

            for client_id in sorted(current_round_clients):
                if client_id not in self.client_z_values_dict:
                    continue
                z_list = self.client_z_values_dict[client_id]
                if len(z_list) == 0:
                    continue

                client_z = np.array(z_list)
                orth_label = self.client_orthogonal_labels_dict.get(
                    client_id, -1)

                z_values_list.append(client_z)
                client_labels_list.extend(
                    [client_id] * len(client_z))
                orthogonal_labels_list.extend(
                    [orth_label] * len(client_z))

            if len(z_values_list) == 0:
                return

            all_z = np.concatenate(z_values_list, axis=0)

            # Collect client mus for the current round's
            # participating clients from the GP prior store
            client_mus_dict = {}
            if (self.vpl_gp_prior_mus is not None and
                    hasattr(self, '_vpl_gp_client_ids')):
                cid_to_idx = {
                    cid: idx for idx, cid
                    in enumerate(self._vpl_gp_client_ids)}
                for cid in current_round_clients:
                    if cid in cid_to_idx:
                        idx = cid_to_idx[cid]
                        mu = self.vpl_gp_prior_mus[idx]
                        if isinstance(mu, torch.Tensor):
                            mu = mu.cpu().numpy()
                        client_mus_dict[cid] = mu
            
            # Get orthogonal prototypes if available (only if orthogonal loss is enabled)
            orthogonal_prototypes = None
            if (hasattr(self._cfg.llm, 'vpl_orthogonal_weight') and 
                self._cfg.llm.vpl_orthogonal_weight > 0 and
                hasattr(self, 'client_orthogonal_prototypes_dict') and 
                len(self.client_orthogonal_prototypes_dict) > 0):
                # Take the first client's prototypes (they should be the same across clients after QR decomposition)
                first_client_id = next(iter(self.client_orthogonal_prototypes_dict.keys()))
                prototypes = self.client_orthogonal_prototypes_dict[first_client_id]
                if prototypes is not None:
                    if isinstance(prototypes, torch.Tensor):
                        prototypes = prototypes.cpu().numpy()
                    elif isinstance(prototypes, list):
                        prototypes = np.array(prototypes)
                    orthogonal_prototypes = prototypes
            
            visualize_cross_client_z(
                z_values=all_z,
                client_labels=client_labels_list,
                orthogonal_labels=orthogonal_labels_list if len(orthogonal_labels_list) > 0 else None,
                orthogonal_prototypes=orthogonal_prototypes,
                client_mus=client_mus_dict if len(client_mus_dict) > 0 else None,
                round_num=self.state,
                output_dir=self._cfg.outdir,
                wandb_project=self._cfg.wandb.name_project if self._cfg.wandb.use else None
            )
        except Exception as e:
            logger.warning(f"Failed to visualize cross-client z: {e}")
    
    def _compute_client_average_z_for_checkpoint(self):
        """
        Compute average z for each client from stored z_values_dict.
        Uses the most recent z values (latest round) for each client.
        This is used to save client-specific z information in checkpoint for RL training.
        Ensures ALL clients (1 to client_num) have z values saved, even if they didn't participate in the final round.
        """
        self.client_average_z_dict = {}
        
        # Get the most recent round's z values from train_msg_buffer
        train_msg_buffer = self.msg_buffer.get('train', {}).get(self.state, {})
        
        for client_id in range(1, self.client_num + 1):
            # Priority 1: Use z values from the most recent round (current round)
            if client_id in train_msg_buffer:
                if self.model_num == 1:
                    _, model_para = train_msg_buffer[client_id]
                else:
                    _, model_para_multiple = train_msg_buffer[client_id]
                    model_para = model_para_multiple[0]
                
                if 'client_z_values' in model_para:
                    z_values = model_para['client_z_values']
                    if isinstance(z_values, torch.Tensor):
                        z_values = z_values.detach().cpu().numpy()
                    elif isinstance(z_values, list):
                        z_values = np.array(z_values)
                    
                    if len(z_values.shape) == 1:
                        z_values = z_values.reshape(1, -1)
                    
                    # Compute average z from the most recent round's z values
                    avg_z = np.mean(z_values, axis=0)  # (latent_dim,)
                    avg_z_tensor = torch.tensor(avg_z, dtype=torch.float32)
                    self.client_average_z_dict[client_id] = avg_z_tensor
                    logger.debug(f"Computed average z for client {client_id} from most recent round: shape {avg_z_tensor.shape}, from {len(z_values)} z samples")
                    continue
            
            # Priority 2: Fallback to stored z_values_dict (if current round data not available)
            # This ensures we get z values for clients that didn't participate in the final round
            if client_id in self.client_z_values_dict and len(self.client_z_values_dict[client_id]) > 0:
                z_list = self.client_z_values_dict[client_id]
                z_array = np.array(z_list)  # (num_samples, latent_dim)
                
                # Compute average z from stored z values
                avg_z = np.mean(z_array, axis=0)  # (latent_dim,)
                avg_z_tensor = torch.tensor(avg_z, dtype=torch.float32)
                self.client_average_z_dict[client_id] = avg_z_tensor
                logger.debug(f"Computed average z for client {client_id} from stored z_values_dict: shape {avg_z_tensor.shape}, from {len(z_list)} z samples")
            else:
                # Priority 3: Try to get z from previous rounds' msg_buffer
                # Search backwards through rounds to find z values for this client
                found_z = False
                for round_num in sorted(self.msg_buffer.get('train', {}).keys(), reverse=True):
                    if round_num == self.state:
                        continue  # Already checked current round
                    round_buffer = self.msg_buffer.get('train', {}).get(round_num, {})
                    if client_id in round_buffer:
                        if self.model_num == 1:
                            _, model_para = round_buffer[client_id]
                        else:
                            _, model_para_multiple = round_buffer[client_id]
                            model_para = model_para_multiple[0]
                        
                        if 'client_z_values' in model_para:
                            z_values = model_para['client_z_values']
                            if isinstance(z_values, torch.Tensor):
                                z_values = z_values.detach().cpu().numpy()
                            elif isinstance(z_values, list):
                                z_values = np.array(z_values)
                            
                            if len(z_values.shape) == 1:
                                z_values = z_values.reshape(1, -1)
                            
                            avg_z = np.mean(z_values, axis=0)
                            avg_z_tensor = torch.tensor(avg_z, dtype=torch.float32)
                            self.client_average_z_dict[client_id] = avg_z_tensor
                            logger.debug(f"Computed average z for client {client_id} from round {round_num}: shape {avg_z_tensor.shape}, from {len(z_values)} z samples")
                            found_z = True
                            break
                
                if not found_z:
                    logger.warning(f"Could not find z values for client {client_id} in any round or stored z_values_dict. This client will not have z values in checkpoint.")
        
        if len(self.client_average_z_dict) > 0:
            logger.info(f"Computed average z for {len(self.client_average_z_dict)}/{self.client_num} clients for checkpoint saving (for RL training)")
            if len(self.client_average_z_dict) < self.client_num:
                missing = set(range(1, self.client_num + 1)) - set(self.client_average_z_dict.keys())
                logger.warning(f"Missing z values for clients: {sorted(missing)}")
        else:
            logger.warning("No client average z computed (no z values stored)")
    
    def check_and_save(self):
        """
        Override to save client average z information in checkpoint.
        """
        # Call parent's check_and_save logic
        from federatedscope.core.auxiliaries.utils import add_prefix_to_path
        
        # early stopping
        if "Results_weighted_avg" in self.history_results and \
                self._cfg.eval.best_res_update_round_wise_key in \
                self.history_results['Results_weighted_avg']:
            should_stop = self.early_stopper.track_and_check(
                self.history_results['Results_weighted_avg'][
                    self._cfg.eval.best_res_update_round_wise_key])
        elif "Results_avg" in self.history_results and \
                self._cfg.eval.best_res_update_round_wise_key in \
                self.history_results['Results_avg']:
            should_stop = self.early_stopper.track_and_check(
                self.history_results['Results_avg'][
                    self._cfg.eval.best_res_update_round_wise_key])
        else:
            should_stop = False

        if should_stop:
            self._monitor.global_converged()
            self.comm_manager.send(
                Message(
                    msg_type="converged",
                    sender=self.ID,
                    receiver=list(self.comm_manager.neighbors.keys()),
                    timestamp=self.cur_timestamp,
                    state=self.state,
                ))
            self.state = self.total_round_num + 1

        if self.state != self.total_round_num and \
                self.state % self._cfg.federate.save_freq == 0 and \
                self._cfg.federate.save_freq > 0:
            path = add_prefix_to_path(f'{self.state}_',
                                      self._cfg.federate.save_to)
            if self.ds_rank == 0:
                # Compute and save client average z before saving checkpoint (only for VPL)
                client_average_z_dict = None
                if hasattr(self._cfg.llm, 'vpl_latent_dim'):  # VPL is enabled
                    self._compute_client_average_z_for_checkpoint()
                    client_average_z_dict = self.client_average_z_dict
                self.aggregator.save_model(path, self.state, client_average_z_dict=client_average_z_dict)

        if should_stop or self.state == self.total_round_num:
            logger.info('Server: Final evaluation is finished! Starting '
                        'merging results.')
            # last round or early stopped
            self.save_best_results()
            if not self._cfg.federate.make_global_eval:
                self.save_client_eval_results()
            
            # Save final checkpoint with client average z (only for VPL)
            if self._cfg.federate.save_to != '':
                path = add_prefix_to_path('final_', self._cfg.federate.save_to)
                if self.ds_rank == 0:
                    # Compute and save client average z before saving final checkpoint (only for VPL)
                    client_average_z_dict = None
                    if hasattr(self._cfg.llm, 'vpl_latent_dim'):  # VPL is enabled
                        self._compute_client_average_z_for_checkpoint()
                        client_average_z_dict = self.client_average_z_dict
                    self.aggregator.save_model(path, self.state, client_average_z_dict=client_average_z_dict)
            
            self.terminate(msg_type='finish')

        # Clean the clients evaluation msg buffer
        if not self._cfg.federate.make_global_eval:
            round = max(self.msg_buffer['eval'].keys())
            self.msg_buffer['eval'][round].clear()
    
    def broadcast_model_para(self,
                             msg_type='model_para',
                             sample_client_num=-1,
                             filter_unseen_clients=True):
        """
        Override to broadcast VPL-GP prior and orthogonal labels.
        """
        # Call parent method
        super().broadcast_model_para(
            msg_type=msg_type,
            sample_client_num=sample_client_num,
            filter_unseen_clients=filter_unseen_clients
        )
        
        # Broadcast VPL-GP prior if available
        if (hasattr(self._cfg.llm, 'vpl_use_gp_prior') and 
            self._cfg.llm.vpl_use_gp_prior and 
            self.vpl_gp_prior_mus is not None and
            self.state > 0):  # Don't broadcast at round 0
            
            # Prepare prior data
            prior_content = {
                'vpl_gp_prior_mus': self.vpl_gp_prior_mus.cpu().tolist(),
                'vpl_gp_prior_logvars': self.vpl_gp_prior_logvars.cpu().tolist(),
                'vpl_gp_prior_weights': self.vpl_gp_prior_weights.cpu().tolist(),
            }
            
            # Use same logic as parent class: sample if sample_client_num > 0, else broadcast to all
            if sample_client_num > 0:
                # Check if sampler is available and has idle clients
                if self.sampler is not None:
                    idle_clients = np.nonzero(self.sampler.client_state)[0]
                    if len(idle_clients) > 0:
                        selected_clients = self.sampler.sample(size=sample_client_num)
                    else:
                        # All clients are working, use all clients instead
                        selected_clients = list(self.comm_manager.neighbors.keys())
                        logger.warning(f"No idle clients available, broadcasting to all {len(selected_clients)} clients")
                else:
                    selected_clients = list(self.comm_manager.neighbors.keys())
            else:
                # Broadcast to all clients
                selected_clients = list(self.comm_manager.neighbors.keys())
            
            for receiver in selected_clients:
                self.comm_manager.send(
                    Message(msg_type='vpl_gp_prior',
                           sender=self.ID,
                           receiver=[receiver],
                           state=self.state,
                           timestamp=self.cur_timestamp,
                           content=prior_content))
            
            logger.info(f"Broadcasting VPL-GP prior with {len(self.vpl_gp_prior_mus)} client distributions to {len(selected_clients)} clients at round {self.state}")
        
        # Broadcast orthogonal labels if available and orthogonal loss is enabled
        if (hasattr(self._cfg.llm, 'vpl_orthogonal_weight') and 
            self._cfg.llm.vpl_orthogonal_weight > 0 and
            self.vpl_orthogonal_client_labels is not None and
            self.state > 0):  # Don't broadcast at round 0
            
            # Use same logic as parent class: sample if sample_client_num > 0, else broadcast to all
            if sample_client_num > 0:
                # Check if sampler is available and has idle clients
                if self.sampler is not None:
                    idle_clients = np.nonzero(self.sampler.client_state)[0]
                    if len(idle_clients) > 0:
                        selected_clients = self.sampler.sample(size=sample_client_num)
                    else:
                        # All clients are working, use all clients instead
                        selected_clients = list(self.comm_manager.neighbors.keys())
                        logger.warning(f"No idle clients available, broadcasting to all {len(selected_clients)} clients")
                else:
                    selected_clients = list(self.comm_manager.neighbors.keys())
            else:
                # Broadcast to all clients
                selected_clients = list(self.comm_manager.neighbors.keys())
            
            # Include aggregated prototypes if available
            ortho_content = {
                'labels': self.vpl_orthogonal_client_labels,
            }
            aggregator = self.aggregators[0]
            if hasattr(aggregator, 'vpl_components'):
                proto_key = 'orthogonal_prototypes.weight'
                if proto_key in aggregator.vpl_components:
                    ortho_content['prototypes'] = \
                        aggregator.vpl_components[proto_key].cpu()

            for receiver in selected_clients:
                self.comm_manager.send(
                    Message(msg_type='vpl_orthogonal_labels',
                            sender=self.ID,
                            receiver=[receiver],
                            state=self.state,
                            timestamp=self.cur_timestamp,
                            content=ortho_content))

            logger.info(f"Broadcasting orthogonal labels to {len(selected_clients)} clients at round {self.state}")

        # Broadcast aggregated VPL components (feature_extractor,
        # variational_encoder, latent_projection, z_to_embedding) so
        # every client trains the SAME encoder.  Without this the
        # FedAvg of VPL components is computed at the server but never
        # reaches the clients — each client trains a private encoder
        # and the mixture prior built from peer μ's lives in
        # incomparable latent spaces.
        aggregator = self.aggregators[0]
        if (hasattr(self._cfg.llm, 'vpl_latent_dim')
                and hasattr(aggregator, 'vpl_components')
                and aggregator.vpl_components
                and self.state > 0):
            # Split aggregated dict by component
            vpl_prefixes = (
                'feature_extractor.',
                'variational_encoder.',
                'latent_projection.',
                'z_to_embedding.',
            )
            components_by_name = {
                p.rstrip('.'): {} for p in vpl_prefixes
            }
            for key, val in aggregator.vpl_components.items():
                for prefix in vpl_prefixes:
                    if key.startswith(prefix):
                        sub_key = key[len(prefix):]
                        comp_name = prefix.rstrip('.')
                        components_by_name[comp_name][sub_key] = \
                            val.cpu() if isinstance(val, torch.Tensor) \
                            else val
                        break
            # Drop empty components
            components_by_name = {
                k: v for k, v in components_by_name.items() if v
            }
            if components_by_name:
                # Reuse the receiver list logic from the parent
                # broadcast: send to all neighbors.
                vpl_receivers = list(
                    self.comm_manager.neighbors.keys())
                for receiver in vpl_receivers:
                    self.comm_manager.send(
                        Message(
                            msg_type='vpl_components',
                            sender=self.ID,
                            receiver=[receiver],
                            state=self.state,
                            timestamp=self.cur_timestamp,
                            content=components_by_name))
                logger.info(
                    f"Broadcasting VPL components "
                    f"({list(components_by_name.keys())}) to "
                    f"{len(vpl_receivers)} clients at round "
                    f"{self.state}")