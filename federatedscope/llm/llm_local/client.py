import copy
import logging
import torch
import random

from federatedscope.core.message import Message
from federatedscope.core.workers.client import Client
from federatedscope.core.data import ClientData

logger = logging.getLogger(__name__)


def reorg_client_data(cdata: ClientData):
    from torch.utils.data import random_split, ConcatDataset
    data_length = len(cdata.train_data)
    val_data_length = max(min(int(data_length * 0.05), 200), 1)
    train_data_length = data_length - val_data_length
    train_data, val_data = random_split(cdata.train_data,
                                        [train_data_length, val_data_length])
    test_data = ConcatDataset([cdata.val_data, cdata.test_data])
    return ClientData(cdata.client_cfg, train_data, val_data, test_data)


class LLMMultiLoRAClient(Client):
    """
    Client implementation of
    "Offsite-Tuning: Transfer Learning without Full Model" paper
    """
    def __init__(self,
                 ID=-1,
                 server_id=None,
                 state=-1,
                 config=None,
                 data=None,
                 model=None,
                 device='cpu',
                 strategy=None,
                 *args,
                 **kwargs):
        if config.llm.adapter.count > 1:
            data = reorg_client_data(data)
        super(LLMMultiLoRAClient,
              self).__init__(ID, server_id, state, config, data, model, device,
                             strategy, *args, **kwargs)

    def _register_default_handlers(self):
        super()._register_default_handlers()
        self.register_handlers('adapter_eval',
                               self.callback_funcs_for_adapter_eval,
                               ['grouping'])
        self.register_handlers('set_active_adapter_idx',
                               self.callback_funcs_for_setting_adapter_idx,
                               [None])
        # Register VPL-GP prior handler
        self.register_handlers('vpl_gp_prior',
                               self.callback_funcs_for_vpl_gp_prior,
                               [None])
        # Register VPL orthogonal labels handler
        self.register_handlers('vpl_orthogonal_labels',
                               self.callback_funcs_for_vpl_orthogonal_labels,
                               [None])
        # Register VPL components handler (aggregated encoder weights)
        self.register_handlers('vpl_components',
                               self.callback_funcs_for_vpl_components,
                               [None])

    def callback_funcs_for_model_para(self, message: Message):
        round = message.state
        sender = message.sender
        timestamp = message.timestamp
        content = message.content

        # When clients share the local model, we must set strict=True to
        # ensure all the model params (which might be updated by other
        # clients in the previous local training process) are overwritten
        # and synchronized with the received model
        if self._cfg.federate.process_num > 1:
            for k, v in content.items():
                content[k] = v.to(self.device)
        self.trainer.update(content,
                            strict=self._cfg.federate.share_local_model)
        self.state = round
        skip_train_isolated_or_global_mode = \
            self.early_stopper.early_stopped and \
            self._cfg.federate.method in ["local", "global"]
        if self.is_unseen_client or skip_train_isolated_or_global_mode:
            # for these cases (1) unseen client (2) isolated_global_mode,
            # we do not local train and upload local model
            sample_size, model_para_all, results = \
                0, self.trainer.get_model_para(), {}
            if skip_train_isolated_or_global_mode:
                logger.info(f"[Local/Global mode] Client #{self.ID} has been "
                            f"early stopped, we will skip the local training")
                self._monitor.local_converged()
        else:
            if self.early_stopper.early_stopped and \
                    self._monitor.local_convergence_round == 0:
                logger.info(
                    f"[Normal FL Mode] Client #{self.ID} has been locally "
                    f"early stopped. "
                    f"The next FL update may result in negative effect")
                self._monitor.local_converged()

            # Two mode of multilora training: Client-wise and clustering
            if self._cfg.llm.adapter.local_only:
                # This is client-wise
                self.model.set_active_adapter(f'Adapter_{self.ID}')
                adapter_idx = self.ID
            elif self._cfg.llm.adapter.count > 1:
                # This is clustering
                warmup_round = self._cfg.llm.adapter.warmup.round
                total_warmup_round = \
                    warmup_round * self._cfg.llm.adapter.count
                if self._cfg.llm.adapter.warmup.use and \
                        self.state < total_warmup_round:
                    # Initialization for all adapters
                    adapter_idx = self.state // warmup_round
                elif self._cfg.llm.adapter.grouping.use:
                    adapter_idx = self.adapter_idx
                else:
                    # select the adapter with min val loss
                    with torch.no_grad():
                        min_loss, adapter_indices = 10.0, []
                        for i in range(self._cfg.llm.adapter.count):
                            if len(self.data.val_data) == 0:
                                adapter_indices.append(i)
                                continue
                            self.model.set_active_adapter(f'Adapter_{i}')
                            self.model.eval()
                            metrics = self.trainer.evaluate(
                                target_data_split_name='val')
                            logger.info(
                                f'Adapter {i} with the results: {metrics}')
                            
                            # Check if metrics is None before accessing
                            if metrics is None:
                                logger.warning(f"Client {self.ID}: metrics is None for adapter {i}, adding to indices anyway")
                                adapter_indices.append(i)
                                continue
                            
                            if i == 0 or min_loss > metrics['val_avg_loss']:
                                min_loss, adapter_indices = metrics[
                                    'val_avg_loss'], [i]
                            elif min_loss == metrics['val_avg_loss']:
                                adapter_indices.append(i)
                        
                        # If no adapters were evaluated successfully, use all adapters
                        if len(adapter_indices) == 0:
                            logger.warning(f"Client {self.ID}: No adapters evaluated successfully, using all adapters")
                            adapter_indices = list(range(self._cfg.llm.adapter.count))
                        
                        logger.info(f"Client {self.ID}: Selected adapter indices: {adapter_indices}")
                        adapter_idx = random.choice(adapter_indices)
                # activate the selected adapter for further training
                logger.info(
                    f'Activate the adapter {adapter_idx} for training...')
                self.model.set_active_adapter(f'Adapter_{adapter_idx}')
                self.model.train()
            else:
                raise ValueError(
                    'You should set llm.adapter.local_only to True '
                    'or llm.adapter.count > 1')

            sample_size, model_para_all, results = self.trainer.train()
            train_data_size = len(self.data.train_data)
            if self._cfg.federate.share_local_model and not \
                    self._cfg.federate.online_aggr:
                model_para_all = copy.deepcopy(model_para_all)
                model_para_all = {
                    key: value
                    for key, value in model_para_all.items()
                    if f'Adapter_{adapter_idx}.' in key
                }
            
            # VPL: Always send z distribution and values — they're
            # needed for the per-client average-z checkpoint, not just
            # t-SNE visualization.  Previously this was gated by
            # `vpl_tsne_visualize_freq`, which meant non-participating
            # clients in viz rounds had no z saved, producing the
            # missing-client gaps in `client_average_z_dict`.
            if hasattr(self.trainer, 'get_client_z_values'):
                z_values = self.trainer.get_client_z_values()
                if z_values is not None:
                    model_para_all['client_z_values'] = z_values.cpu() if isinstance(z_values, torch.Tensor) else z_values
            
            # VPL-GP: Add z distribution for GP prior (only if GP prior is enabled)
            if (hasattr(self._cfg.llm, 'vpl_use_gp_prior') and 
                self._cfg.llm.vpl_use_gp_prior and
                hasattr(self.trainer, 'get_client_z_distribution')):
                z_dist = self.trainer.get_client_z_distribution()
                if z_dist is not None:
                    mu, logvar = z_dist
                    model_para_all['client_z_mu'] = mu.cpu() if isinstance(mu, torch.Tensor) else mu
                    model_para_all['client_z_logvar'] = logvar.cpu() if isinstance(logvar, torch.Tensor) else logvar
                    model_para_all['sample_size'] = sample_size
            
            # Add orthogonal prototypes (if available)
            if hasattr(self.trainer, 'get_client_orthogonal_prototypes'):
                prototypes = self.trainer.get_client_orthogonal_prototypes()
                if prototypes is not None:
                    model_para_all['client_orthogonal_prototypes'] = prototypes.cpu() if isinstance(prototypes, torch.Tensor) else prototypes
            
            # VPL: Add VPL components (variational_encoder, feature_extractor, latent_projection, z_to_embedding) to model parameters
            if hasattr(self.trainer, 'variational_encoder') and self.trainer.variational_encoder is not None:
                vpl_state_dict = {}
                # Exclude prior_logits from aggregation — each client
                # must keep its own Gumbel-Softmax weights so that
                # similar-preference clients can discover each other
                # (per "Variational Multi-Task Learning with
                # Gumbel-Softmax Priors")
                ve_state = {
                    k: v for k, v in
                    self.trainer.variational_encoder.state_dict().items()
                    if 'prior_logits' not in k
                }
                vpl_state_dict['variational_encoder'] = ve_state
                if hasattr(self.trainer, 'feature_extractor') and self.trainer.feature_extractor is not None:
                    vpl_state_dict['feature_extractor'] = self.trainer.feature_extractor.state_dict()
                if hasattr(self.trainer, 'latent_projection') and self.trainer.latent_projection is not None:
                    vpl_state_dict['latent_projection'] = self.trainer.latent_projection.state_dict()
                if hasattr(self.trainer, 'z_to_embedding') and self.trainer.z_to_embedding is not None:
                    vpl_state_dict['z_to_embedding'] = self.trainer.z_to_embedding.state_dict()
                # Add learnable orthogonal prototypes
                if hasattr(self.trainer, 'orthogonal_prototypes') \
                        and self.trainer.orthogonal_prototypes is not None \
                        and isinstance(self.trainer.orthogonal_prototypes,
                                       torch.nn.Parameter):
                    vpl_state_dict['orthogonal_prototypes'] = {
                        'weight': self.trainer.orthogonal_prototypes.data
                    }

                # Add VPL components to model_para_all with prefixes
                for component_name, component_state_dict in vpl_state_dict.items():
                    for key, value in component_state_dict.items():
                        model_para_all[f'{component_name}.{key}'] = value.cpu() if isinstance(value, torch.Tensor) else value
            
            train_log_res = self._monitor.format_eval_res(
                results,
                rnd=self.state,
                role='Client #{}'.format(self.ID),
                return_raw=True)
            logger.info(train_log_res)
            if self._cfg.wandb.use and self._cfg.wandb.client_train_info:
                self._monitor.save_formatted_results(train_log_res,
                                                     save_file_name="")

        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    timestamp=self._gen_timestamp(init_timestamp=timestamp,
                                                  instance_number=sample_size),
                    content=(sample_size, model_para_all)))

    def callback_funcs_for_adapter_eval(self, message: Message):
        sender, timestamp = message.sender, message.timestamp
        self.state = message.state
        if message.content is not None:
            self.trainer.update(message.content,
                                strict=self._cfg.federate.share_local_model)

        metrics = {}
        with torch.no_grad():
            for i in range(self._cfg.llm.adapter.count):
                if len(self.data.val_data) == 0:
                    metrics[f'adapter_{i}_avg_loss'] = random.random()
                    continue
                self.model.set_active_adapter(f'Adapter_{i}')
                self.model.eval()
                adap_metrics = self.trainer.evaluate(
                    target_data_split_name='val')
                logger.info(f'Client {self.ID} Adapter {i} with '
                            f'the results: {adap_metrics}')
                if adap_metrics is not None and 'val_avg_loss' in adap_metrics:
                    metrics[f'adapter_{i}_avg_loss'] = adap_metrics['val_avg_loss']
                else:
                    # Fallback if evaluation returns None or missing key
                    metrics[f'adapter_{i}_avg_loss'] = random.random()
                    logger.warning(f'Client {self.ID} Adapter {i} evaluation returned None, using random value')

        self.comm_manager.send(
            Message(msg_type='grouping',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    timestamp=timestamp,
                    content=metrics))

    def callback_funcs_for_setting_adapter_idx(self, message: Message):
        self.adapter_idx = message.content
    
    def callback_funcs_for_vpl_gp_prior(self, message: Message):
        """
        Handle VPL-GP prior update from server.
        """
        if hasattr(self.trainer, 'update_prior_from_server'):
            prior_mus = message.content.get('vpl_gp_prior_mus')
            prior_logvars = message.content.get('vpl_gp_prior_logvars')
            prior_weights = message.content.get('vpl_gp_prior_weights')
            
            if prior_mus is not None and prior_logvars is not None:
                # Convert to tensors if needed
                if not isinstance(prior_mus, torch.Tensor):
                    prior_mus = torch.tensor(prior_mus, dtype=torch.float32).to(self.device)
                if not isinstance(prior_logvars, torch.Tensor):
                    prior_logvars = torch.tensor(prior_logvars, dtype=torch.float32).to(self.device)
                if not isinstance(prior_weights, torch.Tensor):
                    prior_weights = torch.tensor(prior_weights, dtype=torch.float32).to(self.device)
                
                self.trainer.update_prior_from_server(
                    prior_mus, prior_logvars, prior_weights,
                    current_round=message.state
                )
                logger.info(f"Client {self.ID} updated VPL-GP prior from server with "
                          f"{len(prior_mus)} client distributions at round {message.state}")
    
    def callback_funcs_for_vpl_orthogonal_labels(self, message: Message):
        """
        Handle orthogonal labels and aggregated prototypes from server.
        Content format: dict with 'labels' and optional 'prototypes'.
        """
        content = message.content
        if not isinstance(content, dict):
            return

        # Update orthogonal label
        labels = content.get('labels')
        if labels is not None and hasattr(
                self.trainer, 'update_orthogonal_label_from_server'):
            if isinstance(labels, dict) and self.ID in labels:
                self.trainer.update_orthogonal_label_from_server(
                    labels[self.ID])
                logger.info(
                    f"Client {self.ID} updated orthogonal "
                    f"label from server: {labels[self.ID]}")

        # Update aggregated prototypes
        prototypes = content.get('prototypes')
        if prototypes is not None and hasattr(
                self.trainer, 'orthogonal_prototypes'):
            if not isinstance(prototypes, torch.Tensor):
                prototypes = torch.tensor(
                    prototypes, dtype=torch.float32)
            prototypes = prototypes.to(self.device)
            if isinstance(self.trainer.orthogonal_prototypes,
                          torch.nn.Parameter):
                self.trainer.orthogonal_prototypes.data.copy_(
                    prototypes)
                logger.info(
                    f"Client {self.ID} updated orthogonal "
                    f"prototypes from server "
                    f"(shape={prototypes.shape})")

    def callback_funcs_for_vpl_components(self, message: Message):
        """
        Handle aggregated VPL components from server.

        Content is a dict ``{component_name: state_dict}`` for any of
        ``feature_extractor``, ``variational_encoder``,
        ``latent_projection``, ``z_to_embedding``.  Loads each into
        the matching trainer module so every client trains the same
        encoder — required for the Federated Mixture Prior to be
        meaningful (peer μ's must live in the same latent space).
        """
        content = message.content
        if not isinstance(content, dict):
            return

        loaded = []
        for comp_name, state_dict in content.items():
            comp = getattr(self.trainer, comp_name, None)
            if comp is None or not hasattr(comp, 'load_state_dict'):
                continue
            # Move tensors to the client's device and match dtype of
            # the existing parameters to avoid dtype drift.
            target_state = {}
            for k, v in state_dict.items():
                if not isinstance(v, torch.Tensor):
                    target_state[k] = v
                    continue
                v = v.to(self.device)
                # Find the matching parameter to copy dtype from.
                ref = dict(comp.state_dict()).get(k)
                if ref is not None and ref.dtype != v.dtype:
                    v = v.to(dtype=ref.dtype)
                target_state[k] = v
            try:
                comp.load_state_dict(target_state, strict=False)
                loaded.append(comp_name)
            except Exception as exc:
                logger.warning(
                    f"Client {self.ID} failed to load VPL "
                    f"component {comp_name} from server: {exc}")

        if loaded:
            logger.info(
                f"Client {self.ID} loaded aggregated VPL "
                f"components from server: {loaded} "
                f"(round {message.state})")