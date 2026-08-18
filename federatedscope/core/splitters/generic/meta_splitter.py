import random
import numpy as np
import logging

from federatedscope.core.splitters import BaseSplitter
from federatedscope.core.splitters.generic import IIDSplitter

logger = logging.getLogger(__name__)


class MetaSplitter(BaseSplitter):
    """
    This splitter split dataset with meta information with LLM dataset.

    Args:
        client_num: the dataset will be split into ``client_num`` pieces
    """
    def __init__(self, client_num, **kwargs):
        super(MetaSplitter, self).__init__(client_num)
        # Create an IID spliter in case that num_client < categories
        self.iid_spliter = IIDSplitter(client_num)
        # Optional per-category client counts, e.g. [3, 7] for
        # 3 harmless + 7 helpful.  Set via
        # ``data.meta_split_clients_per_cat``.
        self.clients_per_cat = kwargs.get('clients_per_cat', None)

    def __call__(self, dataset, prior=None, **kwargs):
        from torch.utils.data import Dataset, Subset

        tmp_dataset = [ds for ds in dataset]
        if isinstance(tmp_dataset[0], tuple):
            label = np.array([y for x, y in tmp_dataset])
        elif isinstance(tmp_dataset[0], dict):
            label = np.array([x['categories'] for x in tmp_dataset])
        else:
            raise TypeError(
                f'Unsupported data formats {type(tmp_dataset[0])}')

        # Split by categories in SORTED order for deterministic assignment
        categories = sorted(set(label))
        idx_slice = []
        for cat in categories:
            idxs = np.where(np.array(label) == cat)[0].tolist()
            random.shuffle(idxs)  # shuffle within category only
            idx_slice.append(idxs)

        # Log category sizes
        tot_size = 0
        for i, cat in enumerate(categories):
            logger.info(f'Index: {i}\t'
                        f'Category: {cat}\t'
                        f'Size: {len(idx_slice[i])}')
            tot_size += len(idx_slice[i])
        logger.info(f'Total size: {tot_size}')

        num_cats = len(categories)

        if num_cats < self.client_num:
            # Fewer categories than clients: distribute each category's
            # data across multiple clients (Non-IID split).
            # E.g. 2 categories, 10 clients -> 5 clients per category.

            # Compute per-category client counts
            if (self.clients_per_cat is not None
                    and len(self.clients_per_cat) == num_cats):
                cat_client_counts = list(self.clients_per_cat)
                if sum(cat_client_counts) != self.client_num:
                    logger.warning(
                        f"clients_per_cat {cat_client_counts} sums "
                        f"to {sum(cat_client_counts)}, expected "
                        f"{self.client_num}. Falling back to equal.")
                    cat_client_counts = None
            else:
                cat_client_counts = None

            if cat_client_counts is None:
                # Default equal split
                base = self.client_num // num_cats
                rem = self.client_num % num_cats
                cat_client_counts = [
                    base + (1 if i < rem else 0)
                    for i in range(num_cats)]

            new_idx_slice = []
            for i in range(num_cats):
                idxs = idx_slice[i]
                n_clients = cat_client_counts[i]
                chunk_size = (len(idxs) // n_clients
                              if n_clients > 0 else len(idxs))
                for c in range(n_clients):
                    start = c * chunk_size
                    if c == n_clients - 1:
                        new_idx_slice.append(idxs[start:])
                    else:
                        new_idx_slice.append(
                            idxs[start:start + chunk_size])

            assigned = 0
            for i, cat in enumerate(categories):
                n_clients = cat_client_counts[i]
                client_ids = list(
                    range(assigned + 1, assigned + n_clients + 1))
                logger.info(
                    f'Category "{cat}": {len(idx_slice[i])} samples '
                    f'-> {n_clients} clients {client_ids}')
                assigned += n_clients

        elif num_cats >= self.client_num:
            # More categories than clients: merge categories
            new_idx_slice = []
            for i in range(num_cats):
                if i < self.client_num:
                    new_idx_slice.append(idx_slice[i])
                else:
                    new_idx_slice[i % self.client_num] += idx_slice[i]
        else:
            return self.iid_spliter(dataset)

        if isinstance(dataset, Dataset):
            data_list = [Subset(dataset, idxs)
                         for idxs in new_idx_slice]
        else:
            data_list = [[dataset[idx] for idx in idxs]
                         for idxs in new_idx_slice]
        return data_list
