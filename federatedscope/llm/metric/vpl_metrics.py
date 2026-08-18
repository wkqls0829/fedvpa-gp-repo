import logging
import federatedscope.register as register

logger = logging.getLogger(__name__)


def _safe_div(numerator, denominator, default=0.0):
    try:
        if denominator is None or denominator == 0:
            return default
        result = numerator / denominator
        # Convert tensor to float for JSON serialization
        if hasattr(result, 'item'):
            return result.item()
        return float(result)
    except Exception:
        return default


def eval_vpl_kl_loss(ctx, **kwargs):
    total = getattr(ctx, "vpl_kl_loss_total", None)
    num = getattr(ctx, "num_samples", None)
    return _safe_div(total, num, 0.0)


def eval_vpl_reconstruction_loss(ctx, **kwargs):
    total = getattr(ctx, "vpl_reconstruction_loss_total", None)
    num = getattr(ctx, "num_samples", None)
    return _safe_div(total, num, 0.0)


def eval_vpl_orthogonal_loss(ctx, **kwargs):
    total = getattr(ctx, "vpl_orthogonal_loss_total", None)
    num = getattr(ctx, "num_samples", None)
    return _safe_div(total, num, 0.0)


def register_vpl_kl(types):
    if 'vpl_kl_loss' in types:
        return 'vpl_kl_loss', eval_vpl_kl_loss, True
    return None


def register_vpl_recon(types):
    if 'vpl_reconstruction_loss' in types:
        return 'vpl_reconstruction_loss', eval_vpl_reconstruction_loss, True
    return None


def register_vpl_ortho(types):
    if 'vpl_orthogonal_loss' in types:
        return 'vpl_orthogonal_loss', eval_vpl_orthogonal_loss, True
    return None


register.register_metric('vpl_kl_loss', register_vpl_kl)
register.register_metric('vpl_reconstruction_loss', register_vpl_recon)
register.register_metric('vpl_orthogonal_loss', register_vpl_ortho)
