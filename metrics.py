import torch
import torch.nn.functional as F


def epe(input_flow, target_flow, mean=True):
    epe_map = torch.norm(target_flow - input_flow, p=2, dim=1)
    batch_size = epe_map.size(0)
    if mean:
        return epe_map.mean()
    else:
        return epe_map.sum() / batch_size


def sparse_max_pool(input, size):
    '''Downsample the input by considering 0 values as invalid.

    Unfortunately, no generic interpolation mode can resize a sparse map correctly,
    the strategy here is to use max pooling for positive values and "min pooling"
    for negative values, the two results are then summed.
    This technique allows sparsity to be minized, contrary to nearest interpolation,
    which could potentially lose information for isolated data points.'''

    positive = (input > 0).float()
    negative = (input < 0).float()
    output = F.adaptive_max_pool2d(input * positive, size) - F.adaptive_max_pool2d(-input * negative, size)
    return output


def multiscale_epe(network_output, target_flow, weights=None):
    def one_scale(output, target):
        b, _, h, w = output.size()
        target_scaled = F.interpolate(target, (h, w), mode='area')
        return epe(output, target_scaled, mean=False)

    if type(network_output) not in [tuple, list]:
        network_output = [network_output]

    if weights is None:
        weights = [0.005, 0.01, 0.02, 0.08, 0.32]  # as in original article

    assert (len(weights) == len(network_output))

    loss = 0
    for output, weight in zip(network_output, weights):
        loss += weight * one_scale(output, target_flow)
    return loss


def real_epe(output, target):
    b, _, h, w = target.size()
    upsampled_output = F.interpolate(output, (h, w), mode='bilinear', align_corners=False)
    return epe(upsampled_output, target, mean=True)
