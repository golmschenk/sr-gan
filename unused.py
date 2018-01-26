import torch


def feature_corrcoef(x):
    transposed_x = x.transpose(0, 1)
    return corrcoef(transposed_x)


def corrcoef(x):
    mean_x = x.mean(1, keepdim=True)
    xm = x.sub(mean_x)
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())
    c = torch.clamp(c, -1.0, 1.0)
    return c

class WeightClipper(object):
    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(min=-1, max=1)

clipper = WeightClipper()
# D.apply(clipper)

def torch_variable_contains_nan_or_inf(variable):
    if np.isnan(cpu(variable).data.numpy()).any():
        return True
    elif np.isinf(cpu(variable).data.numpy()).any():
        return True
    else:
        return False