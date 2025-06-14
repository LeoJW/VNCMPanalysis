import numpy as np
import torch
import torch.nn.functional as F

# Check if CUDA or MPS is running
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = "cpu"


def logmeanexp_diag(x, device=device):
    """Compute logmeanexp over the diagonal elements of x."""
    batch_size = x.size(0)

    logsumexp = torch.logsumexp(x.diag(), dim=(0,))
    num_elem = batch_size

    return logsumexp - torch.log(torch.tensor(num_elem).float()).to(device)


def logmeanexp_nodiag(x, dim=None, device=device):
    batch_size = x.size(0)
    if dim is None:
        dim = (0, 1)

    logsumexp = torch.logsumexp(
        x - torch.diag(np.inf * torch.ones(batch_size).to(device)), dim=dim)

    try:
        if len(dim) == 1:
            num_elem = batch_size - 1.
        else:
            num_elem = batch_size * (batch_size - 1.)
    except ValueError:
        num_elem = batch_size - 1
    return logsumexp - torch.log(torch.tensor(num_elem)).to(device)


def log_interpolate(log_a, log_b, alpha_logit):
    # Numerically stable implementation of log(alpha * a + (1-alpha) * b).
    log_alpha = -F.softplus(-torch.tensor(alpha_logit))
    log_1_minus_alpha = -F.softplus(torch.tensor(alpha_logit))
    y = torch.logsumexp(torch.stack((log_alpha + torch.tensor(log_a), log_1_minus_alpha + torch.tensor(log_b))), dim=0)
    return y

# Define a stable softplus inverse
def soft_plus_inverse(x):
    return x + torch.log(-torch.expm1(-x))

def compute_log_loomean(scores):
    # Compute the log leave-one-out mean of the exponentiated scores.
    max_scores, _ = torch.max(scores, dim=1, keepdim=True)
    lse_minus_max = torch.logsumexp(scores - max_scores, dim=1, keepdim=True)
    d = lse_minus_max + (max_scores - scores)
    d_ok = d != 0.
    safe_d = torch.where(d_ok, d, torch.ones_like(d))
    loo_lse = scores + soft_plus_inverse(safe_d) # This should be the softplus inverse function we coded
    # Normalize to get the leave one out log mean exp
    loo_lme = loo_lse - torch.log(torch.tensor(scores.shape[1]) - 1.)
    return loo_lme

def interpolated_lower_bound(scores, baseline, alpha_logit):
    # Interpolated lower bound on mutual information.
    # Compute InfoNCE baseline
    if baseline == None:
        baseline=torch.e*torch.ones(scores.shape[0],device=device)
    nce_baseline = compute_log_loomean(scores)
    # Interpolated baseline interpolates the InfoNCE baseline with a learned baseline
    interpolated_baseline = log_interpolate(
        nce_baseline, baseline.view(-1, 1).repeat(1, scores.shape[0]), alpha_logit)
    # Marginal term.
    critic_marg = scores - torch.diag(interpolated_baseline)
    marg_term = torch.exp(logmeanexp_nodiag(critic_marg))
    # Joint term.
    critic_joint = torch.diag(scores) - interpolated_baseline
    joint_term = (torch.sum(critic_joint) - torch.sum(torch.diag(critic_joint))) / (scores.shape[0] * (scores.shape[0] - 1.))
    return 1 + joint_term - marg_term


def tuba_lower_bound(scores, log_baseline=None):
    if log_baseline is not None:
        scores -= log_baseline[:, None]

    # First term is an expectation over samples from the joint,
    # which are the diagonal elmements of the scores matrix.
    joint_term = scores.diag().mean()

    # Second term is an expectation over samples from the marginal,
    # which are the off-diagonal elements of the scores matrix.
    marg_term = logmeanexp_nodiag(scores).exp()
    return 1. + joint_term - marg_term


def nwj_lower_bound(scores):
    return tuba_lower_bound(scores - 1.)


def infonce_lower_bound(scores):
    mi = scores.diag().mean() - scores.logsumexp(dim=1) + torch.as_tensor(scores.size(0)).log()
    mi = mi.mean()
    return mi


def js_fgan_lower_bound(f):
    """Lower bound on Jensen-Shannon divergence from Nowozin et al. (2016)."""
    f_diag = f.diag()
    first_term = -F.softplus(-f_diag).mean()
    n = f.size(0)
    second_term = (torch.sum(F.softplus(f)) -
                   torch.sum(F.softplus(f_diag))) / (n * (n - 1.))
    return first_term - second_term


def js_lower_bound(f):
    """Obtain density ratio from JS lower bound then output MI estimate from NWJ bound."""
    nwj = nwj_lower_bound(f)
    js = js_fgan_lower_bound(f)

    with torch.no_grad():
        nwj_js = nwj - js

    return js + nwj_js


def dv_upper_lower_bound(f):
    """
    Donsker-Varadhan lower bound, but upper bounded by using log outside. 
    Similar to MINE, but did not involve the term for moving averages.
    """
    first_term = f.diag().mean()
    second_term = logmeanexp_nodiag(f)

    return first_term - second_term


def mine_lower_bound(f, buffer=None, momentum=0.9):
    """
    MINE lower bound based on DV inequality. 
    """
    if buffer is None:
        buffer = torch.tensor(1.0).to(device)
    first_term = f.diag().mean()

    buffer_update = logmeanexp_nodiag(f).exp()
    with torch.no_grad():
        second_term = logmeanexp_nodiag(f)
        buffer_new = buffer * momentum + buffer_update * (1 - momentum)
        buffer_new = torch.clamp(buffer_new, min=1e-4)
        third_term_no_grad = buffer_update / buffer_new

    third_term_grad = buffer_update / buffer_new

    return first_term - second_term - third_term_grad + third_term_no_grad#, buffer_update


def smile_lower_bound(f, clip=None):
    if clip is not None:
        f_ = torch.clamp(f, -clip, clip)
    else:
        f_ = f
    z = logmeanexp_nodiag(f_, dim=(0, 1))
    dv = f.diag().mean() - z

    js = js_fgan_lower_bound(f)

    with torch.no_grad():
        dv_js = dv - js

    return js + dv_js


def estimate_mutual_information(estimator, scores, log_baseline=None):
    """Estimate variational lower bounds on mutual information.
    Args:
    estimator: string specifying estimator, one of:
        'nwj', 'infonce', 'tuba', 'js', 'interpolated'
    scores: Tensor/matrix of critic scores, shape [batch_size, batch_size]
    log_baseline: If not None, should be torch.squeeze(baseline_fn(y))
    baseline_fn (optional): callable that takes y as input 
        outputs a [batch_size]  or [batch_size, 1] vector
    alpha_logit (optional): logit(alpha) for interpolated bound, equivalent to momentum
    Returns:
    scalar estimate of mutual information
    """

    if estimator == 'infonce':
        mi = infonce_lower_bound(scores)
    elif estimator == 'nwj':
        mi = nwj_lower_bound(scores)
    elif estimator == 'js':
        mi = js_lower_bound(scores)
    elif estimator == 'js_fgan':
        mi = js_fgan_lower_bound(scores)
    elif estimator == 'dv':
        mi = dv_upper_lower_bound(scores)
        
    elif estimator == 'tuba':
        mi = tuba_lower_bound(scores, log_baseline)
        
    elif estimator == 'smile':
        mi = smile_lower_bound(scores)
    elif estimator == 'smile_1':
        mi = smile_lower_bound(scores, clip=1.0)
    elif estimator == 'smile_5':
        mi = smile_lower_bound(scores, clip=5.0)

    elif estimator == 'mine_0':
        mi = mine_lower_bound(scores, momentum=0)
    elif estimator == 'mine_0.1':
        mi = mine_lower_bound(scores, momentum=0.1)
    elif estimator == 'mine_0.5':
        mi = mine_lower_bound(scores, momentum=0.5)
    elif estimator == 'mine_0.9':
        mi = mine_lower_bound(scores, momentum=0.9)
    elif estimator == 'mine_1':
        mi = mine_lower_bound(scores, momentum=1)

    elif estimator == 'ialpha_0.1':
        mi = interpolated_lower_bound(scores, log_baseline, alpha_logit=0.1)
    elif estimator == 'ialpha_0.5':
        mi = interpolated_lower_bound(scores, log_baseline, alpha_logit=0.5)
    elif estimator == 'ialpha_0.9':
        mi = interpolated_lower_bound(scores, log_baseline, alpha_logit=0.9)
        
    return mi
