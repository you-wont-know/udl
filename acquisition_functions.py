import torch

def random_acquisition(batch, model, T):
    return torch.rand(batch.shape[0], device=batch.device)

def bald_acquisition(batch, model, T):
    # model.train() having been called is assumed
    entropy_of_avg = 0
    average_entropy = 0
    
    avg_prediction = torch.zeros(batch.shape[0], 10, device=batch.device)
    for _ in range(T):
        pred_prob = model.forward_probs(batch, t=1)
        avg_prediction += pred_prob / T
        average_entropy += (pred_prob * torch.log(pred_prob + 1e-10)).sum(dim=1) / T

    entropy_of_avg = (avg_prediction * torch.log(avg_prediction + 1e-10)).sum(dim=1)

    mutual_info = -entropy_of_avg + average_entropy
    return mutual_info


def predictive_entropy_acquisition(batch, model, T):
    avg_prediction = torch.zeros(batch.shape[0], 10, device=batch.device)
    for _ in range(T):
        pred_prob = model.forward_probs(batch, t=1)
        avg_prediction += pred_prob / T

    entropy_of_avg = (avg_prediction * torch.log(avg_prediction + 1e-10)).sum(dim=1)

    return -entropy_of_avg

def variation_ratio_acquisition(batch, model, T):
    max_prob = model.forward_probs(batch, t=T).max(dim=1).values
    return 1.0 - max_prob


def mean_standard_deviation_acquisition(batch, model, T):
    sum_probs = torch.zeros(batch.shape[0], 10, device=batch.device)
    sum_sq_probs = torch.zeros(batch.shape[0], 10, device=batch.device)
    for _ in range(T):
        pred_prob = model.forward_probs(batch, t=1)
        sum_probs += pred_prob
        sum_sq_probs += pred_prob * pred_prob
    mean_probs = sum_probs / T
    mean_sq_probs = sum_sq_probs / T

    sigma = torch.sqrt(mean_sq_probs - mean_probs * mean_probs + 1e-10)

    return sigma.mean(dim=1)


def determinant_acquisition(batch, model, T):
    mean, cov = model(batch)
    _ = mean
    det = torch.linalg.det(cov)
    return det


def bayesian_entropy_acquisition(batch, model, T):
    mean, _ = model(batch)
    probs = torch.softmax(mean, dim=1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
    return entropy


def determinant_entropy_acquisition(batch, model, T):
    mean, cov = model(batch)
    probs = torch.softmax(mean, dim=1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
    det = torch.linalg.det(cov)
    return entropy * det


def max_diag_acquisition(batch, model, T):
    mean, cov = model(batch)
    _ = mean
    diag = cov.diagonal(dim1=1, dim2=2)
    return diag.max(dim=1).values


def trace_acquisition(batch, model, T):
    mean, cov = model(batch)
    _ = mean
    diag = cov.diagonal(dim1=1, dim2=2)
    return diag.sum(dim=1)

def bayesian_variation_ratio_acquisition(batch, model, T):
    mean, _ = model(batch)
    probs = torch.softmax(mean, dim=1)
    max_prob = probs.max(dim=1).values
    return 1.0 - max_prob
