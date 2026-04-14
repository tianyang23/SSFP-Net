import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import sobel


#def compute_similarity_weights(features, sigma=1.0):
#    D, H, W = features.shape[-3], features.shape[-2], features.shape[-1]
#    num_voxels = D * H * W
#    flat_features = features.view(num_voxels, -1)
#    weights = torch.zeros((num_voxels, num_voxels)).to(features.device)

#    for i in range(num_voxels):
#        for j in range(num_voxels):
#            dist = torch.norm(flat_features[i] - flat_features[j])
#            weights[i, j] = torch.exp(-dist ** 2 / (2 * sigma ** 2))
#    return weights


#def laplacian_regularization(features, adjacency_matrix):
#    D, H, W = features.shape[-3], features.shape[-2], features.shape[-1]
#    num_voxels = D * H * W
#    flat_features = features.view(num_voxels, -1)
#    laplacian_loss = 0.0
#    for i in range(num_voxels):
#        for j in range(num_voxels):
#            laplacian_loss += adjacency_matrix[i, j] * torch.norm(flat_features[i] - flat_features[j]) ** 2
#    return laplacian_loss


def compute_gradient_map(features):
    """Compute voxel-wise gradient magnitudes for 3D feature maps."""
    N, C, D, H, W = features.shape
    gradient_map = torch.zeros_like(features)

    # Compute Sobel gradients channel by channel.
    for n in range(N):
        for c in range(C):

            grad_z = torch.tensor(sobel(features[n, c].detach().cpu().numpy(), axis=0), device=features.device)
            grad_y = torch.tensor(sobel(features[n, c].detach().cpu().numpy(), axis=1), device=features.device)
            grad_x = torch.tensor(sobel(features[n, c].detach().cpu().numpy(), axis=2), device=features.device)


            gradient_map[n, c] = torch.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2)

    return gradient_map


def compute_similarity_weights(features, sigma=1.0, gamma=1.0, delta=5):
    """Build local graph weights from feature and gradient similarity."""
    N, C, D, H, W = features.shape
    num_voxels = D * H * W


    flat_features = features.view(N, C, -1).transpose(1, 2)

    # Compute gradient features for edge-aware weighting.
    gradient_map = compute_gradient_map(features.detach())
    flat_gradient_map = gradient_map.view(N, C, -1).transpose(1, 2)

    # Build voxel coordinates for local neighborhood search.
    coords = torch.stack(torch.meshgrid(
        torch.arange(D, device=features.device),
        torch.arange(H, device=features.device),
        torch.arange(W, device=features.device),
        indexing='ij'
    ), dim=-1).view(-1, 3)

    # Restrict graph edges to local voxel neighborhoods.
    neighbor_dist = torch.cdist(coords.float(), coords.float())
    neighbor_indices = (neighbor_dist <= delta).nonzero(as_tuple=False)

    i_indices, j_indices = neighbor_indices[:, 0], neighbor_indices[:, 1]

    # Measure feature distances on local graph edges.
    feature_dist = torch.norm(flat_features[:, i_indices, :] - flat_features[:, j_indices, :], dim=-1) ** 2
    del flat_features
    # feature_similarity = torch.exp(-feature_dist / (2 * sigma ** 2))

    # Measure gradient distances on the same edges.
    gradient_dist = torch.norm(flat_gradient_map[:, i_indices, :] - flat_gradient_map[:, j_indices, :], dim=-1) ** 2
    del flat_gradient_map
    # gradient_similarity = torch.exp(-torch.log(1 + gradient_dist / (gamma ** 2)))

    # Combine feature and gradient cues into one similarity score.
    enhanced_similarity = torch.exp(-feature_dist / (2 * sigma ** 2) - torch.log(1 + gradient_dist / (gamma ** 2)))

    return i_indices, j_indices, enhanced_similarity


def laplacian_regularization(features, i_indices, j_indices, weights):
    """Penalize feature differences over weighted local graph edges."""
    N, C, D, H, W = features.shape
    flat_features = features.view(N, C, -1).transpose(1, 2)  # (N, D*H*W, C)
    laplacian_loss = 0.0


    for n in range(N):

        diffs = flat_features[n, i_indices] - flat_features[n, j_indices]
        laplacian_loss += torch.sum(weights * torch.norm(diffs, dim=-1) ** 2)

    return laplacian_loss / (N * len(i_indices))