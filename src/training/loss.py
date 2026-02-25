"""
Contrastive loss (InfoNCE) with in-batch negatives and self-adversarial re-weighting.
Harder negatives (higher query-product similarity) get higher weight.
"""

from __future__ import annotations  # Enable postponed evaluation of type hints

import torch  # PyTorch tensor operations
import torch.nn.functional as F  # Functional API (e.g., cross_entropy)

import torch
import torch.nn.functional as F


def contrastive_loss_with_reweighting(
    query_emb: torch.Tensor,
    product_emb: torch.Tensor,
    temperature: float = 0.05,
    reweight_hard: bool = True,
    hard_weight_power: float = 1.0,
) -> torch.Tensor:
    """
    Contrastive loss with hard-negative reweighting (HCL; Robinson et al., ICLR 2021).

    - When `reweight_hard` is False or `hard_weight_power <= 0`, this reduces to **standard InfoNCE**:
      softmax over all products in the batch with the positive on the diagonal.
    - When enabled, we approximate the debiased **Hardness-aware Contrastive Loss (HCL)**:
      negatives that currently look very similar to the query (hard negatives) get **higher weight**
      in the denominator of the softmax, so the model learns more from them.

    Shapes
    ------
    query_emb:
        [B, D] – batch of B L2‑normalized query embeddings.
    product_emb:
        [B, D] – batch of B L2‑normalized product embeddings; row i is the positive for query i.
    temperature:
        Standard InfoNCE temperature τ; lower = sharper softmax.
    reweight_hard:
        If False, skip all HCL logic and compute plain InfoNCE.
    hard_weight_power:
        β ≥ 0 in the HCL paper; larger β puts more emphasis on hard negatives.
    """
    # Batch size B and device for convenience
    B = query_emb.size(0)
    device = query_emb.device
    # Pairwise similarities between every (query i, product j), scaled by temperature.
    # Because embeddings are L2‑normalized, dot‑product ≈ cosine similarity.
    logits = torch.mm(query_emb, product_emb.t()) / temperature  # [B, B]

    # Labels: for each query i, the positive product is at index i (diagonal of the matrix).
    labels = torch.arange(B, device=device, dtype=torch.long)

    # If hard‑negative reweighting is disabled, this is just standard InfoNCE / softmax cross‑entropy.
    if not reweight_hard or hard_weight_power <= 0:
        return F.cross_entropy(logits, labels)

    # ------------------------------
    # Hardness‑aware Contrastive Loss (HCL)
    # ------------------------------
    # See "Hardness‑Aware Deep Metric Learning" (Robinson et al., ICLR 2021).
    # We approximate expectations over the positive and negative distributions using the batch.

    # β controls how aggressively we emphasise hard negatives: higher β -> more mass on high‑sim negatives.
    beta = hard_weight_power
    # τ_pos is the assumed prior probability of a positive pair; small but non‑zero.
    # Values in [0.01, 0.2] are typical; this can be tuned.
    tau_pos = 0.1
    tau_neg = 1.0 - tau_pos

    # Z_beta ≈ E_p[exp(β * s)] under the **mixed** (pos+neg) similarity distribution p(s).
    # We estimate this via a batch average over all similarities.
    Z_beta = torch.logsumexp(beta * logits, dim=1, keepdim=True) - torch.log(
        torch.tensor(B, dtype=torch.float, device=device)
    )
    Z_beta = torch.exp(Z_beta)  # [B,1]

    # Z_pos_beta ≈ E_{p+}[exp(β * s)] under the **positive** similarity distribution.
    # We only have B positives (the diagonal), so we approximate with their batch mean.
    diag_logits = torch.diag(logits)
    Z_pos_beta = torch.exp(beta * diag_logits).mean()  # scalar

    # From the mixture p(s) = τ_pos * p_+(s) + τ_neg * p_-(s):
    #   E_p[exp(β s)] = τ_pos * E_{p+}[exp(β s)] + τ_neg * E_{p-}[exp(β s)]
    # Solve for the negative expectation E_{p-}[exp(β s)] to get a debiased estimate for negatives.
    neg_exp = (Z_beta - tau_pos * Z_pos_beta) / tau_neg  # [B,1]

    # We now construct *effective* logits so that cross‑entropy(q, p) sees:
    #   numerator   ≈ exp(sim_pos / τ)
    #   denominator ≈ exp(sim_pos / τ) + B * E_{p-}[exp(sim_neg / τ)]
    # i.e. we collapse the entire negative distribution into a single term proportional to neg_exp.
    pos_logits = torch.diag(logits).unsqueeze(1)  # [B,1]
    logits_eff = pos_logits + torch.log(B * neg_exp + 1e-8)  # Safe log-denom approx

    return F.cross_entropy(logits_eff.squeeze(1), labels)
