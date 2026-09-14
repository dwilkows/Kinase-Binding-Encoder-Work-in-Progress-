import torch
import torch.nn.functional as F

def similarity(vec1, vec2, temperature):
    vec1_norm = F.normalize(vec1, p=2, dim=-1)
    vec2_norm = F.normalize(vec2, p=2, dim=-1)
    cos_sim = (vec1_norm * vec2_norm).sum(dim=-1)
    return cos_sim / temperature

def evaluate_binding(val_loader, lig_encoder, prot_encoder, temperature, device):
    lig_encoder.eval()
    prot_encoder.eval()

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for graph, tok, labels in val_loader:
            graph = graph.to(device)
            tok = tok.to(device)

            z_L = lig_encoder(graph)
            z_P = prot_encoder(tok)
            scores = similarity(z_L, z_P, temperature)
            probs = torch.sigmoid(scores)

            all_scores.append(probs.cpu())
            all_labels.append(labels.cpu())

    all_scores = torch.cat(all_scores)
    all_labels = torch.cat(all_labels)

    preds = (all_scores > 0.5).float()
    accuracy = (preds == all_labels).float().mean().item()

    pos_mask = all_labels == 1
    neg_mask = all_labels == 0
    pos_mean_sim = all_scores[pos_mask].mean().item() if pos_mask.any() else float("nan")
    neg_mean_sim = all_scores[neg_mask].mean().item() if neg_mask.any() else float("nan")

    metrics = {
        "accuracy": accuracy,
        "pos_mean_similarity": pos_mean_sim,
        "neg_mean_similarity": neg_mean_sim,
        "n_examples": len(all_labels),
    }

    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        metrics["auroc"] = roc_auc_score(all_labels.numpy(), all_scores.numpy())
        metrics["auprc"] = average_precision_score(all_labels.numpy(), all_scores.numpy())
    except ImportError:
        pass

    print(
        f"[bind eval] acc={metrics['accuracy']:.3f} "
        f"pos_sim={metrics['pos_mean_similarity']:.3f} "
        f"neg_sim={metrics['neg_mean_similarity']:.3f}"
        + (f" auroc={metrics['auroc']:.3f} auprc={metrics['auprc']:.3f}" if "auroc" in metrics else "")
    )

    lig_encoder.train()
    prot_encoder.train()
    return metrics


def evaluate_mechanism(val_loader, lig_encoder, prot_encoder, device, margin=1.0):
    lig_encoder.eval()
    prot_encoder.eval()

    correct = 0
    total = 0
    margins = []

    with torch.no_grad():
        for a_graph, a_tok, p_graph, p_tok, n_graph, n_tok in val_loader:
            a_graph = a_graph.to(device)
            p_graph = p_graph.to(device)
            n_graph = n_graph.to(device)

            zA_L = lig_encoder(a_graph)
            zP_L = lig_encoder(p_graph)
            zN_L = lig_encoder(n_graph)

            dist_pos = F.pairwise_distance(zA_L, zP_L, p=2)
            dist_neg = F.pairwise_distance(zA_L, zN_L, p=2)

            correct += (dist_pos < dist_neg).sum().item()
            total += dist_pos.shape[0]

            margins.append((dist_neg - dist_pos).cpu())

    triplet_accuracy = correct / total if total > 0 else float("nan")
    margins = torch.cat(margins)
    mean_margin = margins.mean().item()
    margin_satisfied_rate = (margins > margin).float().mean().item()

    metrics = {
        "triplet_accuracy": triplet_accuracy,
        "mean_margin": mean_margin,
        "margin_satisfied_rate": margin_satisfied_rate,
        "n_triplets": total,
    }

    print(
        f"[mech eval] triplet_acc={metrics['triplet_accuracy']:.3f} "
        f"mean_margin={metrics['mean_margin']:.3f} "
        f"margin_satisfied={metrics['margin_satisfied_rate']:.3f} "
        f"(n={total})"
    )

    lig_encoder.train()
    return metrics


def evaluate_encodings(phase, val_loader, lig_encoder, prot_encoder, device, temperature=0.07, margin=1.0):
    if phase == "bind":
        return evaluate_binding(val_loader, lig_encoder, prot_encoder, temperature, device)
    elif phase == "mech":
        return evaluate_mechanism(val_loader, lig_encoder, prot_encoder, device, margin)
    else:
        raise ValueError(f"Unknown phase '{phase}', expected 'bind' or 'mech'")
