#!/usr/bin/env python3
"""Single-epoch training for sandbox time constraints. Saves/loads state between calls."""
import os, sys, json, time, torch, numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.dataset import generate_synthetic_aliccp, prepare_dataloaders
from src.models.ple import PLEModel
from src.models.baselines import MMoEModel, CGCModel
from src.losses.uncertainty_weight import MultiTaskLoss

MODEL_MAP = {"ple": PLEModel, "mmoe": MMoEModel, "cgc": CGCModel}

def get_config(info):
    return {
        'num_tasks': 2, 'embedding_dim': 8, 'num_extraction_layers': 2,
        'expert_dim': 64, 'num_task_experts': 2, 'num_shared_experts': 1,
        'tower_hidden_dim': 32, 'dropout': 0.1, 'initial_temperature': 2.0,
        'use_esmm': True, 'use_feature_mask': True, 'mask_ratio': 0.15,
        'use_uncertainty_weight': True, 'initial_log_sigma': 0.0,
        'mask_loss_weight': 0.1, 'load_balance_weight': 0.01,
        'num_experts': 3,  # for mmoe
        'num_sparse_features': info['num_sparse'],
        'sparse_feature_dims': [d+1 for d in info['sparse_dims']],
        'num_dense_features': info['num_dense'],
    }

def train_one_epoch(model_name, epoch, total_epochs=8):
    torch.manual_seed(42); np.random.seed(42)
    data = generate_synthetic_aliccp(num_samples=5000, seed=42)
    train_ld, val_ld, test_ld, info = prepare_dataloaders(data, batch_size=1024, num_workers=0, seed=42)
    mc = get_config(info)
    
    ModelClass = MODEL_MAP[model_name]
    model = ModelClass(mc)
    criterion = MultiTaskLoss(mc)
    
    log_dir = f"logs/{model_name}"
    os.makedirs(log_dir, exist_ok=True)
    state_file = os.path.join(log_dir, "state.pt")
    hist_file = os.path.join(log_dir, "training_history.json")
    
    # Load previous state
    history = []
    if os.path.exists(state_file) and epoch > 0:
        ckpt = torch.load(state_file, weights_only=False)
        model.load_state_dict(ckpt["model"])
        criterion.load_state_dict(ckpt["criterion"])
        history = ckpt.get("history", [])
    
    params = list(model.parameters()) + list(criterion.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=1e-5)
    if os.path.exists(state_file) and epoch > 0:
        optimizer.load_state_dict(ckpt["optimizer"])
    
    t0 = time.time()
    
    # Train
    model.train()
    train_loss = 0; n = 0
    for batch in train_ld:
        preds = model(batch["sparse_features"], batch["dense_features"], apply_mask=(model_name=="ple"))
        labels = {"click": batch["click"], "conversion": batch["conversion"]}
        loss_d = criterion(preds, labels, preds.get("gate_weights"))
        optimizer.zero_grad(); loss_d["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss_d["total_loss"].item(); n += 1
    
    if hasattr(model, "anneal_all_temperatures"):
        model.anneal_all_temperatures(0.95)
    
    # Eval
    model.eval()
    cp, cl, vp, vl = [], [], [], []
    vloss = 0; vn = 0
    with torch.no_grad():
        for batch in val_ld:
            preds = model(batch["sparse_features"], batch["dense_features"], apply_mask=False)
            ld = criterion(preds, {"click": batch["click"], "conversion": batch["conversion"]})
            vloss += ld["total_loss"].item(); vn += 1
            cp.extend(preds["ctr_pred"].numpy()); cl.extend(batch["click"].numpy())
            vp.extend(preds["cvr_pred"].numpy()); vl.extend(batch["conversion"].numpy())
    
    try: ctr_auc = roc_auc_score(cl, cp)
    except: ctr_auc = 0.5
    try: cvr_auc = roc_auc_score(vl, vp)
    except: cvr_auc = 0.5
    
    tw_ctr, tw_cvr, ls_ctr, ls_cvr = 1.0, 1.0, 0.0, 0.0
    if hasattr(criterion, "uncertainty_weight"):
        uw = criterion.uncertainty_weight
        tw_ctr = torch.exp(-2*uw.log_sigma[0]).item()
        tw_cvr = torch.exp(-2*uw.log_sigma[1]).item()
        ls_ctr = uw.log_sigma[0].item()
        ls_cvr = uw.log_sigma[1].item()
    
    temps = model.get_gate_temperatures() if hasattr(model, "get_gate_temperatures") else None
    
    ep = {
        "epoch": epoch+1, "time": round(time.time()-t0, 2),
        "train_loss": round(train_loss/max(n,1), 4), "val_loss": round(vloss/max(vn,1), 4),
        "ctr_auc": round(ctr_auc, 6), "cvr_auc": round(cvr_auc, 6),
        "avg_auc": round((ctr_auc+cvr_auc)/2, 6),
        "task_weight_ctr": round(tw_ctr, 4), "task_weight_cvr": round(tw_cvr, 4),
        "log_sigma_ctr": round(ls_ctr, 4), "log_sigma_cvr": round(ls_cvr, 4),
        "gate_temps": temps
    }
    history.append(ep)
    
    # Save state
    torch.save({
        "model": model.state_dict(), "criterion": criterion.state_dict(),
        "optimizer": optimizer.state_dict(), "history": history
    }, state_file)
    
    with open(hist_file, "w") as f:
        json.dump(history, f, indent=2)
    
    # Test on final epoch
    test_res = None
    if epoch == total_epochs - 1:
        cp, cl, vp, vl = [], [], [], []
        with torch.no_grad():
            for batch in test_ld:
                preds = model(batch["sparse_features"], batch["dense_features"], apply_mask=False)
                cp.extend(preds["ctr_pred"].numpy()); cl.extend(batch["click"].numpy())
                vp.extend(preds["cvr_pred"].numpy()); vl.extend(batch["conversion"].numpy())
        try: tc = roc_auc_score(cl, cp)
        except: tc = 0.5
        try: tv = roc_auc_score(vl, vp)
        except: tv = 0.5
        test_res = {"ctr_auc": round(tc,6), "cvr_auc": round(tv,6), "avg_auc": round((tc+tv)/2,6)}
        with open(os.path.join(log_dir, "test_results.json"), "w") as f:
            json.dump(test_res, f, indent=2)
        if hasattr(criterion, "uncertainty_weight"):
            frozen = criterion.uncertainty_weight.get_frozen_weights()
            with open(os.path.join(log_dir, "frozen_weights.json"), "w") as f:
                json.dump({"frozen_task_weights": frozen}, f, indent=2)
    
    print(json.dumps(ep))
    return ep, test_res

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ple")
    p.add_argument("--epoch", type=int, default=0)
    p.add_argument("--total", type=int, default=8)
    a = p.parse_args()
    train_one_epoch(a.model, a.epoch, a.total)
