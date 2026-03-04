#!/usr/bin/env python3
"""Fast training script for sandbox environment (no gradient conflict check to save time)."""
import os, sys, json, time, torch, numpy as np, logging
from datetime import datetime
from sklearn.metrics import roc_auc_score, log_loss
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.dataset import generate_synthetic_aliccp, prepare_dataloaders
from src.models.ple import PLEModel
from src.losses.uncertainty_weight import MultiTaskLoss
import torch.nn.functional as F

def run_fast_train(model_name, ModelClass, mc, train_loader, val_loader, test_loader, log_dir, epochs=5):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(model_name)
    
    model = ModelClass(mc)
    criterion = MultiTaskLoss(mc)
    params = list(model.parameters()) + list(criterion.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=1e-5)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"{model_name.upper()} | Params: {total_params:,}")
    
    history = []
    best_auc = 0; best_state = None
    
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        train_loss = 0; n = 0
        for batch in train_loader:
            preds = model(batch["sparse_features"], batch["dense_features"], apply_mask=(model_name=="ple"))
            labels = {"click": batch["click"], "conversion": batch["conversion"]}
            loss_d = criterion(preds, labels, preds.get("gate_weights"))
            optimizer.zero_grad()
            loss_d["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss_d["total_loss"].item(); n += 1
        
        # Anneal temperatures
        if hasattr(model, "anneal_all_temperatures"):
            model.anneal_all_temperatures(0.95)
        
        # Eval
        model.eval()
        ctr_p, ctr_l, cvr_p, cvr_l = [], [], [], []
        val_loss = 0; vn = 0
        with torch.no_grad():
            for batch in val_loader:
                preds = model(batch["sparse_features"], batch["dense_features"], apply_mask=False)
                labels = {"click": batch["click"], "conversion": batch["conversion"]}
                ld = criterion(preds, labels)
                val_loss += ld["total_loss"].item(); vn += 1
                ctr_p.extend(preds["ctr_pred"].numpy()); ctr_l.extend(batch["click"].numpy())
                cvr_p.extend(preds["cvr_pred"].numpy()); cvr_l.extend(batch["conversion"].numpy())
        
        try: ctr_auc = roc_auc_score(ctr_l, ctr_p)
        except: ctr_auc = 0.5
        try: cvr_auc = roc_auc_score(cvr_l, cvr_p)
        except: cvr_auc = 0.5
        avg_auc = (ctr_auc + cvr_auc) / 2
        
        # Task weights
        tw = {"ctr": 1.0, "cvr": 1.0}
        ls = {"ctr": 0.0, "cvr": 0.0}
        if hasattr(criterion, "uncertainty_weight"):
            uw = criterion.uncertainty_weight
            tw["ctr"] = torch.exp(-2*uw.log_sigma[0]).item()
            tw["cvr"] = torch.exp(-2*uw.log_sigma[1]).item()
            ls["ctr"] = uw.log_sigma[0].item()
            ls["cvr"] = uw.log_sigma[1].item()
        
        temps = model.get_gate_temperatures() if hasattr(model, "get_gate_temperatures") else None
        
        ep_log = {
            "epoch": epoch+1, "time": round(time.time()-t0, 2),
            "train_loss": round(train_loss/max(n,1), 6),
            "val_loss": round(val_loss/max(vn,1), 6),
            "ctr_auc": round(ctr_auc, 6), "cvr_auc": round(cvr_auc, 6),
            "avg_auc": round(avg_auc, 6),
            "task_weight_ctr": round(tw["ctr"], 4), "task_weight_cvr": round(tw["cvr"], 4),
            "log_sigma_ctr": round(ls["ctr"], 4), "log_sigma_cvr": round(ls["cvr"], 4),
            "gate_temps": temps
        }
        history.append(ep_log)
        logger.info(f"Epoch {epoch+1} | Loss: {ep_log['train_loss']:.4f}/{ep_log['val_loss']:.4f} | "
                    f"CTR: {ctr_auc:.4f} | CVR: {cvr_auc:.4f} | Avg: {avg_auc:.4f} | "
                    f"W: [{tw['ctr']:.3f},{tw['cvr']:.3f}] | T: {temps[0] if temps else 'N/A'}")
        
        if avg_auc > best_auc:
            best_auc = avg_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    # Test
    if best_state: model.load_state_dict(best_state)
    model.eval()
    ctr_p, ctr_l, cvr_p, cvr_l = [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            preds = model(batch["sparse_features"], batch["dense_features"], apply_mask=False)
            ctr_p.extend(preds["ctr_pred"].numpy()); ctr_l.extend(batch["click"].numpy())
            cvr_p.extend(preds["cvr_pred"].numpy()); cvr_l.extend(batch["conversion"].numpy())
    
    try: test_ctr = roc_auc_score(ctr_l, ctr_p)
    except: test_ctr = 0.5
    try: test_cvr = roc_auc_score(cvr_l, cvr_p)
    except: test_cvr = 0.5
    
    test_res = {"ctr_auc": round(test_ctr, 6), "cvr_auc": round(test_cvr, 6), "avg_auc": round((test_ctr+test_cvr)/2, 6)}
    
    # Frozen weights
    frozen = None
    if hasattr(criterion, "uncertainty_weight"):
        frozen = criterion.uncertainty_weight.get_frozen_weights()
    
    # Save
    with open(os.path.join(log_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(log_dir, "test_results.json"), "w") as f:
        json.dump(test_res, f, indent=2)
    if frozen:
        with open(os.path.join(log_dir, "frozen_weights.json"), "w") as f:
            json.dump({"frozen_task_weights": frozen}, f, indent=2)
    
    logger.info(f"Test: {json.dumps(test_res)}")
    return {"test": test_res, "history": history, "params": total_params, "frozen_weights": frozen}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ple", choices=["ple", "mmoe", "cgc"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--samples", type=int, default=5000)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)])
    
    torch.manual_seed(42); np.random.seed(42)
    data = generate_synthetic_aliccp(num_samples=args.samples, seed=42)
    train_ld, val_ld, test_ld, info = prepare_dataloaders(data, batch_size=1024, num_workers=0, seed=42)
    
    mc = {
        'num_tasks': 2, 'embedding_dim': 8, 'num_extraction_layers': 2,
        'expert_dim': 64, 'num_task_experts': 2, 'num_shared_experts': 1,
        'tower_hidden_dim': 32, 'dropout': 0.1, 'initial_temperature': 2.0,
        'use_esmm': True, 'use_feature_mask': True, 'mask_ratio': 0.15,
        'use_uncertainty_weight': True, 'initial_log_sigma': 0.0,
        'mask_loss_weight': 0.1, 'load_balance_weight': 0.01,
        'num_sparse_features': info['num_sparse'],
        'sparse_feature_dims': [d+1 for d in info['sparse_dims']],
        'num_dense_features': info['num_dense'],
    }
    
    if args.model == "ple":
        from src.models.ple import PLEModel as MC
    elif args.model == "mmoe":
        from src.models.baselines import MMoEModel as MC
        mc["num_experts"] = mc["num_task_experts"] + mc["num_shared_experts"]
    else:
        from src.models.baselines import CGCModel as MC
    
    run_fast_train(args.model, MC, mc, train_ld, val_ld, test_ld, f"logs/{args.model}", args.epochs)
    print(f"DONE_{args.model.upper()}")
