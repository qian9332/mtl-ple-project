#!/usr/bin/env python3
"""
Full Comparison Training: MMoE vs CGC vs PLE
Runs on CPU/GPU with full 500K dataset, detailed logging.

ANALYSIS: Why CGC > PLE in previous run:
============================================================
Root Cause: PLE's multi-layer extraction amplifies noise on synthetic data
with insufficient signal-to-noise ratio.

1. Overfitting via Over-Parameterization:
   - PLE (1,059,178 params) vs CGC (425,338 params) — 2.5x parameters
   - 3 extraction layers compound noise: each layer's gate error propagates
   - On synthetic data with weak feature-label correlation, more params = more overfitting

2. Information Bottleneck at Layer Transitions:
   - PLE feeds `mean(task_outputs)` as next layer input — loses task-specific info
   - CGC processes raw embeddings directly — stronger gradient signal

3. ESMM Interaction with Multi-Layer:
   - PLE's CVR = CTR * CVR_given_click through 3 layers creates long gradient path
   - CGC has direct 1-layer path — cleaner CVR gradient

4. Temperature Annealing Over-Regularization:
   - 3 layers × 2 tasks × initial_temp=2.0 → gates too uniform early on
   - PLE needs more epochs to "warm up" than CGC

FIX Strategy for this run:
- Reduce PLE to 2 extraction layers (less noise amplification)
- Use per-task input to next layer (not mean)
- Lower initial temperature to 1.5
- Add dropout in tower layers
- Use larger batch size for stable gradients
============================================================
"""

import os
import sys
import json
import time
import logging
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# Setup
# ============================================================
PROJ_DIR = "/home/user/mtl-project"
LOG_DIR = os.path.join(PROJ_DIR, "logs", "full_v2")
os.makedirs(LOG_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"training_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device: {DEVICE}")
logger.info(f"PyTorch: {torch.__version__}")
if DEVICE == "cuda":
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# Config
# ============================================================
SPARSE_DIMS = [1000, 500, 50, 20, 100, 5, 30, 200, 100, 24,
               7, 12, 50, 50, 30, 30, 10, 20, 15, 8]
NUM_SPARSE = 20
NUM_DENSE = 10
EMB_DIM = 8
TOTAL_EMBED = (NUM_SPARSE + NUM_DENSE) * EMB_DIM
EXPERT_DIM = 64          # Reduced from 128 for faster training
TOWER_HIDDEN = 64
DROPOUT = 0.15
NUM_SAMPLES = 500000
BATCH_SIZE = 4096
NUM_EPOCHS = 20
LR = 0.001
WEIGHT_DECAY = 1e-5

# ============================================================
# Dataset
# ============================================================
class MTLDataset(Dataset):
    def __init__(self, sparse, dense, click, conv):
        self.sparse = torch.LongTensor(sparse)
        self.dense = torch.FloatTensor(dense)
        self.click = torch.FloatTensor(click)
        self.conv = torch.FloatTensor(conv)

    def __len__(self):
        return len(self.click)

    def __getitem__(self, idx):
        return self.sparse[idx], self.dense[idx], self.click[idx], self.conv[idx]


def generate_data(n=NUM_SAMPLES, seed=42):
    """Generate synthetic Ali-CCP style data with stronger signal."""
    np.random.seed(seed)
    logger.info(f"Generating {n} samples...")

    sparse = np.zeros((n, NUM_SPARSE), dtype=np.int64)
    for i, d in enumerate(SPARSE_DIMS):
        sparse[:, i] = np.random.randint(0, d, size=n)

    dense = MinMaxScaler().fit_transform(np.random.randn(n, NUM_DENSE).astype(np.float32))

    # Stronger signal for more meaningful comparison
    user_aff = np.random.randn(1000) * 1.5
    item_qual = np.random.randn(500) * 1.5
    cat_bias = np.random.randn(50) * 0.5

    click_score = (
        user_aff[sparse[:, 0]] * 0.35 +
        item_qual[sparse[:, 1]] * 0.30 +
        cat_bias[sparse[:, 2]] * 0.15 +
        dense[:, 0] * 0.25 +
        dense[:, 1] * 0.15 +
        dense[:, 2] * 0.10 +
        np.random.randn(n) * 0.4
    )
    click_prob = 1.0 / (1.0 + np.exp(-(click_score - np.percentile(click_score, 75))))
    click = (np.random.random(n) < click_prob).astype(np.float32)

    conv_score = (
        user_aff[sparse[:, 0]] * 0.20 +
        item_qual[sparse[:, 1]] * 0.45 +
        cat_bias[sparse[:, 2]] * 0.20 +
        dense[:, 3] * 0.20 +
        dense[:, 4] * 0.15 +
        np.random.randn(n) * 0.3
    )
    cvr_prob = 1.0 / (1.0 + np.exp(-(conv_score - np.percentile(conv_score, 90))))
    conv = (click * (np.random.random(n) < cvr_prob)).astype(np.float32)

    ctr_rate = click.mean()
    cvr_rate = conv.mean()
    cvr_click_rate = conv[click == 1].mean() if click.sum() > 0 else 0
    logger.info(f"CTR={ctr_rate:.4f}, CVR={cvr_rate:.4f}, CVR|Click={cvr_click_rate:.4f}")

    return sparse, dense.astype(np.float32), click, conv


def make_loaders(sparse, dense, click, conv):
    ds = MTLDataset(sparse, dense, click, conv)
    n = len(ds)
    tr_n = int(n * 0.7)
    va_n = int(n * 0.15)
    te_n = n - tr_n - va_n
    tr, va, te = random_split(ds, [tr_n, va_n, te_n], generator=torch.Generator().manual_seed(42))
    tr_dl = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    va_dl = DataLoader(va, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2, pin_memory=True)
    te_dl = DataLoader(te, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2, pin_memory=True)
    return tr_dl, va_dl, te_dl


# ============================================================
# Model Components
# ============================================================
class Expert(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Gate(nn.Module):
    def __init__(self, in_dim, num_experts, init_temp=1.0):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_experts, bias=False)
        self.temp = init_temp
        self.min_temp = 0.1

    def forward(self, x):
        return F.softmax(self.fc(x) / max(self.temp, self.min_temp), dim=-1)

    def anneal(self, rate=0.95):
        self.temp = max(self.min_temp, self.temp * rate)


class EmbeddingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(d, EMB_DIM) for d in SPARSE_DIMS])
        self.dense_proj = nn.Linear(NUM_DENSE, NUM_DENSE * EMB_DIM)

    def forward(self, sparse, dense):
        sp = torch.cat([self.embeddings[i](sparse[:, i]) for i in range(NUM_SPARSE)], dim=1)
        dp = self.dense_proj(dense)
        return torch.cat([sp, dp], dim=1)


class Tower(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============================================================
# MMoE
# ============================================================
class MMoE(nn.Module):
    def __init__(self, num_experts=6):
        super().__init__()
        self.embed = EmbeddingLayer()
        self.experts = nn.ModuleList([Expert(TOTAL_EMBED, EXPERT_DIM, DROPOUT) for _ in range(num_experts)])
        self.gate_ctr = Gate(TOTAL_EMBED, num_experts)
        self.gate_cvr = Gate(TOTAL_EMBED, num_experts)
        self.tower_ctr = Tower(EXPERT_DIM, TOWER_HIDDEN, DROPOUT)
        self.tower_cvr = Tower(EXPERT_DIM, TOWER_HIDDEN, DROPOUT)

    def forward(self, sparse, dense):
        e = self.embed(sparse, dense)
        exp_out = torch.stack([ex(e) for ex in self.experts], dim=1)
        g_ctr = self.gate_ctr(e).unsqueeze(1)
        g_cvr = self.gate_cvr(e).unsqueeze(1)
        ctr_in = torch.bmm(g_ctr, exp_out).squeeze(1)
        cvr_in = torch.bmm(g_cvr, exp_out).squeeze(1)
        ctr_p = torch.sigmoid(self.tower_ctr(ctr_in))
        cvr_p = torch.sigmoid(self.tower_cvr(cvr_in))
        return ctr_p, cvr_p, [self.gate_ctr(e), self.gate_cvr(e)]


# ============================================================
# CGC
# ============================================================
class CGC(nn.Module):
    def __init__(self, n_task_exp=3, n_shared_exp=2):
        super().__init__()
        self.embed = EmbeddingLayer()
        self.ctr_experts = nn.ModuleList([Expert(TOTAL_EMBED, EXPERT_DIM, DROPOUT) for _ in range(n_task_exp)])
        self.cvr_experts = nn.ModuleList([Expert(TOTAL_EMBED, EXPERT_DIM, DROPOUT) for _ in range(n_task_exp)])
        self.shared_experts = nn.ModuleList([Expert(TOTAL_EMBED, EXPERT_DIM, DROPOUT) for _ in range(n_shared_exp)])
        total = n_task_exp + n_shared_exp
        self.gate_ctr = Gate(TOTAL_EMBED, total)
        self.gate_cvr = Gate(TOTAL_EMBED, total)
        self.tower_ctr = Tower(EXPERT_DIM, TOWER_HIDDEN, DROPOUT)
        self.tower_cvr = Tower(EXPERT_DIM, TOWER_HIDDEN, DROPOUT)

    def forward(self, sparse, dense):
        e = self.embed(sparse, dense)
        shared_out = [ex(e) for ex in self.shared_experts]
        ctr_out = [ex(e) for ex in self.ctr_experts] + shared_out
        cvr_out = [ex(e) for ex in self.cvr_experts] + shared_out
        ctr_stack = torch.stack(ctr_out, dim=1)
        cvr_stack = torch.stack(cvr_out, dim=1)
        g_ctr = self.gate_ctr(e).unsqueeze(1)
        g_cvr = self.gate_cvr(e).unsqueeze(1)
        ctr_in = torch.bmm(g_ctr, ctr_stack).squeeze(1)
        cvr_in = torch.bmm(g_cvr, cvr_stack).squeeze(1)
        ctr_p = torch.sigmoid(self.tower_ctr(ctr_in))
        cvr_p = torch.sigmoid(self.tower_cvr(cvr_in))
        return ctr_p, cvr_p, [self.gate_ctr(e), self.gate_cvr(e)]


# ============================================================
# PLE (Fixed version)
# ============================================================
class PLELayer(nn.Module):
    """Single PLE extraction layer with per-task input/output."""
    def __init__(self, in_dim, out_dim, n_task_exp=2, n_shared_exp=2, n_tasks=2, dropout=0.1, temp=1.5):
        super().__init__()
        self.n_tasks = n_tasks
        total = n_task_exp + n_shared_exp
        self.task_experts = nn.ModuleList()
        for _ in range(n_tasks):
            self.task_experts.append(nn.ModuleList([Expert(in_dim, out_dim, dropout) for _ in range(n_task_exp)]))
        self.shared_experts = nn.ModuleList([Expert(in_dim, out_dim, dropout) for _ in range(n_shared_exp)])
        self.gates = nn.ModuleList([Gate(in_dim, total, temp) for _ in range(n_tasks)])

    def forward(self, task_inputs):
        """
        task_inputs: list of (batch, in_dim) per task
        Returns: list of (batch, out_dim) per task, gate_weights
        """
        shared_out = [ex(task_inputs[0]) for ex in self.shared_experts]  # use first task input for shared
        outputs = []
        gw_list = []
        for t in range(self.n_tasks):
            t_out = [ex(task_inputs[t]) for ex in self.task_experts[t]]
            all_out = t_out + shared_out
            stack = torch.stack(all_out, dim=1)
            gw = self.gates[t](task_inputs[t])
            gw_list.append(gw)
            out = torch.bmm(gw.unsqueeze(1), stack).squeeze(1)
            outputs.append(out)
        return outputs, gw_list

    def anneal(self, rate):
        for g in self.gates:
            g.anneal(rate)


class PLE(nn.Module):
    """
    Progressive Layered Extraction with fixes:
    - 2 layers (reduced from 3)
    - Per-task input propagation (not mean)
    - ESMM CVR branch
    - Feature mask self-supervised auxiliary
    - Expert utilization monitoring
    """
    def __init__(self, n_layers=2, n_task_exp=2, n_shared_exp=2, temp=1.5):
        super().__init__()
        self.embed = EmbeddingLayer()
        self.n_layers = n_layers

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            in_d = TOTAL_EMBED if i == 0 else EXPERT_DIM
            self.layers.append(PLELayer(in_d, EXPERT_DIM, n_task_exp, n_shared_exp, 2, DROPOUT, temp))

        self.tower_ctr = Tower(EXPERT_DIM, TOWER_HIDDEN, DROPOUT)
        self.tower_cvr = Tower(EXPERT_DIM, TOWER_HIDDEN, DROPOUT)

        # ESMM
        self.use_esmm = True

        # Feature mask reconstructor
        self.mask_recon = nn.Sequential(
            nn.Linear(EXPERT_DIM, EXPERT_DIM * 2),
            nn.ReLU(inplace=True),
            nn.Linear(EXPERT_DIM * 2, TOTAL_EMBED)
        )
        self.mask_ratio = 0.15

        # Expert utilization tracking
        self.gate_weight_accum = []

    def forward(self, sparse, dense, apply_mask=False):
        e = self.embed(sparse, dense)
        original_e = e.clone() if apply_mask and self.training else None

        # Feature masking
        mask_loss = torch.tensor(0.0, device=e.device)
        if apply_mask and self.training:
            mask = torch.bernoulli(torch.full_like(e, 1 - self.mask_ratio))
            e = e * mask

        # PLE layers — per-task propagation
        task_inputs = [e, e]  # both tasks start from same embedding
        all_gw = []
        for layer in self.layers:
            task_inputs, gw = layer(task_inputs)
            all_gw.append(gw)

        # Towers
        ctr_logit = self.tower_ctr(task_inputs[0])
        cvr_logit = self.tower_cvr(task_inputs[1])
        ctr_p = torch.sigmoid(ctr_logit)
        cvr_raw = torch.sigmoid(cvr_logit)

        # ESMM
        if self.use_esmm:
            cvr_p = ctr_p * cvr_raw
        else:
            cvr_p = cvr_raw

        # Mask reconstruction loss
        if apply_mask and self.training and original_e is not None:
            recon = self.mask_recon(task_inputs[0])
            mask_loss = F.mse_loss(recon, original_e)

        # Flatten gate weights for monitoring
        flat_gw = []
        for layer_gw in all_gw:
            flat_gw.extend(layer_gw)

        return ctr_p, cvr_p, flat_gw, mask_loss

    def anneal_temps(self, rate=0.95):
        for layer in self.layers:
            layer.anneal(rate)

    def get_temps(self):
        temps = []
        for layer in self.layers:
            temps.append([g.temp for g in layer.gates])
        return temps


# ============================================================
# Uncertainty Weight Loss
# ============================================================
class UncertaintyWeight(nn.Module):
    def __init__(self, n_tasks=2):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses):
        total = torch.tensor(0.0, device=self.log_sigma.device)
        weights = []
        for i, l in enumerate(losses):
            prec = torch.exp(-2 * self.log_sigma[i])
            total = total + prec * l + self.log_sigma[i]
            weights.append(prec.item())
        return total, weights

    def frozen_weights(self):
        with torch.no_grad():
            return torch.exp(-2 * self.log_sigma).cpu().tolist()


# ============================================================
# Gradient Conflict Detector
# ============================================================
class GradConflict:
    def __init__(self, ema_alpha=0.1):
        self.ema_alpha = ema_alpha
        self.ema_sim = 0.0
        self.history = []
        self.conflicts = 0
        self.total = 0

    def check(self, model, ctr_loss, cvr_loss):
        """Compute gradient cosine similarity on shared params."""
        shared = [p for n, p in model.named_parameters()
                  if p.requires_grad and ("shared" in n or "embed" in n or "dense_proj" in n)]
        if not shared:
            return 0.0, False

        g1 = torch.autograd.grad(ctr_loss, shared, retain_graph=True, allow_unused=True)
        g2 = torch.autograd.grad(cvr_loss, shared, retain_graph=True, allow_unused=True)

        flat1 = torch.cat([g.flatten() if g is not None else torch.zeros_like(p.flatten())
                          for g, p in zip(g1, shared)])
        flat2 = torch.cat([g.flatten() if g is not None else torch.zeros_like(p.flatten())
                          for g, p in zip(g2, shared)])

        cos = F.cosine_similarity(flat1.unsqueeze(0), flat2.unsqueeze(0)).item()
        self.history.append(cos)
        self.ema_sim = self.ema_alpha * cos + (1 - self.ema_alpha) * self.ema_sim
        self.total += 1

        # Adaptive threshold
        is_conflict = False
        if len(self.history) >= 10:
            h = np.array(self.history[-50:])
            threshold = h.mean() - 1.5 * h.std()
            is_conflict = cos < threshold
            if is_conflict:
                self.conflicts += 1

        return cos, is_conflict


# ============================================================
# Load balance loss
# ============================================================
def load_balance_loss(gate_weights_list):
    lb = torch.tensor(0.0)
    for gw in gate_weights_list:
        if gw.device != lb.device:
            lb = lb.to(gw.device)
        imp = gw.sum(0)
        lb = lb + imp.var() / (imp.mean() ** 2 + 1e-10)
    return lb


# ============================================================
# Training Loop
# ============================================================
def train_model(model, name, tr_dl, va_dl, te_dl, num_epochs=NUM_EPOCHS):
    logger.info("=" * 80)
    logger.info(f"Training: {name}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {n_params:,}")
    logger.info("=" * 80)

    model = model.to(DEVICE)
    uw = UncertaintyWeight(2).to(DEVICE)
    optimizer = torch.optim.Adam([
        {"params": model.parameters(), "lr": LR, "weight_decay": WEIGHT_DECAY},
        {"params": uw.parameters(), "lr": LR * 0.1, "weight_decay": 0},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=LR * 0.01)

    gc = GradConflict()
    history = []
    best_auc = 0.0
    best_state = None
    patience_counter = 0
    patience = 8

    is_ple = name.upper() == "PLE"

    for epoch in range(num_epochs):
        t0 = time.time()
        model.train()
        uw.train()
        ep_loss = 0.0; ep_ctr = 0.0; ep_cvr = 0.0; ep_mask = 0.0; ep_lb = 0.0
        n_batch = 0
        ep_cos_sims = []
        ep_conflicts = 0

        all_ctr_p, all_ctr_y, all_cvr_p, all_cvr_y = [], [], [], []

        for batch_idx, (sp, dn, cl, cv) in enumerate(tr_dl):
            sp, dn, cl, cv = sp.to(DEVICE), dn.to(DEVICE), cl.to(DEVICE), cv.to(DEVICE)

            if is_ple:
                ctr_p, cvr_p, gw_list, m_loss = model(sp, dn, apply_mask=True)
            else:
                ctr_p, cvr_p, gw_list = model(sp, dn)
                m_loss = torch.tensor(0.0, device=DEVICE)

            ctr_loss = F.binary_cross_entropy(ctr_p, cl)
            cvr_loss = F.binary_cross_entropy(cvr_p, cv)

            total, tw = uw([ctr_loss, cvr_loss])
            total = total + 0.1 * m_loss + 0.01 * load_balance_loss(gw_list)

            # Gradient conflict check every 20 batches
            if batch_idx % 20 == 0 and batch_idx > 0:
                try:
                    cs, is_conf = gc.check(model, ctr_loss, cvr_loss)
                    ep_cos_sims.append(cs)
                    if is_conf:
                        ep_conflicts += 1
                except:
                    pass

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Soft freeze if many conflicts
            if gc.total > 20 and gc.conflicts / max(gc.total, 1) > 0.5:
                for n, p in model.named_parameters():
                    if p.grad is not None and ("shared" in n or "embed" in n):
                        p.grad.data *= 0.1

            optimizer.step()

            ep_loss += total.item(); ep_ctr += ctr_loss.item(); ep_cvr += cvr_loss.item()
            ep_mask += m_loss.item(); ep_lb += 0; n_batch += 1

            with torch.no_grad():
                all_ctr_p.extend(ctr_p.cpu().numpy()); all_ctr_y.extend(cl.cpu().numpy())
                all_cvr_p.extend(cvr_p.cpu().numpy()); all_cvr_y.extend(cv.cpu().numpy())

        scheduler.step()

        # Temperature annealing for PLE
        if is_ple:
            model.anneal_temps(0.95)

        # Train metrics
        ep_loss /= n_batch; ep_ctr /= n_batch; ep_cvr /= n_batch; ep_mask /= n_batch
        try:
            tr_ctr_auc = roc_auc_score(all_ctr_y, all_ctr_p)
        except:
            tr_ctr_auc = 0.5
        try:
            tr_cvr_auc = roc_auc_score(all_cvr_y, all_cvr_p)
        except:
            tr_cvr_auc = 0.5

        # Validation
        model.eval()
        v_ctr_p, v_ctr_y, v_cvr_p, v_cvr_y = [], [], [], []
        v_loss = 0.0; v_n = 0
        with torch.no_grad():
            for sp, dn, cl, cv in va_dl:
                sp, dn, cl, cv = sp.to(DEVICE), dn.to(DEVICE), cl.to(DEVICE), cv.to(DEVICE)
                if is_ple:
                    cp, vp, gw, _ = model(sp, dn, apply_mask=False)
                else:
                    cp, vp, gw = model(sp, dn)
                ctr_l = F.binary_cross_entropy(cp, cl)
                cvr_l = F.binary_cross_entropy(vp, cv)
                v_loss += (ctr_l.item() + cvr_l.item()); v_n += 1
                v_ctr_p.extend(cp.cpu().numpy()); v_ctr_y.extend(cl.cpu().numpy())
                v_cvr_p.extend(vp.cpu().numpy()); v_cvr_y.extend(cv.cpu().numpy())

        v_loss /= max(v_n, 1)
        try:
            va_ctr_auc = roc_auc_score(v_ctr_y, v_ctr_p)
        except:
            va_ctr_auc = 0.5
        try:
            va_cvr_auc = roc_auc_score(v_cvr_y, v_cvr_p)
        except:
            va_cvr_auc = 0.5
        va_avg_auc = (va_ctr_auc + va_cvr_auc) / 2

        # Convergence diagnosis
        diagnosis_ctr = "WARMING_UP"
        diagnosis_cvr = "WARMING_UP"
        if len(history) >= 5:
            last5_ctr = [h["val_ctr_auc"] for h in history[-5:]]
            last5_cvr = [h["val_cvr_auc"] for h in history[-5:]]
            ctr_delta = max(last5_ctr) - min(last5_ctr)
            cvr_std = np.std(last5_cvr)
            ctr_slope = np.polyfit(range(5), last5_ctr, 1)[0]
            cvr_diffs = np.diff(last5_cvr)
            sign_changes = np.sum(np.abs(np.diff(np.sign(cvr_diffs))) > 0)

            if ctr_delta < 0.001:
                diagnosis_ctr = f"CONVERGED (delta={ctr_delta:.6f})"
            elif ctr_slope > 0:
                diagnosis_ctr = f"IMPROVING (slope={ctr_slope:.6f})"
            else:
                diagnosis_ctr = f"DEGRADING (slope={ctr_slope:.6f})"

            if sign_changes >= 3:
                diagnosis_cvr = f"OSCILLATING (sign_changes={sign_changes}, std={cvr_std:.6f})"
            elif cvr_std < 0.002:
                diagnosis_cvr = f"CONVERGED (std={cvr_std:.6f})"
            elif np.polyfit(range(5), last5_cvr, 1)[0] > 0:
                diagnosis_cvr = f"IMPROVING (slope={np.polyfit(range(5), last5_cvr, 1)[0]:.6f})"
            else:
                diagnosis_cvr = f"DEGRADING"

        # Expert utilization info
        gate_info = ""
        if is_ple:
            temps = model.get_temps()
            gate_info = f" | Temps: {[[round(t, 3) for t in lt] for lt in temps]}"

        avg_cos = np.mean(ep_cos_sims) if ep_cos_sims else 0.0
        conflict_ratio = gc.conflicts / max(gc.total, 1)

        uw_frozen = uw.frozen_weights()

        elapsed = time.time() - t0

        epoch_record = {
            "epoch": epoch + 1,
            "time_s": round(elapsed, 1),
            "train_loss": round(ep_loss, 6),
            "train_ctr_loss": round(ep_ctr, 6),
            "train_cvr_loss": round(ep_cvr, 6),
            "train_mask_loss": round(ep_mask, 6),
            "train_ctr_auc": round(tr_ctr_auc, 6),
            "train_cvr_auc": round(tr_cvr_auc, 6),
            "val_loss": round(v_loss, 6),
            "val_ctr_auc": round(va_ctr_auc, 6),
            "val_cvr_auc": round(va_cvr_auc, 6),
            "val_avg_auc": round(va_avg_auc, 6),
            "lr": round(optimizer.param_groups[0]["lr"], 8),
            "uw_weights": [round(w, 4) for w in uw_frozen],
            "uw_log_sigma": [round(s, 4) for s in uw.log_sigma.detach().cpu().tolist()],
            "avg_cos_sim": round(avg_cos, 4),
            "conflict_ratio": round(conflict_ratio, 4),
            "ep_conflicts": ep_conflicts,
            "diagnosis_ctr": diagnosis_ctr,
            "diagnosis_cvr": diagnosis_cvr,
        }
        history.append(epoch_record)

        logger.info(
            f"[{name}] Ep {epoch+1:2d}/{num_epochs} | {elapsed:.0f}s | "
            f"TrLoss={ep_loss:.4f} CTR={ep_ctr:.4f} CVR={ep_cvr:.4f} | "
            f"ValAUC: CTR={va_ctr_auc:.4f} CVR={va_cvr_auc:.4f} Avg={va_avg_auc:.4f} | "
            f"UW={[round(w,3) for w in uw_frozen]} | "
            f"CosSim={avg_cos:.4f} ConflR={conflict_ratio:.3f}{gate_info}"
        )
        logger.info(f"  Diagnosis: CTR={diagnosis_ctr} | CVR={diagnosis_cvr}")

        # Best model tracking
        if va_avg_auc > best_auc:
            best_auc = va_avg_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            logger.info(f"  ★ New best! AUC={best_auc:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

    # Test evaluation with best model
    if best_state:
        model.load_state_dict(best_state)
    model.to(DEVICE)
    model.eval()
    t_ctr_p, t_ctr_y, t_cvr_p, t_cvr_y = [], [], [], []
    with torch.no_grad():
        for sp, dn, cl, cv in te_dl:
            sp, dn, cl, cv = sp.to(DEVICE), dn.to(DEVICE), cl.to(DEVICE), cv.to(DEVICE)
            if is_ple:
                cp, vp, _, _ = model(sp, dn, apply_mask=False)
            else:
                cp, vp, _ = model(sp, dn)
            t_ctr_p.extend(cp.cpu().numpy()); t_ctr_y.extend(cl.cpu().numpy())
            t_cvr_p.extend(vp.cpu().numpy()); t_cvr_y.extend(cv.cpu().numpy())

    try:
        test_ctr_auc = roc_auc_score(t_ctr_y, t_ctr_p)
    except:
        test_ctr_auc = 0.5
    try:
        test_cvr_auc = roc_auc_score(t_cvr_y, t_cvr_p)
    except:
        test_cvr_auc = 0.5
    try:
        test_ctr_ll = log_loss(t_ctr_y, np.clip(t_ctr_p, 1e-7, 1 - 1e-7))
    except:
        test_ctr_ll = 999
    try:
        test_cvr_ll = log_loss(t_cvr_y, np.clip(t_cvr_p, 1e-7, 1 - 1e-7))
    except:
        test_cvr_ll = 999

    test_avg_auc = (test_ctr_auc + test_cvr_auc) / 2

    result = {
        "model": name,
        "n_params": n_params,
        "best_val_auc": round(best_auc, 6),
        "test_ctr_auc": round(test_ctr_auc, 6),
        "test_cvr_auc": round(test_cvr_auc, 6),
        "test_avg_auc": round(test_avg_auc, 6),
        "test_ctr_logloss": round(test_ctr_ll, 6),
        "test_cvr_logloss": round(test_cvr_ll, 6),
        "uw_frozen": [round(w, 4) for w in uw.frozen_weights()],
        "final_conflict_ratio": round(gc.conflicts / max(gc.total, 1), 4),
        "final_ema_cos_sim": round(gc.ema_sim, 4),
        "epochs_trained": len(history),
        "history": history,
    }

    logger.info("=" * 80)
    logger.info(f"[{name}] TEST RESULTS:")
    logger.info(f"  CTR AUC:     {test_ctr_auc:.6f}")
    logger.info(f"  CVR AUC:     {test_cvr_auc:.6f}")
    logger.info(f"  Avg AUC:     {test_avg_auc:.6f}")
    logger.info(f"  CTR LogLoss: {test_ctr_ll:.6f}")
    logger.info(f"  CVR LogLoss: {test_cvr_ll:.6f}")
    logger.info(f"  UW Frozen:   {uw.frozen_weights()}")
    logger.info(f"  Params:      {n_params:,}")
    logger.info("=" * 80)

    return result


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # Generate data
    sparse, dense, click, conv = generate_data()
    tr_dl, va_dl, te_dl = make_loaders(sparse, dense, click, conv)

    data_info = {
        "n_samples": NUM_SAMPLES,
        "ctr_rate": float(click.mean()),
        "cvr_rate": float(conv.mean()),
        "cvr_click_rate": float(conv[click == 1].mean()) if click.sum() > 0 else 0,
        "train_batches": len(tr_dl),
        "val_batches": len(va_dl),
        "test_batches": len(te_dl),
    }
    logger.info(f"Data Info: {json.dumps(data_info, indent=2)}")

    results = {}

    # 1. Train MMoE
    mmoe = MMoE(num_experts=6)
    results["MMoE"] = train_model(mmoe, "MMoE", tr_dl, va_dl, te_dl)

    # 2. Train CGC
    cgc = CGC(n_task_exp=3, n_shared_exp=2)
    results["CGC"] = train_model(cgc, "CGC", tr_dl, va_dl, te_dl)

    # 3. Train PLE (fixed)
    ple = PLE(n_layers=2, n_task_exp=2, n_shared_exp=2, temp=1.5)
    results["PLE"] = train_model(ple, "PLE", tr_dl, va_dl, te_dl)

    # ============================================================
    # Final Comparison Report
    # ============================================================
    logger.info("\n" + "=" * 100)
    logger.info("FINAL COMPARISON REPORT")
    logger.info("=" * 100)

    header = f"{'Model':<8} {'Params':>10} {'CTR AUC':>10} {'CVR AUC':>10} {'Avg AUC':>10} {'CTR LL':>10} {'CVR LL':>10} {'UW Frozen':>20}"
    logger.info(header)
    logger.info("-" * 100)

    for name in ["MMoE", "CGC", "PLE"]:
        r = results[name]
        logger.info(
            f"{name:<8} {r['n_params']:>10,} {r['test_ctr_auc']:>10.6f} {r['test_cvr_auc']:>10.6f} "
            f"{r['test_avg_auc']:>10.6f} {r['test_ctr_logloss']:>10.6f} {r['test_cvr_logloss']:>10.6f} "
            f"{str(r['uw_frozen']):>20}"
        )

    winner = max(results.keys(), key=lambda k: results[k]["test_avg_auc"])
    logger.info(f"\n🏆 Winner: {winner} (Avg AUC = {results[winner]['test_avg_auc']:.6f})")

    logger.info("\n" + "=" * 100)
    logger.info("WHY CGC > PLE (Previous Run) - ROOT CAUSE ANALYSIS")
    logger.info("=" * 100)
    logger.info("""
1. OVER-PARAMETERIZATION ON SYNTHETIC DATA:
   PLE(1.06M params) vs CGC(425K params) → 2.5x parameters on weak-signal data.
   3 extraction layers compound gate noise → overfitting.

2. INFORMATION BOTTLENECK:
   Previous PLE fed mean(task_outputs) to next layer → lost task-specific signal.
   Fix: per-task input propagation.

3. ESMM + MULTI-LAYER = LONG GRADIENT PATH:
   CVR = CTR * CVR_given_click through 3 layers → gradient vanishing.
   Fix: reduced to 2 layers + stronger initial signal.

4. TEMPERATURE OVER-REGULARIZATION:
   3 layers × T=2.0 → gates too uniform for too long.
   Fix: T=1.5 + faster decay (0.95 vs 0.995).

5. THIS RUN'S FIXES:
   - 2 layers instead of 3
   - Per-task input propagation (not mean)
   - Temp 1.5 + decay 0.95
   - Stronger data signal (more feature-label correlation)
   - Expert dim 64 (right-sized for data complexity)
""")

    # Save all results
    save_results = {}
    for name in results:
        r = dict(results[name])
        r.pop("history", None)  # remove history for summary file
        save_results[name] = r

    with open(os.path.join(LOG_DIR, "final_comparison.json"), "w") as f:
        json.dump({"data_info": data_info, "results": save_results, "winner": winner}, f, indent=2)

    # Save full history per model
    for name in results:
        with open(os.path.join(LOG_DIR, f"{name.lower()}_history.json"), "w") as f:
            json.dump(results[name]["history"], f, indent=2)

    logger.info(f"\nAll logs saved to: {LOG_DIR}")
    logger.info(f"Log file: {log_file}")
    print(f"\n✅ DONE. Results in {LOG_DIR}")
