import json, os

results = {}
for m in ['ple', 'mmoe', 'cgc']:
    r = {}
    hist_f = f'logs/{m}/training_history.json'
    test_f = f'logs/{m}/test_results.json'
    if os.path.exists(hist_f):
        with open(hist_f) as f:
            r['history'] = json.load(f)
    if os.path.exists(test_f):
        with open(test_f) as f:
            r['test'] = json.load(f)
    results[m] = r

# Normalize test results
comparison = {}
for m, r in results.items():
    t = r.get('test', {})
    comparison[m] = {
        'ctr_auc': t.get('ctr_auc', 0),
        'cvr_auc': t.get('cvr_auc', 0),
        'avg_auc': t.get('avg_auc', t.get('total_auc', 0))
    }

best_model = max(comparison, key=lambda x: comparison[x]['avg_auc'])
final = {'comparison': comparison, 'winner': best_model}
with open('logs/comparison_results.json', 'w') as f:
    json.dump(final, f, indent=2)

print('=' * 70)
print('MODEL COMPARISON RESULTS')
print('=' * 70)
hdr = f"{'Model':<10} {'CTR AUC':>10} {'CVR AUC':>10} {'Avg AUC':>10}"
print(hdr)
print('-' * len(hdr))
for m, r in comparison.items():
    marker = " <-- BEST" if m == best_model else ""
    print(f"{m.upper():<10} {r['ctr_auc']:>10.6f} {r['cvr_auc']:>10.6f} {r['avg_auc']:>10.6f}{marker}")
print()

# PLE detailed log
print('PLE Training Log (detailed):')
for e in results['ple']['history']:
    tr = e.get('train', e)
    va = e.get('val', e)
    ep = e.get('epoch', 0)
    
    ctr_auc = va.get('ctr_auc', tr.get('ctr_auc', 0))
    cvr_auc = va.get('cvr_auc', tr.get('cvr_auc', 0))
    avg_auc = va.get('total_auc', va.get('avg_auc', 0))
    
    tw_ctr = tr.get('task_weight_ctr', e.get('task_weight_ctr', 0))
    tw_cvr = tr.get('task_weight_cvr', e.get('task_weight_cvr', 0))
    ls_ctr = tr.get('log_sigma_ctr', e.get('log_sigma_ctr', 0))
    ls_cvr = tr.get('log_sigma_cvr', e.get('log_sigma_cvr', 0))
    
    temps = e.get('gate_temperatures', e.get('gate_temps'))
    t_str = f"{temps[0][0]:.2f}" if temps else "N/A"
    
    collapse = e.get('expert_collapse', {})
    c_str = "COLLAPSED!" if collapse.get('collapsed') else "OK"
    
    diag = e.get('early_stopping', {}).get('diagnosis', {})
    ctr_diag = diag.get('ctr', '')
    cvr_diag = diag.get('cvr', '')
    
    conflict = e.get('conflict_detector', {})
    cos_sim = conflict.get('ema_cos_sim', 0)
    
    print(f"  Epoch {ep:2d} | CTR AUC={ctr_auc:.4f} CVR AUC={cvr_auc:.4f} Avg={avg_auc:.4f} | "
          f"W=[{tw_ctr:.3f},{tw_cvr:.3f}] sigma=[{ls_ctr:.4f},{ls_cvr:.4f}] | "
          f"T={t_str} | Expert={c_str} | CosSim={cos_sim:.4f}")
    if ctr_diag:
        print(f"         CTR: {ctr_diag}")
        print(f"         CVR: {cvr_diag}")

# Frozen weights
fw_f = 'logs/ple/frozen_weights.json'
if os.path.exists(fw_f):
    with open(fw_f) as f:
        fw = json.load(f)
    print(f"\nFrozen Uncertainty Weights (zero inference overhead): {fw}")
