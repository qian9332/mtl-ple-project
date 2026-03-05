#!/usr/bin/env python3
"""Train models one epoch at a time to avoid 90s gateway timeout."""
import os, sys, json, time, pickle, numpy as np, logging
from datetime import datetime
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import MinMaxScaler

PROJ="/home/user/mtl-project"
LOGDIR=os.path.join(PROJ,"logs","gpu_run")
STATEDIR=os.path.join(LOGDIR,"state")
os.makedirs(STATEDIR,exist_ok=True)

logging.basicConfig(level=logging.INFO,format="%(asctime)s %(message)s",
                    handlers=[logging.FileHandler(os.path.join(LOGDIR,"train.log"),"a"),logging.StreamHandler()])
log=logging.getLogger()

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
SPARSE_DIMS=[1000,500,50,20,100,5,30,200,100,24,7,12,50,50,30,30,10,20,15,8]
NS,ND,ED=20,10,8; TE=(NS+ND)*ED; EX=64; TH=64; DR=0.15
N=500000; BS=4096; LR=0.001; N_EP=20; PAT=8

class DS(Dataset):
    def __init__(s,sp,dn,cl,cv):
        s.sp=torch.LongTensor(sp);s.dn=torch.FloatTensor(dn);s.cl=torch.FloatTensor(cl);s.cv=torch.FloatTensor(cv)
    def __len__(s):return len(s.cl)
    def __getitem__(s,i):return s.sp[i],s.dn[i],s.cl[i],s.cv[i]

def gen():
    dp=os.path.join(STATEDIR,"data.pkl")
    if os.path.exists(dp):
        with open(dp,"rb") as f: return pickle.load(f)
    np.random.seed(42)
    sp=np.zeros((N,NS),dtype=np.int64)
    for i,d in enumerate(SPARSE_DIMS):sp[:,i]=np.random.randint(0,d,N)
    dn=MinMaxScaler().fit_transform(np.random.randn(N,ND).astype(np.float32))
    ua=np.random.randn(1000)*1.5;iq=np.random.randn(500)*1.5;cb=np.random.randn(50)*0.5
    cs=ua[sp[:,0]]*0.35+iq[sp[:,1]]*0.30+cb[sp[:,2]]*0.15+dn[:,0]*0.25+dn[:,1]*0.15+dn[:,2]*0.10+np.random.randn(N)*0.4
    cp=1/(1+np.exp(-(cs-np.percentile(cs,75))))
    cl=(np.random.random(N)<cp).astype(np.float32)
    cvs=ua[sp[:,0]]*0.20+iq[sp[:,1]]*0.45+cb[sp[:,2]]*0.20+dn[:,3]*0.20+dn[:,4]*0.15+np.random.randn(N)*0.3
    cvp=1/(1+np.exp(-(cvs-np.percentile(cvs,90))))
    cv=(cl*(np.random.random(N)<cvp)).astype(np.float32)
    data=(sp,dn.astype(np.float32),cl,cv)
    with open(dp,"wb") as f:pickle.dump(data,f)
    return data

def mkdl(sp,dn,cl,cv):
    ds=DS(sp,dn,cl,cv);n_=len(ds);tr=int(n_*0.7);va=int(n_*0.15);te=n_-tr-va
    trs,vas,tes=random_split(ds,[tr,va,te],generator=torch.Generator().manual_seed(42))
    return(DataLoader(trs,BS,True,num_workers=0,drop_last=True),DataLoader(vas,BS*2,False,num_workers=0),DataLoader(tes,BS*2,False,num_workers=0))

class Exp(nn.Module):
    def __init__(s,i,o):super().__init__();s.net=nn.Sequential(nn.Linear(i,o),nn.BatchNorm1d(o),nn.ReLU(True),nn.Dropout(DR),nn.Linear(o,o),nn.BatchNorm1d(o),nn.ReLU(True))
    def forward(s,x):return s.net(x)
class Gt(nn.Module):
    def __init__(s,i,n,t=1.0):super().__init__();s.fc=nn.Linear(i,n,bias=False);s.t=t
    def forward(s,x):return F.softmax(s.fc(x)/max(s.t,0.1),dim=-1)
    def anneal(s,r=0.95):s.t=max(0.1,s.t*r)
class Emb(nn.Module):
    def __init__(s):super().__init__();s.es=nn.ModuleList([nn.Embedding(d,ED) for d in SPARSE_DIMS]);s.dp=nn.Linear(ND,ND*ED)
    def forward(s,sp,dn):return torch.cat([torch.cat([s.es[i](sp[:,i]) for i in range(NS)],1),s.dp(dn)],1)
class Tow(nn.Module):
    def __init__(s,i,h):super().__init__();s.net=nn.Sequential(nn.Linear(i,h),nn.BatchNorm1d(h),nn.ReLU(True),nn.Dropout(DR),nn.Linear(h,h//2),nn.ReLU(True),nn.Linear(h//2,1))
    def forward(s,x):return s.net(x).squeeze(-1)

class MMoE(nn.Module):
    def __init__(s):super().__init__();s.emb=Emb();s.exps=nn.ModuleList([Exp(TE,EX) for _ in range(6)]);s.gc=Gt(TE,6);s.gv=Gt(TE,6);s.tc=Tow(EX,TH);s.tv=Tow(EX,TH)
    def forward(s,sp,dn):
        e=s.emb(sp,dn);es=torch.stack([x(e) for x in s.exps],1)
        return torch.sigmoid(s.tc(torch.bmm(s.gc(e).unsqueeze(1),es).squeeze(1))),torch.sigmoid(s.tv(torch.bmm(s.gv(e).unsqueeze(1),es).squeeze(1))),[s.gc(e),s.gv(e)]

class CGC(nn.Module):
    def __init__(s):
        super().__init__();s.emb=Emb()
        s.ce=nn.ModuleList([Exp(TE,EX) for _ in range(3)]);s.ve=nn.ModuleList([Exp(TE,EX) for _ in range(3)]);s.se=nn.ModuleList([Exp(TE,EX) for _ in range(2)])
        s.gc=Gt(TE,5);s.gv=Gt(TE,5);s.tc=Tow(EX,TH);s.tv=Tow(EX,TH)
    def forward(s,sp,dn):
        e=s.emb(sp,dn);so=[x(e) for x in s.se]
        return torch.sigmoid(s.tc(torch.bmm(s.gc(e).unsqueeze(1),torch.stack([x(e) for x in s.ce]+so,1)).squeeze(1))),torch.sigmoid(s.tv(torch.bmm(s.gv(e).unsqueeze(1),torch.stack([x(e) for x in s.ve]+so,1)).squeeze(1))),[s.gc(e),s.gv(e)]

class PLE(nn.Module):
    def __init__(s):
        super().__init__();s.emb=Emb()
        # Layer 1
        s.t0a=nn.ModuleList([Exp(TE,EX) for _ in range(2)]);s.t1a=nn.ModuleList([Exp(TE,EX) for _ in range(2)])
        s.sa=nn.ModuleList([Exp(TE,EX) for _ in range(2)]);s.g0a=Gt(TE,4,1.5);s.g1a=Gt(TE,4,1.5)
        # Layer 2
        s.t0b=nn.ModuleList([Exp(EX,EX) for _ in range(2)]);s.t1b=nn.ModuleList([Exp(EX,EX) for _ in range(2)])
        s.sb=nn.ModuleList([Exp(EX,EX) for _ in range(2)]);s.g0b=Gt(EX,4,1.5);s.g1b=Gt(EX,4,1.5)
        s.tc=Tow(EX,TH);s.tv=Tow(EX,TH)
        s.mr=nn.Sequential(nn.Linear(EX,EX*2),nn.ReLU(True),nn.Linear(EX*2,TE))
        s.gates=[s.g0a,s.g1a,s.g0b,s.g1b]
    def forward(s,sp,dn,mask=False):
        e=s.emb(sp,dn);oe=e.clone() if mask and s.training else None
        ml=torch.tensor(0.0,device=e.device)
        if mask and s.training:e=e*torch.bernoulli(torch.full_like(e,0.85))
        # L1
        so1=[x(e) for x in s.sa]
        o0=torch.bmm(s.g0a(e).unsqueeze(1),torch.stack([x(e) for x in s.t0a]+so1,1)).squeeze(1)
        o1=torch.bmm(s.g1a(e).unsqueeze(1),torch.stack([x(e) for x in s.t1a]+so1,1)).squeeze(1)
        # L2
        so2=[x(o0) for x in s.sb]
        o0=torch.bmm(s.g0b(o0).unsqueeze(1),torch.stack([x(o0) for x in s.t0b]+so2,1)).squeeze(1)
        o1=torch.bmm(s.g1b(o1).unsqueeze(1),torch.stack([x(o1) for x in s.t1b]+so2,1)).squeeze(1)
        cp=torch.sigmoid(s.tc(o0));vr=torch.sigmoid(s.tv(o1))
        vp=cp*vr  # ESMM
        if mask and s.training and oe is not None:ml=F.mse_loss(s.mr(o0),oe)
        gws=[s.g0a(e),s.g1a(e),s.g0b(o0),s.g1b(o1)]
        return cp,vp,gws,ml
    def anneal(s,r=0.95):
        for g in s.gates:g.anneal(r)
    def temps(s):return[[g.t for g in s.gates[:2]],[g.t for g in s.gates[2:]]]

class UW(nn.Module):
    def __init__(s):super().__init__();s.ls=nn.Parameter(torch.zeros(2))
    def forward(s,losses):
        t=torch.tensor(0.0,device=s.ls.device);ws=[]
        for i,l in enumerate(losses):p=torch.exp(-2*s.ls[i]);t=t+p*l+s.ls[i];ws.append(p.item())
        return t,ws
    def frozen(s):
        with torch.no_grad():return torch.exp(-2*s.ls).cpu().tolist()

def train_epoch(model,uw,opt,tr,is_ple):
    model.train();uw.train()
    el=ec=ev=em=0;nb=0;acp=[];acy=[];avp=[];avy=[]
    for sp,dn,cl,cv in tr:
        sp,dn,cl,cv=sp.to(DEVICE),dn.to(DEVICE),cl.to(DEVICE),cv.to(DEVICE)
        if is_ple:cp,vp,gw,ml=model(sp,dn,mask=True)
        else:cp,vp,gw=model(sp,dn);ml=torch.tensor(0.0,device=DEVICE)
        cl_=F.binary_cross_entropy(cp,cl);vl=F.binary_cross_entropy(vp,cv)
        tot,tw=uw([cl_,vl])
        lb=torch.tensor(0.0,device=DEVICE)
        for g in gw:imp=g.sum(0);lb=lb+imp.var()/(imp.mean()**2+1e-10)
        tot=tot+0.1*ml+0.01*lb
        opt.zero_grad();tot.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        el+=tot.item();ec+=cl_.item();ev+=vl.item();em+=ml.item();nb+=1
        with torch.no_grad():acp.extend(cp.cpu().numpy());acy.extend(cl.cpu().numpy());avp.extend(vp.cpu().numpy());avy.extend(cv.cpu().numpy())
    el/=nb;ec/=nb;ev/=nb;em/=nb
    try:tca=roc_auc_score(acy,acp)
    except:tca=0.5
    try:tva=roc_auc_score(avy,avp)
    except:tva=0.5
    return{"loss":el,"ctr_l":ec,"cvr_l":ev,"mask_l":em,"tr_ctr":tca,"tr_cvr":tva}

def evaluate(model,dl,is_ple):
    model.eval();cp_=[];cy_=[];vp_=[];vy_=[];vl=0;vn=0
    with torch.no_grad():
        for sp,dn,cl,cv in dl:
            sp,dn,cl,cv=sp.to(DEVICE),dn.to(DEVICE),cl.to(DEVICE),cv.to(DEVICE)
            if is_ple:cp,vp,_,_=model(sp,dn,mask=False)
            else:cp,vp,_=model(sp,dn)
            vl+=(F.binary_cross_entropy(cp,cl).item()+F.binary_cross_entropy(vp,cv).item());vn+=1
            cp_.extend(cp.cpu().numpy());cy_.extend(cl.cpu().numpy());vp_.extend(vp.cpu().numpy());vy_.extend(cv.cpu().numpy())
    vl/=max(vn,1)
    try:ca=roc_auc_score(cy_,cp_)
    except:ca=0.5
    try:va=roc_auc_score(vy_,vp_)
    except:va=0.5
    try:cll=log_loss(cy_,np.clip(cp_,1e-7,1-1e-7))
    except:cll=999
    try:vll=log_loss(vy_,np.clip(vp_,1e-7,1-1e-7))
    except:vll=999
    return{"vl":vl,"ctr_auc":ca,"cvr_auc":va,"avg_auc":(ca+va)/2,"ctr_ll":cll,"cvr_ll":vll}

def gc_check(model,tr,is_ple):
    """One-batch gradient conflict check."""
    model.eval()
    try:
        for sp,dn,cl,cv in tr:
            sp,dn,cl,cv=sp.to(DEVICE),dn.to(DEVICE),cl.to(DEVICE),cv.to(DEVICE)
            if is_ple:cp,vp,_,_=model(sp,dn,mask=False)
            else:cp,vp,_=model(sp,dn)
            cl_=F.binary_cross_entropy(cp,cl);vl=F.binary_cross_entropy(vp,cv)
            sh=[p for n,p in model.named_parameters() if p.requires_grad and ("se" in n or "es" in n or "emb" in n or "dp" in n)]
            if not sh:return 0.0
            g1=torch.autograd.grad(cl_,sh,retain_graph=True,allow_unused=True)
            g2=torch.autograd.grad(vl,sh,retain_graph=True,allow_unused=True)
            f1=torch.cat([g.flatten() if g is not None else torch.zeros_like(p.flatten()) for g,p in zip(g1,sh)])
            f2=torch.cat([g.flatten() if g is not None else torch.zeros_like(p.flatten()) for g,p in zip(g2,sh)])
            return F.cosine_similarity(f1.unsqueeze(0),f2.unsqueeze(0)).item()
    except:return 0.0
    return 0.0

def run_model(name,model,tr,va,te,is_ple):
    sf=os.path.join(STATEDIR,f"{name}_state.pkl")

    # Check if already done
    rf=os.path.join(LOGDIR,f"{name}_result.json")
    if os.path.exists(rf):
        with open(rf) as f:r=json.load(f)
        if r.get("complete"):
            log.info(f"[{name}] Already complete. Skipping.")
            return r

    # Load or init state
    model.to(DEVICE)
    uw=UW().to(DEVICE)
    opt=torch.optim.Adam([{"params":model.parameters(),"lr":LR,"weight_decay":1e-5},{"params":uw.parameters(),"lr":LR*0.1}])
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=N_EP,eta_min=LR*0.01)

    start_ep=0;hist=[];best_auc=0;best_st=None;pat_c=0

    if os.path.exists(sf):
        st=torch.load(sf,map_location=DEVICE,weights_only=False)
        model.load_state_dict(st["model"]);uw.load_state_dict(st["uw"]);opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start_ep=st["epoch"];hist=st["hist"];best_auc=st["best_auc"];pat_c=st["pat_c"]
        if st.get("best_st"):best_st=st["best_st"]
        log.info(f"[{name}] Resuming from epoch {start_ep}")

    for ep in range(start_ep,N_EP):
        t0=time.time()
        tm=train_epoch(model,uw,opt,tr,is_ple)
        sched.step()
        if is_ple and hasattr(model,'anneal'):model.anneal(0.95)
        vm=evaluate(model,va,is_ple)
        cs=gc_check(model,tr,is_ple)
        t_=time.time()-t0

        # Diagnosis
        dctr=dcvr="WARM"
        if len(hist)>=5:
            l5c=[h["val"]["ctr_auc"] for h in hist[-5:]];l5v=[h["val"]["cvr_auc"] for h in hist[-5:]]
            cd=max(l5c)-min(l5c);sl=np.polyfit(range(5),l5c,1)[0]
            vs=np.std(l5v);vd=np.diff(l5v);sc=int(np.sum(np.abs(np.diff(np.sign(vd)))>0))
            dctr=f"CONV(d={cd:.4f})" if cd<0.001 else (f"IMP(s={sl:.5f})" if sl>0 else f"DEG(s={sl:.5f})")
            dcvr=f"OSC(sc={sc},s={vs:.4f})" if sc>=3 else (f"CONV(s={vs:.4f})" if vs<0.002 else "IMP" if np.polyfit(range(5),l5v,1)[0]>0 else "DEG")

        extra=""
        if is_ple and hasattr(model,'temps'):extra=f" T={[[round(t,2) for t in l] for l in model.temps()]}"

        rec={"ep":ep+1,"t":round(t_,1),"train":tm,"val":vm,"uw":uw.frozen(),"cos":round(cs,4),"dx_ctr":dctr,"dx_cvr":dcvr}
        hist.append(rec)

        log.info(f"[{name}] Ep{ep+1:2d} {t_:.0f}s Loss={tm['loss']:.4f} CTR_L={tm['ctr_l']:.4f} CVR_L={tm['cvr_l']:.4f} | "
                 f"Val: CTR={vm['ctr_auc']:.4f} CVR={vm['cvr_auc']:.4f} Avg={vm['avg_auc']:.4f} | "
                 f"UW={[round(w,3) for w in uw.frozen()]} CosSim={cs:.4f} | {dctr} {dcvr}{extra}")

        if vm["avg_auc"]>best_auc:
            best_auc=vm["avg_auc"];best_st={k:v.cpu().clone() for k,v in model.state_dict().items()};pat_c=0
            log.info(f"  ★ Best={best_auc:.6f}")
        else:
            pat_c+=1
            if pat_c>=PAT:log.info(f"  EarlyStop ep{ep+1}");ep+=1;break

        # Save state for resume
        torch.save({"model":model.state_dict(),"uw":uw.state_dict(),"opt":opt.state_dict(),
                     "sched":sched.state_dict(),"epoch":ep+1,"hist":hist,"best_auc":best_auc,
                     "pat_c":pat_c,"best_st":best_st},sf)

    # Test
    if best_st:model.load_state_dict(best_st)
    model.to(DEVICE)
    tm_=evaluate(model,te,is_ple)
    res={"model":name,"params":sum(p.numel() for p in model.parameters() if p.requires_grad),
         "test":tm_,"uw_frozen":uw.frozen(),"epochs":len(hist),"best_val_auc":best_auc,
         "gc_final":hist[-1]["cos"] if hist else 0,"history":hist,"complete":True}
    log.info("="*80)
    log.info(f"[{name}] TEST: CTR={tm_['ctr_auc']:.6f} CVR={tm_['cvr_auc']:.6f} Avg={tm_['avg_auc']:.6f} CTR_LL={tm_['ctr_ll']:.6f} CVR_LL={tm_['cvr_ll']:.6f}")
    log.info("="*80)
    with open(rf,"w") as f:json.dump(res,f,indent=2)
    return res

if __name__=="__main__":
    cmd=sys.argv[1] if len(sys.argv)>1 else "mmoe"
    sp,dn,cl,cv=gen()
    log.info(f"Data: CTR={cl.mean():.4f} CVR={cv.mean():.4f}")
    tr,va,te=mkdl(sp,dn,cl,cv)

    if cmd=="mmoe":
        run_model("mmoe",MMoE(),tr,va,te,False)
    elif cmd=="cgc":
        run_model("cgc",CGC(),tr,va,te,False)
    elif cmd=="ple":
        run_model("ple",PLE(),tr,va,te,True)
    elif cmd=="report":
        # Generate comparison report
        results={}
        for n in ["mmoe","cgc","ple"]:
            fp=os.path.join(LOGDIR,f"{n}_result.json")
            if os.path.exists(fp):
                with open(fp) as f:results[n]=json.load(f)
        if len(results)==3:
            log.info("\n"+"="*100)
            log.info("FINAL COMPARISON (500K samples, 20 epochs, CPU)")
            log.info("="*100)
            log.info(f"{'Model':<6} {'Params':>8} {'CTR_AUC':>10} {'CVR_AUC':>10} {'Avg_AUC':>10} {'CTR_LL':>10} {'CVR_LL':>10} {'UW':>20} {'Epochs':>6}")
            log.info("-"*100)
            for n in ["mmoe","cgc","ple"]:
                r=results[n];t=r["test"]
                log.info(f"{n:<6} {r['params']:>8,} {t['ctr_auc']:>10.6f} {t['cvr_auc']:>10.6f} {t['avg_auc']:>10.6f} {t['ctr_ll']:>10.6f} {t['cvr_ll']:>10.6f} {str([round(w,3) for w in r['uw_frozen']]):>20} {r['epochs']:>6}")
            w=max(results,key=lambda k:results[k]["test"]["avg_auc"])
            log.info(f"\n🏆 Winner: {w.upper()} (Avg AUC={results[w]['test']['avg_auc']:.6f})")
            with open(os.path.join(LOGDIR,"comparison.json"),"w") as f:
                json.dump({"results":{n:{k:v for k,v in results[n].items() if k!="history"} for n in results},"winner":w},f,indent=2)
    print("DONE")
