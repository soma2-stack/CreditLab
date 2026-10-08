"""HARD_V2 four-cell GPU benchmark. Default action is preflight only."""
import argparse, hashlib, json, math, random, subprocess, sys, time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from creditlab.delayed_task import make_split, generate_batch
from creditlab.models import ResidualTanhRNN

SEEDS=(17,29)
VARIANTS=("original","fixed")
def split(n,delay,seed,index,variant):
    if variant=="original":
        return make_split(n,delay,"HARD_V2",run_seed=seed,split_index=index,
                          config={"easy_noise_std":0.3,"hard_noise_std":1.0,"competitor_rate":0.4})
    # Same HARD_V2 competitor semantics, explicitly corrected noise strength.
    # Keep the original random draw ordering to isolate the noise correction.
    combined=(seed*1_000_003+1000*9973+delay*31+index)%(2**31-1)
    return generate_batch(n,delay,"HARD_V2",seed=combined,
                          distractor_noise_std=1.0,competitor_rate=0.4)
def temperature():
    try:
        p=subprocess.run(["nvidia-smi","--query-gpu=temperature.gpu","--format=csv,noheader,nounits"],
                         capture_output=True,text=True,timeout=5,check=True)
        return float(p.stdout.strip().splitlines()[0].strip())
    except (OSError,ValueError,IndexError,subprocess.SubprocessError):
        return None
def safety():
    t=temperature()
    if t is not None and t>86: raise RuntimeError(f"GPU temperature {t} C exceeds 86 C")
    return t
def accuracy(model,x,y,batch=64):
    correct=0
    with torch.no_grad():
        for i in range(0,len(y),batch):
            logits,_=model(x[i:i+batch])
            correct+=int(((logits[:,-1,0]>=0)==(y[i:i+batch]>0)).sum())
    return correct/len(y)
def run(seed,variant,args):
    torch.manual_seed(seed);random.seed(seed)
    tx,ty=split(args.train_samples,args.delay,seed,0,variant)
    vx,vy=split(args.val_samples,args.delay,seed,1,variant)
    ex,ey=split(args.test_samples,args.delay,seed,2,variant)
    digest=lambda x:hashlib.sha256(x.numpy().tobytes()).hexdigest()
    record={"seed":seed,"variant":variant,"delay":args.delay,"noise_std":0.3 if variant=="original" else 1.0,
            "train_sha256":digest(tx),"val_sha256":digest(vx),"test_sha256":digest(ex),
            "updates":args.updates,"gpu":torch.cuda.get_device_name(0)}
    model=ResidualTanhRNN(3,args.hidden,1).cuda()
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    tx,ty=tx.cuda(),ty.cuda();vx,vy=vx.cuda(),vy.cuda();ex,ey=ex.cuda(),ey.cuda()
    rng=random.Random(seed*7919+args.delay)
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter();temps=[]
    for step in range(args.updates):
        t=safety()
        if t is not None:temps.append(t)
        idx=[rng.randrange(len(ty)) for _ in range(args.batch)]
        logits,_=model(tx[idx])
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits[:,-1,0],(ty[idx]+1)/2)
        if not bool(torch.isfinite(loss)):raise FloatingPointError("nonfinite loss")
        opt.zero_grad();loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
        if not bool(torch.isfinite(norm)):raise FloatingPointError("nonfinite gradient")
        opt.step()
    torch.cuda.synchronize()
    record.update(status="ok",train_seconds=round(time.perf_counter()-start,3),
                  val_accuracy=accuracy(model,vx,vy),test_accuracy=accuracy(model,ex,ey),
                  peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_vram_reserved_bytes=torch.cuda.max_memory_reserved(),
                  max_temp_c=max(temps) if temps else None)
    return record
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run",action="store_true",help="Explicitly enable four GPU training runs")
    p.add_argument("--output",default="results/king_of_hill_four_run")
    p.add_argument("--delay",type=int,default=128)
    p.add_argument("--hidden",type=int,default=32)
    p.add_argument("--updates",type=int,default=100)
    p.add_argument("--batch",type=int,default=32)
    p.add_argument("--train-samples",type=int,default=512)
    p.add_argument("--val-samples",type=int,default=128)
    p.add_argument("--test-samples",type=int,default=128)
    p.add_argument("--lr",type=float,default=0.001)
    a=p.parse_args()
    if min(a.delay,a.hidden,a.updates,a.batch,a.train_samples,a.val_samples,a.test_samples)<1: p.error("positive values required")
    if not math.isfinite(a.lr) or a.lr<=0:p.error("lr must be positive and finite")
    out=ROOT/a.output
    if out.exists():p.error(f"refusing to overwrite existing output: {out}")
    plan=[{"variant":v,"seed":s} for v in VARIANTS for s in SEEDS]
    if not a.run:
        print(json.dumps({"status":"preflight_only","training_started":False,"runs":plan,
                          "cuda_available":torch.cuda.is_available()},indent=2));return
    if not torch.cuda.is_available():p.error("CUDA GPU required; CPU fallback prohibited")
    out.mkdir(parents=True,exist_ok=False)
    for item in plan:
        target=out/f"{item['variant']}_seed{item['seed']}.json"
        try:r=run(item["seed"],item["variant"],a)
        except Exception as e:
            r={**item,"status":"failure","error":f"{type(e).__name__}: {e}"}
            target.write_text(json.dumps(r,indent=2))
            raise
        target.write_text(json.dumps(r,indent=2))
        print(json.dumps(r),flush=True)
if __name__=="__main__":main()
