"""Reproduce the setup of enhancing.pdf Table I; also compare the existing 13x13 grid.

A validation harness, not an example API wrapper. The paper's random coefficient
realization and initial weights are unavailable; save this realization and every
native CUDA initialization so each run can be replayed exactly.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cpde import Data, Network, Session, table

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--server', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
requests = ['xxx', 'xxy', 'xyy', 'yyy']
schema = table(requests, variables='xy')
coefficients = np.random.default_rng(2026).uniform(-1, 1, (4, 10, 10))
sets = {}
for side in [13, 27, 155]:
    x, y = np.meshgrid(np.linspace(-1, 1, side), np.linspace(-1, 1, side), indexing='ij')
    points = np.column_stack((x.ravel(), y.ravel())).astype(np.float32)
    x, y = points.astype(float).transpose()
    target = np.zeros((len(schema), 1, len(points)))
    for row, (_, dx, dy) in enumerate(schema):
        for n in range(1, 11):
            for k in range(1, 11):
                a, b, c, d = coefficients[:, n-1, k-1]*n**int(dx)*k**int(dy)/(n*k)
                sx, cx = np.sin(n*x+dx*np.pi/2), np.cos(n*x+dx*np.pi/2)
                sy, cy = np.sin(k*y+dy*np.pi/2), np.cos(k*y+dy*np.pi/2)
                target[row, 0] += a*sx*sy+b*sx*cy+c*cx*sy+d*cx*cy
    sets[side] = (points, target)
    np.savez_compressed(out/f'data-{side}.npz', points=points, target=target,
                        schema=schema, coefficients=coefficients)

results = json.loads((out/'results.json').read_text()) if (out/'results.json').exists() else []
metadata = dict(architecture=[2,128,128,128,128,1], derivative_requests=requests,
    epochs=1000, initial_step=2e-4, optimizer='rprop', periodic_restoration=False,
    coefficient_seed=2026, initialization='native CUDA random initialization, paired between grids',
    test_grid=[155,155], normalization='training target population std for cost; both train-std and own-set-std metrics saved',
    paper_order_train_percent=[.07,.18,.77,1.9], paper_order_test_percent=[.06,.17,.61,2.8],
    paper_total_train_percent=4., paper_total_test_percent=5.6,
    python=sys.version, platform=platform.platform(), numpy=np.__version__,
    server_sha256=hashlib.sha256(Path(args.server).read_bytes()).hexdigest())
(out/'setup.json').write_text(json.dumps(metadata, indent=2))
for trial in range(1,4):
    with Session(args.server) as session:
        net = Network(metadata['architecture'], session=session).name_outputs('u')
        if (out/f'initial-{trial}.npz').exists():
            net.load(out/f'initial-{trial}.npz')
        initial = net.state_dict()
        net.save(out/f'initial-{trial}.npz')
        for side in [27,13]:
            if any(r['grid']==side and r['trial']==trial for r in results):
                continue
            net.load_state_dict(initial)
            points, target = sets[side]
            samples = np.zeros((len(schema),2,len(points)), np.float32)
            for row,(order,dx,dy) in enumerate(schema):
                if order == 0: samples[row] = points.transpose()
                elif order == 1: samples[row] = np.array([dx,dy])[:,None]
            inputs = Data(samples, schema, session=session).name_variables('xy')
            targets = Data(target, schema, session=session).name_variables('xy')
            std = target[:,0].std(axis=1)
            weights = (1/std**2)[None,:]
            cost = net.cost(
                inputs=inputs,
                targets=targets,
                weights=weights,
                name=f'order 3 Fourier fit {side}x{side}',
            )
            print(f'START grid={side} trial={trial}', flush=True)
            start=time.perf_counter()
            trained = net.train(cost=cost, epochs=1000, optimizer='rprop', state='reset',
                                reset_steps=2e-4, verbose=1, log_every=50)
            seconds=time.perf_counter()-start
            net.save(out/f'grid{side}-trial{trial}-final.npz')
            metrics={}
            outputs={}
            for split,size in [('train',side),('test',155)]:
                pts, truth = sets[size]
                pieces=[]
                for offset in range(0,len(pts),1024):
                    chunk=pts[offset:offset+1024]
                    values=np.zeros((len(schema),2,len(chunk)),np.float32)
                    for row,(order,dx,dy) in enumerate(schema):
                        if order == 0: values[row]=chunk.transpose()
                        elif order == 1: values[row]=np.array([dx,dy])[:,None]
                    data=Data(values,schema,session=session).name_variables('xy')
                    pred=net.eval(data); pieces.append(pred.to_numpy()[:,0].astype(float))
                    pred.close(); data.close()
                prediction=np.concatenate(pieces,axis=1)
                outputs[split]=prediction
                rms=np.sqrt(np.mean((prediction-truth[:,0])**2,axis=1))
                per_row=100*rms/std
                own=100*rms/truth[:,0].std(axis=1)
                metrics[split]=dict(row_percent_train_std=per_row.tolist(),
                    row_percent_own_std=own.tolist(),
                    order_mean_percent_train_std=[float(per_row[schema[:,0]==order].mean()) for order in range(4)],
                    order_mean_percent_own_std=[float(own[schema[:,0]==order].mean()) for order in range(4)],
                    total_percent_train_std=float(np.linalg.norm(per_row)),
                    total_percent_own_std=float(np.linalg.norm(own)))
            objective=cost.evaluate()
            state=net.state_dict()
            steps=np.concatenate([layer['steps'] for layer in state['optimizer_state']])
            result=dict(grid=side,trial=trial,seconds=seconds,objective=objective,
                sqrt_cost_per_sample_percent=100*np.sqrt(objective/len(points)),
                metrics=metrics,reports=trained['reports'],
                zero_steps=int(np.count_nonzero(steps==0)),
                capped_weights=int(sum(np.count_nonzero(abs(w)>=20) for w in state['weights'])))
            # Server cost vs an independent reduction of the actual float32 targets/weights.
            manual=float(np.sum(weights.astype(np.float32).reshape(-1,1)*(outputs['train']-target[:,0].astype(np.float32))**2))
            result['manual_cost']=manual
            result['cost_relative_difference']=abs(objective-manual)/max(manual,1e-30)
            if not np.isfinite(objective) or not np.all(np.isfinite(outputs['test'])):
                raise RuntimeError('nonfinite benchmark result')
            np.savez_compressed(out/f'grid{side}-trial{trial}-predictions.npz',**outputs)
            results.append(result)
            (out/'results.json').write_text(json.dumps(results,indent=2))
            print('RESULT',json.dumps({k:v for k,v in result.items() if k!='reports'}),flush=True)
            cost.close(); inputs.close(); targets.close()
        net.close()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
colors={27:'#1769aa',13:'#d67a16'}
for result in results:
    side=result['grid']; trial=result['trial']; color=colors[side]
    reports=[r for r in result['reports'] if r['phase']=='before_update']
    reports += [result['reports'][-1]]
    axes[0,0].semilogy([r['update'] for r in reports],
        [100*np.sqrt(r['total']/side**2) for r in reports],color=color,alpha=.7,
        label=f'{side}x{side}' if trial==1 else None)
    for ax,split in [(axes[0,1],'train'),(axes[1,0],'test')]:
        ax.semilogy(range(4),result['metrics'][split]['order_mean_percent_own_std'],
                    'o-',color=color,alpha=.5,label=f'{side}x{side}' if trial==1 else None)
axes[0,0].axhline(4,color='black',linestyle='--',label='Paper: 4% at epoch 1000')
axes[0,0].set(title='Training: sqrt(sum of ten normalized MSEs)',xlabel='Epoch',ylabel='Percent');axes[0,0].legend()
for ax,split,reference in [(axes[0,1],'Training',[.07,.18,.77,1.9]),(axes[1,0],'Test',[.06,.17,.61,2.8])]:
    ax.semilogy(range(4),reference,'ks--',label='Paper Table I')
    ax.set(title=f'{split}: mean normalized RMS by derivative order',xlabel='Derivative order',ylabel='Percent of target std',xticks=range(4));ax.legend()
axes[1,1].axis('off')
lines=['1000 RPROP epochs; 3 paired native initializations',
       '4 hidden layers x 128 sigmoid neurons; linear output',
       'One fixed Fourier realization (seed 2026)',
       'Test grid: 155 x 155; initial RPROP step: 0.0002',
       'No periodic step restoration; no pretraining','',
       'Median aggregate normalized RMS (train / test):']
for side in [27,13]:
    totals=[np.median([r['metrics'][split]['total_percent_train_std'] for r in results if r['grid']==side]) for split in ['train','test']]
    lines.append(f'{side} x {side}: {totals[0]:.3f}% / {totals[1]:.3f}%')
lines.extend(['Paper: about 4% / 5.6%','', 'Curves by order use each set\'s target std.',
              'Aggregates use training std for both sets.',
              'Original paper random realization unavailable.'])
axes[1,1].text(0,1,'\n'.join(lines),va='top',fontsize=10,linespacing=1.65)
for ax in [axes[0,0],axes[0,1],axes[1,0]]:ax.grid(alpha=.2)
fig.savefig(out/'comparison.png',dpi=180)
print('DONE',flush=True)
