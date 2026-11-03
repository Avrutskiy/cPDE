"""Run the standalone examples unchanged and collect scientific plots/checkpoints.

This validation harness owns instrumentation and plotting; examples stay plain
scripts. Full epochs, architectures, grids, optimizers and E updates are retained.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import platform
import runpy
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cpde import Data, Network, Session, table
from cpde.cost import IntegralCost

NAMES = ['fit_function_vanilla', 'fit_function', 'fit_function_explicit',
         'poisson_disk_vanilla', 'poisson_disk', 'schrodinger_circle_vanilla',
         'schrodinger_circle', 'attached_variation']
MAIN_NAMES = NAMES
# Historical result names below remain readable for saved figure provenance.
TITLES = dict(fit_function_explicit='Fourier fitting: values and first derivatives',
              fit_function='Fourier fitting: array matching through third order',
              poisson_disk='Poisson in a disk: extended derivatives', schrodinger_torus='Schrodinger in a deformed torus',
              schrodinger_sphere='Schrodinger in a mildly deformed convex sphere',
              schrodinger_ellipse='Schrodinger in a 2D convex ellipse',
              schrodinger_domain='Schrodinger in an asymmetric rounded triangle',
              schrodinger_circle='Schrodinger ground state in the unit circle · 3 × 128',
              attached_variation='Attached variation on a moving ellipse')
TITLES.update(schrodinger_circle_fixed_energy='Schrodinger circle: fixed exact E, extended derivatives',
              schrodinger_circle_projection='Schrodinger circle: analytically eliminated E, extended derivatives',
              schrodinger_circle_projection_vanilla='Schrodinger circle: analytically eliminated E, vanilla',
              schrodinger_circle_fixed_energy_vanilla='Schrodinger circle: fixed exact E, vanilla',
              fit_function_vanilla='Fourier fitting: vanilla values only',
              poisson_disk_vanilla='Poisson in a disk: vanilla',
              schrodinger_circle_vanilla='Schrodinger circle: vanilla · 3 × 128')
plt.rcParams.update({'font.size': 14, 'axes.titlesize': 14, 'axes.labelsize': 14,
                     'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 11,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': 'white', 'axes.titleweight': 'semibold'})


def write_json(path, data):
    def clean(value):
        if isinstance(value,np.ndarray): return clean(value.tolist())
        if isinstance(value,np.generic): return clean(value.item())
        if isinstance(value,float) and not np.isfinite(value): return None
        if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
        if isinstance(value,(tuple,list)): return [clean(v) for v in value]
        return value
    path.write_text(json.dumps(clean(data), indent=2, allow_nan=False), encoding='utf-8')


def evaluate(net, points, requests, variables):
    schema = table(requests, variables=variables)
    arrays = np.zeros((len(schema), len(variables), len(points)), np.float32)
    for row, counts in enumerate(schema):
        if counts[0] == 0: arrays[row] = points.transpose()
        elif counts[0] == 1: arrays[row] = counts[1:,None]
    data = Data(arrays, schema, session=net.session).name_variables(variables)
    result = net.eval(data)
    values = result.to_numpy().astype(np.float64)
    result.close(); data.close()
    return schema, values


def row_for(schema, suffix, variables):
    counts = [len(suffix)] + [suffix.count(v) for v in variables]
    return int(np.flatnonzero(np.all(schema == counts,axis=1))[0])


def fourier(points, schema, coefficients):
    x,y = points.transpose()
    result = np.zeros((len(schema),len(points)))
    for row,(_,dx,dy) in enumerate(schema):
        for n in range(1,11):
            for k in range(1,11):
                a,b,c,d = coefficients[:,n-1,k-1]*n**int(dx)*k**int(dy)/(n*k)
                sx,cx=np.sin(n*x+dx*np.pi/2),np.cos(n*x+dx*np.pi/2)
                sy,cy=np.sin(k*y+dy*np.pi/2),np.cos(k*y+dy*np.pi/2)
                result[row] += a*sx*sy+b*sx*cy+c*cx*sy+d*cx*cy
    return result


def rounded_boundary(theta, a, b, c3, c2):
    h=1+c3*np.cos(3*theta)+c2*np.sin(2*theta)
    hp=-3*c3*np.sin(3*theta)+2*c2*np.cos(2*theta)
    radius=1-8*c3*np.cos(3*theta)-3*c2*np.sin(2*theta)
    points=np.column_stack((a*(h*np.cos(theta)-hp*np.sin(theta)),b*(h*np.sin(theta)+hp*np.cos(theta))))
    stretch=np.hypot(a*np.sin(theta),b*np.cos(theta))
    tangent=np.column_stack((-a*np.sin(theta),b*np.cos(theta)))/stretch[:,None]
    normal=np.column_stack((tangent[:,1],-tangent[:,0]))
    return points,normal,tangent,a*b/(radius*stretch**3),radius*stretch


def collect(name, scope):
    name = name.removesuffix('_vanilla').removesuffix('_fixed_energy').removesuffix('_projection')
    net=scope['net']; data={}; metrics={}
    if name.startswith('fit_function'):
        x,y=np.meshgrid(np.linspace(-1,1,81),np.linspace(-1,1,81),indexing='ij')
        points=np.column_stack((x.ravel(),y.ravel()))
        requests=['x'*int(dx)+'y'*int(dy) for _,dx,dy in scope['schema']]
        schema,pred=evaluate(net,points,requests,'xy')
        target=fourier(points,schema,scope['coefficients'])
        training=np.column_stack((scope['x'],scope['y']))
        train_target=fourier(training,schema,scope['coefficients'])
        train_pred=scope['values'][:,0].astype(float)
        errors=pred[:,0]-target
        metrics={'training_value_rmse':np.sqrt(np.mean((train_pred[0]-train_target[0])**2)),
                 'dense_value_rmse':np.sqrt(np.mean(errors[0]**2)),
                 'dense_value_relative_l2':np.linalg.norm(errors[0])/np.linalg.norm(target[0]),
                 'dense_rmse_per_row':np.sqrt(np.mean(errors**2,axis=1)),
                 'dense_rmse_over_target_std_per_row':np.sqrt(np.mean(errors**2,axis=1))/target.std(axis=1),
                 'dense_nonfinite_predictions_per_row':np.count_nonzero(~np.isfinite(pred[:,0]),axis=1),
                 'dense_max_abs_value_error':np.abs(errors[0]).max()}
        data=dict(points=points,schema=schema,prediction=pred[:,0],target=target,
                  training_points=training,training_target=train_target,training_prediction=train_pred)
    elif name=='poisson_disk':
        x,y=np.meshgrid(np.linspace(-1,1,121),np.linspace(-1,1,121),indexing='ij')
        mask=x*x+y*y<1; points=np.column_stack((x[mask],y[mask]))
        schema,pred=evaluate(net,points,['xx','yy'],'xy'); pred=pred[:,0]
        target=(np.sum(points**2,axis=1)-1)/4
        residual=pred[row_for(schema,'xx','xy')]+pred[row_for(schema,'yy','xy')]-1
        boundary=scope['boundary']
        # Validation-only derivatives do not change the vanilla training context.
        bs,bv=evaluate(net,boundary.points,['x','y'],'xy')
        wall_tangent=np.column_stack((-boundary.points[:,1],boundary.points[:,0]))
        tangent=bv[row_for(bs,'x','xy'),0]*wall_tangent[:,0]+bv[row_for(bs,'y','xy'),0]*wall_tangent[:,1]
        metrics=dict(dense_value_rmse=np.sqrt(np.mean((pred[0]-target)**2)),
                     dense_value_relative_l2=np.linalg.norm(pred[0]-target)/np.linalg.norm(target),
                     dense_pde_rms=np.sqrt(np.mean(residual**2)),boundary_value_rms=np.sqrt(np.mean(bv[0,0]**2)),
                     boundary_tangent_rms=np.sqrt(np.mean(tangent**2)),mesh_area=scope['interior'].weights.sum())
        data=dict(points=points,prediction=pred,target=target,residual=residual,schema=schema,
                  vertices=scope['interior'].vertices,triangles=scope['interior'].cells,
                  quadrature_points=scope['interior'].points,boundary_points=boundary.points,
                  boundary_value=bv[0,0],boundary_tangent=tangent)
    elif name in ('schrodinger_ellipse','schrodinger_domain','schrodinger_circle'):
        interior,boundary=scope['interior'],scope['boundary']; E=scope['E']
        output=net.eval(scope['interior_inputs']); pred=output.to_numpy().astype(float)[:,0]; output.close()
        # Inspect wall derivatives after training, including for value-only walls.
        bs,bp=evaluate(net,boundary.points,['xx','xy','yy'],'xy'); bp=bp[:,0]
        schema=scope['interior_schema']
        residual=-sum(pred[row_for(schema,v+v,'xy')] for v in 'xy')-E*pred[0]
        wall_tangent=np.column_stack((-boundary.points[:,1],boundary.points[:,0])) if name=='schrodinger_circle' else boundary.T
        tangent=sum(bp[row_for(bs,v,'xy')]*wall_tangent[:,i] for i,v in enumerate('xy'))
        rng=np.random.default_rng(90210); rho=np.sqrt(rng.random(4096)); phi=rng.uniform(0,2*np.pi,len(rho))
        a,b=(1.0,1.0) if name=='schrodinger_circle' else (scope['a'],scope['b'])
        if name=='schrodinger_domain':
            from matplotlib.path import Path as PolygonPath
            outline=PolygonPath(scope['dense_edge'])
            low=scope['dense_edge'].min(axis=0); high=scope['dense_edge'].max(axis=0)
            accepted=[]
            while sum(len(part) for part in accepted)<4096:
                proposed=rng.uniform(low,high,size=(8192,2)); accepted.append(proposed[outline.contains_points(proposed)])
            check_points=np.vstack(accepted)[:4096]
            area=np.pi*a*b*(1-4*scope['shape3']**2-1.5*scope['shape2']**2)
        else:
            check_points=np.column_stack((a*rho*np.cos(phi),b*rho*np.sin(phi)))
            low=np.array([-a,-b]); high=-low; area=np.pi*a*b
        weights=np.full(len(check_points),area/len(check_points))
        cs,cp=evaluate(net,check_points,scope['interior_derivatives'],'xy'); cp=cp[:,0]
        cr=-sum(cp[row_for(cs,v+v,'xy')] for v in 'xy')-E*cp[0]
        mass=float(scope['mass'].evaluate())
        metrics=dict(E=E,normalization_integral=mass,normalization_error=mass-1,
                     training_pde_rms=np.sqrt(np.dot(interior.weights,residual**2)/interior.weights.sum()),
                     heldout_pde_rms=np.sqrt(np.dot(weights,cr**2)/weights.sum()),
                     heldout_normalization_integral=np.dot(weights,cp[0]**2),
                     boundary_value_rms=np.sqrt(np.dot(boundary.weights,bp[0]**2)/boundary.weights.sum()),
                     boundary_tangent_rms=np.sqrt(np.dot(boundary.weights,tangent**2)/boundary.weights.sum()))
        if np.max(schema[:,0])>=3:
            rg=np.column_stack([-sum(pred[row_for(schema,v+v+d,'xy')] for v in 'xy')-E*pred[row_for(schema,d,'xy')] for d in 'xy'])
            cg=np.column_stack([-sum(cp[row_for(cs,v+v+d,'xy')] for v in 'xy')-E*cp[row_for(cs,d,'xy')] for d in 'xy'])
            metrics['training_residual_derivative_rms']=np.sqrt(np.sum(interior.weights[:,None]*rg**2,axis=0)/interior.weights.sum())
            metrics['heldout_residual_derivative_rms']=np.sqrt(np.sum(weights[:,None]*cg**2,axis=0)/weights.sum())
        x,y=np.meshgrid(np.linspace(low[0],high[0],181),np.linspace(low[1],high[1],161),indexing='ij')
        field_candidates=np.column_stack((x.ravel(),y.ravel()))
        mask=outline.contains_points(field_candidates) if name=='schrodinger_domain' else (field_candidates[:,0]/a)**2+(field_candidates[:,1]/b)**2<1
        field_points=field_candidates[mask]
        fs,field=evaluate(net,field_points,['xx','yy'] if name=='schrodinger_circle' else [''],'xy')
        if name in ('schrodinger_domain','schrodinger_circle'):
            wall_normal=boundary.points if name=='schrodinger_circle' else boundary.N
            wall_curvature=1.0 if name=='schrodinger_circle' else scope['curvature']
            hessian=bp[row_for(bs,'xx','xy')]*wall_tangent[:,0]**2+2*bp[row_for(bs,'xy','xy')]*wall_tangent[:,0]*wall_tangent[:,1]+bp[row_for(bs,'yy','xy')]*wall_tangent[:,1]**2
            normal_derivative=sum(bp[row_for(bs,v,'xy')]*wall_normal[:,i] for i,v in enumerate('xy'))
            second=hessian-wall_curvature*normal_derivative
            metrics['boundary_second_tangent_rms']=np.sqrt(np.dot(boundary.weights,second**2)/boundary.weights.sum())
            theta=(np.arange(1024)+.5)*2*np.pi/1024
            cb,cn,ct,ck,speed=rounded_boundary(theta,a,b,scope.get('shape3',0),scope.get('shape2',0))
            cw=speed*2*np.pi/len(theta)
            cbs,cbp=evaluate(net,cb,['xx','xy','yy'],'xy'); cbp=cbp[:,0]
            cbt=sum(cbp[row_for(cbs,v,'xy')]*ct[:,i] for i,v in enumerate('xy'))
            cbn=sum(cbp[row_for(cbs,v,'xy')]*cn[:,i] for i,v in enumerate('xy'))
            cbtt=cbp[row_for(cbs,'xx','xy')]*ct[:,0]**2+2*cbp[row_for(cbs,'xy','xy')]*ct[:,0]*ct[:,1]+cbp[row_for(cbs,'yy','xy')]*ct[:,1]**2-ck*cbn
            cbr=-cbp[row_for(cbs,'xx','xy')]-cbp[row_for(cbs,'yy','xy')]-E*cbp[0]
            for label,values in [('value',cbp[0]),('tangent',cbt),('second_tangent',cbtt),('pde',cbr)]:
                metrics['heldout_boundary_'+label+'_rms']=np.sqrt(np.dot(cw,values**2)/cw.sum())
            metrics['boundary_residual_training_rms']=np.sqrt(np.mean(residual[-len(boundary.points):]**2))
            metrics['bulk_points']=len(scope['bulk_points']);metrics['boundary_points']=len(boundary.points)
            metrics['dense_grid_normalization']=np.sum(field[0,0]**2)*(high[0]-low[0])/180*(high[1]-low[1])/160
        data=dict(points=interior.points,prediction=pred,residual=residual,boundary_points=boundary.points,
                  boundary_value=bp[0],boundary_tangent=tangent,schema=schema,
                  heldout_points=check_points,heldout_residual=cr,field_points=field_points,field=field[0,0])
        if name=='schrodinger_circle':
            data['field_residual']=-field[row_for(fs,'xx','xy'),0]-field[row_for(fs,'yy','xy'),0]-E*field[0,0]
        if name in ('schrodinger_domain','schrodinger_circle'):
            data.update(mesh_triangles=scope['triangles'],boundary_second_tangent=second,
                        heldout_boundary_points=cb,heldout_boundary_value=cbp[0],heldout_boundary_tangent=cbt,
                        heldout_boundary_second_tangent=cbtt,heldout_boundary_theta=theta)
        if name=='schrodinger_circle':
            from scipy.special import jn_zeros, j0, j1
            root=float(jn_zeros(0,1)[0]); exact_energy=root**2
            exact=j0(root*np.linalg.norm(field_points,axis=1))/(np.sqrt(np.pi)*abs(j1(root)))
            sign=1 if np.dot(field[0,0],exact)>=0 else -1
            aligned=sign*field[0,0];error=aligned-exact
            metrics.update(exact_energy=exact_energy,energy_relative_error=abs(E/exact_energy-1),
                           dense_value_relative_l2=np.linalg.norm(error)/np.linalg.norm(exact),
                           dense_value_max_abs_error=np.max(abs(error)),
                           dense_shape_relative_l2=np.linalg.norm(aligned*np.linalg.norm(exact)/np.linalg.norm(aligned)-exact)/np.linalg.norm(exact),
                           exact_training_quadrature_mass=np.dot(interior.weights,(j0(root*np.linalg.norm(interior.points,axis=1))/(np.sqrt(np.pi)*abs(j1(root))))**2),
                           sign_alignment=sign)
            data.update(exact_field=exact,aligned_field=aligned,field_error=error)

    elif name in ('schrodinger_torus','schrodinger_sphere'):
        interior,boundary=scope['interior'],scope['boundary']; E=scope['E']
        output=net.eval(scope['interior_inputs']); pred=output.to_numpy().astype(float)[:,0]; output.close()
        # Inspect wall derivatives after training, including for value-only walls.
        bs,bp=evaluate(net,boundary.points,['xx','xy','yy'],'xy'); bp=bp[:,0]
        schema=scope['interior_schema']
        lap=sum(pred[row_for(schema,v+v,'xyz')] for v in 'xyz')
        residual=-lap-E*pred[0]
        gradient=np.column_stack([bp[row_for(bs,v,'xyz')] for v in 'xyz'])
        tangent=np.cross(gradient,boundary.N)
        if name=='schrodinger_sphere':
            a,b=scope['a'],scope['b']
            x,z=np.meshgrid(np.linspace(-1.05*a,1.05*a,181),np.linspace(-1.05*b,1.05*b,161),indexing='ij')
            mask=(x/a)**2+(z/b)**2<1
        else:
            xmax=1.08*(scope['R']+scope['r'])*1.02; zmax=scope['r']*1.05
            x,z=np.meshgrid(np.linspace(-xmax,xmax,181),np.linspace(-zmax,zmax,81),indexing='ij')
            mask=(np.abs(x/1.08)-scope['R'])**2+z*z<scope['r']**2
        slice_points=np.column_stack((x[mask],np.zeros(mask.sum()),z[mask]))
        _,slice_values=evaluate(net,slice_points,['xx','yy','zz'],'xyz')
        rng=np.random.default_rng(90210)
        if name=='schrodinger_sphere':
            rho=rng.random(4096)**(1/3); mu=rng.uniform(-1,1,4096); phi=rng.uniform(0,2*np.pi,4096)
            X=rho*np.sqrt(1-mu**2)*np.cos(phi); Y=rho*mu; Z=rho*np.sqrt(1-mu**2)*np.sin(phi)
            a,b=scope['a'],scope['b']
            check_points=np.column_stack((a*X,Y,b*Z))
            weights=np.full(len(rho),(4*np.pi/3)/len(rho)*a*b)
        else:
            rho=scope['r']*np.sqrt(rng.random(4096))
            theta=rng.uniform(0,2*np.pi,4096); phi=rng.uniform(0,2*np.pi,4096)
            X=(scope['R']+rho*np.cos(phi))*np.cos(theta); Y=(scope['R']+rho*np.cos(phi))*np.sin(theta); Z=rho*np.sin(phi)
            a=1+.08*np.cos(Y/scope['R']); b=1+.10*np.sin(Y/scope['R'])
            check_points=np.column_stack((a*X,Y,b*Z))
            weights=2*np.pi**2*scope['r']**2*(scope['R']+rho*np.cos(phi))*a*b/len(rho)
        cs,cp=evaluate(net,check_points,scope['interior_derivatives'],'xyz'); cp=cp[:,0]
        cr=-sum(cp[row_for(cs,v+v,'xyz')] for v in 'xyz')-E*cp[0]
        mass=float(scope['mass'].evaluate())
        metrics=dict(E=E,normalization_integral=mass,normalization_error=mass-1,
                     training_pde_rms=np.sqrt(np.dot(interior.weights,residual**2)/interior.weights.sum()),
                     heldout_pde_rms=np.sqrt(np.dot(weights,cr**2)/weights.sum()),
                     heldout_normalization_integral=np.dot(weights,cp[0]**2),
                     boundary_value_rms=np.sqrt(np.dot(boundary.weights,bp[0]**2)/boundary.weights.sum()),
                     boundary_tangent_rms=np.sqrt(np.dot(boundary.weights,np.sum(tangent**2,axis=1))/boundary.weights.sum()))
        if name=='schrodinger_sphere' and np.max(schema[:,0])>=3:
            rg=np.column_stack([-sum(pred[row_for(schema,v+v+d,'xyz')] for v in 'xyz')-E*pred[row_for(schema,d,'xyz')] for d in 'xyz'])
            cg=np.column_stack([-sum(cp[row_for(cs,v+v+d,'xyz')] for v in 'xyz')-E*cp[row_for(cs,d,'xyz')] for d in 'xyz'])
            metrics['training_residual_derivative_rms']=np.sqrt(np.sum(interior.weights[:,None]*rg**2,axis=0)/interior.weights.sum())
            metrics['heldout_residual_derivative_rms']=np.sqrt(np.sum(weights[:,None]*cg**2,axis=0)/weights.sum())
        data=dict(points=interior.points,prediction=pred,residual=residual,boundary_points=boundary.points,
                  boundary_value=bp[0],boundary_normals=boundary.N,slice_points=slice_points,slice_value=slice_values[0,0],
                  heldout_points=check_points,heldout_residual=cr,schema=schema)
    else:
        a,b,c=scope['a'],scope['b'],scope['c']; area=np.pi*c/np.sqrt(a*b); factor=1-.5*c*(a+b)
        expected=area*np.array([-factor/(2*a)-.5*c,-factor/(2*b)-.5*c,factor/c-.5*(a+b)])
        metrics=dict(energy=scope['moving']['value'],exact_energy=area*factor,
                     fixed_gradient=scope['fixed']['gradient'][0],attached_gradient=scope['moving']['gradient'][0],
                     exact_shape_gradient=expected)
        data=dict(points=scope['interior'].points,boundary_points=scope['boundary'].points,
                  weights=scope['interior'].weights,boundary_weights=scope['boundary'].weights)
    return data,metrics


def scatter(fig,ax,points,values,title,*,limits=None,axes=(0,1),cmap='viridis'):
    kwargs={} if limits is None else dict(vmin=limits[0],vmax=limits[1])
    artist=ax.scatter(points[:,axes[0]],points[:,axes[1]],c=values,s=5,cmap=cmap,rasterized=True,**kwargs)
    ax.set_title(title); ax.set_aspect('equal'); fig.colorbar(artist,ax=ax,shrink=.8)
    ax.set_xlabel('xyz'[axes[0]]); ax.set_ylabel('xyz'[axes[1]])


def progress(ax, result, components=False):
    reports=result['reports']; ax.set_title('cost and its terms' if components else 'cost')
    if reports:
        updates=[r['update'] for r in reports]; totals=[max(abs(r['total']),1e-30) for r in reports]
        if 'initial_objective' in result:
            updates.insert(0,result.get('initial_update',0)); totals.insert(0,max(abs(result['initial_objective']),1e-30))
        ax.semilogy(updates,totals,color='#163b65',label='cost',lw=1.8)
        if 'continuation_start_update' in result:
            ax.axvline(result['continuation_start_update'],color='#777777',ls='--',lw=1,
                       label=result['continuation_optimizer'].upper()+' begins')
        if components:
            labels=reports[0]['labels']
            if 'projection A' in labels:
                values=[dict(zip(r['labels'],r['leaf_values'])) for r in reports]
                curves={
                    'A - B**2/C':[v['projection A']-v['projection B']**2/v['projection C'] for v in values],
                    '10*(mass - 1)**2':[10*(v['normalization integral']-1)**2 for v in values],
                    'walls':[sum(x for k,x in v.items() if k.startswith('wall')) for v in values],
                }
                for label,vals in curves.items():
                    ax.semilogy([r['update'] for r in reports],np.maximum(np.abs(vals),1e-30),label=label,lw=1)
                labels=[]
            for i,label in enumerate(labels):
                vals=[r.get('leaf_values',r.get('contributions'))[i] for r in reports]
                if label=='normalization integral':
                    # The torus total contains equation, weighted normalization,
                    # and walls. Use its actual contribution, including its weight.
                    vals=[r['total']-sum(v for j,v in enumerate(r['leaf_values']) if j!=i) for r in reports]
                    label='10*(mass - 1)**2'
                label={'Poisson residual':'equation',
                       'Poisson residual and spatial derivatives':'equation',
                       'Schrodinger residual and spatial derivatives':'equation',
                       'boundary values':'boundary_cost',
                       'boundary values and first/second tangential derivatives':'boundary_cost',
                       'wall values':'walls',
                       'wall values and first/second tangential derivatives':'walls'}.get(label,label)
                ax.semilogy([r['update'] for r in reports],np.maximum(np.abs(vals),1e-30),label=label,lw=1)
        ax.legend()
    ax.set_xlabel('Optimizer updates'); ax.set_ylabel('Magnitude (log scale)' if components else 'cost (log scale)'); ax.grid(alpha=.2)


def grid_field(fig, ax, points, values, title, *, limits, cmap='viridis'):
    """Render Cartesian samples as cells, leaving missing domain cells blank."""
    x, xi = np.unique(points[:, 0], return_inverse=True)
    y, yi = np.unique(points[:, 1], return_inverse=True)
    field = np.full((len(y), len(x)), np.nan)
    field[yi, xi] = values
    artist = ax.pcolormesh(x, y, field, shading='nearest', cmap=cmap,
                          vmin=limits[0], vmax=limits[1], rasterized=True)
    ax.set_title(title); ax.set_aspect('equal')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    fig.colorbar(artist, ax=ax, shrink=.8)


def plot_result(name, data, result, output):
    artifact_name = name
    name = name.removesuffix('_vanilla').removesuffix('_fixed_energy').removesuffix('_projection')
    plt.close('all')
    if name=='attached_variation':
        fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
        axs[0].scatter(*data['points'].transpose(),s=5,label='bulk samples')
        axs[0].plot(*np.vstack((data['boundary_points'],data['boundary_points'][0])).transpose(),color='#cf6b37',label='u = 0 boundary')
        axs[0].set_aspect('equal'); axs[0].legend(); axs[0].set_title('Mapped reference quadrature')
        metrics=result['metrics']; x=np.arange(3)
        for i,(key,label) in enumerate([('fixed_gradient','fixed geometry'),('attached_gradient','attached variation'),('exact_shape_gradient','exact shape derivative')]):
            axs[1].bar(x+(i-1)*.25,metrics[key],width=.25,label=label)
        axs[1].set_xticks(x,['a','b','c']); axs[1].legend(); axs[1].set_title('Weight gradients')
    elif name.startswith('fit_function'):
        fig=plt.figure(figsize=(15,9),layout='constrained')
        layout=fig.add_gridspec(2,3)
        top=[fig.add_subplot(layout[0,i]) for i in range(3)]
        error_ax=fig.add_subplot(layout[1,0])
        cost_ax=fig.add_subplot(layout[1,1:])
        lim=(data['target'][0].min(),data['target'][0].max())
        grid_shape=tuple(len(np.unique(data['training_points'][:,i])) for i in (0,1))
        dense_shape=tuple(len(np.unique(data['points'][:,i])) for i in (0,1))
        grid_field(fig,top[0],data['training_points'],data['training_target'][0],
                   f'Target on {grid_shape[0]} × {grid_shape[1]} training grid',limits=lim)
        grid_field(fig,top[1],data['points'],data['target'][0],
                   f'Target on {dense_shape[0]} × {dense_shape[1]} grid',limits=lim)
        grid_field(fig,top[2],data['points'],data['prediction'][0],'Network at stop',limits=lim)
        error=data['prediction'][0]-data['target'][0]
        error_limit=max(float(np.max(np.abs(error))),1e-12)
        grid_field(fig,error_ax,data['points'],error,'Network − target',
                   cmap='coolwarm',limits=(-error_limit,error_limit))
        progress(cost_ax,result)
    elif name=='schrodinger_circle':
        fig,axs=plt.subplots(3,3,figsize=(15,13),layout='constrained')
        axs[0,0].triplot(data['points'][:,0],data['points'][:,1],data['mesh_triangles'],color='#aab8c5',lw=.4)
        axs[0,0].scatter(*data['points'].transpose(),s=7)
        axs[0,0].scatter(*data['boundary_points'].transpose(),s=12,color='#cf6b37')
        axs[0,0].set_aspect('equal'); axs[0,0].set_title(f"{len(data['points'])-len(data['boundary_points'])} interior + {len(data['boundary_points'])} boundary\nresidual points")
        limits=(min(float(data['aligned_field'].min()),0),max(float(data['aligned_field'].max()),float(data['exact_field'].max())))
        scatter(fig,axs[0,1],data['field_points'],data['exact_field'],'Exact normalized\nBessel ground state',limits=limits)
        scatter(fig,axs[0,2],data['field_points'],data['aligned_field'],'Network output\n(global sign aligned)',limits=limits)
        error_limit=max(float(np.max(abs(data['field_error']))),1e-12)
        scatter(fig,axs[1,0],data['field_points'],data['field_error'],'Network − exact',cmap='coolwarm',limits=(-error_limit,error_limit))
        progress(axs[1,1],result,True)
        energy=result['energy_updates']
        if len(energy)==1:
            axs[1,2].plot([energy[0]['update'],result['updates']],[energy[0]['E']]*2,color='#cf6b37',label='fixed E')
        else:
            axs[1,2].step([e['update'] for e in energy],[e['E'] for e in energy],where='post',color='#cf6b37',label='network')
        axs[1,2].axhline(result['metrics']['exact_energy'],ls='--',color='#163b65',label='exact')
        projected = result.get('energy_mode')=='projection' or '_projection' in artifact_name
        axs[1,2].set_title('energy = B/C\n(sampled reports)' if projected else 'Fixed exact energy' if len(energy)==1 else 'Energy quotient');axs[1,2].set_xlabel('Optimizer updates');axs[1,2].legend();axs[1,2].grid(alpha=.2)
        if 'field_residual' in data:
            limit=max(float(np.max(abs(data['field_residual']))),1e-12)
            grid_field(fig,axs[2,0],data['field_points'],data['field_residual'],
                       'Residual −Δu − E*u',cmap='coolwarm',limits=(-limit,limit))
        else:
            # Older runs contain scattered diagnostics only. Interpolate visibly
            # as a continuous field, and label this distinction from a grid evaluation.
            limit=max(float(np.max(abs(data['heldout_residual']))),1e-12)
            ax=axs[2,0]
            artist=ax.tripcolor(*data['heldout_points'].transpose(),data['heldout_residual'],
                               shading='gouraud',cmap='coolwarm',vmin=-limit,vmax=limit,rasterized=True)
            ax.set_title('Residual −Δu − E*u\n(interpolated samples)'); ax.set_aspect('equal')
            ax.set_xlabel('x'); ax.set_ylabel('y'); fig.colorbar(artist,ax=ax,shrink=.8)
        angle=np.arange(len(data['boundary_value']))*2*np.pi/len(data['boundary_value'])
        axs[2,1].plot(data['heldout_boundary_theta'],data['heldout_boundary_value'],label='independent boundary')
        axs[2,1].scatter(angle,data['boundary_value'],s=10,label='64 training points')
        axs[2,1].set_title('Wall value (exact = 0)');axs[2,1].set_xlabel('Angle');axs[2,1].legend();axs[2,1].grid(alpha=.2)
        axs[2,2].plot(data['heldout_boundary_theta'],data['heldout_boundary_tangent'],label='first tangential derivative')
        axs[2,2].plot(data['heldout_boundary_theta'],data['heldout_boundary_second_tangent'],label='second tangential derivative')
        axs[2,2].set_title('Independent wall derivatives\n(exact = 0)');axs[2,2].set_xlabel('Angle');axs[2,2].legend();axs[2,2].grid(alpha=.2)
    elif name in ('schrodinger_ellipse','schrodinger_domain','schrodinger_circle'):
        fig,axs=plt.subplots(2,3,figsize=(15,9),layout='constrained')
        if 'mesh_triangles' in data:
            axs[0,0].triplot(data['points'][:,0],data['points'][:,1],data['mesh_triangles'],color='#aab8c5',lw=.4)
        axs[0,0].scatter(*data['points'].transpose(),s=6,alpha=.7)
        axs[0,0].plot(*np.vstack((data['boundary_points'],data['boundary_points'][0])).transpose(),color='#cf6b37')
        axs[0,0].scatter(*data['boundary_points'].transpose(),s=10,color='#cf6b37')
        axs[0,0].set_aspect('equal'); axs[0,0].set_title('Full residual grid / boundary points')
        scatter(fig,axs[0,1],data['field_points'],data['field'],'Network output u')
        residual_limit=max(float(np.max(np.abs(data['heldout_residual']))),1e-12)
        scatter(fig,axs[0,2],data['heldout_points'],data['heldout_residual'],'Residual on independent points',cmap='coolwarm',limits=(-residual_limit,residual_limit))
        progress(axs[1,0],result,True)
        energy=result['energy_updates']; axs[1,1].step([e['update'] for e in energy],[e['E'] for e in energy],where='post',color='#cf6b37')
        axs[1,1].set_title('Energy quotient updates'); axs[1,1].set_xlabel('Optimizer updates'); axs[1,1].set_ylabel('E'); axs[1,1].grid(alpha=.2)
        angle=np.linspace(0,2*np.pi,len(data['boundary_value']),endpoint=False)
        axs[1,2].plot(angle,data['boundary_value'],label='u at wall')
        axs[1,2].plot(angle,data['boundary_tangent'],label='first tangential derivative')
        if 'boundary_second_tangent' in data:
            axs[1,2].plot(angle,data['boundary_second_tangent'],label='second tangential derivative')
        axs[1,2].set_title('Boundary conditions'); axs[1,2].set_xlabel('Boundary traversal (0–2π)'); axs[1,2].legend(); axs[1,2].grid(alpha=.2)
    elif name=='poisson_disk':
        fig,axs=plt.subplots(2,3,figsize=(15,9),layout='constrained')
        axs[0,0].triplot(data['vertices'][:,0],data['vertices'][:,1],data['triangles'],lw=.25,color='#6586a5')
        axs[0,0].scatter(*data['boundary_points'].transpose(),s=5,color='#cf6b37'); axs[0,0].set_aspect('equal'); axs[0,0].set_title('Triangulated integration mesh')
        scatter(fig,axs[0,1],data['points'],data['target'],'Exact u = (x² + y² − 1)/4',limits=(-.25,0))
        scatter(fig,axs[0,2],data['points'],data['prediction'][0],'Trained u',limits=(-.25,0))
        scatter(fig,axs[1,0],data['points'],data['prediction'][0]-data['target'],'Value error',cmap='coolwarm')
        progress(axs[1,1],result,True)
        angle=np.arctan2(data['boundary_points'][:,1],data['boundary_points'][:,0]); order=np.argsort(angle)
        axs[1,2].plot(angle[order],data['boundary_value'][order],label='u at wall')
        axs[1,2].plot(angle[order],data['boundary_tangent'][order],label='tangential derivative')
        axs[1,2].legend(); axs[1,2].set_title('Boundary conditions'); axs[1,2].set_xlabel('Angle (radians)'); axs[1,2].grid(alpha=.2)
    else:
        fig=plt.figure(figsize=(15,9),layout='constrained'); axs=[fig.add_subplot(2,3,i+1,projection='3d' if i in (0,1) else None) for i in range(6)]
        edge=data['boundary_points'].reshape(64,32,3)
        axs[0].plot_wireframe(edge[:,:,0],edge[:,:,1],edge[:,:,2],rstride=4,cstride=2,lw=.4,color='#6586a5')
        keep=data['points'][:,1]<0
        axs[0].scatter(*data['points'][keep].transpose(),s=1,alpha=.3,color='#cf6b37'); axs[0].set_title('Full boundary / interior points with y < 0')
        artist=axs[1].scatter(*data['points'][keep].transpose(),c=data['prediction'][0,keep],s=3,cmap='viridis'); fig.colorbar(artist,ax=axs[1],shrink=.6); axs[1].set_title('Network output — points with y < 0')
        for ax in axs[:2]:
            ax.set_box_aspect(np.ptp(data['boundary_points'],axis=0)); ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
            ax.set_zticks([-.65,0,.65]); ax.tick_params(labelsize=8)
        mass=result['metrics']['normalization_integral']
        scatter(fig,axs[2],data['slice_points'],data['slice_value'],f'u on y = 0; integral u² = {mass:.3g}',axes=(0,2))
        progress(axs[3],result,True)
        energy=result['energy_updates']; axs[4].step([e['update'] for e in energy],[e['E'] for e in energy],where='post',marker='.',color='#cf6b37'); axs[4].set_title('Energy quotient updates'); axs[4].set_xlabel('Optimizer updates'); axs[4].set_ylabel('E'); axs[4].grid(alpha=.2)
        axs[5].hist(data['heldout_residual'],bins=60,color='#6586a5'); axs[5].set_title('Residual on independent interior samples'); axs[5].set_xlabel('−Δu − Eu'); axs[5].set_ylabel('Sample count')
    count=result.get('updates',0)
    fig.suptitle(f"{TITLES[artifact_name]}\n{result['backend']} · {count} optimizer updates · {result['status']}",fontsize=18)
    fig.savefig(output/f'{artifact_name}.png',dpi=170)
    fig.savefig(output/f'{artifact_name}.pdf')
    plt.close(fig)


def report_html(output):
    sections=[]
    for name in MAIN_NAMES + [n for n in NAMES if n not in MAIN_NAMES]:
        path=output/f'{name}.json'
        if not path.exists(): continue
        result=json.loads(path.read_text(encoding='utf-8'))
        rows=''.join(f'<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>' for k,v in result.get('metrics',{}).items())
        image=f'<a href="{name}.png"><img src="{name}.png"></a>' if (output/f'{name}.png').exists() else ''
        if (output/f'{name}-pretraining.png').exists():
            image=f'<h3>Profile pretraining</h3><img src="{name}-pretraining.png"><h3>Eigenproblem training</h3>'+image
        error=f'<pre>{html.escape(result.get("error", ""))}</pre>' if result.get('error') else ''
        assessment=f'<p><strong>{html.escape(result["assessment"])}</strong></p>' if result.get('assessment') else ''
        sections.append(f'<section><h2>{html.escape(TITLES[name])}</h2><p>{html.escape(result["status"])} · {html.escape(result["backend"])}</p>{assessment}{error}{image}<table>{rows}</table><p><a href="{name}.json">Raw results and progress</a> · <a href="{name}.npz">Numerical arrays</a> · <a href="{name}.log">Run log</a></p></section>')
    body='''<!doctype html><meta charset="utf-8"><title>cPDE example validation</title><style>body{font:16px system-ui;margin:40px auto;max-width:1250px;padding:0 24px;color:#17324d}img{width:100%}section{margin:40px 0;border-top:1px solid #ccd6df}table{border-collapse:collapse}td{padding:6px 14px;border-bottom:1px solid #e5e9ed}td:last-child{overflow-wrap:anywhere}a{color:#1764a0}</style><h1>cPDE example validation</h1><p>Original example settings unless an epoch override is recorded in the run JSON. Metrics measure these particular runs; successful execution alone is not evidence of convergence. Check normalization, PDE residuals, boundary errors and independent-grid errors.</p>'''
    (output/'index.html').write_text(body+''.join(sections),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server'); parser.add_argument('--output',required=True)
    parser.add_argument('--only',nargs='+',choices=NAMES,default=MAIN_NAMES)
    parser.add_argument('--plots-only',action='store_true')
    parser.add_argument('--initial-state-dir',help='Replay saved initial checkpoints, retaining the example training settings')
    parser.add_argument('--epochs-per-call',type=int,help='Diagnostic override of each train call; actual epochs are recorded in the run JSON')
    args=parser.parse_args(); output=Path(args.output).resolve(); output.mkdir(parents=True,exist_ok=True)
    if args.epochs_per_call is not None and args.epochs_per_call < 1:
        parser.error('--epochs-per-call must be positive')
    if args.plots_only:
        for name in args.only:
            path=output/f'{name}.json'
            if path.exists() and (output/f'{name}.npz').exists():
                with np.load(output/f'{name}.npz') as data: plot_result(name,data,json.loads(path.read_text(encoding='utf-8')),output)
        report_html(output); return
    if not args.server: parser.error('--server is required for training')
    executable=Path(args.server).resolve()
    backend='CPU host probe' if 'host_probe' in executable.name else 'CUDA server'
    metadata=dict(started_utc=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),python=sys.version,
                  numpy=np.__version__,matplotlib=matplotlib.__version__,server=str(executable),backend=backend,
                  visible_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ('cpde','cuda','examples') for p in (ROOT/folder).glob('*') if p.is_file()})
    try: metadata['gpus']=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv'],text=True)
    except (OSError,subprocess.CalledProcessError): pass
    write_json(output/'environment.json',metadata)
    original_train=Network.train; original_update=IntegralCost.set_parameters; console=sys.stdout; failed=False
    for name in args.only:
        result=dict(name=name,backend=backend,status='running',reports=[],train_calls=[],energy_updates=[])
        write_json(output/f'{name}.json',result)
        start=time.perf_counter(); print('START',name,flush=True)
        try:
            with Session(executable) as session, (output/f'{name}.log').open('w',encoding='utf-8',buffering=1) as log:
                def train(net, **options):
                    if args.epochs_per_call is not None:
                        result['epoch_override_per_call']=args.epochs_per_call
                        options['epochs']=args.epochs_per_call
                    if not result['train_calls']:
                        if args.initial_state_dir:
                            initial=Path(args.initial_state_dir)/f'{name}-initial.npz'
                            net.load(initial)
                            result['replayed_initial_state']=str(initial.resolve())
                        net.save(output/f'{name}-initial.npz')
                        result['initial_objective']=options['cost'].evaluate()
                    terms,_=net._costs(options['cost'])
                    projection={t.label:t for t in terms if isinstance(t,IntegralCost) and t.label.startswith('projection ')}
                    if projection:
                        result['energy_mode']='projection'
                        result['energy_updates'].append(dict(update=net.state_dict()['metadata']['updates'],E=projection['projection B'].evaluate()/projection['projection C'].evaluate()))
                    energies=[float(t.parameters['E']) for t in terms if isinstance(t,IntegralCost) and 'E' in t.parameters]
                    pretraining=name in ('schrodinger_torus','schrodinger_sphere') and not energies
                    if energies and not result['energy_updates']:
                        initial_update=net.state_dict()['metadata']['updates']
                        result['energy_updates'].append(dict(update=initial_update,E=energies[0]))
                        result['initial_update']=initial_update
                        result['initial_objective']=options['cost'].evaluate()
                    stamp=time.perf_counter()
                    try:
                        answer=original_train(net,**options)
                    except RuntimeError as error:
                        # Recover already received reports from this failed call. Do not
                        # issue any new server command until train has consumed its error.
                        tb=error.__traceback__
                        while tb is not None:
                            if tb.tb_frame.f_code is original_train.__code__:
                                result['reports'].extend(tb.tb_frame.f_locals.get('reports',[]))
                            tb=tb.tb_next
                        net.save(output/f'{name}-failed.npz')
                        result.update(status='failed',error=str(error),updates=net._last_meta['updates'],architecture=net.layers)
                        result['train_calls'].append(dict(epochs=options['epochs'],optimizer=options.get('optimizer'),seconds=time.perf_counter()-stamp,failed=True))
                        write_json(output/f'{name}.json',result)
                        # These single-train scripts only inspect their output next.
                        # Let that finish so failed fits also get numerical plots.
                        if name.startswith('fit_function'): return None
                        raise
                    if pretraining:
                        result['pretraining_reports']=answer['reports']
                        result['pretraining_initial_objective']=result.pop('initial_objective')
                        result['pretraining_final_objective']=options['cost'].evaluate()
                        net.save(output/f'{name}-pretrained.npz')
                        prediction=net.eval(terms[0].inputs)
                        predicted=prediction.to_numpy()[0,0];prediction.close()
                        target=terms[0].targets.to_numpy()[0,0]
                        points=terms[0].inputs.to_numpy()[0].transpose()
                        np.savez_compressed(output/f'{name}-pretraining.npz',points=points,target=target,prediction=predicted)
                        result['pretraining_value_rmse']=float(np.sqrt(np.mean((predicted.astype(float)-target)**2)))
                        fig=plt.figure(figsize=(15,4.5),layout='constrained')
                        keep=points[:,1]<0
                        for index,values,title in [(1,target,'Target profile'),(2,predicted,'After pretraining')]:
                            ax=fig.add_subplot(1,3,index,projection='3d')
                            artist=ax.scatter(*points[keep].transpose(),c=values[keep],s=3,vmin=0,vmax=float(target.max()))
                            ax.set_box_aspect(np.ptp(points,axis=0));ax.set_title(title+' — cutaway')
                            ax.set_xlabel('x');ax.set_ylabel('y');ax.set_zlabel('z');ax.set_zticks([-.65,0,.65])
                            fig.colorbar(artist,ax=ax,shrink=.65)
                        ax=fig.add_subplot(1,3,3)
                        progress(ax,dict(reports=answer['reports'],initial_objective=result['pretraining_initial_objective']))
                        fig.suptitle(f'Value-only profile pretraining · {options["epochs"]} RPROP epochs')
                        fig.savefig(output/f'{name}-pretraining.png',dpi=170);fig.savefig(output/f'{name}-pretraining.pdf');plt.close(fig)
                    else: result['reports'].extend(answer['reports'])
                    if projection:
                        for r in answer['reports']:
                            v=dict(zip(r['labels'],r['leaf_values']))
                            result['energy_updates'].append(dict(update=r['update'],E=v['projection B']/v['projection C']))
                    result['train_calls'].append(dict(epochs=options['epochs'],optimizer=options.get('optimizer'),seconds=time.perf_counter()-stamp,E=energies[0] if energies else None,phase='pretraining' if pretraining else 'main',restore_every=options.get('restore_every',0),restore_step=options.get('restore_step')))
                    result['updates']=answer['metadata']['updates']; result['architecture']=net.layers
                    write_json(output/f'{name}.json',result)
                    print(name,'updates',result['updates'],'last objective',answer['reports'][-1]['total'],file=console,flush=True)
                    if answer['stopped']: raise RuntimeError('Training was interrupted')
                    return answer
                def set_parameters(cost, **parameters):
                    answer=original_update(cost,**parameters)
                    if 'E' in parameters:
                        result['energy_updates'].append(dict(update=result.get('updates',0),E=float(parameters['E'])))
                        write_json(output/f'{name}.json',result)
                    return answer
                with patch.dict(os.environ,{'CPDE_SERVER':str(executable)}),patch('cpde.Session',return_value=contextlib.nullcontext(session)),patch.object(Network,'train',train),patch.object(IntegralCost,'set_parameters',set_parameters),contextlib.redirect_stdout(log):
                    scope=runpy.run_module('examples.'+name,run_name='__main__')
                scope['net'].save(output/f'{name}-final.npz')
                data,metrics=collect(name,scope); result['metrics']=metrics
                if all(k in scope for k in ('A','B','C')):
                    a,b,c=[scope[k].evaluate() for k in 'ABC']
                    # Independent float64 residual calculation on the same network outputs.
                    schema=scope['interior_schema']; pred=data['prediction'].astype(float)
                    density=data['residual'].astype(float)**2
                    if not name.endswith('_vanilla'):
                        for d in 'xy':
                            rd=-sum(pred[row_for(schema,v+v+d,'xy')] for v in 'xy')-scope['E']*pred[row_for(schema,d,'xy')]
                            density+=0.5*rd**2
                    direct=float(np.dot(scope['interior'].weights,density))
                    metrics.update(projection_A=a,projection_B=b,projection_C=c,
                                   reduced_residual=a-b*b/c,direct_residual=direct,
                                   cancellation_absolute_error=abs(a-b*b/c-direct))
                try: result['final_objective']=scope['cost'].evaluate()
                except RuntimeError as error:
                    result['final_objective_error']=str(error)
                    result['status']='failed'
                    result.setdefault('error',str(error))
                if result['status']!='failed': result['status']='completed'
                result['seconds']=time.perf_counter()-start
                np.savez_compressed(output/f'{name}.npz',**data)
                write_json(output/f'{name}.json',result)
            plot_result(name,data,result,output)
        except Exception:
            result['status']='failed'; result['error']=traceback.format_exc(); result['seconds']=time.perf_counter()-start
            write_json(output/f'{name}.json',result); print(result['error'],file=console,flush=True)
        report_html(output); print('FINISH',name,result['status'],flush=True)
        failed |= result['status']=='failed'
    if failed: raise SystemExit(1)


if __name__=='__main__': main()
