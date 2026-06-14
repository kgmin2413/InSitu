#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

ROOT=Path('/workspace/InSitu')
DATA=ROOT/'third_party/InSitu-A4/data/chair_insitu_a4pose_fgmask_20260611'
MODEL=ROOT/'third_party/InSitu-A4/output/chair_insitu_a4pose_fgmask_result_20260611/point_cloud/iteration_30000/point_cloud.ply'
MASK_DIR=DATA/'mask_undistorted/images'
TXT=DATA/'sparse_txt'

def qvec_to_rotmat(q):
    qw,qx,qy,qz=q
    return np.array([
        [1-2*qy*qy-2*qz*qz, 2*qx*qy-2*qz*qw, 2*qz*qx+2*qy*qw],
        [2*qx*qy+2*qz*qw, 1-2*qx*qx-2*qz*qz, 2*qy*qz-2*qx*qw],
        [2*qz*qx-2*qy*qw, 2*qy*qz+2*qx*qw, 1-2*qx*qx-2*qy*qy],
    ], dtype=np.float32)

def cameras():
    out={}
    for line in (TXT/'cameras.txt').read_text().splitlines():
        s=line.strip()
        if not s or s.startswith('#'): continue
        p=s.split(); cid=int(p[0]); model=p[1]; w=int(p[2]); h=int(p[3]); params=list(map(float,p[4:]))
        if model=='PINHOLE': fx,fy,cx,cy=params[:4]
        elif model=='SIMPLE_PINHOLE': fx=fy=params[0]; cx,cy=params[1:3]
        else: raise RuntimeError(model)
        out[cid]=(w,h,fx,fy,cx,cy)
    return out

def images():
    cams=cameras(); out=[]; lines=(TXT/'images.txt').read_text().splitlines(); i=0
    while i<len(lines):
        s=lines[i].strip(); i+=1
        if not s or s.startswith('#'): continue
        p=s.split(); q=list(map(float,p[1:5])); t=np.array(list(map(float,p[5:8])),dtype=np.float32); cid=int(p[8]); name=Path(' '.join(p[9:])).name
        R=qvec_to_rotmat(q); out.append((name,R,t,cams[cid])); i+=1
    return out

def vote(vertices):
    xyz=np.column_stack([vertices['x'],vertices['y'],vertices['z']]).astype(np.float32)
    total=np.zeros(len(xyz), dtype=np.uint16)
    fg=np.zeros(len(xyz), dtype=np.uint16)
    for name,R,t,cam in images():
        mask=np.array(Image.open(MASK_DIR/name).convert('L'))
        w,h,fx,fy,cx,cy=cam
        pts=xyz @ R.T + t[None,:]
        z=pts[:,2]
        valid=z>1e-6
        u=np.rint(fx*(pts[:,0]/z)+cx).astype(np.int32)
        v=np.rint(fy*(pts[:,1]/z)+cy).astype(np.int32)
        valid &= (u>=0)&(u<w)&(v>=0)&(v<h)
        idx=np.flatnonzero(valid)
        if len(idx):
            total[idx]+=1
            fg[idx]+=(mask[v[idx],u[idx]]>=128).astype(np.uint16)
    return total,fg

def main():
    ply=PlyData.read(MODEL); verts=ply['vertex'].data
    total,fg=vote(verts)
    ratio=np.divide(fg,total,out=np.zeros(len(fg),dtype=np.float32),where=total>0)
    print(json.dumps({
      'points': len(verts),
      'visible_min': int(total.min()), 'visible_median': float(np.median(total)), 'visible_max': int(total.max()),
      'fg_median': float(np.median(fg)),
      'ratio_percentiles': {str(p): float(np.percentile(ratio,p)) for p in [0,1,5,10,25,50,75,90,95,99,100]},
      'kept_by_threshold': {f'fg2_ratio{r}': int(((fg>=2)&(ratio>=r)).sum()) for r in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]},
      'kept_fg5': {f'ratio{r}': int(((fg>=5)&(ratio>=r)).sum()) for r in [0.1,0.2,0.3,0.4,0.5,0.6]},
    }, indent=2))
    np.savez(ROOT/'data/intermediate/chair_a4pose_fgmask_votes.npz', total=total, fg=fg, ratio=ratio)
if __name__=='__main__': main()
