import SimpleITK as sitk, numpy as np, glob, pandas as pd
S='/mnt/sdb/shared/dang.cpm/Exchange-ML/'
cases=sorted(pd.read_csv('outputs/stage3_h1/uncertainty_scores.csv',dtype={'case_id':str}).case_id)
def rd(p): return sitk.GetArrayFromImage(sitk.ReadImage(p))
rows=[]
for c in cases:
    gt=rd(glob.glob(f'dataset/PENGWIN26_task1_2/*/{c}/label.mha')[0])
    src={'stage2':f'outputs/stage2_backend/validation_predictions/{c}.mha'}
    for m in ['member_01','member_02','member_03']: src[m]=S+f'outputs/stage3_h1/members/{m}/validation_predictions/{c}.mha'
    gm=[(gt>0)&((gt-1)//50==g) for g in range(3)]
    r={'case':c}
    for k,p in src.items():
        v=rd(p); fg=v>0
        for g,name in enumerate(['sac','lhip','rhip']):
            if gm[g].sum()==0: r[f'{k}:{name}']=np.nan; continue
            # recall of the GT bone by any foreground, and by the correctly-labelled group
            r[f'{k}:{name}']=round((fg&gm[g]).sum()/gm[g].sum(),2)
            r[f'{k}:{name}_lab']=round((((v-1)//50==g)&fg&gm[g]).sum()/gm[g].sum(),2)
    rows.append(r); print(c,{k:v for k,v in r.items() if not k.endswith('_lab') and k!='case'},flush=True)
d=pd.DataFrame(rows); d.to_csv('outputs/stage3_h1/diagnostics/bone_recall_by_model.csv',index=False)
print('\nmedian bone recall (any fg / correct-label):')
for k in ['stage2','member_01','member_02','member_03']:
    print(k,{b:(d[f'{k}:{b}'].median(),d[f'{k}:{b}_lab'].median(),int((d[f'{k}:{b}']<0.5).sum())) for b in ['sac','lhip','rhip']})
