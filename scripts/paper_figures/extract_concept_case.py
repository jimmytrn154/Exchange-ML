# Extract bone-cropped CT + label volumes to npz (SimpleITK env).
import SimpleITK as sitk, numpy as np, sys
root='D:/EMBC/dataset/PENGWIN26_task1_2_train_part1'; out=sys.argv[1]
for spec in sys.argv[2:]:
    c,base=spec.split(':'); base=int(base)
    try:
        im=sitk.ReadImage(f"{root}/{c}/image.mha"); lb=sitk.ReadImage(f"{root}/{c}/label.mha")
    except RuntimeError: print(c,"unreadable"); continue
    img=sitk.GetArrayFromImage(im); lab=sitk.GetArrayFromImage(lb)
    bone=(lab>=base)&(lab<base+50)
    z,y,x=np.where(bone); m=20
    sl=tuple(slice(max(a.min()-m,0),a.max()+m+1) for a in (z,y,x))
    np.savez_compressed(f'{out}/{c}_{base}.npz',img=img[sl].astype(np.int16),lab=np.where(bone[sl],lab[sl],0).astype(np.int16),
        spacing=np.array(im.GetSpacing()),direction=np.array(im.GetDirection()))
    print(c,base,img[sl].shape,im.GetSpacing(),np.round(im.GetDirection(),2))
