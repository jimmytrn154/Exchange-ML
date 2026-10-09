import pandas as pd
s=pd.read_csv('outputs/stage3_h1/uncertainty_scores.csv',dtype={'case_id':str})
c=['foreground_jaccard_disagreement','fa_partition_nvi','fa_weighted_partition_contribution','fa_ipd','union_foreground_nvi','fragment_count_variance','hungarian_structural_disagreement','mean_member_fragment_count','roi_voxels']
print(s[c].describe().T.round(4).to_string())
r=s.fa_weighted_partition_contribution/s.fa_ipd
print('\nNVI-term share of FA-IPD: median %.4f  max %.4f'%(r.median(),r.max()))
print('\nSpearman:'); print(s[c].corr('spearman').round(2).to_string())
