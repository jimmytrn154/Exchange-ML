#!/usr/bin/env python3
"""Build manuscript reporting artifacts from saved, hash-bound tables only."""
from __future__ import annotations
import csv, hashlib, json, os, subprocess, sys
from pathlib import Path
import numpy as np
PAPER=Path(__file__).resolve().parent
ROOT=PAPER.parent
sys.path.insert(0,str(ROOT/'src'))

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
def rows(path):
 with path.open(newline='') as f:return list(csv.DictReader(f))
def verify(folder,name):
 m=json.loads((folder/name).read_text())
 if m['status']!='complete':raise ValueError(name+' incomplete')
 for file,digest in {**m['output_sha256'],**m.get('cache_sha256',{})}.items():
  if sha(folder/file)!=digest:raise ValueError('hash mismatch '+file)
 return m

def main():
 sources={}
 for folder,name in [('outputs/stage3_h1_v2/heldout','e3_manifest.json'),('outputs/count_ablation_development_v1','rank_manifest.json'),('outputs/evaluator_sensitivity_v1','sensitivity_manifest.json')]:
  verify(ROOT/folder,name);sources[folder+'/'+name]=sha(ROOT/folder/name)
 mainfile=ROOT/'outputs/stage3_h1_v2/heldout/main_table.csv'
 mainrows=rows(mainfile);assert len(mainrows)==12
 aliases={'mean_entropy':'Mean entropy','p95_entropy':'95th-percentile entropy','pairwise_dice':'Foreground Dice disagreement','presence':'Presence (Jaccard disagreement)','union_nvi':'Union-normalized VI','count_variance':'Instance-count variance','hungarian':'Hungarian matching disagreement','fa_v1':r'$D_{\rm vox}$ (combined)','partition_v1':r'$wQ_{\rm vox}$ (partition only)','partition_v2':r'$wQ_{\rm joint}$ (partition only)','fa_v2':r'$D_{\rm joint}$ (combined)','q_joint_unweighted':r'$Q_{\rm joint}$ (unweighted VI/H)'}
 index={r['score']:r for r in mainrows}
 assert np.isclose(float(index['partition_v1']['heldout_merge_split_aurc']),.27196549882394705)
 table=['\\begin{table}[t]','\\caption{Bone-level AURC for all 12 scores; lower is better. M+S denotes normalized merge-plus-split risk. Subscripts vox and joint denote normalization by voxel count and joint entropy, respectively. Dev.: 40 development CTs; Eval.: 100 separate evaluation CTs.}','\\label{tab:main}','\\centering\\footnotesize','\\setlength{\\tabcolsep}{3pt}','\\begin{tabular}{lrrr}','\\toprule','Score & Dev. M+S & Eval. M+S & Eval. $1-F_1$ \\\\','\\midrule']
 for r in mainrows:
  vals=[float(r[k]) for k in ['validation_merge_split_aurc','heldout_merge_split_aurc','heldout_one_minus_f1_aurc']]
  table.append(aliases[r['score']]+' & '+' & '.join(f'{v:.4f}' for v in vals)+' \\\\')
 table+=['\\bottomrule','\\end{tabular}','\\end{table}'];(PAPER/'main_table.tex').write_text('\n'.join(table)+'\n')
 contrasts=rows(ROOT/'outputs/evaluator_sensitivity_v1/paired_differences.csv')
 sel={r['endpoint']:r for r in contrasts if r['dataset_split']=='heldout' and r['stratum']=='pooled' and r['risk_source']=='member_mean' and r['score']=='fa_v2' and r['baseline']=='partition_v1'}
 endpoint_names={'original_merge_split':'Original M+S','one_to_one_one_minus_f1':'One-to-one $1-F_1$','significant_0p01_merge_split':'Significant overlap, $\\tau=.01$','significant_0p1_merge_split':'Significant overlap, $\\tau=.10$','missing_volume':'Missing volume'}
 table=['\\begin{table}[t]','\\caption{Supplementary endpoint sensitivity on the same 100 evaluation CTs: joint-entropy combined score minus voxel-normalized partition-only score ($D_{\\rm joint}-wQ_{\\rm vox}$). Negative AURC differences favor the combined score. Intervals are marginal 95\\% paired CT-bootstrap intervals.}','\\label{tab:sensitivity}','\\centering\\footnotesize','\\begin{tabular}{lrr}','\\toprule','Failure definition & $\\Delta$AURC & 95\\% CI \\\\','\\midrule']
 for key,label in endpoint_names.items():
  r=sel[key];table.append(f"{label} & {float(r['delta_aurc']):+.4f} & [{float(r['ci_low']):+.4f}, {float(r['ci_high']):+.4f}] \\\\")
 table+=['\\bottomrule','\\end{tabular}','\\end{table}'];(PAPER/'endpoint_table.tex').write_text('\n'.join(table)+'\n')
 os.environ.setdefault('MPLCONFIGDIR','/tmp/exchange-paper-mpl')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'legend.fontsize':7,'pdf.fonttype':42})
 fig,axes=plt.subplots(1,2,figsize=(6.9,2.25))
 repl=rows(ROOT/'outputs/stage3_h1_v2/perturbations/replication.csv');sub=[r for r in repl if r['source']=='phantom']
 sub=sorted(sub,key=lambda r:float(r['r']))
 for key,label,col,marker in [('Q1',r'$\mathrm{VI}/\log(rN)$','#0072B2','o'),('Q2',r'$\mathrm{VI}/H$','#D55E00','s')]:
  axes[0].plot([float(r['r']) for r in sub],[float(r[key]) for r in sub],label=label,color=col,marker=marker,ms=3,lw=1.3)
 axes[0].set(xscale='log',xlabel='Count-replication factor r',ylabel='Partition disagreement',title='A  Fixed-probability mechanism check',ylim=(0,1.08));axes[0].set_xticks([1,2,4,8,16],['1','2','4','8','16']);axes[0].legend(loc='center right',frameon=False)
 curvefile=ROOT/'outputs/stage3_h1_v2/heldout/risk_coverage.csv';curves=rows(curvefile)
 from exchange_ml.e1_statistics import expected_curve
 score_rows=rows(ROOT/'outputs/stage3_h1_v2/heldout/scores.csv');risk_rows=rows(ROOT/'outputs/stage3_h1_v2/heldout/risks.csv')
 risk_index={(r['case_id'],r['bone']):r for r in risk_rows}
 y=np.array([float(risk_index[r['case_id'],r['bone']]['member_mean_merge_split']) for r in score_rows])
 for key,label,col,ls in [('partition_v1','Voxel-normalized partition','#009E73','-'),('fa_v2','Joint-entropy combined','#0072B2','-'),('presence','Presence disagreement','#D55E00','--'),('mean_entropy','Mean entropy','#666666',':')]:
  x=np.array([float(r[key]) for r in score_rows]);values=expected_curve(x,y)
  assert len(values)==200
  axes[1].plot(np.arange(1,len(y)+1)/len(y),values,label=label,color=col,ls=ls,lw=1.3)
 axes[1].set(xlabel='Fraction of bones retained',ylabel='Retained merge/split risk',title='B  Failure ranking on 100 separate CTs');axes[1].legend(frameon=True,facecolor='white',edgecolor='none',framealpha=.95,loc='upper left',fontsize=6.5)
 for ax in axes:ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
 fig.tight_layout(pad=.6,w_pad=2)
 fig.savefig(PAPER/'figures/mechanism_ranking.pdf',bbox_inches='tight');fig.savefig(PAPER/'figures/mechanism_ranking.png',dpi=220,bbox_inches='tight');plt.close(fig)
 for relative in ['outputs/stage3_h1_v2/heldout/main_table.csv','outputs/stage3_h1_v2/heldout/risk_coverage.csv','outputs/stage3_h1_v2/heldout/scores.csv','outputs/stage3_h1_v2/heldout/risks.csv','outputs/stage3_h1_v2/perturbations/replication.csv','outputs/evaluator_sensitivity_v1/paired_differences.csv','outputs/count_ablation_development_v1/paired_differences.csv']:
  sources[relative]=sha(ROOT/relative)
 evidence={'source_sha256':sources,'generated_sha256':{str(p.relative_to(PAPER)):sha(p) for p in [PAPER/'main_table.tex',PAPER/'endpoint_table.tex',PAPER/'figures/mechanism_ranking.pdf']},'role':'reporting_only_saved_completed_artifacts_no_new_experiments','pending_results_excluded':['E6_external_RibFrac','new_triage_policy'],'all_current_numbers_descriptive_or_frozen_contrasts_not_acceptance_probability':True}
 evidence['figure_table_trace']=[
  {'artifact_id':'fig:results','source_data':['outputs/stage3_h1_v2/perturbations/replication.csv','outputs/stage3_h1_v2/heldout/scores.csv','outputs/stage3_h1_v2/heldout/risks.csv'],'transformation':{'script':'paper/build_paper.py','sha256':sha(Path(__file__))},'caption_claim':'Fixed-count replication changes voxel-count normalization but not joint-entropy normalization; actual held-out risk curves compare fixed scores.','supported_manuscript_claims':['count-denominator mechanism in Results/Component Contributions and Controlled Responses','joint-entropy combined score improves over presence/entropy but voxel-normalized partition-only score is stronger for original M+S in Results/Held-Out Ranking and Stronger Component Controls'],'limitations':['count replication is not CT resampling','different scores use same risk and case selection','no clinical correction assumed']},
  {'artifact_id':'tab:main','source_data':['outputs/stage3_h1_v2/heldout/main_table.csv'],'transformation':{'script':'paper/build_paper.py','sha256':sha(Path(__file__))},'caption_claim':'All12 fixed scores on validation and held-out under same bone AURC convention.','supported_manuscript_claims':['stronger voxel-normalized partition-only and Hungarian controls in Results/Held-Out Ranking and Stronger Component Controls'],'limitations':['per-bone pooling weights pelvic CT three times femur','one backend family']},
  {'artifact_id':'tab:sensitivity','source_data':['outputs/evaluator_sensitivity_v1/paired_differences.csv'],'transformation':{'script':'paper/build_paper.py','sha256':sha(Path(__file__)),'filter':'heldout/pooled/member_mean/fa_v2-partition_v1'},'caption_claim':'AURC contrasts depend on internal failure endpoint.','supported_manuscript_claims':['original voxel-normalized partition-only advantage not clear under supplementary definitions in Results/Dependence on the Failure Definition'],'limitations':['post-held-out analysis not fresh confirmation','marginal intervals','fixed thresholds not clinically validated','absolute AURCs not comparable across risk definitions']}
 ]
 (PAPER/'evidence/manuscript_sources.json').write_text(json.dumps(evidence,indent=2)+'\n')
 write_markdown()
 print('Verified source manifests and generated figure/tables; no experiment run.')
 if '--compile' in sys.argv:
  subprocess.run(['/home/dang.cpm/.local/bin/tectonic','--only-cached','--keep-logs','manuscript.tex'],cwd=PAPER,check=True)

def write_markdown():
 import re
 s=(PAPER/'manuscript.tex').read_text();s=s[s.index('\\begin{abstract}'):s.index('\\bibliographystyle')]
 s=s.replace('\\begin{abstract}','## Abstract\n').replace('\\end{abstract}','\n').replace('\\begin{IEEEkeywords}','**Keywords:** ').replace('\\end{IEEEkeywords}','\n')
 s=re.sub(r'\\section\*?\{([^}]+)\}',r'\n## \1\n',s);s=re.sub(r'\\subsection\{([^}]+)\}',r'\n### \1\n',s)
 s=re.sub(r'\\cite\{([^}]+)\}',lambda m:'['+', '.join(m[1].split(','))+']',s)
 s=s.replace('\\input{main_table}','\n'+markdown_table('main_table.tex')+'\n').replace('\\input{endpoint_table}','\n'+markdown_table('endpoint_table.tex')+'\n')
 s=re.sub(r'\\begin\{figure\*\}.*?\\end\{figure\*\}',r'\n![Mechanism and actual held-out ranking](figures/mechanism_ranking.png)\n',s,flags=re.S)
 s=s.replace('\\begin{equation}','\n$$\n').replace('\\end{equation}','\n$$\n')
 s=re.sub(r'\\emph\{([^}]+)\}',r'*\1*',s)
 s=s.replace('\\clearpage','').replace('\\%','%').replace('~',' ').replace('\\ref{tab:main}','1').replace('\\ref{tab:sensitivity}','2').replace('\\ref{fig:results}','1')
 (PAPER/'manuscript.md').write_text('# Presence or Partition? Endpoint-Dependent Failure Detection in 3D Fracture Segmentation\n\nAnonymous scientific draft for author review.\n\n'+s+'\n## References\n\n```bibtex\n'+(PAPER/'references.bib').read_text()+'```\n')

def markdown_table(filename):
 import re
 t=(PAPER/filename).read_text();lines=[]
 for line in t.splitlines():
  if '&' in line:
   line=line.replace('\\\\','').strip();lines.append('| '+line.replace('&','|')+' |')
   if len(lines)==1:lines.append('| '+' | '.join(['---']*len(line.split('&')))+' |')
 return '\n'.join(lines)

if __name__=='__main__':main()
