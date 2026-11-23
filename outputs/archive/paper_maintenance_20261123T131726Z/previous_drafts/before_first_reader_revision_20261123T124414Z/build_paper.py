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
 aliases={'mean_entropy':'Mean entropy','p95_entropy':'P95 entropy','pairwise_dice':'Pairwise Dice','presence':'Presence','union_nvi':'Union NVI','count_variance':'Count variance','hungarian':'Hungarian','fa_v1':'FA-v1','partition_v1':'Partition-v1','partition_v2':'Partition-v2','fa_v2':'FA-v2','q_joint_unweighted':'Unweighted VI/H'}
 index={r['score']:r for r in mainrows}
 assert np.isclose(float(index['partition_v1']['heldout_merge_split_aurc']),.27196549882394705)
 table=['\\begin{table}[t]','\\caption{All 12 frozen scores: bone-level AURC, lower is better. M+S denotes normalized merge-plus-split risk.}','\\label{tab:main}','\\centering\\footnotesize','\\begin{tabular}{lrrr}','\\hline','Score & Val. M+S & Held-out M+S & Held-out $1-F_1$ \\\\','\\hline']
 for r in mainrows:
  vals=[float(r[k]) for k in ['validation_merge_split_aurc','heldout_merge_split_aurc','heldout_one_minus_f1_aurc']]
  table.append(aliases[r['score']]+' & '+' & '.join(f'{v:.4f}' for v in vals)+' \\\\')
 table+=['\\hline','\\end{tabular}','\\end{table}'];(PAPER/'main_table.tex').write_text('\n'.join(table)+'\n')
 contrasts=rows(ROOT/'outputs/evaluator_sensitivity_v1/paired_differences.csv')
 sel={r['endpoint']:r for r in contrasts if r['dataset_split']=='heldout' and r['stratum']=='pooled' and r['risk_source']=='member_mean' and r['score']=='fa_v2' and r['baseline']=='partition_v1'}
 endpoint_names={'original_merge_split':'Original M+S','one_to_one_one_minus_f1':'One-to-one $1-F_1$','significant_0p01_merge_split':'Significant overlap, $\\tau=.01$','significant_0p1_merge_split':'Significant overlap, $\\tau=.10$','missing_volume':'Missing volume'}
 table=['\\begin{table}[t]','\\caption{Post-held-out endpoint sensitivity: FA-v2 minus partition-v1 AURC on the same 100 CTs. Negative differences favor FA-v2. Intervals are marginal 95\\% paired CT-bootstrap intervals.}','\\label{tab:sensitivity}','\\centering\\footnotesize','\\begin{tabular}{lrr}','\\hline','Endpoint & $\\Delta$AURC & 95\\% CI \\\\','\\hline']
 for key,label in endpoint_names.items():
  r=sel[key];table.append(f"{label} & {float(r['delta_aurc']):+.4f} & [{float(r['ci_low']):+.4f}, {float(r['ci_high']):+.4f}] \\\\")
 table+=['\\hline','\\end{tabular}','\\end{table}'];(PAPER/'endpoint_table.tex').write_text('\n'.join(table)+'\n')
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
 for key,label,col,ls in [('partition_v1','Partition-v1','#009E73','-'),('fa_v2','FA-v2','#0072B2','-'),('presence','Presence','#D55E00','--'),('mean_entropy','Mean entropy','#666666',':')]:
  x=np.array([float(r[key]) for r in score_rows]);values=expected_curve(x,y)
  assert len(values)==200
  axes[1].plot(np.arange(1,len(y)+1)/len(y),values,label=label,color=col,ls=ls,lw=1.3)
 axes[1].set(xlabel='Retained bone coverage',ylabel='Retained M+S risk',title='B  Held-out failure ranking');axes[1].legend(frameon=False,loc='upper left')
 for ax in axes:ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
 fig.tight_layout(pad=.6,w_pad=2)
 fig.savefig(PAPER/'figures/mechanism_ranking.pdf',bbox_inches='tight');fig.savefig(PAPER/'figures/mechanism_ranking.png',dpi=220,bbox_inches='tight');plt.close(fig)
 for relative in ['outputs/stage3_h1_v2/heldout/main_table.csv','outputs/stage3_h1_v2/heldout/risk_coverage.csv','outputs/stage3_h1_v2/heldout/scores.csv','outputs/stage3_h1_v2/heldout/risks.csv','outputs/stage3_h1_v2/perturbations/replication.csv','outputs/evaluator_sensitivity_v1/paired_differences.csv','outputs/count_ablation_development_v1/paired_differences.csv']:
  sources[relative]=sha(ROOT/relative)
 evidence={'source_sha256':sources,'generated_sha256':{str(p.relative_to(PAPER)):sha(p) for p in [PAPER/'main_table.tex',PAPER/'endpoint_table.tex',PAPER/'figures/mechanism_ranking.pdf']},'role':'reporting_only_saved_completed_artifacts_no_new_experiments','pending_results_excluded':['E6_external_RibFrac','new_triage_policy'],'all_current_numbers_descriptive_or_frozen_contrasts_not_acceptance_probability':True}
 evidence['figure_table_trace']=[
  {'artifact_id':'fig:results','source_data':['outputs/stage3_h1_v2/perturbations/replication.csv','outputs/stage3_h1_v2/heldout/scores.csv','outputs/stage3_h1_v2/heldout/risks.csv'],'transformation':{'script':'paper/build_paper.py','sha256':sha(Path(__file__))},'caption_claim':'Fixed-count replication changes normalization; actual held-out risk curves compare fixed scores.','supported_manuscript_claims':['count-denominator mechanism','FAv2 improves over presence/entropy but P1 stronger for original M+S'],'limitations':['count replication is not CT resampling','different scores use same risk and case selection','no clinical correction assumed']},
  {'artifact_id':'tab:main','source_data':['outputs/stage3_h1_v2/heldout/main_table.csv'],'transformation':{'script':'paper/build_paper.py','sha256':sha(Path(__file__))},'caption_claim':'All12 fixed scores on validation and held-out under same bone AURC convention.','supported_manuscript_claims':['stronger partition-only and Hungarian controls'],'limitations':['per-bone pooling weights pelvic CT three times femur','one backend family']},
  {'artifact_id':'tab:sensitivity','source_data':['outputs/evaluator_sensitivity_v1/paired_differences.csv'],'transformation':{'script':'paper/build_paper.py','sha256':sha(Path(__file__)),'filter':'heldout/pooled/member_mean/fa_v2-partition_v1'},'caption_claim':'AURC contrasts depend on internal failure endpoint.','supported_manuscript_claims':['original P1 advantage not clear under supplementary definitions'],'limitations':['post-held-out analysis not fresh confirmation','marginal intervals','fixed thresholds not clinically validated','absolute AURCs not comparable across risk definitions']}
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
