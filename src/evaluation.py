"""Actual A/B/C experiments against the unchanged evaluation_questions.json."""
import argparse,json,re,sys
from collections import Counter
from statistics import mean
try:
    from .pipeline import Pipeline,ROOT,SETTINGS,write_json,fingerprint,utcnow
except ImportError:
    from pipeline import Pipeline,ROOT,SETTINGS,write_json,fingerprint,utcnow

def normalize(value): return re.sub(r'\s+',' ',str(value).lower()).strip()
def source_matches(source,expected):
    return normalize(expected) in normalize(source.rsplit('.',1)[0])

def score_verification(trace,gold):
    if trace['evidence_gate']['status'] not in gold['acceptable_verdicts']: return False
    values=gold.get('required_conflict_values')
    if not values: return True
    # These are evaluation-only reference annotations, never model inputs.
    return any((values[0] in c['left_quote'] and values[1] in c['right_quote']) or
               (values[1] in c['left_quote'] and values[0] in c['right_quote'])
               for c in trace['cross_evidence_verification']['conflicts'])

def metrics(rows,question,text_key='text'):
    expected=question.get('document','')
    hit=any(source_matches(r['source'],expected) for r in rows) if expected and expected!='cross_document' else None
    ranks=[n for n,r in enumerate(rows,1) if source_matches(r['source'],expected)] if hit is not None else []
    keys=question.get('keywords',[])
    text=normalize(' '.join(r.get(text_key,'') for r in rows))
    found=[k for k in keys if normalize(k) in text]
    return {'source_hit':hit,'reciprocal_source_rank':1/min(ranks) if ranks else (0 if hit is not None else None),
            'keyword_coverage':len(found)/len(keys) if keys else None,'matched_keywords':found}

def evaluate(limit=None,modes=('A','B','C')):
    questions=json.loads((ROOT/'evaluation_questions.json').read_text(encoding='utf-8'))['questions']
    if limit: questions=questions[:limit]
    annotations_path=ROOT/'evaluation_annotations.json'
    annotations=json.loads(annotations_path.read_text(encoding='utf-8')) if annotations_path.exists() else {}
    pipe=Pipeline()
    report_dir=ROOT/'data/reports'
    experiment=report_dir/'experiments'/utcnow().replace(':','').replace('.','')
    results=[]
    print(f'ACTUAL A/B/C EVALUATION | {len(questions)} questions | modes {modes}',flush=True)
    for question in questions:
        print('Running',question['id'],question['question'],flush=True)
        try:
            prepared=pipe.prepare(question['question'])
        except Exception as exc:
            results.append({'id':question['id'],'error':str(exc),'modes':{}})
            write_json(experiment/'partial_results.json',results)
            continue
        candidate_metrics=metrics(prepared[0],question)
        raw_metrics=metrics(prepared[0][:SETTINGS.rerank_top_k],question)
        ranked_metrics=metrics(prepared[1][:SETTINGS.rerank_top_k],question)
        item={'id':question['id'],'question':question['question'],'type':question.get('type'),
              'expected_document':question.get('document'),'candidate_metrics':candidate_metrics,
              'raw_top_n_metrics':raw_metrics,'reranked_top_n_metrics':ranked_metrics,'modes':{}}
        for mode in modes:
            try:
                trace=pipe.run(question['question'],mode,prepared)
                rel=f"{question['id']}_{mode}.json"
                write_json(experiment/rel,trace)
                gold=annotations.get(question['id']) if mode=='C' else None
                # No inferred gold labels from question type: irrelevant questions may themselves be answerable.
                errors=[r['verification']['error'] for r in trace['slm_verification'] if r['verification'].get('error')]
                if trace['cross_evidence_verification'].get('error'): errors.append(trace['cross_evidence_verification']['error'])
                item['modes'][mode]={'trace':str(experiment/rel),'gate_status':trace['evidence_gate']['status'],
                    'evidence_metrics':metrics(trace['final_evidence'],question),'final_answer':trace['final_answer'],
                    'answer_keyword_coverage':metrics([{'source':'','text':trace['final_answer']}],question)['keyword_coverage'],
                    'gold':gold,'verification_correct':score_verification(trace,gold) if gold else None,
                    'parent_expansions':sum(r['parent_expansion'] for r in trace['slm_verification']),
                    'verdict_counts':dict(Counter(r['verification']['verdict'] for r in trace['slm_verification'])),
                    'timings':trace['timings'],'generation_provider':trace['generation']['provider'],
                    'verification_errors':errors,'generation_error':trace['generation'].get('error')}
                print(f"  {mode}: {trace['evidence_gate']['status']} | evidence keyword coverage {item['modes'][mode]['evidence_metrics']['keyword_coverage']:.1%}",flush=True)
            except Exception as exc:
                item['modes'][mode]={'error':type(exc).__name__+': '+str(exc)}
        results.append(item)
        write_json(experiment/'partial_results.json',results)
    valid=[r for r in results if 'error' not in r]
    def avg(values):
        vals=[v for v in values if v is not None]
        return mean(vals) if vals else None
    summary={}
    for mode in modes:
        completed=[r['modes'][mode] for r in valid if mode in r['modes'] and 'error' not in r['modes'][mode]]
        summary[mode]={'completed_questions':len(completed),
            'source_hit_rate':avg([x['evidence_metrics']['source_hit'] for x in completed]),
            'mean_evidence_keyword_coverage':avg([x['evidence_metrics']['keyword_coverage'] for x in completed]),
            'mean_answer_keyword_coverage':avg([x['answer_keyword_coverage'] for x in completed]),
            'mean_stage_seconds':avg([x['timings']['total_stage_seconds'] for x in completed]),
            'gate_status_counts':dict(Counter(x['gate_status'] for x in completed)),
            'verification_labeled_count':sum(x['verification_correct'] is not None for x in completed),
            'verification_accuracy_on_annotated_queries':avg([x['verification_correct'] for x in completed]),
            'verification_error_count':sum(len(x['verification_errors']) for x in completed),
            'generation_error_count':sum(bool(x['generation_error']) for x in completed)}
    summary['retrieval']={key:{
        'source_hit_rate':avg([r[key]['source_hit'] for r in valid]),
        'mean_reciprocal_source_rank':avg([r[key]['reciprocal_source_rank'] for r in valid]),
        'mean_keyword_coverage':avg([r[key]['keyword_coverage'] for r in valid])}
        for key in ('candidate_metrics','raw_top_n_metrics','reranked_top_n_metrics')}
    notes=[
        'Original evaluation_questions.json is unchanged. Expected answers are never passed to retrieval, verification, or generation.',
        'Retrieval accuracy is operationalized as expected-source hit rate and reciprocal source rank, not gold passage relevance accuracy.',
        'Keyword coverage is literal normalized substring coverage, not semantic correctness or a hallucination metric. It may decrease when irrelevant material is blocked.',
        'Baseline A and B receive the same final evidence count and final generator settings. Retrieval and reranking are shared, and their measured times are accounted for per mode.',
        'Model confidence is uncalibrated. No claim of research novelty or statistical significance.',
        'Only explicitly annotated query outcomes are scored for verification accuracy. Per-chunk accuracy requires independent human labels and is not claimed.',
        'An answer to a question about irrelevant content can be SUPPORTED; the question type irrelevant is not automatically an INSUFFICIENT gold label.',
        'All PDF contents, including synthetic labels and test instructions, are treated as untrusted evidence.',
        'Experiments are development-set measurements; prompts have been debugged on this dataset. They are not held-out benchmark estimates.'
    ]
    report={'timestamp':utcnow(),'question_count':len(questions),'config':SETTINGS.public(),'fingerprints':fingerprint(),
            'model_manifest':json.loads((ROOT/'models/manifest.json').read_text()) if (ROOT/'models/manifest.json').exists() else None,
            'summary':summary,'limitations':notes,'results':results,'experiment_directory':str(experiment)}
    write_json(report_dir/'evaluation_report.json',report)
    write_json(experiment/'evaluation_report.json',report)
    text='SLM-RAG ACTUAL EXPERIMENT REPORT\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n\nLIMITATIONS\n'+'\n'.join(notes)+'\n\n'
    for row in results:
        text+=f"{row['id']} | {row.get('question','')}\n"
        for mode,item in row['modes'].items():
            text+=f"  MODE {mode}: {item.get('gate_status',item.get('error'))}\n  {item.get('final_answer','')}\n"
    (report_dir/'evaluation_report.txt').write_text(text,encoding='utf-8')
    retrieval_logs=[];verification_logs=[]
    for row in valid:
        for mode,item in row['modes'].items():
            if 'trace' not in item: continue
            trace=json.loads(__import__('pathlib').Path(item['trace']).read_text(encoding='utf-8'))
            retrieval_logs.append({k:trace[k] for k in ('run_id','timestamp','query','mode','retrieved_candidates','reranked_candidates')})
            if mode=='C': verification_logs.append({k:trace[k] for k in ('run_id','timestamp','query','slm_verification','cross_evidence_verification','evidence_gate','final_evidence','final_answer')})
    write_json(report_dir/'retrieval_results.json',retrieval_logs)
    write_json(report_dir/'slm_verification_results.json',verification_logs)
    print(json.dumps(summary,indent=2),flush=True)
    print('Reports:',report_dir,flush=True)
    return report

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--modes',nargs='+',choices=['A','B','C'],default=['A','B','C'])
    args=parser.parse_args()
    result=evaluate(args.limit,args.modes)
    raise SystemExit(1 if any(r.get('error') or any(m.get('error') for m in r['modes'].values()) for r in result['results']) else 0)
