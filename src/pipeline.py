"""CLI: python src/pipeline.py --query "..." --mode C"""
import argparse,copy,hashlib,json,sys,time,uuid
from datetime import datetime,timezone
try:
    from .config import ROOT,SETTINGS
    from .retrieval import retrieve, retrieve_candidates, focused_query
    from .reranking import rerank
    from .verifier import EvidenceVerifier
    from .evidence_gate import apply_gate
    from .generation import Generator
except ImportError:
    from config import ROOT,SETTINGS
    from retrieval import retrieve, retrieve_candidates, focused_query
    from reranking import rerank
    from verifier import EvidenceVerifier
    from evidence_gate import apply_gate
    from generation import Generator

def utcnow(): return datetime.now(timezone.utc).isoformat()

def fingerprint():
    paths=['vectorstore/index.faiss','vectorstore/metadata.json','evaluation_questions.json']
    paths += ['src/'+p.name for p in sorted((ROOT/'src').glob('*.py'))]
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in paths}

def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path)

class Pipeline:
    def __init__(self,settings=SETTINGS,verifier=None,generator=None):
        self.settings=settings
        self.verifier=verifier or EvidenceVerifier(settings=settings)
        self.generator=generator or Generator(settings=settings)

    def prepare(self,query):
        start=time.perf_counter()
        candidates=retrieve_candidates(query,self.settings.retrieval_top_k)
        retrieval_seconds=time.perf_counter()-start
        start=time.perf_counter()
        ranked=rerank(focused_query(query),candidates)
        return candidates,ranked,{'retrieval_seconds':retrieval_seconds,'reranking_seconds':time.perf_counter()-start}

    def run(self,query,mode='C',prepared=None):
        if mode not in ('A','B','C'): raise ValueError('Mode must be A, B or C')
        start=time.perf_counter()
        if prepared is None:
            if mode=='A':
                candidates=retrieve_candidates(query,self.settings.retrieval_top_k)
                ranked=[]
                timings={'retrieval_seconds':time.perf_counter()-start,'reranking_seconds':0.0}
            else: candidates,ranked,timings=self.prepare(query)
        else: candidates,ranked,timings=copy.deepcopy(prepared)
        chosen=(candidates if mode=='A' else ranked)[:self.settings.rerank_top_k]
        verification=[]; comparison={'conflicts':[],'reason':'Not applicable in baseline mode'}
        first_call=len(self.verifier.client.calls)
        llm_first=len(self.generator.client.calls) if self.generator.client else 0
        events=['retrieval']+(['reranking'] if mode!='A' else [])
        verify_start=time.perf_counter()
        if mode=='C':
            verification,comparison=self.verifier.verify(query,chosen)
            gate=apply_gate(verification,comparison)
            events+=['child_verification','conditional_parent_expansion','cross_evidence_verification','evidence_gate']
        else:
            gate={'status':'UNVERIFIED_BASELINE','approved_evidence':[
                {k:row[k] for k in ('chunk_id','source','page','text')} for row in chosen],
                'blocked_ids':[],'warnings':['Baseline mode: evidence was not SLM-verified.'],'conflicts':[]}
        timings['verification_seconds']=time.perf_counter()-verify_start
        generation_start=time.perf_counter()
        final=self.generator.generate(query,gate,verified=mode=='C')
        events+=['final_generation' if gate['approved_evidence'] else 'abstention']
        timings['generation_seconds']=time.perf_counter()-generation_start
        timings['total_stage_seconds']=timings['retrieval_seconds']+(timings['reranking_seconds'] if mode!='A' else 0)+timings['verification_seconds']+timings['generation_seconds']
        return {'run_id':str(uuid.uuid4()),'timestamp':utcnow(),'query':query,'mode':mode,
                'config':self.settings.public(),'events':events,'retrieved_candidates':candidates,
                'reranked_candidates':ranked if mode!='A' else [],'selected_evidence':chosen,
                'slm_verification':verification,'cross_evidence_verification':comparison,
                'evidence_gate':gate,'final_evidence':gate['approved_evidence'],'final_answer':final['answer'],
                'generation':final,'timings':timings,
                'slm_calls':self.verifier.client.calls[first_call:] if mode=='C' else [],
                'llm_calls':self.generator.client.calls[llm_first:] if self.generator.client else []}

def format_trace(result):
    lines=['='*72,'SLM-RAG VERIFICATION PIPELINE','='*72,f"MODE: {result['mode']}",f"QUESTION: {result['query']}",f"TIMESTAMP: {result['timestamp']}",'\nRETRIEVED CANDIDATES']
    for row in result['retrieved_candidates']:
        lines.append(f"{row['chunk_id']} | {row['source']} | page {row['page']} | similarity {row['similarity_score']:.6f}")
    variants=list(dict.fromkeys(q['query'] for r in result['retrieved_candidates'] for q in r.get('retrieval_queries',[])))
    lines.append('Retrieval query variants: '+json.dumps(variants,ensure_ascii=False))
    lines.append('\nRERANKED EVIDENCE')
    for row in result['selected_evidence']:
        lines.append(f"{row['chunk_id']} | reranker {row.get('reranker_score','N/A')}\n{row['text']}")
    lines.append('\nSLM VERIFICATION')
    if result['mode']!='C': lines.append('Bypassed only for explicit experimental baseline.')
    for row in result['slm_verification']:
        lines.append(f"\n{row['chunk_id']} | {row['source']} | page {row['page']}")
        lines.append('Inspection question: '+row.get('verification_question',result['query']))
        lines.append('CHILD: '+json.dumps(row['child_verification'],ensure_ascii=False))
        lines.append('PARENT EXPANSION: '+str(row['parent_expansion']))
        if row.get('parent_verification'): lines.append('PARENT: '+json.dumps(row['parent_verification'],ensure_ascii=False))
        v=row['verification']
        lines.append(f"VERDICT: {v['verdict']} | CONFIDENCE: {v['confidence']}\nREASON: {v['reason']}")
    lines.append('\nCROSS-EVIDENCE INSPECTION\n'+json.dumps(result['cross_evidence_verification'],ensure_ascii=False,indent=2))
    gate=result['evidence_gate']
    lines.append('\nEVIDENCE GATE\nStatus: '+gate['status'])
    lines.extend(gate['warnings'])
    lines.append('Blocked: '+', '.join(gate['blocked_ids']))
    lines.append('\nFINAL VERIFIED EVIDENCE' if result['mode']=='C' else '\nFINAL BASELINE EVIDENCE (UNVERIFIED)')
    for row in result['final_evidence']:
        lines.append(f"[{row['chunk_id']}] {row['source']} page {row['page']}\n{row['text']}")
    lines.extend(['\nFINAL LLM ANSWER',f"Provider: {result['generation']['provider']}",result['final_answer'],'\nTIMINGS: '+json.dumps(result['timings']),'='*72])
    return '\n'.join(lines)

def save_run(result):
    report=ROOT/'data/reports'
    folder=report/'runs'/result['run_id']
    write_json(folder/'trace.json',dict(result,fingerprints=fingerprint()))
    (folder/'terminal.txt').write_text(format_trace(result),encoding='utf-8')
    write_json(report/'retrieval_results.json',{'run_id':result['run_id'],'query':result['query'],'candidates':result['retrieved_candidates'],'reranked':result['reranked_candidates']})
    write_json(report/'slm_verification_results.json',{'run_id':result['run_id'],'query':result['query'],'verification':result['slm_verification'],'comparison':result['cross_evidence_verification'],'gate':result['evidence_gate'],'answer':result['final_answer']})
    return folder

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query')
    parser.add_argument('--mode',choices=['A','B','C'],default='C')
    args=parser.parse_args()
    try:
        result=Pipeline().run(args.query or input('Question: '),args.mode)
        print(format_trace(result))
        print('Full trace saved:',save_run(result))
        if result['evidence_gate'].get('error') or result['generation'].get('error') or any(r['verification'].get('error') for r in result['slm_verification']): return 2
        return 0
    except Exception as exc:
        print(f'Pipeline stopped safely: {type(exc).__name__}: {exc}',file=sys.stderr)
        return 1

if __name__=='__main__': raise SystemExit(main())
