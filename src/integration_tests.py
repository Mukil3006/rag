"""Six requested real-model categories. Corpus fixtures are test data only."""
import json,sys
try:
    from .pipeline import Pipeline,ROOT,write_json,utcnow,fingerprint
    from .retrieval import load_store
except ImportError:
    from pipeline import Pipeline,ROOT,write_json,utcnow,fingerprint
    from retrieval import load_store

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p=Pipeline();results=[]
    demo_path=ROOT/'data/reports/demo_results.json'
    demos=json.loads(demo_path.read_text(encoding='utf-8')) if demo_path.exists() else []
    queries=[
        ('direct','What are the components of a RAG system?',{'SUPPORTED'}),
        ('numerical','What was the commercial paper issuance value?',{'SUPPORTED','CONTRADICTORY'}),
        ('contradiction','Does the synthetic memorandum agree with the original financial report about commercial paper issuance?',{'CONTRADICTORY'}),
        ('partial','What is the complete Q3 average daily money market volume in the synthetic memorandum?',{'PARTIAL','INSUFFICIENT'}),
        ('unsupported','What is the measured atmospheric pressure on the fictional planet Zorblax?',{'INSUFFICIENT'})]
    for category,query,expected in queries:
        cached=next((r for r in demos if r['query']==query and r['config']==p.settings.public() and r.get('fingerprints')==fingerprint()),None)
        trace=cached or p.run(query)
        passed=trace['evidence_gate']['status'] in expected
        if category=='numerical':
            passed=passed and any('finance' in x['source'].lower() for x in trace['retrieved_candidates'])
        if category=='contradiction':
            passed=passed and any(
                ('16.9' in c['left_quote'] and '18.2' in c['right_quote']) or
                ('18.2' in c['left_quote'] and '16.9' in c['right_quote'])
                for c in trace['cross_evidence_verification']['conflicts'])
        if category=='unsupported':
            passed=passed and not trace['final_evidence'] and trace['generation']['provider']=='abstention'
        results.append({'category':category,'question':query,'expected_verdicts':sorted(expected),'passed':passed,'trace':trace,'reused_demo':bool(cached)})
        print(category,trace['evidence_gate']['status'],'PASS' if passed else 'FAIL',flush=True)
        write_json(ROOT/'data/reports/integration_tests.json',{'timestamp':utcnow(),'results':results})
    metadata=load_store()[1]
    wellness=next(r for r in metadata if r['chunk_id']=='DOC09_P01_PAR01_CH02')
    # Inspect this actual child alone: a finance query must not approve wellness figures.
    v=p.verifier.inspect('What was the commercial paper issuance value?',wellness,wellness['text'],'child')
    results.append({'category':'irrelevant','question':'What was the commercial paper issuance value?',
                    'fixture_chunk_id':wellness['chunk_id'],'fixture_text':wellness['text'],
                    'verification':v,'expected_verdicts':['INSUFFICIENT'],
                    'passed':v['verdict']=='INSUFFICIENT' and not v['supporting_claims'] and not v.get('error')})
    print('irrelevant',v['verdict'],'PASS' if results[-1]['passed'] else 'FAIL',flush=True)
    report={'timestamp':utcnow(),'test_count':len(results),'passed':sum(x['passed'] for x in results),'results':results}
    write_json(ROOT/'data/reports/integration_tests.json',report)
    print(f"{report['passed']}/{report['test_count']} actual model category tests passed",flush=True)
    return 0 if all(r['passed'] for r in results) else 1

if __name__=='__main__':raise SystemExit(main())
