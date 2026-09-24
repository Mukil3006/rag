"""Three requested end-to-end demonstrations with complete terminal traces."""
import sys
try:
    from .pipeline import Pipeline,ROOT,format_trace,save_run,write_json,fingerprint
except ImportError:
    from pipeline import Pipeline,ROOT,format_trace,save_run,write_json,fingerprint

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    pipeline=Pipeline();traces=[];outputs=[]
    for query in [
        'What are the components of a RAG system?',
        'What was the commercial paper issuance value?',
        'Does the synthetic memorandum agree with the original financial report about commercial paper issuance?']:
        result=pipeline.run(query)
        result['fingerprints']=fingerprint()
        save_run(result)
        trace=format_trace(result)
        print(trace,flush=True)
        traces.append(trace);outputs.append(result)
        (ROOT/'data/reports/demo_terminal_output.txt').write_text('\n\n'.join(traces),encoding='utf-8')
        write_json(ROOT/'data/reports/demo_results.json',outputs)

if __name__=='__main__':main()
