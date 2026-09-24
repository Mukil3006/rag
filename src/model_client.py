"""Small stdlib HTTP adapter. No API keys are stored in source or trace logs."""
import json,os,time,urllib.request,urllib.error
from urllib.parse import urlparse
try:
    from .config import SETTINGS
except ImportError:
    from config import SETTINGS

class ModelError(RuntimeError):
    pass

class ModelClient:
    def __init__(self,role='slm',settings=SETTINGS):
        self.settings=settings
        self.provider=getattr(settings,role+'_provider')
        self.model=getattr(settings,role+'_model')
        self.url=getattr(settings,role+'_url').rstrip('/')
        self.role=role
        self.calls=[]
        if role=='slm' and urlparse(self.url).hostname not in ('localhost','127.0.0.1','::1'):
            raise ValueError('SLM_URL must be a loopback URL: verification remains local.')

    def complete(self,system,user,schema=None):
        messages=[{'role':'system','content':system},{'role':'user','content':user}]
        headers={'Content-Type':'application/json'}
        if self.provider=='ollama':
            url=self.url.removesuffix('/v1')+'/api/chat'
            payload={'model':self.model,'messages':messages,'stream':False,'options':{'temperature':0,'seed':self.settings.seed,'num_predict':self.settings.max_tokens}}
            if schema: payload['format']=schema
        else:
            url=self.url+'/chat/completions'
            payload={'model':self.model,'messages':messages,'temperature':0,'seed':self.settings.seed,'max_tokens':self.settings.max_tokens}
            if schema:
                payload['response_format']={'type':'json_schema','json_schema':{'name':'evidence_result','strict':True,'schema':schema}}
            if self.provider=='openai-compatible':
                key=os.getenv('LLM_API_KEY')
                if not key: raise ModelError('LLM_API_KEY is missing')
                headers['Authorization']='Bearer '+key
        start=time.perf_counter()
        record={'role':self.role,'model':self.model,'messages':messages,'schema':schema}
        self.calls.append(record)
        try:
            request=urllib.request.Request(url,json.dumps(payload).encode('utf-8'),headers)
            with urllib.request.urlopen(request,timeout=self.settings.request_timeout) as response:
                body=json.load(response)
            if self.provider=='ollama':
                content=body['message']['content']
                record['raw_response']=content
                if body.get('done_reason')=='length': raise ModelError('Model output truncated')
            else:
                choice=body['choices'][0]
                record['raw_response']=choice['message']['content']
                if choice.get('finish_reason')=='length': raise ModelError('Model output truncated')
                content=choice['message']['content']
            record['raw_response']=content
            record['seconds']=time.perf_counter()-start
            return json.loads(content) if schema else content
        except (OSError,ValueError,KeyError,IndexError,ModelError) as exc:
            record['error']=type(exc).__name__+': '+str(exc)
            record['seconds']=time.perf_counter()-start
            raise ModelError(record['error']) from exc
