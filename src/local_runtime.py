"""Start the configured local model when needed; no downloads or cloud services."""
import json
import os
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen
from .config import ROOT, SETTINGS


def model_ready(settings=SETTINGS):
    base=settings.slm_url.removesuffix('/v1')
    endpoint=base+('/api/tags' if settings.slm_provider=='ollama' else '/v1/models')
    try:
        with urlopen(endpoint,timeout=2) as response:body=json.load(response)
        if settings.slm_provider=='ollama':
            return any(m.get('name')==settings.slm_model for m in body.get('models',[]))
        return any(m.get('id')==settings.slm_model for m in body.get('data',[]))
    except (OSError,ValueError):return False


def ensure_local_model(settings=SETTINGS):
    if model_ready(settings):return
    if settings.slm_provider!='local':
        raise RuntimeError('Start Ollama with the configured model before running app.py.')
    address=urlparse(settings.slm_url)
    if address.hostname not in ('127.0.0.1','localhost','::1'):
        raise RuntimeError('The verifier endpoint must remain local.')
    executable=ROOT/'runtime/llama/llama-server.exe'
    weights=Path(settings.local_gguf_path)
    if not executable.exists() or not weights.exists():
        raise RuntimeError('Local model files are missing. Restore models/ and runtime/, or run setup_models.py.')
    logs=ROOT/'data/logs';logs.mkdir(parents=True,exist_ok=True)
    command=[str(executable),'-m',str(weights),'--host','127.0.0.1','--port',str(address.port or 8089),
             '-c',str(settings.local_context_tokens),'-ngl',str(settings.local_gpu_layers),
             '-t',str(settings.cpu_threads),'--parallel','1','--alias',settings.slm_model,'--no-webui']
    print('Starting local verifier model…',flush=True)
    with (logs/'model_server_stdout.log').open('ab') as out,(logs/'model_server_stderr.log').open('ab') as err:
        process=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    (logs/'model_server.pid').write_text(str(process.pid),encoding='utf-8')
    deadline=time.monotonic()+settings.local_startup_timeout
    while time.monotonic()<deadline:
        if process.poll() is not None:
            raise RuntimeError('Local model failed to start. See data/logs/model_server_stderr.log.')
        if model_ready(settings):
            print('Local verifier model ready.',flush=True)
            return
        time.sleep(0.4)
    process.terminate()
    raise RuntimeError('Local model startup timed out. See data/logs/model_server_stderr.log.')


if __name__=='__main__':ensure_local_model()
