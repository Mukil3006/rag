"""Local web interface for the existing SLM-RAG pipeline. Run: python app.py."""
import argparse
import os
import subprocess
import sys
from pathlib import Path
# Running with another Python interpreter still uses this project's existing environment.
if __name__=='__main__':
    project_python=Path(__file__).resolve().parent/'.venv/Scripts/python.exe'
    if project_python.exists() and Path(sys.executable).resolve()!=project_python.resolve():
        raise SystemExit(subprocess.call([str(project_python),str(Path(__file__).resolve()),*sys.argv[1:]]))
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from urllib.request import urlopen

from src.config import ROOT, SETTINGS
from src.pipeline import Pipeline, fingerprint, format_trace, save_run, write_json
from src.local_runtime import ensure_local_model, model_ready


class ResearchApp:
    def __init__(self):
        self.lock = threading.Lock()
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.jobs = {}
        self.pipeline = None
        self.active = None
        self.history_path = ROOT / 'data/reports/web_ui_runs.json'
        self.history = []
        if self.history_path.exists():
            self.history = json.loads(self.history_path.read_text(encoding='utf-8'))[-20:]

    def submit(self, query, mode):
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError('Enter a question between 1 and 2,000 characters.')
        if mode not in ('A', 'B', 'C'):
            raise ValueError('Choose mode A, B, or C.')
        with self.lock:
            if self.active:
                raise RuntimeError('A question is already running. Wait for it to finish.')
            job_id = str(uuid.uuid4())
            self.jobs[job_id] = {'id': job_id, 'status': 'running', 'query': query.strip(),
                                 'mode': mode, 'started': time.time()}
            self.active = job_id
        self.worker.submit(self.execute, job_id)
        return job_id

    def execute(self, job_id):
        try:
            with self.lock:
                job = dict(self.jobs[job_id])
            if self.pipeline is None:
                self.pipeline = Pipeline()
            result = self.pipeline.run(job['query'], job['mode'])
            result['fingerprints'] = fingerprint()
            result['fingerprints']['app.py'] = __import__('hashlib').sha256((ROOT/'app.py').read_bytes()).hexdigest()
            save_run(result)
            terminal = format_trace(result)
            with self.lock:
                self.history.append({'id': result['run_id'], 'query': result['query'], 'mode': result['mode'],
                                     'timestamp': result['timestamp'], 'status': result['evidence_gate']['status']})
                self.history = self.history[-20:]
                write_json(self.history_path, self.history)
                self.jobs[job_id].update(status='complete', result=result, terminal=terminal)
        except Exception as exc:
            with self.lock:
                self.jobs[job_id].update(status='error', error=f'{type(exc).__name__}: {exc}')
        finally:
            with self.lock:
                self.active = None
                # Keep recent browser jobs bounded; full traces remain on disk.
                for old_id in list(self.jobs)[:-20]:
                    self.jobs.pop(old_id, None)

    def snapshot(self, job_id):
        with self.lock:
            return dict(self.jobs[job_id]) if job_id in self.jobs else None

    def saved(self, run_id):
        with self.lock:
            allowed = {row['id'] for row in self.history}
        if run_id not in allowed:
            return None
        result = json.loads((ROOT/'data/reports/runs'/run_id/'trace.json').read_text(encoding='utf-8'))
        return {'status': 'complete', 'result': result, 'terminal': format_trace(result)}


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body, content_type='application/json; charset=utf-8'):
            data = json.dumps(body, ensure_ascii=False).encode('utf-8') if not isinstance(body, bytes) else body
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def local_request(self):
            allowed = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
            if self.headers.get('Host') not in allowed:
                self.reply(403, {'error': 'This interface is available only on this computer.'})
                return False
            origin = self.headers.get('Origin')
            if origin and origin not in {'http://' + host for host in allowed}:
                self.reply(403, {'error': 'Cross-origin requests are not allowed.'})
                return False
            return True

        def do_GET(self):
            if not self.local_request():
                return
            path = urlparse(self.path).path
            static = {'/': ('index.html', 'text/html; charset=utf-8'),
                      '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                      '/style.css': ('style.css', 'text/css; charset=utf-8')}
            if path in static:
                name, kind = static[path]
                return self.reply(200, (ROOT/'web'/name).read_bytes(), kind)
            if path == '/api/status':
                ready = model_ready()
                with app.lock:
                    active = app.active
                return self.reply(200, {'application':'slm-rag','project_root':str(ROOT),'model_ready': ready, 'model': SETTINGS.slm_model,
                                       'provider': SETTINGS.llm_provider, 'active_job': active,
                                       'retrieval_top_k': SETTINGS.retrieval_top_k, 'rerank_top_k': SETTINGS.rerank_top_k})
            if path == '/api/history':
                with app.lock:
                    history = list(reversed(app.history))
                return self.reply(200, history)
            if path.startswith('/api/jobs/'):
                job = app.snapshot(path.removeprefix('/api/jobs/'))
                return self.reply(200 if job else 404, job or {'error': 'Run not found.'})
            if path.startswith('/api/saved/'):
                try:
                    result = app.saved(path.removeprefix('/api/saved/'))
                except (OSError, ValueError):
                    result = None
                return self.reply(200 if result else 404, result or {'error': 'Saved run not found.'})
            return self.reply(404, {'error': 'Not found.'})

        def do_POST(self):
            if not self.local_request():
                return
            if urlparse(self.path).path != '/api/runs':
                return self.reply(404, {'error': 'Not found.'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 12000:
                    raise ValueError('Invalid request size.')
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError('Expected a question and mode.')
                job_id = app.submit(payload.get('query'), payload.get('mode', 'C'))
                return self.reply(202, {'id': job_id})
            except RuntimeError as exc:
                return self.reply(409, {'error': str(exc)})
            except (ValueError, TypeError) as exc:
                return self.reply(400, {'error': str(exc)})
    return Handler


def main():
    import webbrowser
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=SETTINGS.web_ui_port)
    args = parser.parse_args()
    url=f'http://127.0.0.1:{args.port}'
    try:
        with urlopen(url+'/api/status',timeout=2) as response:existing=json.load(response)
        if existing.get('application')=='slm-rag' and existing.get('project_root')==str(ROOT):
            ensure_local_model()
            print(f'Project already running: {url}',flush=True)
            if SETTINGS.open_browser:webbrowser.open(url)
            return 0
    except (OSError,ValueError):pass
    ensure_local_model()
    app=ResearchApp()
    server=ThreadingHTTPServer(('127.0.0.1', args.port),make_handler(app))
    print(f'SLM-RAG web interface: {url}',flush=True)
    if SETTINGS.open_browser:threading.Timer(0.5,lambda:webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.worker.shutdown(wait=False)
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        print(f'Unable to start the project: {exc}',file=sys.stderr)
        raise SystemExit(1)
