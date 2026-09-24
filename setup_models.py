import os, json, hashlib, urllib.request, zipfile
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(__file__).resolve().parent
os.environ['HF_HUB_DISABLE_XET'] = '1'
models = root / 'models'
models.mkdir(exist_ok=True)
manifest = json.loads((models/'manifest.json').read_text()) if (models/'manifest.json').exists() else {}
for repo, folder, revision, patterns, weight, expected in [
    ('BAAI/bge-m3', 'bge-m3', '5617a9f61b028005a4858fdac845db406aefb181', ['*.json','pytorch_model.bin','sentencepiece.bpe.model','1_Pooling/*'], 'pytorch_model.bin', 'b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38'),
    ('cross-encoder/ms-marco-MiniLM-L6-v2', 'reranker', '233902d25c440f23af6f7d6e94d2946bac0bee0a', ['*.json','model.safetensors','vocab.txt'], 'model.safetensors', None),
    ('lmstudio-community/Qwen3-4B-Instruct-2507-GGUF', 'verifier', '4edb920b6f14e3b9284d4502a6485103d72cde05', ['Qwen3-4B-Instruct-2507-Q4_K_M.gguf'], 'Qwen3-4B-Instruct-2507-Q4_K_M.gguf', '8cdb57cbb880d313736a9bc4e3d3d2485f145b5e19cf33783746e753e82641fc'),
]:
    print('Downloading', repo, revision, flush=True)
    path=models/folder/weight
    if path.exists() and expected:
        with path.open('rb') as f: matches=hashlib.file_digest(f,'sha256').hexdigest()==expected
        if matches: patterns=[p for p in patterns if p!=weight]
    if patterns: snapshot_download(repo, revision=revision, local_dir=models/folder, allow_patterns=patterns, max_workers=2)
    with path.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
    if expected and digest!=expected: raise RuntimeError('Weight checksum mismatch: '+str(path))
    manifest[folder] = {'repo':repo,'revision':revision,'file':weight,'sha256':digest}
    print('Ready:',folder,flush=True)
runtime = root / 'runtime'
runtime.mkdir(exist_ok=True)
url = 'https://github.com/ggml-org/llama.cpp/releases/download/b11065/llama-b11065-bin-win-vulkan-x64.zip'
archive=runtime/'llama-b11065-bin-win-vulkan-x64.zip'
if not archive.exists():
    urllib.request.urlretrieve(url,archive)
with zipfile.ZipFile(archive) as z:
    z.extractall(runtime/'llama')
manifest['runtime']={'url':url,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
(models/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('All local models and runtime ready.',flush=True)
