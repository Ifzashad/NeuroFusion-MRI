import hashlib,json
from pathlib import Path
r=Path(__file__).resolve().parent
m=json.loads((r/'ARTIFACT_SHA256.json').read_text())
for n,h in m.items():
 p=r/n
 assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==h,n
print('ALL EXPORTED ARTIFACTS VERIFIED:',len(m))
