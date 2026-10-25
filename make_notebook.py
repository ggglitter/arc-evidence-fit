"""Package the original pure-stdlib solver into a self-contained Kaggle notebook."""
import json
from pathlib import Path

root=Path(__file__).parent
source=(root/'solver.py').read_text()
runner='''
import json
from pathlib import Path

# Locate only a challenge file with the expected static-grid schema.
paths=sorted(Path("/kaggle/input").rglob("*test*challenges*.json"))
valid=[]
for candidate in paths:
    data=json.loads(candidate.read_text())
    if (isinstance(data,dict) and data and all(isinstance(v,dict) and "train" in v and "test" in v for v in data.values())):
        valid.append((candidate,data))
if len(valid)!=1:
    raise RuntimeError("Attach the official ARC-AGI-2 competition dataset; expected exactly one test challenge file, found "+str(len(valid)))
input_path,challenges=valid[0]
predictions=solve(challenges)
assert set(predictions)==set(challenges)
for task_id,attempts in predictions.items():
    assert len(attempts)==len(challenges[task_id]["test"])
    assert all(set(a)=={"attempt_1","attempt_2"} and all(valid_grid(g) for g in a.values()) for a in attempts)
Path("/kaggle/working/submission.json").write_text(json.dumps(predictions))
print("Wrote predictions for",len(predictions),"tasks. No private score is claimed.")
'''
cells=[
 dict(cell_type="markdown",metadata={},source=["# Evidence-Fit ARC-AGI-2 baseline\n","Original AI-assisted, open-source, CPU-only hypothesis search. No network or external API at inference.\n","This notebook is prepared for submission; successful local generation is not a Kaggle judge verdict.\n"]),
 dict(cell_type="code",execution_count=None,metadata={},outputs=[],source=source.splitlines(keepends=True)),
 dict(cell_type="code",execution_count=None,metadata={},outputs=[],source=runner.splitlines(keepends=True))
]
notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name="Python 3",language="python",name="python3"),language_info=dict(name="python",version="3.11")),nbformat=4,nbformat_minor=5)
(root/"evidence_fit_arc_agi_2.ipynb").write_text(json.dumps(notebook,indent=2))
print("Prepared",root/"evidence_fit_arc_agi_2.ipynb")
