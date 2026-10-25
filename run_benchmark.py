"""Aggregate evaluation without printing task contents."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from solver import predict_task, valid_grid


def evaluate(folder, allow_recolor=True):
    files=sorted(Path(folder).glob('*.json'))
    if not files:
        raise ValueError('No task files')
    correct_inputs=total_inputs=correct_tasks=coverage=0
    started=time.perf_counter()
    rows=[]
    for file in files:
        task=json.loads(file.read_text())
        clean=dict(train=task['train'],test=[dict(input=p['input']) for p in task['test']])
        predictions,trace=predict_task(clean,allow_recolor=allow_recolor)
        matches=[any(grid==case['output'] for grid in pred.values()) for pred,case in zip(predictions,task['test'])]
        total_inputs+=len(matches)
        correct_inputs+=sum(matches)
        correct_tasks+=all(matches)
        coverage+=trace['training_consistent_hypotheses']>0
        assert all(valid_grid(grid) for p in predictions for grid in p.values())
        rows.append(dict(id=file.stem,correct_inputs=sum(matches),test_inputs=len(matches),
                         consistent_hypotheses=trace['training_consistent_hypotheses']))
    return dict(tasks=len(files),test_inputs=total_inputs,correct_inputs=correct_inputs,
                output_accuracy=correct_inputs/total_inputs,fully_correct_tasks=correct_tasks,
                task_accuracy=correct_tasks/len(files),training_fit_coverage=coverage,
                elapsed_seconds=round(time.perf_counter()-started,4),details=rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('folder')
    parser.add_argument('--output',required=True)
    parser.add_argument('--no-recolor',action='store_true')
    args=parser.parse_args()
    digest=hashlib.sha256(Path(__file__).with_name('solver.py').read_bytes()).hexdigest()
    result=evaluate(args.folder,allow_recolor=not args.no_recolor)
    result.update(solver_sha256=digest,split=str(Path(args.folder)),allow_recolor=not args.no_recolor,
                  note='Public data only. Not a Kaggle/private-leaderboard result.')
    Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='details'},indent=2))
