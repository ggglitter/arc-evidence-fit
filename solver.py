"""Evidence-Fit: a small, auditable, offline ARC hypothesis baseline.

No task-ID dispatch, network calls, learned weights or test-output access.
Catalog fixed before inspecting the public evaluation result.
"""
from collections import Counter, deque
from dataclasses import dataclass, field


def valid_grid(g):
    return (isinstance(g, list) and 1 <= len(g) <= 30 and isinstance(g[0], list)
            and 1 <= len(g[0]) <= 30
            and all(isinstance(row, list) and len(row) == len(g[0]) for row in g)
            and all(type(c) is int and 0 <= c <= 9 for row in g for c in row))


def transpose(g):
    return [list(row) for row in zip(*g)]


def rotate(g):
    return [list(row) for row in zip(*g[::-1])]


def geometry(g, k):
    out = [row[:] for row in g]
    if k >= 4:
        out = [row[::-1] for row in out]
    for _ in range(k % 4):
        out = rotate(out)
    return out


def crop(g, bg):
    cells = [(y, x) for y, row in enumerate(g) for x, c in enumerate(row) if c != bg]
    if not cells:
        return [row[:] for row in g]
    ys, xs = zip(*cells)
    return [row[min(xs):max(xs)+1] for row in g[min(ys):max(ys)+1]]


def largest_component(g, bg, same_color):
    h, w = len(g), len(g[0])
    visited, groups = set(), []
    for y in range(h):
        for x in range(w):
            if g[y][x] == bg or (y, x) in visited:
                continue
            color, component = g[y][x], []
            pending = deque([(y, x)])
            visited.add((y, x))
            while pending:
                cy, cx = pending.popleft()
                component.append((cy, cx))
                for ny, nx in [(cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)]:
                    if (0 <= ny < h and 0 <= nx < w and (ny,nx) not in visited
                            and g[ny][nx] != bg and (not same_color or g[ny][nx] == color)):
                        visited.add((ny,nx))
                        pending.append((ny,nx))
            groups.append(component)
    if not groups:
        return [row[:] for row in g]
    group = sorted(groups, key=lambda c: (-len(c), min(c)))[0]
    ys, xs = zip(*group)
    keep = set(group)
    return [[g[y][x] if (y,x) in keep else bg for x in range(min(xs),max(xs)+1)]
            for y in range(min(ys),max(ys)+1)]


def compress(g):
    rows = []
    for row in g:
        if not rows or row != rows[-1]:
            rows.append(row[:])
    columns = []
    for col in zip(*rows):
        if not columns or list(col) != columns[-1]:
            columns.append(list(col))
    return transpose(columns)


def gravity(g, bg, direction):
    out = [row[:] for row in g]
    vertical = direction in ('up', 'down')
    lines = transpose(out) if vertical else out
    moved = []
    for line in lines:
        active = [c for c in line if c != bg]
        space = [bg] * (len(line)-len(active))
        moved.append(space+active if direction in ('down','right') else active+space)
    return transpose(moved) if vertical else moved


def transform(g, kind, params):
    if kind == 'geometry':
        return geometry(g, params[0])
    if kind == 'crop':
        return geometry(crop(g, params[0]), params[1])
    if kind == 'component':
        return largest_component(g, params[0], params[1])
    if kind == 'scale':
        k = params[0]
        return [[c for c in row for _ in range(k)] for row in g for _ in range(k)]
    if kind == 'tile':
        ny, nx = params
        return [row * nx for _ in range(ny) for row in g]
    if kind == 'compress':
        return compress(g)
    if kind == 'gravity':
        return gravity(g, *params)
    if kind == 'mirror_join':
        axis = params[0]
        if axis == 'horizontal':
            return [row + row[::-1] for row in g]
        return g + g[::-1]
    if kind == 'fill':
        colors = Counter(c for row in g for c in row)
        ordered = sorted(colors, key=lambda c: (-colors[c], c))
        color = ordered[0] if params[0] == 'majority' else ordered[-1]
        return [[color] * len(g[0]) for _ in g]
    raise ValueError('Unknown transform')


def catalog():
    for k in range(8):
        yield 'geometry', (k,), 1 + (k != 0)
    for bg in range(10):
        for k in range(8):
            yield 'crop', (bg, k), 3 + (k != 0)
    for bg in range(10):
        for same in (False, True):
            yield 'component', (bg, same), 5
    for k in (2, 3):
        yield 'scale', (k,), 3
    for ny, nx in ((1,2),(2,1),(2,2),(1,3),(3,1),(3,3)):
        yield 'tile', (ny,nx), 4
    yield 'compress', (), 4
    for bg in range(10):
        for direction in ('down','up','left','right'):
            yield 'gravity', (bg,direction), 5
    for axis in ('horizontal','vertical'):
        yield 'mirror_join', (axis,), 4
    for mode in ('majority','minority'):
        yield 'fill', (mode,), 5


def infer_colors(pairs):
    mapping = {}
    for source, target in pairs:
        if len(source) != len(target) or len(source[0]) != len(target[0]):
            return None
        for a, b in zip((c for row in source for c in row), (c for row in target for c in row)):
            if a in mapping and mapping[a] != b:
                return None
            mapping[a] = b
    return mapping


def recolor(g, mapping):
    return [[mapping.get(c,c) for c in row] for row in g]


@dataclass
class Hypothesis:
    kind: str
    params: tuple
    cost: float
    mapping: dict = field(default_factory=dict)
    constant: list | None = None

    def apply(self, grid):
        if self.kind == 'constant':
            return [row[:] for row in self.constant]
        return recolor(transform(grid,self.kind,self.params),self.mapping)

    def describe(self):
        return dict(kind=self.kind,params=list(self.params),mapping=self.mapping,cost=self.cost)


def fit_hypotheses(train, allow_recolor=True):
    if not train:
        return []
    for pair in train:
        if not valid_grid(pair['input']) or not valid_grid(pair['output']):
            raise ValueError('Invalid training grid')
    found = []
    for kind, params, cost in catalog():
        produced = [transform(pair['input'],kind,params) for pair in train]
        if not all(valid_grid(g) for g in produced):
            continue
        targets = [p['output'] for p in train]
        if produced == targets:
            found.append(Hypothesis(kind,params,cost))
        elif allow_recolor:
            mapping = infer_colors(list(zip(produced,targets)))
            if mapping is not None:
                changed = sum(k != v for k,v in mapping.items())
                found.append(Hypothesis(kind,params,cost+1+changed/10,mapping))
    if all(pair['output'] == train[0]['output'] for pair in train):
        found.append(Hypothesis('constant',(),10,constant=train[0]['output']))
    return sorted(found,key=lambda h:(h.cost,h.kind,str(h.params),str(sorted(h.mapping.items()))))


def predict_task(task, allow_recolor=True):
    """Use demonstration pairs only; test output labels are never read."""
    hypotheses = fit_hypotheses(task.get('train',[]),allow_recolor=allow_recolor)
    predictions, choices = [], []
    for case in task['test']:
        grid = case['input']
        if not valid_grid(grid):
            raise ValueError('Invalid test input')
        candidates, selected = [], []
        for hypothesis in hypotheses:
            out = hypothesis.apply(grid)
            if valid_grid(out) and out not in candidates:
                candidates.append(out)
                selected.append(hypothesis.describe())
                if len(candidates) == 2:
                    break
        if not candidates:
            candidates.append([row[:] for row in grid])
            selected.append(dict(kind='fallback_identity'))
        if len(candidates) == 1:
            alternatives = [p['output'] for p in task.get('train',[])] + [geometry(grid,1)]
            extra = next((g for g in alternatives if valid_grid(g) and g != candidates[0]),candidates[0])
            candidates.append([row[:] for row in extra])
            selected.append(dict(kind='fallback_demonstration_or_rotation'))
        predictions.append(dict(attempt_1=candidates[0],attempt_2=candidates[1]))
        choices.append(selected)
    return predictions, dict(training_consistent_hypotheses=len(hypotheses),choices=choices)


def solve(challenges, allow_recolor=True):
    return {key:predict_task(task,allow_recolor=allow_recolor)[0] for key,task in challenges.items()}
