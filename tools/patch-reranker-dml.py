"""Repair the bge-reranker export for DirectML the NARROW way: in every head-split reshape target
Concat(Shape(input_ids)[0:1], Shape(attention_mask)[1:2], [-1], [64]) replace ONLY the [-1] with the
literal head count, keeping batch and seq dynamic. Nothing else is touched."""
import onnx, sys, shutil, os
from onnx import numpy_helper, helper
src, dst, heads = sys.argv[1], sys.argv[2], int(sys.argv[3])
m = onnx.load(os.path.join(src, 'model.onnx'), load_external_data=False)
g = m.graph
inits = {t.name: t for t in g.initializer}
producers = {o: n for n in g.node for o in n.output}
def is_shape_of(name, inp, start, end):
    p = producers.get(name)
    if p is None or p.op_type != 'Shape' or p.input[0] != inp: return False
    a = {x.name: helper.get_attribute_value(x) for x in p.attribute}
    return a.get('start') == start and a.get('end') == end
def const_val(name):
    t = inits.get(name)
    if t is not None and t.data_type == onnx.TensorProto.INT64:
        return numpy_helper.to_array(t).tolist()
    p = producers.get(name)
    if p is not None and p.op_type == 'Constant':
        for a in p.attribute:
            if a.name == 'value': return numpy_helper.to_array(a.t).tolist()
    return None
patched = 0
new_init_name = f'__heads_{heads}'
g.initializer.append(numpy_helper.from_array(__import__('numpy').array([heads], dtype='int64'), new_init_name))
for n in g.node:
    if n.op_type != 'Concat': continue
    if len(n.input) != 4: continue
    if not (is_shape_of(n.input[0], 'input_ids', 0, 1) and is_shape_of(n.input[1], 'attention_mask', 1, 2)): continue
    if const_val(n.input[2]) != [-1] or const_val(n.input[3]) != [64]: continue
    n.input[2] = new_init_name
    patched += 1
# Second shared vector: the merge-heads target [batch, seq, -1]. DirectML rejects this -1 as well
# (the failing node was `node_view_3`, one per layer), so it becomes the literal hidden size.
hidden_name = f'__hidden_{heads * 64}'
g.initializer.append(numpy_helper.from_array(__import__('numpy').array([heads * 64], dtype='int64'), hidden_name))
merged = 0
for n in g.node:
    if n.op_type != 'Concat' or len(n.input) != 3: continue
    if not (is_shape_of(n.input[0], 'input_ids', 0, 1) and is_shape_of(n.input[1], 'attention_mask', 1, 2)): continue
    if const_val(n.input[2]) != [-1]: continue
    n.input[2] = hidden_name
    merged += 1
merged_outs = {o for n in g.node if n.op_type == 'Concat' and len(n.input) == 3 and n.input[2] == hidden_name for o in n.output}
mk = collections.Counter(n.op_type for n in g.node if any(i in merged_outs for i in n.input)) if (collections := __import__('collections')) else None
print('shared merge-heads Concats patched:', merged, '| consumers:', dict(mk))
assert mk.get('Reshape', 0) == 24 and set(mk) == {'Reshape'}, f'expected 24 consuming reshapes, got {mk}'
patched_outs = {o for n in g.node if n.op_type == 'Concat' and len(n.input) == 4 and n.input[2] == new_init_name for o in n.output}
consumers = [(n.op_type) for n in g.node if any(i in patched_outs for i in n.input)]
import collections
kinds = collections.Counter(consumers)
print('shared head-split Concats patched:', patched, '| consumers of their output:', dict(kinds))
assert kinds.get('Reshape', 0) == 72, f'expected 72 consuming reshapes (24 layers x q/k/v), got {kinds}'
assert set(kinds) == {'Reshape'}, f'unexpected consumer op(s): {set(kinds) - {"Reshape"}}'
# The checker resolves external data relative to the CWD, not the model's directory.
_cwd = os.getcwd(); os.chdir(src)
try: onnx.checker.check_model(m, full_check=False)
finally: os.chdir(_cwd)
os.makedirs(dst, exist_ok=True)
onnx.save(m, os.path.join(dst, 'model.onnx'))
for f in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'README.md'):
    if os.path.exists(os.path.join(src, f)): shutil.copy(os.path.join(src, f), dst)
print('wrote', dst)
