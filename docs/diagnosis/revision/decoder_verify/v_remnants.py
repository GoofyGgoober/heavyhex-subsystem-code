"""Probe: decomposition pieces that exist only inside decomposed errors (remnants), day-5 r1 2-round circuits;
whether stim can decompose without introducing them; and how many circuit fault locations feed
the D0 / D5 boundary edges of the X d3 counterexample."""
import json

from vcommon import HERE, MEM, model
f="ibm_fez-2026-10-08-r1"
out={}
for basis in "XZ":
    m=model(f,MEM[(basis,2)])
    dem=m.detector_error_model(decompose_errors=True)
    whole=set(); pieces=set()
    for ins in dem.flattened():
        if ins.type!="error": continue
        parts=[[]]
        for t in ins.targets_copy():
            if t.is_separator(): parts.append([]); continue
            if t.is_relative_detector_id(): parts[-1].append(t.val)
        if len(parts)==1: whole.add(tuple(sorted(parts[0])))
        else:
            for p in parts: pieces.add(tuple(sorted(p)))
    only=sorted(pieces-whole)
    try:
        m.detector_error_model(decompose_errors=True, block_decomposition_from_introducing_remnant_edges=True)
        blocked="ok"
    except Exception as ex:
        blocked="fails: "+str(ex).splitlines()[0][:200]
    out[basis]={"pieces_not_standalone_detector_sets":[list(x) for x in only],"n":len(only),"block_remnants":blocked}
    print(basis, len(only), only, blocked)
# circuit fault locations feeding D0 and D5 boundary edges (X d3, 2 rounds)
m=model(f,MEM[("X",2)])
dem=m.detector_error_model(decompose_errors=True)
# map undecomposed symptom -> decomposed components via the decomposed DEM line with that full symptom
comp_of={}
for ins in dem.flattened():
    if ins.type!="error": continue
    parts=[[]]; full=set(); fo=0
    for t in ins.targets_copy():
        if t.is_separator(): parts.append([]); continue
        if t.is_relative_detector_id(): parts[-1].append(t.val); full^={t.val}
        else: fo^=1<<t.val
    comp_of.setdefault((tuple(sorted(full)),fo),[]).append([tuple(sorted(p)) for p in parts])
cnt={0:0,5:0}
lines={0:0,5:0}
for e in m.explain_detector_error_model_errors(reduce_to_one_representative_error=False):
    d=tuple(sorted(t.dem_target.val for t in e.dem_error_terms if t.dem_target.is_relative_detector_id()))
    o=sum(1<<t.dem_target.val for t in e.dem_error_terms if t.dem_target.is_logical_observable_id())
    for decomp in comp_of.get((d,o),[]):
        for b in (0,5):
            if (b,) in decomp:
                cnt[b]+=len(e.circuit_error_locations); lines[b]+=1
        break
print("circuit locations feeding boundary edges:",cnt, "DEM lines:",lines)
out["boundary_feed"]={"locations":cnt,"dem_lines":lines}
(HERE/"v_remnants.json").write_text(json.dumps(out,indent=1))
