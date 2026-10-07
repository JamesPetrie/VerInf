import re,sys,json,dukpy
sys.path.insert(0,'analysis/bench')
src=open('analysis/bench/wc_bridge_explainer.html').read()
markup=src[:src.index('<script>')]; js=src[src.index('<script>')+8:src.rindex('</script>')]
els={}
for m in re.finditer(r'<(input|select)\b([^>]*)>', markup):
    idm=re.search(r'id="([^"]+)"',m.group(2))
    if not idm: continue
    a=m.group(2); eid=idm.group(1)
    els[eid]={'value':(re.search(r'value="([^"]+)"',a).group(1) if 'value="' in a else ''),
              'checked':'checked' in a,'type':(re.search(r'type="([^"]+)"',a).group(1) if 'type=' in a else 'select')}
for m in re.finditer(r'<select[^>]*id="([^"]+)"[^>]*>(.*?)</select>', markup, re.S):
    opts=re.findall(r'<option([^>]*)>([^<]*)</option>', m.group(2))
    vals=[((re.search(r'value="([^"]+)"',a).group(1) if 'value="' in a else t.strip()),'selected' in a) for a,t in opts]
    els[m.group(1)]['value']=next((v for v,s_ in vals if s_), vals[0][0])
sinks=set(re.findall(r'id="([A-Za-z0-9_]+)"',markup))-set(els)
dom={k:{'value':v['value'],'checked':v['checked'],'type':v['type']} for k,v in els.items()}
for x in sinks: dom[x]={'value':'','checked':False,'type':'text'}
SHIM=open('analysis/bench/knob_selftest.py').read()
shim=SHIM[SHIM.index('SHIM = """')+10:SHIM.index('""" % json.dumps(dom)')]
def run(seq):
    code=(shim % json.dumps(dom)) if '%s' in shim else shim.replace('%s', json.dumps(dom))
    code=code.replace('%s', json.dumps(dom)) if '%s' in code else code
    code+=js+"\n__dom['cSeq'].value='%d';\n" % seq
    code+="var ls=__listeners['cSeq']||[]; for(var i=0;i<ls.length;i++) ls[i][1]();\n"
    code+="JSON.stringify(__out);"
    return json.loads(dukpy.evaljs(code))
print(f"{'контекст':>9} {'итог':>9} | прогоны / упаковка / умножения")
for seq in (1093,4096,16384,65536):
    o=run(seq)
    print(f"{seq:9d} {o.get('oTotal'):>9} | {o.get('sWit')} / {o.get('sStr')} / {o.get('sQd')}")
