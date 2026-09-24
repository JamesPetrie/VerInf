"""Исполняет ручку страницы в настоящем JS-движке с DOM-заглушкой
и проверяет, что каждый элемент управления меняет результат."""
import re, sys, json, dukpy

PAGE = sys.argv[1] if len(sys.argv) > 1 else 'analysis/bench/wc_bridge_explainer.html'
src = open(PAGE).read()
markup = src[:src.index('<script>')]
js = src[src.index('<script>')+8 : src.rindex('</script>')]

# --- собираем элементы из разметки: id, тег, тип, начальное значение ---
els = {}
for m in re.finditer(r'<(input|select|button)\b([^>]*)>', markup):
    tag, attrs = m.group(1), m.group(2)
    idm = re.search(r'id="([^"]+)"', attrs)
    if not idm: continue
    eid = idm.group(1)
    typ = (re.search(r'type="([^"]+)"', attrs) or [None,''])[1] if 'type=' in attrs else ('select' if tag=='select' else 'button')
    val = (re.search(r'value="([^"]+)"', attrs).group(1) if 'value="' in attrs else '')
    els[eid] = {'tag':tag,'type':typ,'value':val,'checked':'checked' in attrs}
# опции select: значение по умолчанию — selected, иначе первая
for m in re.finditer(r'<select[^>]*id="([^"]+)"[^>]*>(.*?)</select>', markup, re.S):
    eid, body = m.group(1), m.group(2)
    opts = re.findall(r'<option([^>]*)>([^<]*)</option>', body)
    vals = [(re.search(r'value="([^"]+)"',a).group(1) if 'value="' in a else t.strip(), 'selected' in a) for a,t in opts]
    els[eid]['value'] = next((v for v,sel in vals if sel), vals[0][0] if vals else '')
# текстовые приёмники (то, куда пишет compute)
sinks = set(re.findall(r'id="([A-Za-z0-9_]+)"', markup)) - set(els)

dom = {eid: {'value': e['value'], 'checked': e['checked'], 'type': e['type']} for eid, e in els.items()}
for s_ in sinks: dom[s_] = {'value':'', 'checked':False, 'type':'text'}

SHIM = """
var __dom = %s, __out = {};
function __El(id){ this.id=id; this._d=__dom[id]; }
Object.defineProperty(__El.prototype,'value',{get:function(){return this._d.value;},set:function(v){this._d.value=String(v);}});
Object.defineProperty(__El.prototype,'checked',{get:function(){return !!this._d.checked;},set:function(v){this._d.checked=!!v;}});
Object.defineProperty(__El.prototype,'type',{get:function(){return this._d.type;}});
Object.defineProperty(__El.prototype,'textContent',{get:function(){return __out[this.id]||'';},set:function(v){__out[this.id]=String(v);}});
Object.defineProperty(__El.prototype,'innerHTML',{get:function(){return __out[this.id]||'';},set:function(v){__out[this.id]=String(v);}});
Object.defineProperty(__El.prototype,'className',{get:function(){return __out[this.id+'#class']||'';},set:function(v){__out[this.id+'#class']=String(v);}});
__El.prototype.addEventListener=function(ev,fn){ (__listeners[this.id]=__listeners[this.id]||[]).push([ev,fn]); };
Object.defineProperty(__El.prototype,'style',{get:function(){ return (__styles[this.id]=__styles[this.id]||{}); }});
var __styles={};
var __listeners={};
var document={ getElementById:function(id){ return (id in __dom)? new __El(id) : null; },
               querySelector:function(){return null;}, addEventListener:function(){},
               documentElement:{style:{},dataset:{}}, body:{} };
var window={addEventListener:function(){},matchMedia:function(){return{matches:false,addEventListener:function(){}};}};
var console={log:function(){},warn:function(){},error:function(){}};
""" % json.dumps(dom)

DRIVER = """
function __fire(id){
  var ls=__listeners[id]||[];
  for(var i=0;i<ls.length;i++){ ls[i][1](); }
  return ls.length;
}
function __snapshot(){ return JSON.stringify(__out); }
"""

def run(actions):
    """actions: список [id, вид, значение]; возвращает (снимок, число слушателей)"""
    code = SHIM + js + DRIVER + "var __r={fired:0};\n"
    for eid, kind, val in actions:
        if kind == 'check':
            code += "__dom[%s].checked=%s; __r.fired+=__fire(%s);\n" % (json.dumps(eid), 'true' if val else 'false', json.dumps(eid))
        else:
            code += "__dom[%s].value=%s; __r.fired+=__fire(%s);\n" % (json.dumps(eid), json.dumps(str(val)), json.dumps(eid))
    code += "JSON.stringify({out:__out, styles:__styles, fired:__r.fired});"
    try:
        res = json.loads(dukpy.evaljs(code))
        return dict(res['out'], **{'#style:'+k: json.dumps(v) for k,v in res.get('styles',{}).items()}), res['fired'], None
    except Exception as e:
        return None, 0, str(e).split('\n')[0][:200]

base, fired0, err = run([])
if err:
    print("СТРАНИЦА НЕ ИСПОЛНЯЕТСЯ:", err); sys.exit(1)
key = 'oTotal'
print(f"базовое значение {key}: {base.get(key,'—')}   (обработчиков навешано при загрузке: см. ниже)\n")

bad = []
for eid, e in sorted(els.items()):
    if e['tag'] == 'button': continue
    if e['type'] == 'checkbox':
        acts = [(eid,'check', not e['checked'])]
        label = ('снять' if e['checked'] else 'поставить') + ' ' + eid
    elif e['type'] == 'range':
        try: newv = float(e['value']) * 1.5 or 1
        except: newv = 1
        acts = [(eid,'value', newv)]; label = f"сдвинуть {eid}"
    else:
        acts = None
        m = re.search(r'<select[^>]*id="%s"[^>]*>(.*?)</select>' % re.escape(eid), markup, re.S)
        if m:
            vals = [ (re.search(r'value="([^"]+)"',a).group(1) if 'value="' in a else t.strip())
                     for a,t in re.findall(r'<option([^>]*)>([^<]*)</option>', m.group(1)) ]
            alt = next((v for v in vals if v != e['value']), None)
            if alt is not None: acts = [(eid,'value',alt)]; label = f"переключить {eid} → {alt}"
        if acts is None: continue
    out, fired, err = run(acts)
    if err:
        print(f"  ✗ {label:34s} ОШИБКА JS: {err}"); bad.append(eid); continue
    if fired == 0:
        print(f"  ✗ {label:34s} НЕТ ОБРАБОТЧИКА — клик ничего не вызывает"); bad.append(eid); continue
    changed = out.get(key) != base.get(key)
    other = sum(1 for k in out if out.get(k) != base.get(k))
    mark = '✓' if changed else ('~' if other else '✗')
    if not changed and not other: bad.append(eid)
    print(f"  {mark} {label:34s} {base.get(key,'')} → {out.get(key,'')}" + ('' if changed else f'   (изменилось полей: {other})'))


# --- условные элементы: проверяем в конфигурации, где они обязаны работать ---
print()
print("проверка условных элементов в их рабочей конфигурации:")
for eid, enable, label in [('cBw', [('cSpill','check',True)], 'скорость диска при включённом отказе от повторных прогонов'),
                           ('cBw', [('cWcDisk','check',True)], 'скорость диска при регистрации на диске')]:
    b2,_,_ = run(enable)
    o2,f2,e2 = run(enable + [(eid,'value','2')])
    b2, o2 = b2 or {}, o2 or {}
    diff = [k for k in o2 if o2.get(k) != b2.get(k)]
    ok = bool(diff)
    detail = f"{b2.get('oTotal')} → {o2.get('oTotal')}"
    if ok and 'oTotal' not in diff:
        detail += f"  (итог не сдвинулся заметно; изменились: {', '.join(diff[:3])})"
    print(f"  {'✓' if ok else '✗'} {label:58s} {detail}")


# --- отрисовка: полоски разной длины, цвета не повторяются ---
print()
print("отрисовка:")
widths = {k[len('#style:'):]: json.loads(v).get('width') for k,v in base.items() if k.startswith('#style:')}
bars = {k:v for k,v in widths.items() if k.startswith('b') and v}
uniq = len(set(bars.values()))
print(f"  {'✓' if uniq>1 else '✗'} длины полосок различаются: {uniq} разных значений из {len(bars)}  {bars}")

rows = re.findall(r'<div class="oterm">.*?<span class="chip ([a-z0-9]+)".*?</div>', markup, re.S)
dup = [c for c in set(rows) if rows.count(c) > 1]
print(f"  {'✓' if not dup else '✗'} цвета строк уникальны" + (f" — ПОВТОР: {dup}" if dup else ""))

css = markup[markup.index('<style>'):markup.index('</style>')]
colors = dict(re.findall(r'\.(c[a-z0-9]+)\{background:var\((--[a-z]+)\)\}', css))
same = {}
for cls, tok in colors.items(): same.setdefault(tok, []).append(cls)
clash = {t:c for t,c in same.items() if len(c)>1 and any(x in rows for x in c)}
print(f"  {'✓' if not clash else '✗'} один цвет — одна строка" + (f" — СТОЛКНОВЕНИЕ: {clash}" if clash else ""))

print()
print("ИТОГ:", "все элементы управления живые" if not bad else "требуют условия: " + ", ".join(bad))
sys.exit(0)
