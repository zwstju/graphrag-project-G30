const API='http://127.0.0.1:8000';
const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');

async function health(){
  try{
    const r=await fetch(API+'/health');
    const d=await r.json();
    if(!r.ok||d.status!=='ok') throw new Error();
    $('dot').style.background='#18a46b'; $('healthText').textContent='后端已连接';
  }catch{
    $('dot').style.background='#d84b4b'; $('healthText').textContent='后端未连接';
  }
}

document.querySelectorAll('[data-q]').forEach(b=>b.onclick=()=>{$('query').value=b.dataset.q});
document.querySelectorAll('input[name=mode]').forEach(r=>r.onchange=()=>{$('badge').textContent=r.value.toUpperCase()});

document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>{
  document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('active',x===t));
  document.querySelectorAll('.panel').forEach(x=>x.classList.remove('active'));
  $(t.dataset.tab).classList.add('active');
});

$('ask').onclick=async()=>{
  const query=$('query').value.trim();
  const mode=document.querySelector('input[name=mode]:checked').value;
  $('error').classList.add('hidden');
  if(!query){$('error').textContent='请输入问题。';$('error').classList.remove('hidden');return}
  $('ask').disabled=true;$('ask').textContent='正在检索并生成...';
  try{
    const r=await fetch(API+'/ask',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({query,mode})
    });
    const d=await r.json();
    if(!r.ok) throw new Error(d.detail||('HTTP '+r.status));
    render(d);
  }catch(e){
    $('error').textContent='请求失败：'+e.message;
    $('error').classList.remove('hidden');
  }finally{
    $('ask').disabled=false;$('ask').textContent='开始提问';
  }
};

function render(d){
  $('answer').classList.remove('muted');
  $('answer').innerHTML = DOMPurify.sanitize(
    marked.parse(d.answer || '')
  );
  $('badge').textContent=(d.mode||'').toUpperCase();

  const vr=Array.isArray(d.vector_results)?d.vector_results:[];
  $('vector').innerHTML=vr.length?vr.map((x,i)=>`
    <div class="item">
      <strong>${i+1}. ${esc(x.title||x.paper_id||'Untitled')}</strong>
      <div class="meta">paper_id=${esc(x.paper_id)} · score=${esc(x.score)} ${x.section?'· '+esc(x.section):''}</div>
      <div>${esc(x.text)}</div>
    </div>`).join(''):'<p class="muted">没有 Vector evidence。</p>';

  const g=d.graph_results||{};
  const entities=Array.isArray(g.entities)?g.entities:[];
  const triples=Array.isArray(g.triples)?g.triples:[];
  const paths=Array.isArray(g.paths)?g.paths:[];
  $('graph').innerHTML=`
    <h3>Entities</h3><div>${entities.length?entities.map(x=>`<span class="chip">${esc(x)}</span>`).join(''):'<span class="muted">无</span>'}</div>
    <h3>Triples</h3>${triples.length?triples.map(x=>`<div class="triple">${esc(x.source)} --${esc(x.relation)}--&gt; ${esc(x.target)}<br>paper_id=${esc(x.paper_id)}</div>`).join(''):'<p class="muted">无</p>'}
    <h3>Paths</h3>${paths.length?paths.map(x=>`<div class="path">${esc((x.nodes||[]).join(' → '))}<br>${esc((x.relations||[]).join(' / '))}</div>`).join(''):'<p class="muted">无</p>'}`;

  const refs=Array.isArray(d.references)?d.references:[];
  $('references').innerHTML=refs.length?refs.map(x=>`<div class="item"><strong>${esc(x.paper_id)}</strong><div>${esc(x.title||'')}</div></div>`).join(''):'<p class="muted">没有 references。</p>';
}
health();
