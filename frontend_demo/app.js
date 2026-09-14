
const D = window.DEMO_DATA;
const $ = id => document.getElementById(id);
const caseData = D.case;

let currentStep = 0;
let playing = false;
let timer = null;
let evidenceMode = "before";

const steps = [
  {title:"Ready", desc:"点击播放，依次展示 Hybrid evidence、Reranking、Answer 与 Verification。", delay:900},
  {title:"Hybrid retrieval", desc:"Vector 与 Graph evidence 同时进入候选池；右侧展示当前问题的局部图关系。", delay:3000},
  {title:"Evidence reranking", desc:"异构证据统一打分，Graph 候选若不够有用会被降权或移出最终 Top-k。", delay:3000},
  {title:"Grounded generation", desc:"LLM 只基于最终选中的证据生成回答；本案例 judge score 从 0.50 恢复到 1.00。", delay:3500},
  {title:"Claim verification", desc:"冻结原答案后逐 claim 对照实际 evidence；本案例 7/9 claims 得到支持。", delay:5000}
];

function esc(s){
  return String(s ?? "").replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;");
}

function fmtScore(v){
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(3) : "—";
}

function clip(s,n=150){
  const t = String(s ?? "").replace(/\s+/g," ").trim();
  return t.length > n ? t.slice(0,n-1)+"…" : t;
}

function basicMarkdown(text){
  let s = esc(text || "");
  s = s.replace(/\*\*(.*?)\*\*/g,"<strong>$1</strong>");
  const lines = s.split("\n");
  let html = "", inList = false;
  for(const line of lines){
    const t = line.trim();
    if(/^[-•]\s+/.test(t)){
      if(!inList){ html += "<ul>"; inList = true; }
      html += `<li>${t.replace(/^[-•]\s+/,"")}</li>`;
    }else{
      if(inList){ html += "</ul>"; inList = false; }
      if(t) html += `<p>${t}</p>`;
    }
  }
  if(inList) html += "</ul>";
  return html;
}

function initMetrics(){
  const p = D.project;
  $("mPapers").textContent = p.papers;
  $("mChunks").textContent = p.retrieval_chunks;
  $("mGraph").textContent = Number(p.graph_nodes).toLocaleString();
  $("mSupport").textContent = (p.global_claim_support_rate*100).toFixed(1)+"%";
  $("caseQid").textContent = caseData.qid;
  $("queryBox").textContent = caseData.question;
  $("scoreVector").textContent = Number(caseData.scores.vector).toFixed(2);
  $("scoreHybrid").textContent = Number(caseData.scores.hybrid).toFixed(2);
  $("scoreRerank").textContent = Number(caseData.scores.rerank).toFixed(2);
}

function setPipeline(step){
  document.querySelectorAll(".pipe-step").forEach((el,i)=>{
    el.classList.toggle("active", i+1===step);
    el.classList.toggle("done", i+1<step);
  });
  document.querySelectorAll(".pipe-line").forEach((el,i)=>{
    el.classList.toggle("done", i+1<step);
  });
}

function parseTriple(content){
  const m = String(content||"").match(/^\s*(.*?)\s+--([A-Z0-9_ -]+)-->\s+(.*?)\s*$/i);
  return m ? {source:m[1].trim(), relation:m[2].trim(), target:m[3].trim()} : null;
}

function graphTriples(){
  const triples = [];
  for(const e of caseData.vanilla_evidence){
    if(e.source_type === "triple"){
      const t = parseTriple(e.content);
      if(t) triples.push({...t, paper_id:e.paper_id});
    }
  }
  return triples;
}

function renderGraph(active=true, reranked=false){
  const triples = graphTriples();
  const edgesG = $("graphEdges");
  const nodesG = $("graphNodes");
  edgesG.innerHTML = "";
  nodesG.innerHTML = "";

  const nodePos = {
    "GraphSAGE": [205,138],
    "GCN": [510,138],
    "graphsage_2017": [190,286],
    "diffpool_2018": [515,286]
  };

  const uniquePapers = [...new Set(triples.map(t=>t.paper_id).filter(Boolean))];
  const rels = [];
  for(const t of triples){
    if(!rels.some(x=>x.source===t.source && x.target===t.target && x.relation===t.relation)){
      rels.push(t);
    }
  }

  const ns = new Set();
  rels.forEach(t=>{ns.add(t.source);ns.add(t.target);});
  uniquePapers.forEach(p=>ns.add(p));

  // positions for unexpected nodes
  let extra = 0;
  for(const n of ns){
    if(!nodePos[n]){
      nodePos[n] = [150 + (extra%3)*210, 110 + Math.floor(extra/3)*110];
      extra++;
    }
  }

  function edgePath(x1,y1,x2,y2,curve=0){
    if(!curve) return `M ${x1} ${y1} L ${x2} ${y2}`;
    const mx=(x1+x2)/2, my=(y1+y2)/2-curve;
    return `M ${x1} ${y1} Q ${mx} ${my} ${x2} ${y2}`;
  }

  rels.forEach((t,i)=>{
    const [x1,y1]=nodePos[t.source], [x2,y2]=nodePos[t.target];
    const curve = i===0 ? 34 : -34;
    const p = document.createElementNS("http://www.w3.org/2000/svg","path");
    p.setAttribute("d",edgePath(x1+53,y1,x2-53,y2,curve));
    p.setAttribute("class","graph-edge "+(i>0?"secondary":""));
    p.style.opacity = active ? (reranked ? ".2" : ".95") : ".08";
    edgesG.appendChild(p);

    const tx=document.createElementNS("http://www.w3.org/2000/svg","text");
    tx.setAttribute("x",(x1+x2)/2);
    tx.setAttribute("y",y1 + (i===0 ? -38 : 48));
    tx.setAttribute("class","edge-label");
    tx.style.opacity = active ? (reranked ? ".28" : "1") : ".08";
    tx.textContent=t.relation;
    edgesG.appendChild(tx);
  });

  // provenance links
  uniquePapers.forEach((p,i)=>{
    const model=i===0?"GraphSAGE":"GCN";
    const [x1,y1]=nodePos[model], [x2,y2]=nodePos[p];
    const line=document.createElementNS("http://www.w3.org/2000/svg","path");
    line.setAttribute("d",`M ${x1} ${y1+38} L ${x2} ${y2-27}`);
    line.setAttribute("class","graph-edge secondary");
    line.style.opacity = active ? (reranked ? ".12" : ".5") : ".05";
    edgesG.appendChild(line);
  });

  for(const n of ns){
    const [x,y]=nodePos[n];
    const isPaper=n.includes("_20");
    const g=document.createElementNS("http://www.w3.org/2000/svg","g");
    g.style.opacity = active ? (reranked && !isPaper ? ".55" : "1") : ".15";

    const c=document.createElementNS("http://www.w3.org/2000/svg","rect");
    c.setAttribute("x",x-(isPaper?72:62));
    c.setAttribute("y",y-(isPaper?25:32));
    c.setAttribute("width",isPaper?144:124);
    c.setAttribute("height",isPaper?50:64);
    c.setAttribute("rx",isPaper?14:18);
    c.setAttribute("class","node-circle "+(isPaper?"paper":"model"));
    g.appendChild(c);

    const label=document.createElementNS("http://www.w3.org/2000/svg","text");
    label.setAttribute("x",x);label.setAttribute("y",y+(isPaper?4:0));
    label.setAttribute("class","node-label");
    label.textContent=isPaper?n.replace("_"," · "):n;
    g.appendChild(label);

    if(!isPaper){
      const sub=document.createElementNS("http://www.w3.org/2000/svg","text");
      sub.setAttribute("x",x);sub.setAttribute("y",y+17);
      sub.setAttribute("class","node-sub");
      sub.textContent="MODEL";
      g.appendChild(sub);
    }
    nodesG.appendChild(g);
  }

  $("graphCaption").textContent = !active
    ? "等待 Hybrid retrieval…"
    : reranked
      ? "Reranker 后，当前 Graph triples 未进入最终 Top-6 evidence；图关系被保留为检索 provenance。"
      : "Vanilla Hybrid 的 Graph evidence：GraphSAGE → GCN，关系来自 graphsage_2017 / diffpool_2018。";
}

function renderEvidence(mode){
  evidenceMode = mode;
  $("beforeBtn").classList.toggle("active",mode==="before");
  $("afterBtn").classList.toggle("active",mode==="after");
  const evs = mode==="before" ? caseData.vanilla_evidence : caseData.rerank_evidence;
  const afterIds = new Set(caseData.rerank_evidence.map(e=>e.evidence_id+"|"+e.paper_id+"|"+e.source_type));

  $("evidenceList").innerHTML = evs.map((e,i)=>{
    const keep = afterIds.has(e.evidence_id+"|"+e.paper_id+"|"+e.source_type);
    const filtered = mode==="before" && !keep;
    const score = mode==="after" ? e.rerank_score : e.retrieval_score;
    const source = (e.source_type || "vector").toUpperCase();
    return `
      <article class="evidence-card fade-up ${filtered?"filtered":""}" style="animation-delay:${i*55}ms">
        <div class="ev-top">
          <span class="source-badge ${esc(e.source_type)}">${esc(source)}</span>
          <span class="ev-score">${mode==="after"?"rerank ":"score "}${fmtScore(score)}</span>
        </div>
        <div class="ev-title">${esc(e.title || e.paper_id || e.content)}</div>
        <div class="ev-meta">${esc(e.paper_id || "")}${e.section ? " · "+esc(e.section):""}</div>
        <div class="ev-text">${esc(clip(e.content,165))}</div>
        ${filtered?'<span class="filtered-label">FILTERED</span>':""}
      </article>
    `;
  }).join("");
}

function renderClaims(){
  const cs = caseData.claims || [];
  $("claimList").innerHTML = cs.slice(0,5).map(c=>`
    <div class="claim ${c.supported?"supported":"unsupported"} fade-up">
      <span class="claim-icon">${c.supported?"✓":"×"}</span>
      <span class="claim-text">${esc(clip(c.claim,110))}</span>
      <span class="claim-ev">${c.evidence_ids?.length ? esc(c.evidence_ids.join(", ")) : "no evidence"}</span>
    </div>
  `).join("");
  const pct = Number(caseData.claim_support_rate)*100;
  $("caseSupport").textContent = pct.toFixed(1)+"%";
  $("supportRing").style.background = `conic-gradient(var(--green) 0 ${pct}%,#1a3049 ${pct}% 100%)`;
  $("verifySummary").textContent =
    `${caseData.supported_claims}/${caseData.total_claims} claims supported · answer SHA-256 matched · answer_regenerated=false`;
}

function resetView(){
  currentStep=0;
  playing=false;
  if(timer) clearTimeout(timer);
  timer=null;
  $("playBtn").disabled=false;
  $("playBtn").textContent="▶ 播放演示";
  $("progressBar").style.width="0%";
  $("stepTitle").textContent=steps[0].title;
  $("stepDesc").textContent=steps[0].desc;
  setPipeline(1);
  renderEvidence("before");
  renderGraph(false,false);
  $("answer").className="answer muted";
  $("answer").innerHTML="播放演示后，系统将展示经过证据重排后的最终回答。";
  $("answerState").textContent="WAITING";
  $("answerState").className="answer-state";
  $("caseSupport").textContent="—";
  $("supportRing").style.background="conic-gradient(var(--green) 0 0%,#1a3049 0 100%)";
  $("claimList").innerHTML='<div class="empty-state">验证结果将在最后一步显示。</div>';
  $("verifySummary").textContent="Saved answer hash verified · answer_regenerated=false";
}

function showStep(step){
  currentStep=step;
  const info=steps[step];
  $("stepTitle").textContent=info.title;
  $("stepDesc").textContent=info.desc;
  $("progressBar").style.width=(step/4*100)+"%";

  if(step===1){
    setPipeline(2);
    renderEvidence("before");
    renderGraph(true,false);
    document.querySelector(".graph-panel").classList.add("flash");
    setTimeout(()=>document.querySelector(".graph-panel").classList.remove("flash"),700);
  }
  if(step===2){
    setPipeline(3);
    renderEvidence("after");
    renderGraph(true,true);
    document.querySelector(".evidence-panel").classList.add("flash");
    setTimeout(()=>document.querySelector(".evidence-panel").classList.remove("flash"),700);
  }
  if(step===3){
    setPipeline(4);
    $("answer").className="answer ready fade-up";
    $("answer").innerHTML=basicMarkdown(caseData.answer);
    $("answerState").textContent="GROUNDED";
    $("answerState").className="answer-state ready";
    document.querySelector(".query-panel").classList.add("flash");
    setTimeout(()=>document.querySelector(".query-panel").classList.remove("flash"),700);
  }
  if(step===4){
    setPipeline(5);
    renderClaims();
    document.querySelector(".verify-panel").classList.add("flash");
    setTimeout(()=>document.querySelector(".verify-panel").classList.remove("flash"),700);
  }
}

function playFrom(step=1){
  if(playing) return;
  playing=true;
  $("playBtn").disabled=true;
  const run = s=>{
    if(s>4){
      playing=false;
      $("playBtn").disabled=false;
      $("playBtn").textContent="↻ 再播放一次";
      return;
    }
    showStep(s);
    timer=setTimeout(()=>run(s+1),steps[s].delay);
  };
  run(step);
}

$("playBtn").addEventListener("click",()=>{
  if(currentStep>=4) resetView();
  playFrom(1);
});
$("resetBtn").addEventListener("click",resetView);
$("beforeBtn").addEventListener("click",()=>{
  renderEvidence("before"); renderGraph(true,false);
});
$("afterBtn").addEventListener("click",()=>{
  renderEvidence("after"); renderGraph(true,true);
});

document.addEventListener("keydown",e=>{
  if(e.code==="Space"){
    e.preventDefault();
    if(!playing){
      if(currentStep>=4) resetView();
      playFrom(Math.max(1,currentStep+1));
    }
  }else if(e.code==="ArrowRight" && !playing){
    showStep(Math.min(4,currentStep+1));
  }else if(e.code==="ArrowLeft" && !playing){
    const s=Math.max(0,currentStep-1);
    if(s===0) resetView(); else showStep(s);
  }
});

initMetrics();
resetView();
