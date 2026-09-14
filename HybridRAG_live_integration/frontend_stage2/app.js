
const $ = id => document.getElementById(id);

let current = null;
let evidenceView = "before";

const esc = s =>
  String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");

const clip = (s, n=170) => {
  const t = String(s ?? "").replace(/\s+/g, " ").trim();
  return t.length > n ? t.slice(0, n - 1) + "…" : t;
};

function simpleMarkdown(text) {
  let s = esc(text || "");
  s = s.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  const lines = s.split("\n");
  let html = "", inList = false;
  for (const line of lines) {
    const t = line.trim();
    if (/^[-•]\s+/.test(t)) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += `<li>${t.replace(/^[-•]\s+/, "")}</li>`;
    } else {
      if (inList) { html += "</ul>"; inList = false; }
      if (t) html += `<p>${t}</p>`;
    }
  }
  if (inList) html += "</ul>";
  return html;
}

async function loadHealth() {
  try {
    const r = await fetch("/api/health");
    const d = await r.json();
    if (!r.ok || d.status !== "ok") throw new Error();

    $("status").className = "status online";
    $("statusText").textContent = "Stage2 已连接";

    $("papers").textContent = d.papers ?? "—";
    $("chunks").textContent = d.retrieval_chunks ?? "—";
    $("nodes").textContent = Number(d.graph_nodes || 0).toLocaleString();
    $("edges").textContent = Number(d.graph_edges || 0).toLocaleString();
  } catch {
    $("status").className = "status offline";
    $("statusText").textContent = "后端未连接";
  }
}

document.querySelectorAll("[data-q]").forEach(btn => {
  btn.addEventListener("click", () => {
    $("query").value = btn.dataset.q;
  });
});

document.querySelectorAll('input[name="method"]').forEach(r => {
  r.addEventListener("change", () => {
    $("methodPill").textContent = r.value.toUpperCase();
    const full = r.value === "full";
    $("verify").disabled = !full;
    if (!full) $("verify").checked = false;
  });
});

$("verify").disabled = true;

$("beforeBtn").addEventListener("click", () => {
  evidenceView = "before";
  renderEvidence();
});
$("afterBtn").addEventListener("click", () => {
  evidenceView = "after";
  renderEvidence();
});

$("ask").addEventListener("click", ask);

async function ask() {
  const query = $("query").value.trim();
  if (!query) return;

  const method = document.querySelector('input[name="method"]:checked').value;
  const verify = $("verify").checked;

  $("error").classList.add("hidden");
  $("ask").disabled = true;
  $("ask").textContent = "正在检索并生成…";
  $("answer").className = "answer placeholder";
  $("answer").textContent = "正在运行 Stage2 pipeline…";
  clearGraph();

  try {
    const r = await fetch("/api/ask", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({query, method, verify})
    });

    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || `HTTP ${r.status}`);

    current = d;
    evidenceView = d.before_evidence?.length ? "before" : "after";
    renderAll();
  } catch (e) {
    $("error").textContent = "请求失败：" + e.message;
    $("error").classList.remove("hidden");
    $("answer").className = "answer placeholder";
    $("answer").textContent = "本次请求失败。";
  } finally {
    $("ask").disabled = false;
    $("ask").textContent = "开始提问";
  }
}

function renderAll() {
  $("methodPill").textContent = (current.method || "").toUpperCase();

  $("answer").className = "answer";
  $("answer").innerHTML = simpleMarkdown(current.answer || "没有生成答案。");

  const ms = current.request_latency_ms;
  $("latency").textContent =
    ms != null ? `End-to-end request latency: ${(ms / 1000).toFixed(2)} s` : "";

  renderEvidence();
  renderGraph(current.graph || {nodes:[], edges:[]});
  renderVerification(current.checks || {});
}

function renderEvidence() {
  if (!current) return;

  const hasBefore = Array.isArray(current.before_evidence) && current.before_evidence.length;
  $("beforeBtn").disabled = !hasBefore;
  $("beforeBtn").classList.toggle("active", evidenceView === "before" && hasBefore);
  $("afterBtn").classList.toggle("active", evidenceView === "after" || !hasBefore);

  let rows;
  if (evidenceView === "before" && hasBefore) rows = current.before_evidence;
  else rows = current.evidence || [];

  const afterKeys = new Set(
    (current.evidence || []).map(e =>
      [e.evidence_id, e.paper_id, e.source_type, e.content].join("|")
    )
  );

  $("evidenceList").innerHTML = rows.length
    ? rows.map(e => {
        const key = [e.evidence_id, e.paper_id, e.source_type, e.content].join("|");
        const filtered = hasBefore && evidenceView === "before" && !afterKeys.has(key);
        const score = evidenceView === "after"
          ? (e.rerank_score ?? e.retrieval_score)
          : e.retrieval_score;

        return `
          <article class="evidence-card ${filtered ? "filtered" : ""}">
            <div class="ev-top">
              <span class="source ${esc(e.source_type)}">${esc((e.source_type || "vector").toUpperCase())}</span>
              <span class="score">${score == null ? "—" : Number(score).toFixed(3)}</span>
            </div>
            <div class="ev-title">${esc(e.title || e.paper_id || "Evidence")}</div>
            <div class="ev-meta">${esc(e.paper_id || "")}${e.section ? " · " + esc(e.section) : ""}</div>
            <div class="ev-text">${esc(clip(e.content))}</div>
            ${filtered ? '<span class="filtered-mark">FILTERED BY RERANKER</span>' : ""}
          </article>
        `;
      }).join("")
    : '<div class="verification-empty">当前方法没有可展示的 evidence。</div>';
}

function clearGraph() {
  $("edgeLayer").innerHTML = "";
  $("nodeLayer").innerHTML = "";
  $("graphEmpty").style.display = "grid";
}

function renderGraph(graph) {
  const nodes = Array.isArray(graph.nodes) ? graph.nodes : [];
  const edges = Array.isArray(graph.edges) ? graph.edges : [];

  clearGraph();
  if (!nodes.length) return;
  $("graphEmpty").style.display = "none";

  const W = 760, H = 390;
  const cx = W / 2, cy = H / 2;
  const entityNodes = nodes.filter(n => n.type !== "paper");
  const paperNodes = nodes.filter(n => n.type === "paper");

  const positions = new Map();

  const placeRing = (arr, rx, ry, offset=0) => {
    arr.forEach((n, i) => {
      const angle = offset + (Math.PI * 2 * i / Math.max(arr.length, 1));
      positions.set(n.id, {
        x: cx + Math.cos(angle) * rx,
        y: cy + Math.sin(angle) * ry
      });
    });
  };

  placeRing(entityNodes, Math.min(255, 75 + entityNodes.length * 24), 105, -Math.PI / 2);
  placeRing(paperNodes, Math.min(295, 105 + paperNodes.length * 24), 145, Math.PI / 2);

  const svgNS = "http://www.w3.org/2000/svg";

  for (const edge of edges.slice(0, 24)) {
    const a = positions.get(edge.source);
    const b = positions.get(edge.target);
    if (!a || !b) continue;

    const line = document.createElementNS(svgNS, "path");
    line.setAttribute("d", `M ${a.x} ${a.y} L ${b.x} ${b.y}`);
    line.setAttribute(
      "class",
      "graph-edge " + (edge.relation === "PROVENANCE" ? "provenance" : "")
    );
    $("edgeLayer").appendChild(line);

    if (edge.relation !== "PROVENANCE") {
      const label = document.createElementNS(svgNS, "text");
      label.setAttribute("x", (a.x + b.x) / 2);
      label.setAttribute("y", (a.y + b.y) / 2 - 5);
      label.setAttribute("class", "edge-label");
      label.textContent = edge.relation;
      $("edgeLayer").appendChild(label);
    }
  }

  for (const node of nodes.slice(0, 22)) {
    const p = positions.get(node.id);
    if (!p) continue;

    const g = document.createElementNS(svgNS, "g");
    const isPaper = node.type === "paper";

    const rect = document.createElementNS(svgNS, "rect");
    const w = isPaper ? 126 : 112;
    const h = isPaper ? 42 : 52;
    rect.setAttribute("x", p.x - w/2);
    rect.setAttribute("y", p.y - h/2);
    rect.setAttribute("width", w);
    rect.setAttribute("height", h);
    rect.setAttribute("rx", 13);
    rect.setAttribute("class", `node-rect ${isPaper ? "paper" : "entity"}`);
    g.appendChild(rect);

    const label = document.createElementNS(svgNS, "text");
    label.setAttribute("x", p.x);
    label.setAttribute("y", p.y + 2);
    label.setAttribute("class", "node-label");
    label.textContent = clip(node.label || node.id, isPaper ? 20 : 16);
    g.appendChild(label);

    const sub = document.createElementNS(svgNS, "text");
    sub.setAttribute("x", p.x);
    sub.setAttribute("y", p.y + 15);
    sub.setAttribute("class", "node-sub");
    sub.textContent = isPaper ? "PAPER" : "ENTITY";
    g.appendChild(sub);

    $("nodeLayer").appendChild(g);
  }
}

function renderVerification(checks) {
  const cv = checks?.claim_verification;
  if (!cv) {
    $("verifyState").textContent = "NOT RUN";
    $("verifyState").className = "pill muted-pill";
    $("verification").className = "verification-empty";
    $("verification").textContent =
      "Full 模式下勾选 Claim verification 才会运行实时验证。录屏时建议关闭，避免等待。";
    return;
  }

  $("verifyState").textContent = "VERIFIED";
  $("verifyState").className = "pill";
  $("verification").className = "claim-list";

  const claims = Array.isArray(cv.claims) ? cv.claims : [];
  if (!claims.length) {
    $("verification").innerHTML =
      `<div class="verification-empty">${esc(JSON.stringify(cv))}</div>`;
    return;
  }

  $("verification").innerHTML = claims.map((c, i) => `
    <article class="claim ${c.supported ? "supported" : "unsupported"}">
      <div class="claim-head">
        <strong>${c.supported ? "✓ SUPPORTED" : "× UNSUPPORTED"}</strong>
        <span>Claim ${i + 1}</span>
      </div>
      <p>${esc(c.claim || c.text || "")}</p>
    </article>
  `).join("");
}

loadHealth();
