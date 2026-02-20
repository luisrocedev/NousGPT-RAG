/**
 * NousGPT RAG Lab — app.js v2
 * Dark mode, Ollama/corpus status, markdown rendering, structured sources.
 */

const el = {
  statusBadge:      document.getElementById('statusBadge'),
  ollamaBadge:      document.getElementById('ollamaBadge'),
  darkModeBtn:      document.getElementById('darkModeBtn'),
  corpusInfo:       document.getElementById('corpusInfo'),
  collection:       document.getElementById('collection'),
  corpusDir:        document.getElementById('corpusDir'),
  chunkSize:        document.getElementById('chunkSize'),
  overlap:          document.getElementById('overlap'),
  embedModel:       document.getElementById('embedModel'),
  chatModel:        document.getElementById('chatModel'),
  topK:             document.getElementById('topK'),
  resetCollection:  document.getElementById('resetCollection'),
  trainBtn:         document.getElementById('trainBtn'),
  trainResult:      document.getElementById('trainResult'),
  searchQuery:      document.getElementById('searchQuery'),
  searchBtn:        document.getElementById('searchBtn'),
  searchResults:    document.getElementById('searchResults'),
  askQuery:         document.getElementById('askQuery'),
  askBtn:           document.getElementById('askBtn'),
  answerBox:        document.getElementById('answerBox'),
  answerContent:    document.getElementById('answerContent'),
  answerSources:    document.getElementById('answerSources'),
};

/* ───── API helper ───── */
async function api(url, method = 'GET', body) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(url, opts);
  const data = await res.json();
  if (!res.ok || data.ok === false) throw new Error(data.error || 'Error inesperado');
  return data;
}

/* ───── Helpers ───── */
function getCommonPayload() {
  return {
    collection: el.collection.value.trim(),
    embed_model: el.embedModel.value.trim(),
    top_k: Number(el.topK.value || 4),
  };
}

function setBusy(btn, busy, label) {
  btn.disabled = busy;
  btn.dataset.label ||= btn.textContent;
  btn.textContent = busy ? label : btn.dataset.label;
}

function escapeHtml(v) {
  return String(v).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

/* ───── Simple markdown → HTML ───── */
function md(text) {
  let h = escapeHtml(text);
  // code blocks
  h = h.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
  // inline code
  h = h.replace(/`([^`]+)`/g, '<code>$1</code>');
  // bold
  h = h.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // italic
  h = h.replace(/\*(.+?)\*/g, '<em>$1</em>');
  // newlines → <br>
  h = h.replace(/\n/g, '<br>');
  return h;
}

/* ───── Dark mode ───── */
function applyDark(dark) {
  document.body.classList.toggle('dark', dark);
  localStorage.setItem('nous-rag-dark', dark ? '1' : '0');
}
(function initDark() {
  const stored = localStorage.getItem('nous-rag-dark');
  const prefer = window.matchMedia('(prefers-color-scheme: dark)').matches;
  applyDark(stored !== null ? stored === '1' : prefer);
})();
el.darkModeBtn?.addEventListener('click', () => {
  applyDark(!document.body.classList.contains('dark'));
});

/* ───── Status badges ───── */
async function refreshStatus() {
  try {
    const d = await api(`/api/status?collection=${encodeURIComponent(el.collection.value.trim())}`);
    el.statusBadge.textContent = `Coleccion ${d.collection} · ${d.chunks} chunks`;
  } catch (e) {
    el.statusBadge.textContent = `Estado: ${e.message}`;
  }
}

async function refreshOllama() {
  try {
    const d = await api('/api/ollama');
    const n = (d.models || []).length;
    el.ollamaBadge.textContent = `Ollama OK · ${n} modelos`;
    el.ollamaBadge.classList.add('ollama-status');
  } catch {
    el.ollamaBadge.textContent = 'Ollama offline';
    el.ollamaBadge.classList.remove('ollama-status');
  }
}

async function refreshCorpus() {
  try {
    const d = await api('/api/corpus');
    const files = d.files || [];
    if (files.length) {
      const names = files.map(f => f.name).join(', ');
      el.corpusInfo.innerHTML = `<strong>${files.length} ficheros</strong> en corpus: ${escapeHtml(names)}`;
    } else {
      el.corpusInfo.textContent = 'Sin ficheros en corpus.';
    }
  } catch {
    el.corpusInfo.textContent = 'No se pudo leer el corpus.';
  }
}

/* ───── Render search results ───── */
function renderSearchResults(items = [], target = el.searchResults) {
  if (!items.length) {
    target.innerHTML = '<article class="item"><p>No hay resultados.</p></article>';
    return;
  }
  target.innerHTML = items.map(r => {
    const pct = Math.max(0, Math.min(100, Math.round((1 - r.score) * 100)));
    return `<article class="item">
      <small>#${r.rank} · score ${r.score} · fuente: ${escapeHtml(r.source)}</small>
      <div class="score-bar"><div class="score-bar-fill" style="width:${pct}%"></div></div>
      <p>${escapeHtml(r.text)}</p>
    </article>`;
  }).join('');
}

/* ───── Render answer sources ───── */
function renderSources(items = []) {
  if (!items.length) {
    el.answerSources.textContent = '';
    return;
  }
  const sources = [...new Set(items.map(r => r.source))];
  el.answerSources.innerHTML = `Fuentes: ${sources.map(s => `<strong>${escapeHtml(s)}</strong>`).join(', ')}`;
}

/* ───── Actions ───── */
el.trainBtn.addEventListener('click', async () => {
  try {
    setBusy(el.trainBtn, true, 'Entrenando...');
    const d = await api('/api/train', 'POST', {
      collection: el.collection.value.trim(),
      corpus_dir: el.corpusDir.value.trim(),
      chunk_size: Number(el.chunkSize.value || 700),
      overlap: Number(el.overlap.value || 120),
      embed_model: el.embedModel.value.trim(),
      reset: el.resetCollection.checked,
    });
    el.trainResult.textContent = JSON.stringify(d, null, 2);
    await refreshStatus();
  } catch (e) {
    el.trainResult.textContent = e.message;
  } finally {
    setBusy(el.trainBtn, false);
  }
});

el.searchBtn.addEventListener('click', async () => {
  const q = el.searchQuery.value.trim();
  if (!q) { renderSearchResults([]); return; }
  try {
    setBusy(el.searchBtn, true, 'Buscando...');
    const d = await api('/api/search', 'POST', { ...getCommonPayload(), query: q });
    renderSearchResults(d.results || []);
  } catch (e) {
    el.searchResults.innerHTML = `<article class="item"><p>${escapeHtml(e.message)}</p></article>`;
  } finally {
    setBusy(el.searchBtn, false);
  }
});

el.askBtn.addEventListener('click', async () => {
  const q = el.askQuery.value.trim();
  if (!q) return;
  try {
    setBusy(el.askBtn, true, 'Generando respuesta...');
    const d = await api('/api/ask', 'POST', {
      ...getCommonPayload(),
      query: q,
      chat_model: el.chatModel.value.trim(),
    });
    el.answerBox.classList.remove('hidden');
    el.answerContent.innerHTML = md(d.answer || '(sin respuesta)');
    renderSources(d.results || []);
    renderSearchResults(d.results || []);
  } catch (e) {
    el.answerBox.classList.remove('hidden');
    el.answerContent.innerHTML = `<span style="color:var(--danger)">${escapeHtml(e.message)}</span>`;
    el.answerSources.textContent = '';
  } finally {
    setBusy(el.askBtn, false);
  }
});

el.collection.addEventListener('change', refreshStatus);

/* ───── Boot ───── */
refreshStatus();
refreshOllama();
refreshCorpus();
