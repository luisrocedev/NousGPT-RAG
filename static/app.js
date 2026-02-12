const el = {
  statusBadge: document.getElementById('statusBadge'),
  collection: document.getElementById('collection'),
  corpusDir: document.getElementById('corpusDir'),
  chunkSize: document.getElementById('chunkSize'),
  overlap: document.getElementById('overlap'),
  embedModel: document.getElementById('embedModel'),
  chatModel: document.getElementById('chatModel'),
  topK: document.getElementById('topK'),
  resetCollection: document.getElementById('resetCollection'),
  trainBtn: document.getElementById('trainBtn'),
  trainResult: document.getElementById('trainResult'),
  searchQuery: document.getElementById('searchQuery'),
  searchBtn: document.getElementById('searchBtn'),
  searchResults: document.getElementById('searchResults'),
  askQuery: document.getElementById('askQuery'),
  askBtn: document.getElementById('askBtn'),
  answerBox: document.getElementById('answerBox'),
};

async function api(url, method = 'GET', body) {
  const response = await fetch(url, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await response.json();
  if (!response.ok || data.ok === false) {
    throw new Error(data.error || 'Error inesperado');
  }
  return data;
}

function getCommonPayload() {
  return {
    collection: el.collection.value.trim(),
    embed_model: el.embedModel.value.trim(),
    top_k: Number(el.topK.value || 4),
  };
}

function setBusy(button, busy, textBusy) {
  button.disabled = busy;
  button.dataset.label ||= button.textContent;
  button.textContent = busy ? textBusy : button.dataset.label;
}

function renderSearchResults(items = []) {
  if (!items.length) {
    el.searchResults.innerHTML = '<article class="item"><p>No hay resultados.</p></article>';
    return;
  }

  el.searchResults.innerHTML = items.map((row) => `
    <article class="item">
      <small>#${row.rank} · score ${row.score} · fuente: ${row.source}</small>
      <p>${escapeHtml(row.text)}</p>
    </article>
  `).join('');
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

async function refreshStatus() {
  try {
    const data = await api(`/api/status?collection=${encodeURIComponent(el.collection.value.trim())}`);
    el.statusBadge.textContent = `Estado: colección ${data.collection} · chunks ${data.chunks}`;
  } catch (error) {
    el.statusBadge.textContent = `Estado: ${error.message}`;
  }
}

el.trainBtn.addEventListener('click', async () => {
  try {
    setBusy(el.trainBtn, true, 'Entrenando...');
    const data = await api('/api/train', 'POST', {
      collection: el.collection.value.trim(),
      corpus_dir: el.corpusDir.value.trim(),
      chunk_size: Number(el.chunkSize.value || 700),
      overlap: Number(el.overlap.value || 120),
      embed_model: el.embedModel.value.trim(),
      reset: el.resetCollection.checked,
    });
    el.trainResult.textContent = JSON.stringify(data, null, 2);
    await refreshStatus();
  } catch (error) {
    el.trainResult.textContent = error.message;
  } finally {
    setBusy(el.trainBtn, false, 'Entrenar base vectorial');
  }
});

el.searchBtn.addEventListener('click', async () => {
  const query = el.searchQuery.value.trim();
  if (!query) {
    renderSearchResults([]);
    return;
  }

  try {
    setBusy(el.searchBtn, true, 'Buscando...');
    const data = await api('/api/search', 'POST', {
      ...getCommonPayload(),
      query,
    });
    renderSearchResults(data.results || []);
  } catch (error) {
    el.searchResults.innerHTML = `<article class="item"><p>${escapeHtml(error.message)}</p></article>`;
  } finally {
    setBusy(el.searchBtn, false, 'Buscar contexto');
  }
});

el.askBtn.addEventListener('click', async () => {
  const query = el.askQuery.value.trim();
  if (!query) {
    return;
  }

  try {
    setBusy(el.askBtn, true, 'Generando respuesta...');
    const data = await api('/api/ask', 'POST', {
      ...getCommonPayload(),
      query,
      chat_model: el.chatModel.value.trim(),
    });

    el.answerBox.classList.remove('hidden');
    el.answerBox.textContent = data.answer || '(sin respuesta)';
    renderSearchResults(data.results || []);
  } catch (error) {
    el.answerBox.classList.remove('hidden');
    el.answerBox.textContent = error.message;
  } finally {
    setBusy(el.askBtn, false, 'Responder con RAG');
  }
});

el.collection.addEventListener('change', refreshStatus);

refreshStatus();
