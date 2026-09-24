document.addEventListener("DOMContentLoaded", () => {
  const uploadForm = document.getElementById("uploadForm");
  const fileInput = document.getElementById("fileInput");
  const uploadStatus = document.getElementById("uploadStatus");

  const askForm = document.getElementById("askForm");
  const questionInput = document.getElementById("question");
  const answerArea = document.getElementById("answerArea");
  const dropzone = document.getElementById('dropzone');
  const answerStatus = document.getElementById('answerStatus');
  let history = []; // chat history: {role:'user'|'assistant', content, sources?:[]}
  const clearBtn = document.getElementById('clearChat');

  uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!fileInput.files.length) {
      uploadStatus.textContent = "Please select a file to upload.";
      return;
    }
    const file = fileInput.files[0];
    const form = new FormData();
    form.append("file", file);
    uploadStatus.textContent = "Uploading...";
    uploadStatus.classList.remove('muted');
    try {
      const res = await fetch('/api/upload', { method: 'POST', body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || data.message || 'Upload failed');
      // show the uploaded filename and notes count
      const name = data.filename ? escapeHtml(data.filename) : 'file';
      uploadStatus.innerHTML = `<strong>Uploaded:</strong> ${name} — ${escapeHtml(data.message)} (${data.notes_count} note(s) total)`;
    } catch (err) {
      uploadStatus.textContent = err.message;
    }
  });

  // Drag & drop support
  if (dropzone) {
    ['dragenter','dragover'].forEach(ev => dropzone.addEventListener(ev, (e)=>{e.preventDefault();dropzone.classList.add('dragover')}));
    ['dragleave','drop'].forEach(ev => dropzone.addEventListener(ev, (e)=>{e.preventDefault();dropzone.classList.remove('dragover')}));
    dropzone.addEventListener('drop', (e)=>{
      const f = e.dataTransfer.files[0];
      if(f) fileInput.files = e.dataTransfer.files;
    });
  }

  askForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = questionInput.value.trim();
    if (!question) return;
    // add user message to chat UI and history
    pushMessage('user', question);
    questionInput.value = '';
    const assistantMessage = { role: 'assistant', content: '', thinking: 'Preparing response…' };
    history.push(assistantMessage);
    renderChat();
    answerStatus.textContent = '';
    try {
      const res = await fetch('/api/ask/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, history: history.slice(0, -1) }),
      });
      if (!res.ok || !res.body) throw new Error('Unable to start response stream');
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const messages = buffer.split('\n\n');
        buffer = messages.pop();
        messages.forEach(message => {
          const eventType = message.match(/^event:\s*(.+)$/m)?.[1];
          const rawData = message.match(/^data:\s*(.+)$/m)?.[1];
          if (!eventType || rawData === undefined) return;
          const data = JSON.parse(rawData);
          if (eventType === 'status') assistantMessage.thinking = data;
          if (eventType === 'sources') assistantMessage.sources = data;
          if (eventType === 'token') { assistantMessage.content += data; assistantMessage.thinking = ''; }
          if (eventType === 'error') throw new Error(data);
          if (eventType === 'done') assistantMessage.thinking = '';
          renderChat();
        });
      }
      assistantMessage.thinking = '';
      renderChat();
    } catch (err) {
      assistantMessage.content = `Error: ${err.message}`;
      assistantMessage.thinking = '';
      renderChat();
    }
  });

  // Chat UI helpers
  function pushMessage(role, content, sources) {
    const entry = { role, content };
    if (sources && sources.length) entry.sources = sources;
    history.push(entry);
    renderChat();
    document.getElementById('welcomePanel')?.classList.toggle('is-hidden', history.length > 0);
  }

  function renderChat() {
    if (!history.length) {
      answerArea.innerHTML = '';
      return;
    }
    answerArea.innerHTML = history.map((m, messageIndex) => {
      const cls = m.role === 'user' ? 'chat user' : 'chat assistant';
      const thinking = m.thinking ? `<div class="thinking"><span></span>${escapeHtml(m.thinking)}</div>` : '';
      const content = `<div class="bubble">${m.role === 'assistant' ? (m.content ? renderMarkdown(m.content, m.sources?.length || 0, messageIndex) : '') : escapeHtml(m.content)}${thinking}</div>`;
      let srcHtml = '';
      if (m.sources && m.sources.length) {
        srcHtml = '<aside class="source"><span>Sources used</span><ol>' + m.sources.map((s, index) => {
          const url = s.source || s.url || s;
          const text = (s.snippet || s.title || url);
          const safeUrl = escapeHtml(url);
          const safeText = escapeHtml(text);
          const details = s.chunk_index !== undefined ? ` <small>chunk ${s.chunk_index + 1}</small>` : '';
          const number = index + 1;
          return /^https?:\/\//i.test(String(url))
            ? `<li id="source-${messageIndex}-${number}"><a href="${safeUrl}" target="_blank" rel="noopener noreferrer">${safeText}</a>${details}</li>`
            : `<li id="source-${messageIndex}-${number}"><span>${safeText}</span>${details}</li>`;
        }).join('') + '</ol></aside>';
      }
      return `<div class="${cls}">${content}${srcHtml}</div>`;
    }).join('');
    answerArea.scrollTop = answerArea.scrollHeight;
  }

  function escapeHtml(str){
    if(!str) return '';
    return String(str).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#039;').replaceAll('\n','<br>');
  }

  function renderMarkdown(markdown, citationCount = 0, messageIndex = 0) {
    const lines = String(markdown || '').replace(/\r\n?/g, '\n').split('\n');
    const html = []; let listType = null; let inCode = false; let code = [];
    const closeList = () => { if (listType) { html.push(`</${listType}>`); listType = null; } };
    lines.forEach(line => {
      if (/^\s*```/.test(line)) { if (inCode) { html.push(`<pre><code>${escapeHtml(code.join('\n'))}</code></pre>`); code = []; } closeList(); inCode = !inCode; return; }
      if (inCode) { code.push(line); return; }
      const heading = line.match(/^(#{1,3})\s+(.+)$/); const unordered = line.match(/^\s*[-*+]\s+(.+)$/); const ordered = line.match(/^\s*\d+[.)]\s+(.+)$/);
      if (heading) { closeList(); const level = heading[1].length; html.push(`<h${level + 2}>${inlineMarkdown(heading[2], citationCount, messageIndex)}</h${level + 2}>`); }
      else if (unordered || ordered) { const type = unordered ? 'ul' : 'ol'; if (listType !== type) { closeList(); html.push(`<${type}>`); listType = type; } html.push(`<li>${inlineMarkdown((unordered || ordered)[1], citationCount, messageIndex)}</li>`); }
      else if (!line.trim()) { closeList(); } else { closeList(); html.push(`<p>${inlineMarkdown(line, citationCount, messageIndex)}</p>`); }
    });
    if (inCode) html.push(`<pre><code>${escapeHtml(code.join('\n'))}</code></pre>`); closeList(); return html.join('') || '<p>No answer returned.</p>';
  }
  function inlineMarkdown(text, citationCount = 0, messageIndex = 0) {
    let safe = escapeHtml(text);
    safe = safe.replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>').replace(/__([^_]+)__/g, '<strong>$1</strong>').replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, '<em>$1</em>');
    safe = safe.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
    return citationCount ? safe.replace(/\[(\d+)\]/g, (match, number) => Number(number) <= citationCount ? `<a class="citation" href="#source-${messageIndex}-${number}" aria-label="View source ${number}">[${number}]</a>` : match) : safe;
  }

  // Send on Enter (press Enter to send, Shift+Enter for newline)
  questionInput.addEventListener('keydown', (e)=>{
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      askForm.dispatchEvent(new Event('submit', { cancelable: true }));
    }
  });

  // init
  renderChat();

  document.querySelectorAll('.suggestion').forEach(button => button.addEventListener('click', () => {
    questionInput.value = button.textContent.replace(/^[^\w]+\s*/, '');
    questionInput.focus();
  }));

  // clear chat
  if (clearBtn) {
    clearBtn.addEventListener('click', (e) => {
      history = [];
      renderChat();
      document.getElementById('welcomePanel')?.classList.remove('is-hidden');
    });
  }
});
