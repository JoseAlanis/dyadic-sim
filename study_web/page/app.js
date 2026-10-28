/* Transcript rating page. Everything runs locally: the sessions are embedded in
   this file, work is kept in this browser as a safety copy, and the record is the
   JSON file the rater saves and sends back. Nothing is sent anywhere.
   Evidence offsets count Unicode code points, as Python string slicing does, so
   the importer can check text[start:end] === quote against its own copy. */
'use strict';

const $ = (id) => document.getElementById(id);
const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const clone = (value) => JSON.parse(JSON.stringify(value));
const now = () => new Date().toISOString();
const RETURN_FORMAT = 'transcript-rating-return';
const SCHEMA_VERSION = 1;

let pkg, cb, state = null, storageOk = true, conflict = false, knownRevision = 0;
let current = 0;        // index into pkg.assignments
let focusUid = null;    // the patient turn the panel is about
let draft = null;       // the open claim editor
let pending = null;     // a passage selected while a claim is open
let toastTimer, textTimer;

const storeKey = () => `transcript-rating:${pkg.package_id}`;
const symptomOf = (key) => cb.symptoms.find((s) => s.key === key);
const labelOf = (list, key) => list.find((o) => String(o.key) === String(key))?.label ?? String(key);

// ---------- code points
function u16ToCp(text, index) { return Array.from(text.slice(0, index)).length; }
function cpToU16(text, index) { return Array.from(text).slice(0, index).join('').length; }
function cpSlice(text, start, end) { return Array.from(text).slice(start, end).join(''); }

// ---------- state
function emptyForm() {
  return {expression:{}, frequency:{}, frequency_evidence:{}, most_prominent:null, presenting_complaint:'', realism:null, note:''};
}
function emptyAssignment() {
  return {status:'not_started', turns:{}, claims:[], form:emptyForm(), submissions:[], reopenings:[], opened_at:null, updated_at:null};
}
function normaliseAssignments(input) {
  const out = {};
  for (const a of pkg.assignments) {
    const saved = (input || {})[a.assignment_id] || {};
    out[a.assignment_id] = {...emptyAssignment(), ...saved, form:{...emptyForm(), ...(saved.form || {})}};
  }
  return out;
}
function freshState() {
  return {revision:0, created_at:now(), updated_at:null, assignments:normaliseAssignments({}), events:[], last_file_save:null};
}
const assignment = () => pkg.assignments[current];
const astate = () => state.assignments[assignment().assignment_id];
const utterance = (uid) => assignment().utterances.find((u) => u.id === uid);
const patientTurns = (a = assignment()) => a.utterances.filter((u) => u.speaker === 'patient');
const claimsIn = (uid) => astate().claims.filter((c) => c.utterance_id === uid);
const isReviewed = (uid, s = astate()) => Boolean(s.turns[uid] && s.turns[uid].reviewed);
const reviewedCount = (a = assignment()) => patientTurns(a).filter((u) => isReviewed(u.id, state.assignments[a.assignment_id])).length;
const allReviewed = () => reviewedCount() === patientTurns().length;
const locked = () => conflict || astate().status === 'submitted';
function previousTherapist(uid) {
  const list = assignment().utterances, i = list.findIndex((u) => u.id === uid);
  for (let k = i - 1; k >= 0; k--) if (list[k].speaker === 'therapist') return list[k];
  return null;
}

function commit(event) {
  if (conflict) return;
  state.revision += 1;
  state.updated_at = now();
  const s = astate();
  s.updated_at = state.updated_at;
  if (s.status === 'not_started') s.status = 'in_progress';
  if (event) state.events.push({at:state.updated_at, revision:state.revision, assignment_id:assignment().assignment_id, ...event});
  persist();
}

function persist() {
  if (conflict) return;
  try {
    const raw = localStorage.getItem(storeKey());
    if (raw) {
      const stored = JSON.parse(raw);
      if (stored && Number.isInteger(stored.revision) && stored.revision > knownRevision) { enterConflict(); return; }
    }
    localStorage.setItem(storeKey(), JSON.stringify(state));
    knownRevision = state.revision;
    storageOk = true;
  } catch {
    storageOk = false;
  }
  renderSaveState();
}

function enterConflict() {
  if (conflict) return;
  conflict = true;
  draft = null; pending = null;
  showBanner('<strong>This rating page is open in another tab or window.</strong> To avoid overwriting your work, this tab has stopped saving. Continue in the other tab, or reload this one to pick up the latest work.');
  renderAll();
}

// ---------- files
function returnDocument() {
  return {format:RETURN_FORMAT, schema_version:SCHEMA_VERSION, app_version:pkg.app_version,
    package_id:pkg.package_id, study_version:pkg.study_version, protocol_version:pkg.protocol_version,
    rater_id:pkg.rater_id, revision:state.revision, saved_at:now(),
    assignments:clone(state.assignments), events:clone(state.events)};
}

function saveToFile(message) {
  flushText();
  const doc = returnDocument();
  const name = `rating_${pkg.rater_id}_${pkg.study_version}_rev${doc.revision}.json`;
  const url = URL.createObjectURL(new Blob([JSON.stringify(doc, null, 1)], {type:'application/json'}));
  const anchor = document.createElement('a');
  anchor.href = url; anchor.download = name;
  document.body.append(anchor); anchor.click(); anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
  state.last_file_save = {revision:doc.revision, at:doc.saved_at, file:name};
  persist();
  toast(message || `Saved as ${name}. Keep the newest file and send it when asked.`);
}

function checkReturn(doc) {
  if (!doc || doc.format !== RETURN_FORMAT) return 'That is not a saved rating file.';
  if (doc.schema_version !== SCHEMA_VERSION) return 'That file was saved by a different version of this tool.';
  if (doc.package_id !== pkg.package_id || doc.rater_id !== pkg.rater_id) return 'That file belongs to a different rating package.';
  if (!Number.isInteger(doc.revision) || typeof doc.assignments !== 'object' || !doc.assignments) return 'That file is incomplete.';
  for (const [aid, saved] of Object.entries(doc.assignments)) {
    const a = pkg.assignments.find((x) => x.assignment_id === aid);
    if (!a) return 'That file refers to a session that is not in this package.';
    for (const claim of saved.claims || []) {
      const u = a.utterances.find((x) => x.id === claim.utterance_id);
      if (!u || u.speaker !== 'patient') return 'That file has evidence outside a patient turn.';
      for (const span of claim.spans || []) if (cpSlice(u.text, span.start, span.end) !== span.quote) return 'That file has evidence that does not match the transcript.';
    }
  }
  return '';
}

async function loadFile(file) {
  let doc;
  try { doc = JSON.parse(await file.text()); } catch { toast('That is not a saved rating file.'); return; }
  const problem = checkReturn(doc);
  if (problem) { toast(problem); return; }
  if (doc.revision < state.revision && !confirm(`This file is older than the work in this browser (step ${doc.revision} against step ${state.revision}). Replace the work in this browser with the file?`)) return;
  const revision = Math.max(state.revision, knownRevision, doc.revision) + 1, time = now();
  state = {revision, created_at:state.created_at, updated_at:time, assignments:normaliseAssignments(doc.assignments),
    events:[...(Array.isArray(doc.events) ? doc.events : []), {at:time, revision, type:'loaded_file', file_revision:doc.revision}],
    last_file_save:{revision, at:time, file:file.name}};
  draft = null; pending = null; focusUid = null;
  persist();
  renderAll();
  toast(`Loaded ${file.name}.`);
}

// ---------- selection to evidence
function readSelection() {
  const selection = window.getSelection();
  if (!selection || selection.rangeCount === 0 || selection.isCollapsed) return null;
  const range = selection.getRangeAt(0);
  const turnOf = (node) => (node.nodeType === Node.ELEMENT_NODE ? node : node.parentElement)?.closest('#dialogue .utt');
  let first = turnOf(range.startContainer), last = turnOf(range.endContainer);
  if (!first && !last) return null;
  // A triple click, or a drag that just touches a neighbour, reaches into the next
  // turn without selecting any of its text; treat that as ending in this turn.
  if (first && last && first !== last) {
    const tail = document.createRange(); tail.setStart(last, 0); tail.setEnd(range.endContainer, range.endOffset);
    const head = document.createRange(); head.setStart(range.startContainer, range.startOffset); head.setEnd(first, first.childNodes.length);
    if (!tail.toString().trim()) last = first;
    else if (!head.toString().trim()) first = last;
  }
  if (!first || first !== last) return {error:'Select within one turn. Evidence from another turn goes in a claim of its own.'};
  if (first.dataset.speaker !== 'patient') return {error:'Evidence comes from patient turns. A therapist line can be linked as context inside a claim.'};
  const textEl = first.querySelector('.utt-text');
  if (!textEl || textEl.classList.contains('empty')) return null;
  const u = utterance(first.dataset.uid);
  const inside = (node) => textEl.contains(node);
  if (!inside(range.startContainer) && !inside(range.endContainer) && !range.intersectsNode(textEl)) return null;
  const offset = (node, at, fallback) => {
    if (!inside(node)) return fallback;
    const prefix = document.createRange(); prefix.selectNodeContents(textEl); prefix.setEnd(node, at);
    return prefix.toString().length;
  };
  let start = offset(range.startContainer, range.startOffset, 0);
  let end = offset(range.endContainer, range.endOffset, u.text.length);
  while (start < end && /\s/.test(u.text[start])) start++;
  while (end > start && /\s/.test(u.text[end - 1])) end--;
  if (start >= end) return null;
  const cs = u16ToCp(u.text, start), ce = u16ToCp(u.text, end);
  return {uid:u.id, start:cs, end:ce, quote:cpSlice(u.text, cs, ce)};
}
function clearSelection() { try { window.getSelection().removeAllRanges(); } catch { /* nothing selected */ } }

function spanFrom(p) { return {utterance_id:p.uid, start:p.start, end:p.end, quote:p.quote}; }
function newDraft(spans, template = {}) {
  return {mode:'new', claim_id:null, utterance_id:spans[0].utterance_id, spans:clone(spans),
    symptom:template.symptom || '', code:template.code || '', elicited:template.elicited || '',
    context:template.context ? [...template.context] : [], note:''};
}
function editDraft(claim) {
  return {mode:'edit', claim_id:claim.id, utterance_id:claim.utterance_id, spans:clone(claim.spans),
    symptom:claim.symptom, code:claim.code, elicited:claim.elicited, context:[...claim.context], note:claim.note};
}

function onSelection(p) {
  clearSelection();
  if (draft) {
    if (draft.utterance_id !== p.uid) { toast(`This claim holds evidence from turn ${utterance(draft.utterance_id).turn}. Save or cancel it first.`); return; }
    if (draft.spans.some((s) => s.start === p.start && s.end === p.end)) { toast('That passage is already part of this claim.'); return; }
    readEditor();
    pending = spanFrom(p);
    renderDialogue(); renderPanel();
    return;
  }
  focusUid = p.uid;
  draft = newDraft([spanFrom(p)]);
  renderAll();
  $('d-symptom')?.focus({preventScroll:true});
}

function useWholeTurn() {
  const u = utterance(focusUid);
  let start = 0, end = u.text.length;
  while (start < end && /\s/.test(u.text[start])) start++;
  while (end > start && /\s/.test(u.text[end - 1])) end--;
  if (start >= end) return;
  const cs = u16ToCp(u.text, start), ce = u16ToCp(u.text, end);
  onSelection({uid:u.id, start:cs, end:ce, quote:cpSlice(u.text, cs, ce)});
}

// ---------- claims and turns
function readEditor() {
  if (!draft || !$('d-symptom')) return;
  draft.symptom = $('d-symptom').value;
  draft.code = document.querySelector('input[name=d-code]:checked')?.value || '';
  draft.elicited = document.querySelector('input[name=d-elicited]:checked')?.value || '';
  draft.context = [...document.querySelectorAll('input[name=d-context]:checked')].map((box) => box.value);
  draft.note = $('d-note').value;
}

function saveDraft(another) {
  readEditor();
  const missing = [];
  if (!draft.symptom) missing.push('a symptom');
  if (!draft.code) missing.push('a code');
  if (!draft.elicited) missing.push('whether it was elicited');
  if (missing.length) { $('d-error').textContent = `Choose ${missing.join(', ')}.`; $('d-error').hidden = false; return; }
  const s = astate(), time = now(), fields = {symptom:draft.symptom, code:draft.code, elicited:draft.elicited,
    spans:clone(draft.spans), context:[...draft.context], note:draft.note.trim()};
  if (draft.mode === 'edit') {
    const claim = s.claims.find((c) => c.id === draft.claim_id);
    if (claim.symptom !== fields.symptom) unlinkEvidence(claim.id);
    Object.assign(claim, fields, {updated_at:time});
  } else {
    let id;
    do { id = `c-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`; } while (s.claims.some((c) => c.id === id));
    s.claims.push({id, utterance_id:draft.utterance_id, ...fields, created_at:time, updated_at:time});
  }
  commit({type:draft.mode === 'edit' ? 'claim_edited' : 'claim_added'});
  const spans = draft.spans, context = draft.context;
  draft = another ? newDraft(spans, {context}) : null;
  pending = null;
  renderAll();
  toast(another ? 'Claim saved. Choose another symptom for the same passage.' : 'Claim saved.');
}

function unlinkEvidence(claimId) {
  const links = astate().form.frequency_evidence;
  for (const key of Object.keys(links)) links[key] = links[key].filter((id) => id !== claimId);
}

function deleteDraftClaim() {
  if (!confirm('Delete this claim?')) return;
  const s = astate(), claim = s.claims.find((c) => c.id === draft.claim_id);
  s.claims = s.claims.filter((c) => c.id !== claim.id);
  unlinkEvidence(claim.id);
  let note = 'Claim deleted.';
  if (isReviewed(claim.utterance_id) && !claimsIn(claim.utterance_id).length) {
    delete s.turns[claim.utterance_id];
    note = 'Claim deleted. The turn has no evidence left, so it is marked not reviewed again.';
  }
  commit({type:'claim_deleted'});
  draft = null; pending = null;
  renderAll();
  toast(note);
}

function setReviewed(uid, value) {
  if (locked()) return;
  const s = astate();
  if (value) s.turns[uid] = {reviewed:true, reviewed_at:now()};
  else delete s.turns[uid];
  commit({type:value ? 'turn_reviewed' : 'turn_unreviewed', utterance_id:uid});
  renderAll();
  if (value && allReviewed()) toast('Every patient turn is reviewed. The session judgement below the transcript is now open.');
}

function focusTurn(uid, scroll = true) {
  if (draft && draft.utterance_id !== uid) { toast('Save or cancel the open claim first.'); return; }
  focusUid = uid;
  renderAll();
  const item = document.querySelector(`.utt[data-uid="${uid}"]`);
  if (scroll) item?.scrollIntoView({block:'center', behavior:'smooth'});
  item?.focus({preventScroll:true});
}

function nextUnreviewed() {
  const list = patientTurns();
  const from = list.findIndex((u) => u.id === focusUid);
  for (let k = 1; k <= list.length; k++) {
    const u = list[(from + k + list.length) % list.length];
    if (!isReviewed(u.id)) { focusTurn(u.id); return; }
  }
  toast('Every patient turn is reviewed. The session judgement is below the transcript.');
  $('session-form').scrollIntoView({behavior:'smooth', block:'start'});
}

function switchSession(index) {
  if (index === current) return;
  if (draft && !confirm('Leave this session? The claim you have not saved will be lost.')) return;
  flushText();
  current = index; focusUid = null; draft = null; pending = null;
  const s = astate();
  if (!s.opened_at) { s.opened_at = now(); persist(); }
  renderAll();
  window.scrollTo({top:0});
}

// ---------- session form
function formProblems() {
  const f = astate().form, out = [];
  const missing = (field) => cb.symptoms.filter((s) => !f[field][s.key]).map((s) => s.label);
  if (!allReviewed()) out.push('every patient turn marked reviewed');
  const e = missing('expression'), q = missing('frequency');
  if (e.length) out.push(`expression for ${e.length === cb.symptoms.length ? 'every symptom' : e.join(', ')}`);
  if (q.length) out.push(`frequency for ${q.length === cb.symptoms.length ? 'every symptom' : q.join(', ')}`);
  if (!f.most_prominent) out.push('the most prominent symptom');
  if (!f.presenting_complaint.trim()) out.push('the presenting complaint');
  if (!f.realism) out.push('how realistic the patient is');
  return out;
}

function submitSession() {
  flushText();
  const problems = formProblems();
  if (problems.length) { $('form-errors').textContent = `Still needed: ${problems.join('; ')}.`; return; }
  if (!confirm('Submit this session? Your answers are then fixed. You can reopen it later, with a reason.')) return;
  const s = astate();
  s.submissions.push({revision:state.revision + 1, submitted_at:now(), turns:clone(s.turns), claims:clone(s.claims), form:clone(s.form)});
  s.status = 'submitted';
  commit({type:'submitted'});
  draft = null; pending = null; focusUid = null;
  renderAll();
  saveToFile('Session submitted and saved to a file. Keep the newest file.');
}

function reopenSession() {
  const reason = $('reopen-reason').value.trim();
  if (!reason) { $('reopen-error').hidden = false; return; }
  const s = astate();
  s.reopenings.push({at:now(), reason, after_submission:s.submissions.length});
  s.status = 'in_progress';
  commit({type:'reopened', reason});
  $('reopen-dialog').close();
  renderAll();
  toast('Session reopened. Your earlier submission is kept.');
}

function flushText() {
  if (!textTimer) return;
  clearTimeout(textTimer); textTimer = null;
  commit({type:'form_changed'});
}

// ---------- rendering
function renderAll() {
  renderSessions(); renderHeader(); renderDialogue(); renderPanel(); renderForm(); renderSaveState();
}

function renderSaveState() {
  if (!state) return;
  const last = state.last_file_save;
  const browser = storageOk ? 'Saved in this browser' : 'Not kept in this browser';
  let file = '<span class="warn">Not yet saved to a file</span>';
  if (last) file = state.revision > last.revision ? `<span class="warn">Changes since your last file, ${esc(ago(last.at))}</span>` : `Saved to a file ${esc(ago(last.at))}`;
  $('save-state').innerHTML = `<b>${browser}</b>${file}`;
}
function ago(iso) {
  const minutes = Math.round((Date.now() - Date.parse(iso)) / 60000);
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes} min ago`;
  return new Date(iso).toLocaleString();
}

function renderSessions() {
  $('session-list').innerHTML = pkg.assignments.map((a, index) => {
    const s = state.assignments[a.assignment_id], n = patientTurns(a).length;
    const line = s.status === 'submitted' ? '<span class="done">Submitted</span>'
      : s.status === 'not_started' ? '<span>Not started</span>' : `<span>${reviewedCount(a)} of ${n} turns reviewed</span>`;
    return `<button class="session-item ${index === current ? 'selected' : ''}" data-index="${index}" ${index === current ? 'aria-current="true"' : ''}><b>Session ${a.order}</b>${line}</button>`;
  }).join('');
  const done = pkg.assignments.filter((a) => state.assignments[a.assignment_id].status === 'submitted').length;
  $('overall-progress').textContent = `${done} of ${pkg.assignments.length} submitted`;
}

function renderHeader() {
  const a = assignment(), s = astate(), n = patientTurns().length, r = reviewedCount();
  $('brand-rater').textContent = `Rater ${pkg.rater_id}`;
  $('eyebrow').textContent = `${pkg.study_version} · protocol ${pkg.protocol_version}`;
  $('session-title').textContent = `Session ${a.order} of ${pkg.assignments.length}`;
  $('session-lede').textContent = `Reference ${a.assignment_id}. Read in order, mark every patient turn reviewed, with or without evidence, then complete the session judgement.`;
  const formState = s.status === 'submitted' ? 'Submitted' : !allReviewed() ? 'Not open yet' : formProblems().length ? 'In progress' : 'Ready';
  const status = {not_started:'Not started', in_progress:'In progress', submitted:'Submitted'}[s.status];
  $('metrics').innerHTML = [['Turns reviewed', `${r} / ${n}`], ['Evidence claims', s.claims.length], ['Session judgement', formState], ['Status', status]]
    .map(([label, value]) => `<div><span>${label}</span><strong>${esc(value)}</strong></div>`).join('');
  const note = $('submitted-note');
  if (s.status === 'submitted') {
    const last = s.submissions[s.submissions.length - 1];
    note.innerHTML = `Submitted ${esc(new Date(last.submitted_at).toLocaleString())}. Your answers are fixed.${conflict ? '' : '<button id="reopen" class="button compact">Reopen with a reason</button>'}`;
    note.hidden = false;
    $('reopen')?.addEventListener('click', () => { $('reopen-reason').value = ''; $('reopen-error').hidden = true; $('reopen-dialog').showModal(); });
  } else note.hidden = true;
  $('next-unreviewed').disabled = locked() || allReviewed();
}

function highlighted(textEl, u) {
  const chars = Array.from(u.text);
  const spans = [];
  for (const claim of claimsIn(u.id)) for (const span of claim.spans) spans.push({...span, claim:claim.id, code:claim.code, label:`${symptomOf(claim.symptom)?.label} · ${labelOf(cb.claim_codes, claim.code)}`});
  if (draft && draft.utterance_id === u.id) for (const span of [...draft.spans, ...(pending ? [pending] : [])]) spans.push({...span, claim:'draft', code:'pending', label:'Not saved yet'});
  const cuts = [...new Set([0, chars.length, ...spans.flatMap((s) => [s.start, s.end])])].sort((a, b) => a - b);
  for (let i = 0; i < cuts.length - 1; i++) {
    const a = cuts[i], b = cuts[i + 1], piece = chars.slice(a, b).join('');
    const covering = spans.filter((s) => s.start <= a && s.end >= b);
    if (!covering.length) { textEl.append(document.createTextNode(piece)); continue; }
    const mark = document.createElement('mark');
    const codes = new Set(covering.map((s) => s.code));
    const ids = [...new Set(covering.map((s) => s.claim))];
    mark.className = `hl code-${codes.has('pending') ? 'pending' : codes.size > 1 ? 'mixed' : [...codes][0]}${ids.length > 1 ? ' overlap' : ''}`;
    if (draft?.claim_id && ids.includes(draft.claim_id)) mark.classList.add('active');
    mark.dataset.claims = ids.join(' ');
    mark.title = [...new Set(covering.map((s) => s.label))].join('\n');
    mark.textContent = piece;
    textEl.append(mark);
  }
}

function renderDialogue() {
  const list = $('dialogue');
  list.textContent = '';
  for (const u of assignment().utterances) {
    const item = document.createElement('li');
    item.className = `utt ${u.speaker}`;
    item.dataset.uid = u.id; item.dataset.speaker = u.speaker;
    const head = document.createElement('div'); head.className = 'utt-head';
    const who = document.createElement('span'); who.className = 'who';
    who.textContent = u.speaker === 'therapist' ? (u.turn === 0 ? 'Therapist · opening' : 'Therapist') : `Patient · turn ${u.turn}`;
    head.append(who);
    const text = document.createElement('p'); text.className = 'utt-text';
    if (u.speaker === 'patient') {
      item.tabIndex = 0;
      item.setAttribute('aria-label', `Patient turn ${u.turn}`);
      if (isReviewed(u.id)) item.classList.add('reviewed');
      if (u.id === focusUid) item.classList.add('focused');
      head.append(reviewControls(u));
    }
    if (!u.text.trim()) { text.classList.add('empty'); text.textContent = 'No text was recorded for this line.'; }
    else if (u.speaker === 'patient') highlighted(text, u);
    else text.textContent = u.text;
    item.append(head, text);
    list.append(item);
  }
}

function reviewControls(u) {
  const wrap = document.createElement('div'); wrap.className = 'review';
  const n = claimsIn(u.id).length, reviewed = isReviewed(u.id), claims = `${n} ${n === 1 ? 'claim' : 'claims'}`;
  const chip = document.createElement('span');
  chip.className = `chip ${reviewed ? 'ok' : 'todo'}`;
  chip.textContent = reviewed ? (n ? `Reviewed · ${claims}` : 'Reviewed · no evidence') : (n ? `${claims} · not reviewed` : 'Not reviewed');
  wrap.append(chip);
  if (!locked()) {
    const button = document.createElement('button');
    button.className = 'button compact';
    button.dataset.action = reviewed ? 'unreview' : 'review';
    button.dataset.uid = u.id;
    button.textContent = reviewed ? 'Undo' : n ? 'Mark reviewed' : 'No symptom evidence';
    wrap.append(button);
  }
  return wrap;
}

function option(name, o, value, type = 'radio') {
  const checked = type === 'radio' ? String(value) === String(o.key) : (value || []).includes(o.key);
  return `<label class="option"><input type="${type}" name="${name}" value="${esc(o.key)}" ${checked ? 'checked' : ''}><span>${esc(o.label)}${o.help ? `<small>${esc(o.help)}</small>` : ''}</span></label>`;
}
function snippet(text, length = 70) { const chars = Array.from(text); return chars.length > length ? `${chars.slice(0, length).join('')}…` : text; }

function renderPanel() {
  const panel = $('panel');
  if (draft) { renderEditor(panel); return; }
  if (focusUid) { renderTurnPanel(panel, focusUid); return; }
  panel.innerHTML = `<h2>How to rate</h2>
    <ol class="guide"><li>Read the conversation from the top.</li>
    <li>When a patient turn gives evidence about a symptom, select the words. A short form opens here.</li>
    <li>Choose the symptom, the code, and whether the therapist asked about it.</li>
    <li>Mark every patient turn reviewed, with or without evidence.</li>
    <li>Complete the session judgement below the transcript, then submit.</li></ol>
    <div class="legend"><span><mark class="hl code-present">Present</mark></span><span><mark class="hl code-raised_and_denied">Raised and denied</mark></span><span><mark class="hl code-unclear">Unclear</mark></span><span><mark class="hl code-mixed">Several codes on one passage</mark></span></div>
    <p class="muted small" style="margin-top:14px">The full instructions and codebook are under Instructions at the top.</p>`;
}

function renderTurnPanel(panel, uid) {
  const u = utterance(uid), claims = claimsIn(uid), reviewed = isReviewed(uid), ro = locked();
  const list = claims.length ? `<div class="claim-list">${claims.map((c) => `<button class="claim-row" data-claim="${esc(c.id)}"><b>${esc(symptomOf(c.symptom)?.label)}</b> · ${esc(labelOf(cb.claim_codes, c.code))} · elicited: ${esc(labelOf(cb.elicited, c.elicited).toLowerCase())}<span>“${esc(c.spans.map((s) => s.quote).join(' … '))}”</span></button>`).join('')}</div>`
    : '<p class="muted small" style="margin-top:6px">None yet.</p>';
  panel.innerHTML = `<h2>Patient · turn ${u.turn}</h2><p class="sub">${reviewed ? (claims.length ? 'Reviewed' : 'Reviewed · no symptom evidence') : 'Not reviewed yet'}</p>
    ${ro ? '' : '<p class="guide">Select a phrase in this turn to add evidence. If precise selection is awkward, use the whole turn.</p><div class="panel-actions"><button id="whole-turn" class="button">Use the whole turn as evidence</button></div>'}
    <h3 style="margin-top:18px">Evidence in this turn</h3>${list}
    ${ro ? '' : `<div class="panel-actions"><button id="toggle-review" class="button ${reviewed ? '' : 'primary'}">${reviewed ? 'Mark as not reviewed' : claims.length ? 'Mark this turn reviewed' : 'No symptom evidence in this turn'}</button><button id="panel-next" class="button quiet">Next unreviewed turn</button></div>`}`;
  $('whole-turn')?.addEventListener('click', useWholeTurn);
  $('toggle-review')?.addEventListener('click', () => setReviewed(uid, !reviewed));
  $('panel-next')?.addEventListener('click', nextUnreviewed);
  panel.querySelectorAll('[data-claim]').forEach((button) => button.addEventListener('click', () => {
    if (ro) return;
    draft = editDraft(astate().claims.find((c) => c.id === button.dataset.claim));
    renderAll();
  }));
}

function renderEditor(panel) {
  const u = utterance(draft.utterance_id), before = previousTherapist(u.id);
  const index = assignment().utterances.findIndex((x) => x.id === u.id);
  const earlier = assignment().utterances.slice(0, index).filter((x) => x.id !== before?.id && x.text.trim()).reverse();
  const contextOption = (x, label) => option('d-context', {key:x.id, label:label || `${x.speaker === 'therapist' ? 'Therapist' : 'Patient'} · turn ${x.turn}: “${snippet(x.text, 60)}”`}, draft.context, 'checkbox');
  const item = symptomOf(draft.symptom)?.item || 'Choose the symptom this passage is evidence about.';
  panel.innerHTML = `<h2>${draft.mode === 'new' ? 'New evidence' : 'Edit evidence'}</h2><p class="sub">Patient · turn ${u.turn}</p>
    ${draft.spans.map((s, i) => `<div class="quote"><q>${esc(s.quote)}</q>${draft.spans.length > 1 ? `<button class="text-button" data-remove-span="${i}">Remove</button>` : ''}</div>`).join('')}
    ${pending ? `<div class="note">Selected: “${esc(pending.quote)}”<div class="editor-actions"><button id="add-pending" class="button compact">Add to this claim</button><button id="pending-new" class="button compact quiet">Start a new claim with it</button></div></div>`
      : '<p class="hint">To cite another passage of this turn, select it in the transcript.</p>'}
    <fieldset><label class="field-label" for="d-symptom">Symptom</label><select id="d-symptom"><option value="">Choose a symptom…</option>${cb.symptoms.map((s) => `<option value="${esc(s.key)}" ${draft.symptom === s.key ? 'selected' : ''}>${esc(s.label)}</option>`).join('')}</select><span class="hint" id="d-item">${esc(item)}</span></fieldset>
    <fieldset><legend>Code</legend>${cb.claim_codes.map((o) => option('d-code', o, draft.code)).join('')}</fieldset>
    <fieldset><legend>Elicited<span class="hint">Does this answer a therapist question specifically about this symptom?</span></legend>${cb.elicited.map((o) => option('d-elicited', o, draft.elicited)).join('')}</fieldset>
    <fieldset><legend>Context<span class="hint">Lines needed to understand a short or referential reply.</span></legend>
      ${before ? contextOption(before, 'The therapist line just before this turn') : ''}
      ${earlier.length ? `<details class="context"><summary>Other earlier lines</summary>${earlier.map((x) => contextOption(x)).join('')}</details>` : ''}</fieldset>
    <fieldset><label class="field-label" for="d-note">Note<span class="hint">Optional. Useful for ambiguity or contradictions.</span></label><textarea id="d-note" maxlength="2000">${esc(draft.note)}</textarea></fieldset>
    <p id="d-error" class="error" role="alert" hidden></p>
    <div class="editor-actions"><button id="d-save" class="button primary">Save claim</button><button id="d-another" class="button">Save, then another symptom for this passage</button><button id="d-cancel" class="button quiet">Cancel</button>${draft.mode === 'edit' ? '<button id="d-delete" class="button danger">Delete claim</button>' : ''}</div>`;
  $('d-symptom').addEventListener('change', () => { $('d-item').textContent = symptomOf($('d-symptom').value)?.item || ''; });
  $('d-save').addEventListener('click', () => saveDraft(false));
  $('d-another').addEventListener('click', () => saveDraft(true));
  $('d-cancel').addEventListener('click', () => { draft = null; pending = null; renderAll(); });
  $('d-delete')?.addEventListener('click', deleteDraftClaim);
  $('add-pending')?.addEventListener('click', () => { readEditor(); draft.spans.push(pending); draft.spans.sort((a, b) => a.start - b.start); pending = null; renderDialogue(); renderPanel(); });
  $('pending-new')?.addEventListener('click', () => {
    if (draft.mode === 'new' && !confirm('Discard the claim you have not saved?')) return;
    draft = newDraft([pending]); pending = null; renderAll();
  });
  panel.querySelectorAll('[data-remove-span]').forEach((button) => button.addEventListener('click', () => {
    readEditor(); draft.spans.splice(Number(button.dataset.removeSpan), 1); renderDialogue(); renderPanel();
  }));
}

function select(id, options, value, placeholder, extra = '') {
  return `<select id="${id}" ${extra}><option value="">${placeholder}</option>${options.map((o) => `<option value="${esc(o.key)}" ${String(value) === String(o.key) ? 'selected' : ''}>${esc(o.label)}</option>`).join('')}</select>`;
}

function renderForm() {
  const s = astate(), f = s.form, open = allReviewed(), ro = locked();
  const left = patientTurns().length - reviewedCount();
  const rows = cb.symptoms.map((sym) => {
    const linked = f.frequency_evidence[sym.key] || [];
    const claims = s.claims.filter((c) => c.symptom === sym.key);
    const evidence = claims.length ? `<details><summary>Link evidence (${linked.length})</summary>${claims.map((c) => `<label class="option"><input type="checkbox" data-evidence="${esc(sym.key)}" value="${esc(c.id)}" ${linked.includes(c.id) ? 'checked' : ''}><span>Turn ${utterance(c.utterance_id).turn}: “${esc(snippet(c.spans.map((x) => x.quote).join(' … '), 60))}”</span></label>`).join('')}</details>`
      : '<span class="muted small">No highlighted evidence</span>';
    return `<tr><td>${esc(sym.label)}<small>${esc(sym.item)}</small></td>
      <td>${select(`f-expression-${sym.key}`, cb.expression, f.expression[sym.key], 'Choose…', `data-field="expression" data-symptom="${esc(sym.key)}" aria-label="Expression: ${esc(sym.label)}"`)}</td>
      <td>${select(`f-frequency-${sym.key}`, cb.frequency, f.frequency[sym.key], 'Choose…', `data-field="frequency" data-symptom="${esc(sym.key)}" aria-label="Frequency: ${esc(sym.label)}"`)}</td>
      <td>${evidence}</td></tr>`;
  }).join('');
  const prominence = [...cb.symptoms.map((x) => ({key:x.key, label:x.label})), ...cb.prominence_extra];
  $('session-form').innerHTML = `<h2 id="form-title">Session judgement</h2>
    <p>Answer from the whole session. This is a separate judgement, not a summary of your highlights. Frequency refers to the <strong>last two weeks</strong>. A 0 needs evidence that it did not happen; if the session gives no basis, choose Cannot judge. These are transcript judgements, not PHQ-9 scores.</p>
    ${open || ro ? '' : `<div class="note" style="margin-top:14px">This form opens when every patient turn is marked reviewed. ${left} ${left === 1 ? 'turn' : 'turns'} to go.</div>`}
    <fieldset id="form-fields" ${open && !ro ? '' : 'disabled'}>
      <table class="form-table"><thead><tr><th>Symptom</th><th>Expressed in this session</th><th>Frequency, last two weeks</th><th>Supporting evidence</th></tr></thead><tbody>${rows}</tbody></table>
      <div class="form-grid">
        <label class="field-label" for="f-prominent">Most prominent symptom${select('f-prominent', prominence, f.most_prominent, 'Choose…')}</label>
        <label class="field-label" for="f-complaint">Presenting complaint<span class="hint">One line, in your own words.</span><input type="text" id="f-complaint" maxlength="300" value="${esc(f.presenting_complaint)}"></label>
        <fieldset class="wide"><legend>How realistic is this patient?</legend><div class="realism">${cb.realism.map((o) => option('f-realism', o, f.realism)).join('')}</div></fieldset>
        <label class="field-label wide" for="f-note">Note<span class="hint">Optional.</span><textarea id="f-note" maxlength="2000">${esc(f.note)}</textarea></label>
      </div>
      <div class="form-footer"><div id="form-errors" class="error" role="alert"></div><button id="submit-session" class="button primary">Submit session</button></div>
    </fieldset>`;
  const light = () => { renderSessions(); renderHeader(); };
  $('session-form').querySelectorAll('select[data-field]').forEach((box) => box.addEventListener('change', () => {
    const answers = f[box.dataset.field];
    if (box.value) answers[box.dataset.symptom] = box.value; else delete answers[box.dataset.symptom];
    commit({type:'form_changed'}); light();
  }));
  $('session-form').querySelectorAll('input[data-evidence]').forEach((box) => box.addEventListener('change', () => {
    const key = box.dataset.evidence, links = new Set(f.frequency_evidence[key] || []);
    if (box.checked) links.add(box.value); else links.delete(box.value);
    f.frequency_evidence[key] = [...links];
    box.closest('details').querySelector('summary').textContent = `Link evidence (${links.size})`;
    commit({type:'form_changed'}); light();
  }));
  $('f-prominent').addEventListener('change', () => { f.most_prominent = $('f-prominent').value || null; commit({type:'form_changed'}); light(); });
  $('session-form').querySelectorAll('input[name=f-realism]').forEach((box) => box.addEventListener('change', () => { f.realism = Number(box.value); commit({type:'form_changed'}); light(); }));
  for (const [id, key] of [['f-complaint', 'presenting_complaint'], ['f-note', 'note']]) $(id).addEventListener('input', () => {
    f[key] = $(id).value;
    clearTimeout(textTimer);
    textTimer = setTimeout(() => { textTimer = null; commit({type:'form_changed'}); light(); }, 500);
  });
  $('submit-session').addEventListener('click', submitSession);
}

// ---------- chrome
function toast(message) {
  $('toast').textContent = message; $('toast').hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $('toast').hidden = true; }, 5200);
}
function showBanner(html) { $('banner').innerHTML = html; $('banner').hidden = false; }

function applyTheme(choice) {
  let theme = choice;
  if (!theme) { try { theme = localStorage.getItem('transcript-rating:theme'); } catch { theme = null; } }
  theme = theme === 'light' ? 'light' : 'dark';
  document.documentElement.classList.toggle('dark', theme === 'dark');
  document.documentElement.classList.toggle('light', theme === 'light');
  if (choice) { try { localStorage.setItem('transcript-rating:theme', theme); } catch { /* the choice lasts this visit only */ } }
}

function helpBody() {
  const dl = (list) => `<dl>${list.map((o) => `<dt>${esc(o.label)}</dt><dd>${esc(o.help || o.item || '')}</dd>`).join('')}</dl>`;
  return `<h3>What you are rating</h3>
    <p>Each session is a simulated therapy conversation between two language models, one playing the therapist and one the patient. No real patient is involved. Some patients mention thoughts of death or of hurting themselves; this is simulated, but please be prepared for it.</p>
    <h3>For each patient turn</h3>
    <ol><li>Read the turn in context, including the therapist line before it.</li>
    <li>If it gives evidence about one of the nine symptoms, select the words and fill in the short form. Several symptoms can share one passage, and a claim can cite several passages of the same turn.</li>
    <li>Mark the turn reviewed. A turn you have not marked counts as not yet read, never as “no symptoms”.</li></ol>
    <h3>Codes</h3>${dl(cb.claim_codes)}
    <p>Count the patient's own current or recent experience, within the last two weeks. Past or out-of-window experience, someone else's experience, hypotheticals and quoted speech are Unclear; add a note if it helps.</p>
    <h3>Elicited</h3>${dl(cb.elicited)}
    <h3>Context</h3><p>If a reply only makes sense with an earlier line, for example “Yes, exactly”, tick that line under Context in the claim.</p>
    <h3>The nine symptoms</h3>${dl(cb.symptoms)}
    <h3>Session judgement</h3><p>Once every turn is reviewed: expression and frequency for each symptom, the most prominent symptom, the presenting complaint in one line, and how realistic the patient is. It is a judgement of the whole session, separate from your highlights. Frequency 0 needs evidence of absence over the last two weeks; otherwise choose Cannot judge.</p>
    <h3>Saving and sending your work</h3>
    <ul><li>This browser keeps a safety copy while you work, but only in this browser and for this file in this place. Moving the file or clearing browser data can lose it.</li>
    <li><b>Save to file</b> makes the real record. The page also saves a file each time you submit a session.</li>
    <li>To continue in another browser or on another computer, open this page there and use <b>Load saved file</b>.</li>
    <li>Keep the page open in one tab only. Send the newest saved file when you finish, or when asked.</li>
    <li>This page sends nothing anywhere.</li></ul>
    <p class="small">Draft codebook, protocol ${esc(pkg.protocol_version)}. The wording will be settled during calibration.</p>`;
}

// ---------- test support: select a passage the way a mouse would
function rangeFor(uid, start, end) {
  const textEl = document.querySelector(`.utt[data-uid="${uid}"] .utt-text`), u = utterance(uid);
  const point = (u16) => {
    const walker = document.createTreeWalker(textEl, NodeFilter.SHOW_TEXT);
    let node, seen = 0;
    while ((node = walker.nextNode())) { if (seen + node.length >= u16) return [node, u16 - seen]; seen += node.length; }
    return [textEl, textEl.childNodes.length];
  };
  const range = document.createRange();
  range.setStart(...point(cpToU16(u.text, start)));
  range.setEnd(...point(cpToU16(u.text, end)));
  return range;
}

function boot() {
  pkg = JSON.parse($('package-data').textContent);
  if (pkg.format !== 'transcript-rating-package' || pkg.schema_version !== SCHEMA_VERSION) throw new Error('this page was built for a different version of the rating tool');
  cb = pkg.codebook;
  applyTheme();
  let restored = false;
  try {
    const raw = localStorage.getItem(storeKey());
    if (raw) {
      const saved = JSON.parse(raw);
      if (saved && Number.isInteger(saved.revision) && saved.assignments) {
        state = {...freshState(), ...saved, assignments:normaliseAssignments(saved.assignments)};
        knownRevision = saved.revision;
        restored = true;
      }
    }
  } catch { storageOk = false; }
  if (!state) state = freshState();
  const open = pkg.assignments.findIndex((a) => state.assignments[a.assignment_id].status !== 'submitted');
  current = open === -1 ? 0 : open;

  $('help-body').innerHTML = helpBody();
  document.querySelectorAll('[data-close]').forEach((button) => button.addEventListener('click', () => $(button.dataset.close).close()));
  $('help').addEventListener('click', () => $('help-dialog').showModal());
  $('theme').addEventListener('click', () => applyTheme(document.documentElement.classList.contains('dark') ? 'light' : 'dark'));
  $('save-file').addEventListener('click', () => saveToFile());
  $('load-file').addEventListener('click', () => $('file-input').click());
  $('file-input').addEventListener('change', async (event) => { const file = event.target.files[0]; event.target.value = ''; if (file) await loadFile(file); });
  $('next-unreviewed').addEventListener('click', nextUnreviewed);
  $('reopen-confirm').addEventListener('click', reopenSession);
  $('session-list').addEventListener('click', (event) => { const item = event.target.closest('[data-index]'); if (item) switchSession(Number(item.dataset.index)); });
  const dialogue = $('dialogue');
  const pick = () => {
    const picked = readSelection();
    if (!picked) return false;
    if (picked.error) { toast(picked.error); clearSelection(); return true; }
    if (locked()) { toast(conflict ? 'This tab has stopped saving; see the message at the top.' : 'This session is submitted. Reopen it to change your answers.'); clearSelection(); return true; }
    onSelection(picked);
    return true;
  };
  dialogue.addEventListener('mouseup', (event) => {
    if (event.target.closest('button')) return;
    if (pick()) return;
    const mark = event.target.closest('mark.hl');
    const ids = mark ? mark.dataset.claims.split(' ').filter((id) => id !== 'draft') : [];
    if (ids.length === 1 && !draft && !locked()) {
      draft = editDraft(astate().claims.find((c) => c.id === ids[0]));
      focusUid = draft.utterance_id;
      renderAll();
      return;
    }
    const turn = event.target.closest('.utt.patient');
    if (turn) focusTurn(turn.dataset.uid, false);
  });
  dialogue.addEventListener('keyup', (event) => {
    if (event.shiftKey && pick()) return;
    const turn = event.target.closest?.('.utt.patient');
    if (turn && (event.key === 'Enter' || event.key === ' ') && event.target === turn) focusTurn(turn.dataset.uid, false);
  });
  dialogue.addEventListener('click', (event) => {
    const button = event.target.closest('[data-action]');
    if (button) setReviewed(button.dataset.uid, button.dataset.action === 'review');
  });
  window.addEventListener('storage', (event) => {
    if (event.key !== storeKey() || !event.newValue) return;
    try { if (JSON.parse(event.newValue).revision > knownRevision) enterConflict(); } catch { /* not ours */ }
  });
  window.addEventListener('beforeunload', (event) => {
    if (textTimer) flushText();
    const last = state.last_file_save;
    if (!conflict && state.revision > 0 && (!last || state.revision > last.revision)) { event.preventDefault(); event.returnValue = ''; }
  });
  setInterval(renderSaveState, 30000);

  if (!astate().opened_at) { astate().opened_at = now(); persist(); }
  renderAll();
  $('loading').hidden = true;
  $('app').hidden = false;
  if (!storageOk) showBanner('<strong>This browser is not keeping a copy of your work.</strong> Use Save to file after every session, or you may lose answers.');
  if (restored) toast('Your work in this browser was restored.');
  else $('help-dialog').showModal();
  window.ratingApp = {
    get pkg() { return pkg; }, get state() { return state; }, returnDocument, rangeFor,
    selectPassage(uid, start, end) {
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(rangeFor(uid, start, end));
      document.querySelector(`.utt[data-uid="${uid}"] .utt-text`).dispatchEvent(new MouseEvent('mouseup', {bubbles:true}));
    },
  };
}

try { boot(); } catch (error) {
  $('loading').textContent = `Unable to open the rating page: ${error.message}. Please tell the study team.`;
  console.error(error);
}
