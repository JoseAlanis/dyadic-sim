// Rate one fixture session through the real page in headless Chrome, over the
// DevTools protocol. Node 22+ only (global WebSocket and fetch), no packages.
//
//   node tests/study_web/browser_e2e.mjs <package.html> <download-dir> [screenshot-prefix]
//
// Prints a JSON summary on success; exits non-zero with the failing step otherwise.
import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

const [htmlPath, downloadDir, shots] = process.argv.slice(2);
const profile = mkdtempSync(join(tmpdir(), 'rating-chrome-'));
const chrome = spawn(process.env.CHROME || 'google-chrome', [
  '--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
  `--user-data-dir=${profile}`, '--remote-debugging-port=0', '--window-size=1440,1000', 'about:blank',
], {stdio: ['ignore', 'ignore', 'ignore']});
const steps = [];
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function until(probe, label, timeout = 15000) {
  const start = Date.now();
  while (Date.now() - start < timeout) {
    const value = await probe();
    if (value) return value;
    await sleep(100);
  }
  throw new Error(`timed out waiting for ${label}`);
}

class Connection {
  constructor(url) {
    this.ws = new WebSocket(url);
    this.next = 0;
    this.waiting = new Map();
    this.opened = new Promise((resolve, reject) => { this.ws.onopen = resolve; this.ws.onerror = reject; });
    this.ws.onmessage = (message) => {
      const data = JSON.parse(message.data);
      const waiter = this.waiting.get(data.id);
      if (!waiter) return;
      this.waiting.delete(data.id);
      if (data.error) waiter.reject(new Error(JSON.stringify(data.error)));
      else waiter.resolve(data.result);
    };
  }
  send(method, params = {}) {
    const id = ++this.next;
    this.ws.send(JSON.stringify({id, method, params}));
    return new Promise((resolve, reject) => this.waiting.set(id, {resolve, reject}));
  }
}

let tab;
async function page(expression) {
  const result = await tab.send('Runtime.evaluate', {expression, awaitPromise: true, returnByValue: true});
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
  return result.result.value;
}
function check(condition, label) {
  if (!condition) throw new Error(`check failed: ${label}`);
  steps.push(label);
}
async function screenshot(name) {
  if (!shots) return;
  const {data} = await tab.send('Page.captureScreenshot', {format: 'png'});
  writeFileSync(`${shots}-${name}.png`, Buffer.from(data, 'base64'));
}

// Select the n-th occurrence of a phrase in the current session the way a mouse
// would, and return the code-point offsets the page should record for it.
const select = (phrase, occurrence = 0) => page(`(() => {
  const a = ratingApp.pkg.assignments.find((x) => x.assignment_id === document.getElementById('session-lede').textContent.match(/Reference (\\w+)/)[1]);
  const u = a.utterances.find((x) => x.speaker === 'patient' && x.text.includes(${JSON.stringify(phrase)}));
  let at = -1;
  for (let k = 0; k <= ${occurrence}; k++) at = u.text.indexOf(${JSON.stringify(phrase)}, at + 1);
  const start = Array.from(u.text.slice(0, at)).length, end = start + Array.from(${JSON.stringify(phrase)}).length;
  ratingApp.selectPassage(u.id, start, end);
  return {uid: u.id, start, end};
})()`);
const claim = (symptom, code, elicited, {context = false, another = false} = {}) => page(`(() => {
  const box = document.getElementById('d-symptom');
  box.value = ${JSON.stringify(symptom)}; box.dispatchEvent(new Event('change', {bubbles: true}));
  document.querySelector('input[name=d-code][value=${code}]').click();
  document.querySelector('input[name=d-elicited][value=${elicited}]').click();
  ${context ? "document.querySelector('input[name=d-context]').click();" : ''}
  document.getElementById(${JSON.stringify(another ? 'd-another' : 'd-save')}).click();
  return true;
})()`);
const currentState = () => page(`(() => {
  const id = document.getElementById('session-lede').textContent.match(/Reference (\\w+)/)[1];
  return {id, revision: ratingApp.state.revision, ...ratingApp.state.assignments[id]};
})()`);

try {
  const portFile = join(profile, 'DevToolsActivePort');
  const port = await until(() => existsSync(portFile) && readFileSync(portFile, 'utf8').split('\n')[0], 'Chrome to start');
  const version = await (await fetch(`http://127.0.0.1:${port}/json/version`)).json();
  const target = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find((t) => t.type === 'page');
  const browser = new Connection(version.webSocketDebuggerUrl);
  await browser.opened;
  await browser.send('Browser.setDownloadBehavior', {behavior: 'allow', downloadPath: downloadDir});
  tab = new Connection(target.webSocketDebuggerUrl);
  await tab.opened;
  await tab.send('Runtime.enable');
  await tab.send('Page.enable');
  await tab.send('Page.navigate', {url: pathToFileURL(htmlPath).href});
  await until(() => page(`Boolean(window.ratingApp) && !document.getElementById('app').hidden`), 'the page to open');
  check(await page(`document.getElementById('help-dialog').open`), 'instructions open on first visit');
  await screenshot('1-instructions');
  await page(`document.getElementById('help-dialog').close(); window.confirm = () => true; true`);

  // Go to the session that holds the emoji fixture.
  const index = await page(`ratingApp.pkg.assignments.findIndex((a) => a.utterances.some((u) => u.text.includes("I can't sleep properly")))`);
  await page(`document.querySelector('[data-index="${index}"]').click(); true`);

  // A selection that crosses turns is refused, with a reason.
  const refused = await page(`(() => {
    const a = ratingApp.pkg.assignments[${index}], p = a.utterances.filter((u) => u.speaker === 'patient');
    const one = ratingApp.rangeFor(p[0].id, 0, 5), two = ratingApp.rangeFor(p[1].id, 0, 5), range = document.createRange();
    range.setStart(one.startContainer, one.startOffset); range.setEnd(two.endContainer, two.endOffset);
    const selection = getSelection(); selection.removeAllRanges(); selection.addRange(range);
    document.querySelector('.utt[data-uid="' + p[0].id + '"] .utt-text').dispatchEvent(new MouseEvent('mouseup', {bubbles: true}));
    return document.getElementById('toast').textContent;
  })()`);
  check(refused.startsWith('Select within one turn'), 'a selection across two turns is refused');

  // Claim 1: after an emoji, so UTF-16 and code-point offsets differ.
  const sleep1 = await select("I can't sleep properly");
  check(await page(`document.getElementById('d-symptom') !== null`), 'selecting a passage opens the claim editor');
  await screenshot('2-editor');
  await claim('sleep_problems', 'present', 'no');
  // Claim 2: a referential answer, linked to the therapist question before it.
  await select('Most nights I lie awake until two or three');
  await claim('sleep_problems', 'present', 'yes', {context: true});
  // Claims 3 and 4: the second of two identical phrases, coded two ways, so the turn is mixed.
  const tired = await select("I'm not tired", 1);
  await claim('low_energy', 'raised_and_denied', 'yes', {another: true});
  await claim('low_energy', 'present', 'yes');
  let s = await currentState();
  check(s.claims.length === 4, 'four claims saved');
  const first = s.claims.find((c) => c.spans[0].quote === "I can't sleep properly");
  check(first && first.spans[0].start === sleep1.start && first.spans[0].end === sleep1.end, 'offsets after an emoji are code points');
  check(s.claims.filter((c) => c.symptom === 'low_energy').every((c) => c.spans[0].start === tired.start), 'the second occurrence, not the first, was recorded');
  check(s.claims.some((c) => c.context.length === 1), 'a context link was stored');
  check(await page(`document.querySelectorAll('#dialogue mark.hl').length >= 3`), 'saved claims are highlighted in the transcript');

  check(await page(`document.getElementById('form-fields').disabled`), 'the session form is locked until every turn is reviewed');
  await page(`(() => { let button; while ((button = document.querySelector('#dialogue [data-action=review]'))) button.click(); return true; })()`);
  check(!(await page(`document.getElementById('form-fields').disabled`)), 'the session form opens once every turn is reviewed');

  await page(`(() => {
    for (const box of document.querySelectorAll('#session-form select[data-field]')) {
      const sleep = box.dataset.symptom === 'sleep_problems', energy = box.dataset.symptom === 'low_energy';
      box.value = box.dataset.field === 'expression' ? (sleep ? 'present' : energy ? 'unclear' : 'no_evidence') : (sleep ? '3' : 'cannot_judge');
      box.dispatchEvent(new Event('change', {bubbles: true}));
    }
    const prominent = document.getElementById('f-prominent'); prominent.value = 'sleep_problems'; prominent.dispatchEvent(new Event('change'));
    document.querySelector('input[name=f-realism][value="4"]').click();
    const complaint = document.getElementById('f-complaint'); complaint.value = 'Busy with the garage; sleeps badly.'; complaint.dispatchEvent(new Event('input'));
    document.querySelector('input[data-evidence=sleep_problems]').click();
    return true;
  })()`);
  await page(`document.getElementById('session-form').scrollIntoView(); true`);
  await screenshot('3-form');
  await page(`document.getElementById('submit-session').click(); true`);
  s = await currentState();
  check(s.status === 'submitted' && s.submissions.length === 1, 'the session is submitted and its answers frozen');
  const file = await until(() => readdirSync(downloadDir).find((name) => name.endsWith('.json')), 'the saved file');
  check(file.startsWith('rating_'), 'submitting saves a file');

  await page(`document.querySelector('#dialogue mark.hl').dispatchEvent(new MouseEvent('mouseup', {bubbles: true})); true`);
  check(await page(`document.getElementById('d-symptom') === null`), 'a submitted session opens read-only');

  // Reload: the browser copy restores the same work.
  await tab.send('Page.reload');
  await sleep(300);
  await until(() => page(`Boolean(window.ratingApp) && !document.getElementById('app').hidden`), 'the page to reopen');
  const reopened = await page(`ratingApp.state.assignments[${JSON.stringify(s.id)}].status`);
  check(reopened === 'submitted' && !(await page(`document.getElementById('help-dialog').open`)), 'work is restored after a reload');
  await page(`document.documentElement.classList.contains('dark') && document.getElementById('theme').click(); true`);
  await screenshot('4-light');

  console.log(JSON.stringify({ok: true, file: join(downloadDir, file), revision: s.revision, steps}));
} catch (error) {
  console.log(JSON.stringify({ok: false, error: String(error.stack || error), steps}));
  process.exitCode = 1;
} finally {
  chrome.kill();
  await sleep(300);
  rmSync(profile, {recursive: true, force: true});
  process.exit(process.exitCode || 0);
}
