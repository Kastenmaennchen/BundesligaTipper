/* Kleine Laufzeit für die öffentliche Website: lädt die JSON-Daten aus data/ und zeichnet
   die Bereiche der Seite – dieselbe Schnittstelle wie im Claude-Dashboard (dash.data, dash.draw …). */
(function () {
  'use strict';
  var rows = {}, state = {}, calcs = {}, draws = [], pending = false;

  function load(id) {
    if (state[id]) return;
    state[id] = 'loading';
    fetch('data/' + id + '.json', {cache: 'no-cache'})
      .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(function (j) { rows[id] = Array.isArray(j) ? j : (j.rows || []); state[id] = 'ok'; })
      .catch(function () { state[id] = 'error'; })
      .then(schedule);
  }

  function data(id) {
    var c = calcs[id];
    if (c) {
      var inputs = c.inputs.map(data);
      var bad = inputs.find(function (d) { return d.status !== 'ok'; });
      if (bad) return {status: bad.status, data: null};
      try { return {status: 'ok', data: c.fn.apply(null, inputs.map(function (d) { return d.data; }))}; }
      catch (e) { return {status: 'error', data: null}; }
    }
    load(id);
    return state[id] === 'ok' ? {status: 'ok', data: rows[id]} : {status: state[id], data: []};
  }

  function run(only) {
    draws.forEach(function (d) {
      if (only && d[0] !== only) return;
      document.querySelectorAll('[data-draw="' + d[0] + '"]').forEach(function (el) {
        if (el.closest('[hidden]')) return;
        try { d[1](el); } catch (e) { console.error('Zeichnen fehlgeschlagen:', d[0], e); }
      });
    });
  }

  function schedule() {
    if (pending) return;
    pending = true;
    requestAnimationFrame(function () { pending = false; run(); });
  }

  window.dash = {
    colors: ['#d8b45f', '#c9d1dc'],
    data: data,
    calc: function (id, o) { calcs[id] = o; },
    loader: function () {},
    onData: function (fn) { fn(); },
    params: function () { return {}; },
    draw: function (name, fn) { if (fn) { draws.push([name, fn]); schedule(); } else run(name); },
    setLink: function (name) {
      try { history.replaceState(null, '', '#' + name); } catch (e) { /* ohne Verlauf */ }
      schedule();
    },
  };

  var t;
  window.addEventListener('resize', function () { clearTimeout(t); t = setTimeout(schedule, 150); });
  document.addEventListener('DOMContentLoaded', schedule);
})();

/* Als App installieren: Service Worker, Installieren-Knopf (Chrome, Edge, Android) und Hinweis für iPhone/iPad. */
(function () {
  'use strict';
  if ('serviceWorker' in navigator && (location.protocol === 'https:' || location.hostname === 'localhost')) {
    window.addEventListener('load', function () { navigator.serviceWorker.register('sw.js').catch(function () {}); });
  }
  var KEY = 'bt-install-weg';
  var standalone = (window.matchMedia && window.matchMedia('(display-mode: standalone)').matches) || window.navigator.standalone === true;
  var ua = navigator.userAgent || '';
  var ios = /iPad|iPhone|iPod/.test(ua) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
  var prompt = null;

  function weg() { try { return localStorage.getItem(KEY) === '1'; } catch (e) { return false; } }
  function setWeg() { try { localStorage.setItem(KEY, '1'); } catch (e) { /* ohne Speicher */ } }

  function installieren() {
    if (!prompt) return;
    prompt.prompt();
    prompt.userChoice.then(function () { prompt = null; zeigen(); }).catch(function () {});
  }

  function banner(text, mitKnopf) {
    if (weg() || document.getElementById('bt-install')) return;
    var panel = document.getElementById('spiele');
    if (!panel) return;
    var b = document.createElement('div');
    b.className = 'bt-install'; b.id = 'bt-install'; b.setAttribute('role', 'region'); b.setAttribute('aria-label', 'App installieren');
    var p = document.createElement('p'); p.textContent = text; b.appendChild(p);
    if (mitKnopf) {
      var k = document.createElement('button'); k.type = 'button'; k.className = 'bt-btn'; k.textContent = 'Installieren';
      k.addEventListener('click', installieren); b.appendChild(k);
    }
    var x = document.createElement('button'); x.type = 'button'; x.className = 'x'; x.setAttribute('aria-label', 'Hinweis schließen'); x.textContent = '×';
    x.addEventListener('click', function () { setWeg(); b.remove(); });
    b.appendChild(x);
    panel.insertBefore(b, panel.firstChild);
  }

  function zeigen() {
    var knopf = document.getElementById('app-knopf'), fertig = document.getElementById('app-fertig'),
      wege = document.getElementById('app-wege'), text = document.getElementById('app-text');
    if (standalone) {
      if (fertig) fertig.hidden = false;
      if (wege) wege.hidden = true;
      if (text) text.hidden = true;
      if (knopf) knopf.hidden = true;
      return;
    }
    if (knopf) knopf.hidden = !prompt;
    var b = document.getElementById('bt-install');
    if (prompt) banner('Als App installieren – auch offline nutzbar.', true);
    else if (b && b.querySelector('.bt-btn')) b.remove();
    if (ios && !prompt) banner('Als App: unten auf das Teilen-Symbol tippen, dann „Zum Home-Bildschirm“.', false);
  }

  window.addEventListener('beforeinstallprompt', function (e) { e.preventDefault(); prompt = e; zeigen(); });
  window.addEventListener('appinstalled', function () { prompt = null; setWeg(); var b = document.getElementById('bt-install'); if (b) b.remove(); zeigen(); });
  document.addEventListener('click', function (e) { if (e.target.closest && e.target.closest('[data-app-install]')) installieren(); });
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', zeigen); else zeigen();
})();
