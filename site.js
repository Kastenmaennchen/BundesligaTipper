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
