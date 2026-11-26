/* Service Worker der Website: macht sie installierbar und zeigt die zuletzt geladenen Tipps auch offline.
   Seite, Skript und Daten: erst Netz, bei Ausfall der Zwischenspeicher. Schriften, Symbole, d3: aus dem Speicher. */
'use strict';
var VERSION = 'bt-84c6cc0029a3';
var GRUND = ['./', 'index.html', 'site.js', 'fonts.css', 'd3.min.js', 'icon.svg', 'icon-192.png', 'icon-512.png',
  'apple-touch-icon.png', 'manifest.webmanifest', 'fonts/archivo-var-latin.woff2', 'fonts/archivo-var-latin-ext.woff2',
  'data/prognosen.json', 'data/staerken.json', 'data/modell.json', 'data/protokoll.json'];

self.addEventListener('install', function (e) {
  e.waitUntil(caches.open(VERSION).then(function (c) { return c.addAll(GRUND); }).then(function () { return self.skipWaiting(); }));
});

self.addEventListener('activate', function (e) {
  e.waitUntil(caches.keys().then(function (keys) {
    return Promise.all(keys.filter(function (k) { return k !== VERSION; }).map(function (k) { return caches.delete(k); }));
  }).then(function () { return self.clients.claim(); }));
});

function netzZuerst(req) {
  return fetch(req).then(function (res) {
    if (res && res.ok) {
      var copy = res.clone();
      caches.open(VERSION).then(function (c) { c.put(req, copy); });
    }
    return res;
  }).catch(function () {
    return caches.match(req, {ignoreSearch: true}).then(function (hit) {
      return hit || (req.mode === 'navigate' ? caches.match('index.html') : Response.error());
    });
  });
}

function speicherZuerst(req) {
  return caches.match(req).then(function (hit) {
    return hit || fetch(req).then(function (res) {
      if (res && res.ok) {
        var copy = res.clone();
        caches.open(VERSION).then(function (c) { c.put(req, copy); });
      }
      return res;
    });
  });
}

self.addEventListener('fetch', function (e) {
  var req = e.request;
  if (req.method !== 'GET') return;
  var url = new URL(req.url);
  if (url.origin !== self.location.origin) return;
  var p = url.pathname;
  if (req.mode === 'navigate' || /\/data\/[^/]+\.json$/.test(p) || /\/(index\.html|site\.js|fonts\.css|manifest\.webmanifest)$/.test(p) || p.endsWith('/')) {
    e.respondWith(netzZuerst(req));
  } else {
    e.respondWith(speicherZuerst(req));
  }
});
