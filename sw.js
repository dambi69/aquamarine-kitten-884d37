const CACHE = 'smartair-v1';
const SHELL = ['./index.html', './manifest.json', './icon-192.svg', './icon-512.svg'];

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  /* Navigation requests: serve shell from cache, fall back to network */
  if(e.request.mode === 'navigate'){
    e.respondWith(
      caches.match('./index.html').then(r => r || fetch(e.request))
    );
    return;
  }
  /* Firebase / OpenWeatherMap API calls: always network-first */
  const url = new URL(e.request.url);
  if(url.hostname.includes('firebase') || url.hostname.includes('openweathermap')){
    return;
  }
  /* Static assets: cache-first */
  e.respondWith(
    caches.match(e.request).then(r => r || fetch(e.request))
  );
});

self.addEventListener('push', e => {
  let data = {};
  try { data = e.data ? e.data.json() : {}; } catch (_) {}
  e.waitUntil(
    self.registration.showNotification(data.title || 'SmartAir Alert', {
      body    : data.body || 'ตรวจพบค่าผิดปกติ',
      icon    : './icon-192.svg',
      badge   : './icon-192.svg',
      tag     : 'smartair-alert',
      renotify: true,
    })
  );
});

self.addEventListener('notificationclick', e => {
  e.notification.close();
  e.waitUntil(clients.openWindow('./'));
});
