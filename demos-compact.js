(() => {
  'use strict';

  const section = document.getElementById('demos');
  const media = window.DreamFreeMedia;
  const status = document.getElementById('compact-status');
  if (!section || !media) return;

  // Load nearby clips; the shared player pauses every clip outside the viewport.
  const lazyMedia = new IntersectionObserver(entries => {
    entries.forEach(({ target: video, isIntersecting }) => {
      if (!isIntersecting) return;
      video.src = video.dataset.src;
      delete video.dataset.src;
      video.load();
      media.register(video, video.closest('#failure-cases') ? 'demo-failures' : 'compact-demos');
      lazyMedia.unobserve(video);
    });
  }, { rootMargin: '200px' });

  document.querySelectorAll('#demos video[data-src], #failure-cases video[data-src]').forEach(video => {
    video.addEventListener('error', () => {
      status.textContent = `Could not load ${video.getAttribute('aria-label')}.`;
    });
    lazyMedia.observe(video);
  });
  media.bindToggle(document.getElementById('compact-motion'), 'compact-demos');
  media.bindToggle(document.getElementById('failure-motion'), 'demo-failures');
})();
