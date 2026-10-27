(() => {
  'use strict';

  const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const videos = new Map();
  const scopes = new Map();
  const scopeState = name => {
    if (!scopes.has(name)) scopes.set(name, { paused: motion.matches, buttons: new Set() });
    return scopes.get(name);
  };

  function updateButtons(name) {
    const state = scopeState(name);
    state.buttons.forEach(button => {
      button.textContent = state.paused ? 'Resume videos' : 'Pause videos';
      button.setAttribute('aria-pressed', String(state.paused));
    });
  }

  function refresh(video) {
    const item = videos.get(video);
    if (!item) return;
    const generation = ++item.generation;
    if (!item.visible || document.hidden || scopeState(item.scope).paused) {
      video.pause();
      return;
    }
    const pending = video.play();
    if (!pending?.catch) return;
    pending.catch(error => {
      if (videos.get(video) !== item || generation !== item.generation) return;
      if (error.name === 'NotAllowedError') {
        scopeState(item.scope).paused = true;
        refreshScope(item.scope);
        updateButtons(item.scope);
      }
    });
  }

  function refreshScope(name) {
    videos.forEach((item, video) => { if (item.scope === name) refresh(video); });
  }

  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      const item = videos.get(entry.target);
      if (!item) return;
      item.visible = entry.isIntersecting && entry.intersectionRatio >= 0.15;
      refresh(entry.target);
    });
  }, { threshold: [0, 0.15] });

  window.DreamFreeMedia = {
    register(video, scope) {
      video.controls = false;
      video.autoplay = false; // Visibility-aware playback is managed below.
      video.muted = true;
      video.defaultMuted = true;
      video.loop = true;
      video.playsInline = true;
      video.disablePictureInPicture = true;
      video.disableRemotePlayback = true;
      video.classList.add('ambient-video');
      if (videos.has(video)) return;
      scopeState(scope);
      videos.set(video, { scope, visible: false, generation: 0 });
      observer.observe(video);
    },
    unregister(video) {
      observer.unobserve(video);
      videos.delete(video);
      video.pause();
    },
    refresh,
    bindToggle(button, scope) {
      if (!button) return;
      scopeState(scope).buttons.add(button);
      button.addEventListener('click', () => {
        const state = scopeState(scope);
        state.paused = !state.paused;
        refreshScope(scope);
        updateButtons(scope);
      });
      updateButtons(scope);
    }
  };

  document.addEventListener('visibilitychange', () => videos.forEach((item, video) => refresh(video)));
  motion.addEventListener('change', event => {
    scopes.forEach((state, name) => {
      state.paused = event.matches;
      updateButtons(name);
    });
    videos.forEach((item, video) => refresh(video));
  });
})();
