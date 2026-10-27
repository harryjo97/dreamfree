'use strict';

const config = window.DREAMFREE;
// Empty author URLs remain plain names; configured personal sites open in a new tab.
document.querySelectorAll('[data-author]').forEach(link => {
  const value = config.authorLinks?.[link.dataset.author]?.trim();
  if (!value) return;
  try {
    const url = new URL(/^https?:\/\//i.test(value) ? value : `https://${value}`);
    if (!['https:', 'http:'].includes(url.protocol)) return;
    link.href = url.href;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
  } catch { /* Keep the name readable if a URL is incomplete. */ }
});
const mediaPlayback = window.DreamFreeMedia;
const teaser = document.querySelector('#teaser-video');
if (teaser.querySelector('source').getAttribute('src') !== config.teaser.src) {
  teaser.querySelector('source').src = config.teaser.src;
  teaser.load();
}
teaser.poster = config.teaser.poster;
mediaPlayback.register(teaser, 'teaser');
mediaPlayback.bindToggle(document.querySelector('#teaser-motion'), 'teaser');

// Unpublished links stay visibly unavailable instead of pointing to a guessed URL.
document.querySelectorAll('.resource.unavailable').forEach((button, index) => {
  const url = index === 0 ? config.paperUrl : config.codeUrl;
  if (!url) return;
  const link = document.createElement('a');
  link.className = 'resource';
  link.href = url;
  link.target = '_blank';
  link.rel = 'noopener noreferrer';
  link.innerHTML = button.innerHTML;
  link.querySelector('small').remove();
  button.replaceWith(link);
});

const tabs = [...document.querySelectorAll('#examples [role="tab"]')];
const grid = document.querySelector('#video-grid');
const panel = document.querySelector('#video-panel');
const playAll = document.querySelector('#play-all');
function galleryVideos() { return [...grid.querySelectorAll('video')]; }
function renderDataset(key) {
  galleryVideos().forEach(video => mediaPlayback.unregister(video));
  const dataset = config.datasets[key];
  tabs.forEach(tab => { const active = tab.dataset.dataset === key; tab.setAttribute('aria-selected', String(active)); tab.tabIndex = active ? 0 : -1; });
  panel.setAttribute('aria-labelledby', `tab-${key}`);
  const description = document.querySelector('#dataset-description');
  description.replaceChildren();
  if (dataset.placeholder) { const badge = document.createElement('span'); badge.className = 'placeholder-note'; badge.textContent = 'Placeholder footage'; description.append(badge); }
  dataset.description.split(/(DreamFree)/g).forEach(text => {
    if (text === 'DreamFree') {
      const name = document.createElement('span');
      name.className = 'method-name';
      name.textContent = text;
      description.append(name);
    } else {
      description.append(document.createTextNode(text));
    }
  });
  grid.replaceChildren();
  dataset.clips.forEach((id, index) => {
    const clip = config.clips.find(item => item.id === id);
    const card = document.createElement('figure');
    card.className = 'video-card';
    const frame = document.createElement('div'); frame.className = 'video-frame';
    const video = document.createElement('video');
    const source = document.createElement('source');
    source.src = clip.src || `assets/videos/${id}.mp4`;
    source.type = 'video/mp4';
    video.append(source);
    video.poster = clip.poster || `assets/posters/${id}.jpg`;
    video.controls = false; video.muted = true; video.loop = true; video.playsInline = true; video.preload = 'metadata';
    video.setAttribute('aria-label', `${clip.title}${dataset.placeholder ? ' — DROID placeholder' : ' — DreamFree DROID rollout'}`);
    video.addEventListener('error', () => { document.querySelector('#gallery-status').textContent = 'A video could not load. Check its path in content.js.'; });
    const number = document.createElement('span'); number.className = 'video-number'; number.textContent = String(index + 1).padStart(2, '0');
    frame.append(video, number);
    const caption = document.createElement('figcaption'); caption.textContent = clip.title;
    const meta = document.createElement('span'); meta.className = 'video-meta'; meta.textContent = dataset.placeholder ? 'DROID footage · Placeholder' : 'DROID · Real-world rollout';
    caption.append(meta); card.append(frame, caption); grid.append(card);
    mediaPlayback.register(video, 'gallery');
  });
}
tabs.forEach((tab, index) => {
  tab.addEventListener('click', () => renderDataset(tab.dataset.dataset));
  tab.addEventListener('keydown', event => {
    let next;
    if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
    if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = tabs.length - 1;
    if (next === undefined) return;
    event.preventDefault(); tabs[next].focus(); renderDataset(tabs[next].dataset.dataset);
  });
});
mediaPlayback.bindToggle(playAll, 'gallery');
renderDataset('droid');

const dialog = document.querySelector('#figure-dialog');
const dialogImage = document.querySelector('#dialog-image');
document.querySelectorAll('.figure-zoom').forEach(button => button.addEventListener('click', () => {
  const image = button.querySelector('img'); dialogImage.src = image.src; dialogImage.alt = image.alt; dialog.showModal();
}));
document.querySelector('#close-figure').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });

const copyButton = document.querySelector('#copy-citation');
copyButton.addEventListener('click', async () => {
  const text = document.querySelector('#bibtex').textContent;
  try {
    await navigator.clipboard.writeText(text);
    copyButton.textContent = 'Copied!';
    document.querySelector('#copy-status').textContent = 'BibTeX copied to clipboard.';
    setTimeout(() => { copyButton.textContent = 'Copy BibTeX'; }, 2200);
  } catch {
    const selection = window.getSelection(); const range = document.createRange();
    range.selectNodeContents(document.querySelector('#bibtex')); selection.removeAllRanges(); selection.addRange(range);
    copyButton.textContent = 'Select → copy';
    document.querySelector('#copy-status').textContent = 'Citation selected. Press Command+C or Control+C to copy.';
  }
});
