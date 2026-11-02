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
// Unpublished links stay visibly unavailable instead of pointing to a guessed URL.
document.querySelectorAll('[data-resource]').forEach(button => {
  const url = config[`${button.dataset.resource}Url`];
  if (!url) return;
  if (button.tagName === 'A') {
    button.href = url;
    return;
  }
  const link = document.createElement('a');
  link.className = 'resource';
  link.href = url;
  link.target = '_blank';
  link.rel = 'noopener noreferrer';
  link.innerHTML = button.innerHTML;
  if (!button.dataset.comingSoon) link.querySelector('small')?.remove();
  button.replaceWith(link);
});

// Reveal each Figure 1 text box once as it enters the viewport.
const deploymentStory = document.querySelector('.deployment-evidence .deployment-story');
if (deploymentStory && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
  const revealObserver = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add('is-visible');
      revealObserver.unobserve(entry.target);
    });
  }, { threshold: .15, rootMargin: '0px 0px -32px 0px' });
  deploymentStory.classList.add('reveal-ready');
  Array.from(deploymentStory.children).forEach(card => revealObserver.observe(card));
}

// Reveal each comparison panel together with its explanation, once per visit.
const conceptComparison = document.querySelector('.concept-comparison');
if (conceptComparison && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
  const conceptObserver = new IntersectionObserver(entries => {
    if (!entries.some(entry => entry.isIntersecting && entry.intersectionRatio >= .2)) return;
    conceptComparison.classList.add('is-playing');
    conceptObserver.disconnect();
  }, { threshold: .2, rootMargin: '0px 0px -32px 0px' });
  conceptComparison.classList.add('reveal-ready');
  conceptObserver.observe(conceptComparison.querySelector('.concept-caption > div'));
}

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
