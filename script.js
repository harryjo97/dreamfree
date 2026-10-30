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
