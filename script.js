(() => {
  'use strict';
  const menuButton = document.querySelector('.menu-toggle');
  const menu = document.querySelector('.site-nav');
  const closeMenu = () => {
    menuButton.setAttribute('aria-expanded', 'false');
    menu.classList.remove('is-open');
  };
  menuButton.addEventListener('click', () => {
    const expanded = menuButton.getAttribute('aria-expanded') === 'true';
    menuButton.setAttribute('aria-expanded', String(!expanded));
    menu.classList.toggle('is-open', !expanded);
  });
  menu.querySelectorAll('a').forEach(link => link.addEventListener('click', closeMenu));
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && menuButton.getAttribute('aria-expanded') === 'true') {
      closeMenu();
      menuButton.focus();
    }
  });
  document.addEventListener('click', event => {
    if (!event.target.closest('.nav-shell')) closeMenu();
  });
  window.matchMedia('(min-width: 621px)').addEventListener('change', event => {
    if (event.matches) closeMenu();
  });

  const copyButton = document.querySelector('[data-copy]');
  copyButton.addEventListener('click', async () => {
    const text = document.getElementById(copyButton.dataset.copy).textContent;
    const label = copyButton.querySelector('[data-copy-label]');
    const status = document.querySelector('.copy-status');
    try {
      await navigator.clipboard.writeText(text);
      label.textContent = 'Copied';
      status.textContent = 'BibTeX copied to clipboard.';
      window.setTimeout(() => { label.textContent = 'Copy BibTeX'; }, 2500);
    } catch {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(document.getElementById(copyButton.dataset.copy));
      selection.removeAllRanges();
      selection.addRange(range);
      label.textContent = 'Select & copy';
      status.textContent = 'The citation is selected. Use your device’s copy command.';
    }
  });
})();
