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

  document.querySelectorAll('[data-copy]').forEach(copyButton => {
    const label = copyButton.querySelector('[data-copy-label]');
    const originalLabel = label.textContent;
    let resetTimer;
    copyButton.addEventListener('click', async () => {
      const target = document.getElementById(copyButton.dataset.copy);
      const text = target.textContent.trim();
      const status = copyButton.parentElement.querySelector('.copy-status');
      const name = copyButton.dataset.copyName || 'Text';
      window.clearTimeout(resetTimer);
      try {
        await navigator.clipboard.writeText(text);
        label.textContent = 'Copied';
        status.textContent = `${name} copied to clipboard.`;
        resetTimer = window.setTimeout(() => { label.textContent = originalLabel; }, 2500);
      } catch {
        const selection = window.getSelection();
        const range = document.createRange();
        range.selectNodeContents(target);
        selection.removeAllRanges();
        selection.addRange(range);
        if (document.execCommand('copy')) {
          label.textContent = 'Copied';
          status.textContent = `${name} copied to clipboard.`;
          selection.removeAllRanges();
          resetTimer = window.setTimeout(() => { label.textContent = originalLabel; }, 2500);
        } else {
          label.textContent = 'Select & copy';
          status.textContent = `The ${name.toLowerCase()} is selected. Use your device’s copy command.`;
        }
      }
    });
  });
})();
