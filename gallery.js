(() => {
  'use strict';

  const tasks = window.DROID_GALLERY;
  const models = { ours: 'DreamFree', dreamzero: 'DreamZero', pi: 'π₀.₅' };
  const media = window.DreamFreeMedia;
  const nav = document.querySelector('#droid-task-nav');
  const taskSelect = document.querySelector('#droid-task-select');
  const trials = document.querySelector('#droid-trials');
  const speed = document.querySelector('#droid-speed');
  const tabs = [...document.querySelectorAll('[data-model][role="tab"]')];
  let currentTask = tasks[0];
  let currentModel = 'ours';

  const mean = values => values.reduce((sum, value) => sum + Number(value), 0) / values.length * 100;
  const percentage = value => `${Number(value.toFixed(1))}%`;
  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  Object.entries(models).forEach(([key, name]) => {
    const row = element('tr', key === 'ours' ? 'ours' : '');
    const label = element('th', key === 'ours' ? 'method-name' : '', name);
    label.scope = 'row';
    row.append(label,
      element('td', '', percentage(mean(tasks.flatMap(task => task.outcomes[key])))),
      element('td', '', percentage(mean(tasks.flatMap(task => task.progress[key])))));
    document.querySelector('#overall-results').append(row);
  });

  tasks.forEach(task => {
    const link = element('a');
    link.href = `#${task.id}`;
    link.dataset.task = task.id;
    link.append(element('span', '', task.number), element('span', '', task.category));
    nav.append(link);
    const option = element('option', '', `${task.number} · ${task.category}`);
    option.value = task.id;
    taskSelect.append(option);
  });

  function renderMetrics() {
    const container = document.querySelector('#droid-task-metrics');
    container.replaceChildren();
    Object.entries(models).forEach(([key, name]) => {
      const card = element('div', 'droid-metric');
      card.dataset.model = key;
      const heading = element('h3', key === 'ours' ? 'method-name' : '', name);
      const values = element('div', 'droid-metric-values');
      [['Success', currentTask.outcomes[key]], ['Progress', currentTask.progress[key]]].forEach(([label, scores]) => {
        const metric = element('span', '', label);
        metric.append(element('strong', '', percentage(mean(scores))));
        values.append(metric);
      });
      card.append(heading, values);
      container.append(card);
    });
  }

  function renderTrials() {
    trials.querySelectorAll('video').forEach(video => media.unregister(video));
    trials.replaceChildren();
    tabs.forEach(tab => {
      const active = tab.dataset.model === currentModel;
      tab.setAttribute('aria-selected', String(active));
      tab.tabIndex = active ? 0 : -1;
    });
    trials.setAttribute('aria-labelledby', `droid-model-${currentModel}`);
    currentTask[currentModel].forEach((src, index) => {
      const success = currentTask.outcomes[currentModel][index];
      const progress = currentTask.progress[currentModel][index] * 100;
      const card = element('figure', 'droid-trial');
      const caption = element('figcaption');
      const outcome = element('span', `droid-trial-outcome${success ? ' is-success' : ''}`);
      outcome.append(element('span', '', success ? 'Success' : 'Failure'), element('span', 'droid-trial-progress', `Progress ${percentage(progress)}`));
      caption.append(element('span', '', `Trial ${String(index + 1).padStart(2, '0')}`), outcome);
      const frame = element('div', 'droid-trial-frame');
      const video = element('video');
      video.preload = 'metadata';
      video.src = src;
      video.playbackRate = Number(speed.value);
      video.setAttribute('aria-label', `${models[currentModel]} · ${currentTask.title} · Trial ${index + 1}`);
      const error = element('p', 'droid-video-error', 'This video could not be loaded.');
      error.hidden = true;
      video.addEventListener('error', () => { error.hidden = false; });
      video.addEventListener('loadedmetadata', () => { video.playbackRate = Number(speed.value); });
      frame.append(video, error);
      card.append(caption, frame);
      trials.append(card);
      media.register(video, 'droid-evaluation');
    });
    document.querySelector('#droid-status').textContent = `${currentTask.title}: showing all five ${models[currentModel]} trials.`;
  }

  function renderTask() {
    const hash = location.hash.slice(1);
    currentTask = tasks.find(task => task.id === hash) || tasks[0];
    taskSelect.value = currentTask.id;
    nav.querySelectorAll('a').forEach(link => {
      if (link.dataset.task === currentTask.id) link.setAttribute('aria-current', 'true');
      else link.removeAttribute('aria-current');
    });
    document.querySelector('#droid-task-number').textContent = `TASK ${currentTask.number} / 10`;
    document.querySelector('#droid-task-title').textContent = currentTask.title;
    document.querySelector('#droid-instruction').textContent = currentTask.instruction;
    document.title = `${currentTask.category} · DROID Gallery — DreamFree`;
    renderMetrics();
    renderTrials();
  }

  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => {
      currentModel = tab.dataset.model;
      renderTrials();
    });
    tab.addEventListener('keydown', event => {
      let next;
      if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
      if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = tabs.length - 1;
      if (next === undefined) return;
      event.preventDefault();
      tabs[next].focus();
      tabs[next].click();
    });
  });
  taskSelect.addEventListener('change', () => { location.hash = taskSelect.value; });
  window.addEventListener('hashchange', renderTask);
  speed.addEventListener('change', () => {
    trials.querySelectorAll('video').forEach(video => { video.playbackRate = Number(speed.value); });
  });
  media.bindToggle(document.querySelector('#droid-playback'), 'droid-evaluation');
  renderTask();
})();
