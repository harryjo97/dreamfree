(() => {
  'use strict';

  const datasets = {
    droid: {
      label: 'DROID', tasks: window.DROID_GALLERY,
      description: 'Ten tasks. Five trials per policy. Explore every rollout.',
      training: 'DreamFree: trained on DROID only · No task-specific tuning.'
    },
    yam: {
      label: 'ABC-130k (YAM)', tasks: window.YAM_GALLERY,
      description: 'A small-data pilot on YAM. Six tasks, 51 rollouts, filmed from the top view.',
      training: 'Trained only on 100 hours of ABC-130k video (≈1/35 of its total duration) · No task-specific tuning.'
    }
  };
  const models = { ours: 'DreamFree', dreamzero: 'DreamZero', pi: 'π₀.₅' };
  const media = window.DreamFreeMedia;
  const nav = document.querySelector('#droid-task-nav');
  const taskSelect = document.querySelector('#droid-task-select');
  const trials = document.querySelector('#droid-trials');
  const tabs = [...document.querySelectorAll('[data-model][role="tab"]')];
  const datasetTabs = [...document.querySelectorAll('[data-dataset][role="tab"]')];
  const lastTask = { droid: datasets.droid.tasks[0].id, yam: datasets.yam.tasks[0].id };
  let currentDataset;
  let currentTask;
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
      element('td', '', percentage(mean(datasets.droid.tasks.flatMap(task => task.outcomes[key])))),
      element('td', '', percentage(mean(datasets.droid.tasks.flatMap(task => task.progress[key])))));
    document.querySelector('#overall-results').append(row);
  });

  function loadVideo(video) {
    video.src = video.dataset.src;
    video.preload = 'metadata';
    media.register(video, 'gallery');
  }
  const loader = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      loader.unobserve(entry.target);
      loadVideo(entry.target);
    });
  }, { rootMargin: '300px 0px' });

  function renderCollection(key) {
    currentDataset = key;
    const dataset = datasets[key];
    const isYam = key === 'yam';
    document.body.dataset.gallery = key;
    document.querySelector('#gallery-collection').setAttribute('aria-labelledby', `gallery-dataset-${key}`);
    document.querySelector('#gallery-description').textContent = dataset.description;
    document.querySelector('#gallery-training').textContent = dataset.training;
    document.querySelector('#gallery-summary').hidden = isYam;
    document.querySelector('#droid-task-metrics').hidden = isYam;
    document.querySelector('.droid-model-tabs').hidden = isYam;
    document.querySelector('#gallery-pilot-label').hidden = !isYam;
    document.querySelector('.gallery-timing-note').hidden = isYam;
    document.querySelector('#gallery-task-count').textContent = `01–${String(dataset.tasks.length).padStart(2, '0')}`;
    datasetTabs.forEach(tab => {
      const active = tab.dataset.dataset === key;
      tab.setAttribute('aria-selected', String(active));
      tab.tabIndex = active ? 0 : -1;
    });
    nav.replaceChildren();
    taskSelect.replaceChildren();
    dataset.tasks.forEach(task => {
      const link = element('a');
      link.href = `#${task.id}`;
      link.dataset.task = task.id;
      link.append(element('span', '', task.number), element('span', '', task.category));
      if (isYam) link.append(element('span', 'gallery-nav-count', String(task.runs.length)));
      nav.append(link);
      const option = element('option', '', `${task.number} · ${task.category}${isYam ? ` (${task.runs.length})` : ''}`);
      option.value = task.id;
      taskSelect.append(option);
    });
  }

  function renderMetrics() {
    const container = document.querySelector('#droid-task-metrics');
    container.replaceChildren();
    if (currentDataset === 'yam') return;
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
    trials.querySelectorAll('video').forEach(video => {
      loader.unobserve(video);
      media.unregister(video);
      video.removeAttribute('src');
      video.load();
    });
    trials.replaceChildren();
    const isYam = currentDataset === 'yam';
    tabs.forEach(tab => {
      const active = tab.dataset.model === currentModel;
      tab.setAttribute('aria-selected', String(active));
      tab.tabIndex = active ? 0 : -1;
    });
    trials.setAttribute('role', isYam ? 'region' : 'tabpanel');
    trials.setAttribute('aria-labelledby', isYam ? 'droid-task-title' : `droid-model-${currentModel}`);
    const runs = isYam ? currentTask.runs : currentTask[currentModel].map((src, index) => ({
      src, status: currentTask.outcomes[currentModel][index] ? 'success' : 'fail',
      progress: currentTask.progress[currentModel][index] * 100
    }));
    const orderedRuns = runs.map((run, index) => ({ ...run, trialNumber: index + 1 }))
      .sort((a, b) => Number(b.status === 'success') - Number(a.status === 'success') || a.trialNumber - b.trialNumber);
    orderedRuns.forEach((run, index) => {
      const card = element('figure', 'droid-trial');
      const caption = element('figcaption');
      const trialLabel = `${isYam ? 'Rollout' : 'Trial'} ${String(isYam ? index + 1 : run.trialNumber).padStart(2, '0')}`;
      caption.append(element('span', '', trialLabel));
      if (run.status) {
        const outcome = element('span', `droid-trial-outcome${run.status === 'success' ? ' is-success' : ''}`);
        outcome.append(element('span', '', run.status === 'success' ? 'Success' : 'Failure'));
        if (!isYam) outcome.append(element('span', 'droid-trial-progress', `Progress ${percentage(run.progress)}`));
        caption.append(outcome);
      }
      const frame = element('div', 'droid-trial-frame');
      const video = element('video');
      video.preload = 'none';
      video.dataset.src = run.src;
      video.defaultPlaybackRate = 1;
      video.playbackRate = 1;
      video.setAttribute('aria-label', `${isYam ? 'DreamFree on YAM' : models[currentModel]} · ${currentTask.title} · ${trialLabel}`);
      const error = element('p', 'droid-video-error', 'This video could not be loaded.');
      error.hidden = true;
      video.addEventListener('error', () => { if (video.getAttribute('src')) error.hidden = false; });
      video.addEventListener('loadedmetadata', () => { video.playbackRate = 1; });
      frame.append(video, error);
      card.append(caption, frame);
      trials.append(card);
      loader.observe(video);
    });
    document.querySelector('#droid-status').textContent = `${currentTask.title}: showing ${runs.length} ${isYam ? 'DreamFree rollouts on YAM' : `${models[currentModel]} trials`}.`;
  }

  function renderTask() {
    const hash = location.hash.slice(1);
    const key = datasets.yam.tasks.some(task => task.id === hash) || hash === 'yam' ? 'yam' : 'droid';
    const dataset = datasets[key];
    currentTask = dataset.tasks.find(task => task.id === hash) || dataset.tasks[0];
    lastTask[key] = currentTask.id;
    if (currentDataset !== key) renderCollection(key);
    taskSelect.value = currentTask.id;
    nav.querySelectorAll('a').forEach(link => {
      if (link.dataset.task === currentTask.id) link.setAttribute('aria-current', 'true');
      else link.removeAttribute('aria-current');
    });
    document.querySelector('#droid-task-number').textContent = `TASK ${currentTask.number} / ${String(dataset.tasks.length).padStart(2, '0')}${key === 'yam' ? ` · ${currentTask.runs.length} ROLLOUTS · TOP VIEW` : ''}`;
    document.querySelector('#droid-task-title').textContent = currentTask.title;
    document.querySelector('#droid-instruction').textContent = currentTask.instruction;
    document.title = `${currentTask.category} · ${key === 'yam' ? 'YAM' : 'DROID'} · Gallery — DreamFree`;
    renderMetrics();
    renderTrials();
  }

  function bindTabKeys(buttons) {
    buttons.forEach((tab, index) => {
      tab.addEventListener('keydown', event => {
        let next;
        if (event.key === 'ArrowRight') next = (index + 1) % buttons.length;
        if (event.key === 'ArrowLeft') next = (index - 1 + buttons.length) % buttons.length;
        if (event.key === 'Home') next = 0;
        if (event.key === 'End') next = buttons.length - 1;
        if (next === undefined) return;
        event.preventDefault();
        buttons[next].focus();
        buttons[next].click();
      });
    });
  }
  tabs.forEach(tab => tab.addEventListener('click', () => {
    currentModel = tab.dataset.model;
    renderTrials();
  }));
  datasetTabs.forEach(tab => tab.addEventListener('click', () => {
    location.hash = lastTask[tab.dataset.dataset];
  }));
  bindTabKeys(tabs);
  bindTabKeys(datasetTabs);
  taskSelect.addEventListener('change', () => { location.hash = taskSelect.value; });
  window.addEventListener('hashchange', renderTask);
  media.bindToggle(document.querySelector('#droid-playback'), 'gallery');
  renderTask();
})();
