(() => {
  'use strict';

  const root = document.querySelector('#video-transfer');
  const data = window.DREAMFREE?.transfer;
  if (!root || !data) return;

  const find = id => root.querySelector(`#transfer-${id}`);
  const tabs = [...root.querySelectorAll('[data-transfer-group]')];
  const taskSelect = find('task');
  const viewSelect = find('view');
  const mediaPlayback = window.DreamFreeMedia;
  const sampleButtons = find('samples');
  const videos = [find('training'), find('before'), find('after')];
  const tasks = new Map(data.tasks.map(task => [task.id, task]));
  let currentGroup;
  let currentTask;
  let currentSample = 0;

  function setEmbodiment(role, robot) {
    const label = find(`${role}-robot`);
    label.textContent = robot;
    const colors = { 'Dual-arm Franka': 'franka', 'ALOHA-AgileX': 'aloha' };
    label.hidden = !colors[robot];
    label.closest('.transfer-video-card').dataset.embodiment = colors[robot] || 'same';
  }

  function setMedia(video, media, label) {
    video.pause();
    video.setAttribute('aria-label', label);
    if (video.getAttribute('src') !== media.src) {
      video.src = media.src;
      video.load();
    }
    video.defaultPlaybackRate = currentGroup.speed;
    video.playbackRate = currentGroup.speed;
    mediaPlayback.refresh(video);
  }

  function selectSample(index) {
    currentSample = index;
    setMedia(videos[0], currentTask.training[index], `${currentTask.title}: ${currentGroup.trainingTitle.toLowerCase()}, sample ${index + 1} of ${currentTask.training.length}`);
    [...sampleButtons.children].forEach((button, i) => button.setAttribute('aria-pressed', String(i === index)));
    find('status').textContent = `Training sample ${index + 1} of ${currentTask.training.length} selected.`;
  }

  function renderTask(id, sampleIndex = 0) {
    videos.forEach(video => video.pause());
    currentTask = tasks.get(id);
    find('goal').textContent = currentTask.goal;
    find('training-note').textContent = currentTask.trainingNote;
    find('training-title').textContent = currentGroup.trainingTitle;
    find('training-stage').textContent = currentGroup.trainingStage;
    find('after-title').textContent = currentGroup.afterTitle;
    find('after-stage').textContent = currentGroup.afterStage;
    ['before', 'after'].forEach(role => { find(`${role}-instruction`).textContent = currentTask.instruction; });
    const rates = currentTask.successRate;
    ['before', 'after'].forEach(role => {
      find(`${role}-rate`).textContent = rates ? `Success: ${rates[role]}%` : 'Success: not reported';
      find(`${role}-progress`).textContent = currentTask.progress
        ? `Progress: ${currentTask.progress[role]}%`
        : 'Progress: not reported';
    });
    find('evaluation-note').textContent = currentTask.evaluationNote || (rates
      ? `Paper evaluation: ${data.evaluationTrials} trials per task. Progress measures average subgoal completion; videos are selected examples.`
      : 'Qualitative example only. Success and progress are not reported for this task in the paper’s transfer figure.');
    find('pair-note').textContent = currentTask.pairNote;
    sampleButtons.replaceChildren();
    currentTask.training.forEach((media, index) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = `Sample ${index + 1}`;
      button.setAttribute('aria-label', `Training sample ${index + 1} of ${currentTask.training.length}`);
      button.addEventListener('click', () => selectSample(index));
      sampleButtons.append(button);
    });
    const rolloutLabel = [currentGroup.rolloutRobot, currentTask.title].filter(Boolean).join(', ');
    setMedia(videos[1], currentTask.before, `${rolloutLabel}: selected atomic-only baseline failure`);
    setMedia(videos[2], currentTask.after, `${rolloutLabel}: selected success, ${currentGroup.afterTitle.toLowerCase()}`);
    selectSample(Math.min(sampleIndex, currentTask.training.length - 1));
    find('status').textContent = `${currentGroup.title}: ${currentTask.title}. ${currentTask.training.length} training samples and a before-and-after comparison.`;
  }

  function selectGroup(id, sampleIndex = 0) {
    const group = data.groups.find(group => group.id === id);
    currentGroup = { ...group, ...group.variants?.[viewSelect.value] };
    tabs.forEach(tab => {
      const active = tab.dataset.transferGroup === id;
      tab.setAttribute('aria-selected', String(active));
      tab.tabIndex = active ? 0 : -1;
    });
    find('panel').setAttribute('aria-labelledby', `transfer-tab-${id}`);
    find('setting-title-text').textContent = currentGroup.title;
    find('setting-title-text').hidden = Boolean(group.variants);
    find('view-heading').hidden = !group.variants;
    find('description').textContent = currentGroup.description;
    find('speed').textContent = `${currentGroup.speed}× playback`;
    find('view-note').textContent = currentGroup.viewNote;
    find('view-note').hidden = !currentGroup.viewNote;
    setEmbodiment('training', currentGroup.trainingRobot);
    setEmbodiment('before', currentGroup.rolloutRobot);
    setEmbodiment('after', currentGroup.rolloutRobot);
    taskSelect.replaceChildren();
    currentGroup.tasks.forEach(id => {
      const option = document.createElement('option');
      option.value = id;
      option.textContent = tasks.get(id).title;
      taskSelect.append(option);
    });
    taskSelect.disabled = currentGroup.tasks.length === 1;
    renderTask(currentGroup.tasks[0], sampleIndex);
  }

  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => selectGroup(tab.dataset.transferGroup));
    tab.addEventListener('keydown', event => {
      let next;
      if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
      if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = tabs.length - 1;
      if (next === undefined) return;
      event.preventDefault();
      tabs[next].focus();
      selectGroup(tabs[next].dataset.transferGroup);
    });
  });
  taskSelect.addEventListener('change', () => renderTask(taskSelect.value));
  viewSelect.addEventListener('change', () => selectGroup('adapt', currentSample));
  videos.forEach(video => {
    mediaPlayback.register(video, 'transfer');
    video.addEventListener('loadedmetadata', () => { video.playbackRate = currentGroup.speed; });
    video.addEventListener('error', () => { find('status').textContent = 'A video could not load. Try another sample or reload the page.'; });
  });
  mediaPlayback.bindToggle(find('motion'), 'transfer');

  root.querySelector('.transfer-tabs').hidden = false;
  root.querySelector('.transfer-toolbar').hidden = false;
  sampleButtons.hidden = false;
  selectGroup(data.groups[0].id);
})();
