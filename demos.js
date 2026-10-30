(() => {
  'use strict';

  const media = window.DreamFreeMedia;
  const clips = {
    bottle: { file: 'droid_bottle', title: 'Stand the bottle upright', label: 'Bottle' },
    pour: { file: 'droid-pour', title: 'Pour the contents of the cup into the pan.', label: 'Pour' },
    bolts: { file: 'droid-bolts', title: 'Put the bolts in the bowl', label: 'Bolts' },
    ring: { file: 'droid-hole', title: 'Put ring around pole', label: 'Ring' },
    hat: { file: 'droid-hat', title: 'Hang the hat on the tripod', label: 'Hat' },
    towel: { file: 'droid-fold-towel', title: 'Fold the towel', label: 'Towel' }
  };
  Object.entries({ red: '#b54235', blue: '#35699a', green: '#48734a', yellow: '#d4a900' }).forEach(([color, hex]) => {
    clips[color] = { file: `droid-block-${color}`, label: color[0].toUpperCase() + color.slice(1), title: `“put the ${color} block in the bowl”`, color: hex, colorName: color };
  });
  const chapters = {
    precision: { number: '01', lines: ['Object', 'manipulation.'], description: 'Stand a bottle upright, pour into a pan, pick up bolts, and place a ring around a pole.', clips: ['bottle', 'pour', 'bolts', 'ring'], note: '' },
    deformable: { number: '02', lines: ['Deformable', 'objects.'], description: 'Hang a hat on a tripod and fold a towel.', clips: ['hat', 'towel'], note: '' },
    language: { number: '03', lines: ['Following', 'language.'], description: '“put the <color> block in the bowl” in the same setting, with different requested colors.', clips: ['red', 'blue', 'green', 'yellow'], note: '' }
  };
  const chapterKeys = Object.keys(chapters);
  const selections = { precision: 'bottle', deformable: 'hat', language: 'red' };
  let chapterKey = 'precision';
  const find = id => document.getElementById(id);
  const asset = (clip, extension) => `assets/demos/${clip.file}.${extension}`;

  function nextChapterKey() {
    return chapterKeys[(chapterKeys.indexOf(chapterKey) + 1) % chapterKeys.length];
  }

  function nextDemo() {
    const chapter = chapters[chapterKey];
    const index = chapter.clips.indexOf(selections[chapterKey]);
    if (index + 1 < chapter.clips.length) return { chapter: chapterKey, clip: chapter.clips[index + 1] };
    const next = nextChapterKey();
    return { chapter: next, clip: chapters[next].clips[0] };
  }

  function chapterLabel(key) {
    return chapters[key].lines.join(' ').replace(/\.$/, '');
  }

  function renderInstruction(caption, clip) {
    if (!clip.colorName) { caption.textContent = clip.title; return; }
    const start = clip.title.indexOf(clip.colorName);
    const colorWord = document.createElement('span');
    colorWord.className = 'demo-instruction-color';
    colorWord.style.setProperty('--instruction-color', clip.color);
    colorWord.textContent = clip.colorName;
    caption.replaceChildren(clip.title.slice(0, start), colorWord, clip.title.slice(start + clip.colorName.length));
  }

  // Sources in the lower section are fetched only as it approaches the viewport.
  const lazyMedia = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      const video = entry.target;
      if (video.dataset.src) { video.src = video.dataset.src; delete video.dataset.src; video.load(); }
      media.register(video, video.closest('#failure-cases') ? 'demo-failures' : 'demos');
      lazyMedia.unobserve(video);
    });
  }, { rootMargin: '300px' });

  function showClip(id, clip) {
    const video = find(id);
    media.unregister(video);
    lazyMedia.unobserve(video);
    video.poster = asset(clip, 'jpg');
    video.setAttribute('aria-label', clip.title);
    video.dataset.src = asset(clip, 'mp4');
    video.removeAttribute('src');
    video.load();
    const caption = find(`${id}-caption`);
    renderInstruction(caption, clip);
    const outcome = find(`${id}-outcome`);
    if (outcome) { outcome.hidden = !clip.failure; outcome.textContent = clip.failure ? 'Unsuccessful attempt' : ''; }
    lazyMedia.observe(video);
  }

  function chooseClip(key) {
    selections[chapterKey] = key;
    const next = nextDemo();
    showClip('demo-primary', clips[key]);
    showClip('demo-secondary', clips[next.clip]);
    const crossesChapter = next.chapter !== chapterKey;
    find('demo-next-clip').setAttribute('aria-label', `Show next ${crossesChapter ? `topic, ${chapterLabel(next.chapter)}` : 'demo'}: ${clips[next.clip].title}`);
    find('demo-next-topic').hidden = !crossesChapter;
    find('demo-next-topic').textContent = crossesChapter ? `Next · ${chapterLabel(next.chapter)}` : '';
    find('demo-choices').querySelectorAll('button').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.clip === key)));
    find('demo-status').textContent = `${chapterLabel(chapterKey)}. Showing ${clips[key].title} Next: ${clips[next.clip].title}`;
  }

  function chooseChapter(key) {
    chapterKey = key;
    const chapter = chapters[key];
    const nextLabel = `Next topic: ${chapterLabel(nextChapterKey())}`;
    find('demo-next-chapter').setAttribute('aria-label', nextLabel);
    find('demo-next-chapter').title = nextLabel;
    find('demo-number').textContent = chapter.number;
    const title = find('demos-title');
    title.replaceChildren(chapter.lines[0], document.createElement('br'), chapter.lines[1]);
    find('demo-description').textContent = chapter.description;
    find('demo-note').textContent = chapter.note;
    find('demo-note').hidden = !chapter.note;
    const choices = find('demo-choices');
    choices.replaceChildren();
    chapter.clips.forEach(id => {
      const clip = clips[id];
      const button = document.createElement('button');
      button.type = 'button'; button.dataset.clip = id;
      if (clip.color) { const dot = document.createElement('span'); dot.className = 'demo-color-dot'; dot.style.setProperty('--block-color', clip.color); dot.setAttribute('aria-hidden', 'true'); button.append(dot); }
      button.append(clip.label);
      if (clip.failure) button.setAttribute('aria-label', `${clip.label} block — unsuccessful attempt`);
      button.addEventListener('click', () => chooseClip(id));
      choices.append(button);
    });
    chooseClip(selections[key]);
  }

  find('demo-next-chapter').addEventListener('click', () => {
    const next = nextChapterKey();
    selections[next] = chapters[next].clips[0];
    chooseChapter(next);
  });
  find('demo-next-clip').addEventListener('click', () => {
    const next = nextDemo();
    if (next.chapter === chapterKey) chooseClip(next.clip);
    else {
      selections[next.chapter] = next.clip;
      chooseChapter(next.chapter);
    }
  });
  document.querySelectorAll('.demo-film video, .compact-abc video, #failure-cases video').forEach(video => {
    if (video.dataset.playbackRate) {
      const rate = Number(video.dataset.playbackRate);
      video.defaultPlaybackRate = rate;
      video.playbackRate = rate;
      video.addEventListener('loadedmetadata', () => { video.playbackRate = rate; });
    }
    video.addEventListener('timeupdate', () => {
      const progress = video.duration ? video.currentTime / video.duration : 0;
      video.parentElement.style.setProperty('--played', progress);
    });
    video.addEventListener('error', () => {
      if (video.getAttribute('src')) find('demo-status').textContent = `Could not load ${video.getAttribute('aria-label')}.`;
    });
  });
  media.bindToggle(find('demo-motion'), 'demos');
  chooseChapter('precision');
  document.querySelectorAll('.compact-abc video[data-src], #failure-cases video[data-src]').forEach(video => lazyMedia.observe(video));
  media.bindToggle(find('failure-motion'), 'demo-failures');
})();
