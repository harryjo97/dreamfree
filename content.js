// Replace media paths and resource links here as the project is finalized.
// Video-only placeholders intentionally reuse the supplied DROID rollouts.
window.DREAMFREE = {
  paperUrl: null,
  codeUrl: null,
  // 개인사이트 URL을 따옴표 안에 입력하세요. 빈칸이면 일반 이름으로 표시됩니다.
  authorLinks: {
    jaehyeongJo: 'https://harryjo97.github.io/',   // Jaehyeong Jo
    sangwonJang: 'https://agwmon.github.io/',   // Sangwon Jang
    sihyunYu: 'https://sihyun.me/',      // Sihyun Yu
    seonghyeonYe: 'https://seonghyeonye.github.io/',  // Seonghyeon Ye
    yekyungNah: '',    // Yekyung Nah
    hojungJung: 'https://scholar.google.com/citations?user=Cq4lgFgAAAAJ',    // Hojung Jung
    kiminLee: 'https://sites.google.com/view/kiminlee',      // Kimin Lee
    sungJuHwang: 'http://www.sungjuhwang.com/'    // Sung Ju Hwang
  },
  teaser: { src: 'assets/videos/hat.mp4', poster: 'assets/posters/hat.jpg' },
  clips: [
    { id: 'bottle', title: 'Stand the bottle upright' },
    { id: 'glove', title: 'Fold the glove' },
    { id: 'towel', title: 'Cover the strawberry with a towel' },
    { id: 'bolts', title: 'Put the bolts in the bowl' },
    { id: 'hat', title: 'Hang the hat on the tripod' },
    { id: 'cups', title: 'Nest the cups' }
  ],
  datasets: {
    droid: {
      name: 'DROID · Real',
      description: 'Selected real-world rollouts from the supplementary material. Each clip shows a DreamFree policy execution.',
      placeholder: false,
      clips: ['bottle', 'glove', 'towel', 'bolts', 'hat', 'cups']
    },
    robocasa: {
      name: 'RoboCasa',
      description: 'RoboCasa videos will be added here. These preview clips reuse the DROID footage to show the gallery layout.',
      placeholder: true,
      clips: ['bolts', 'cups', 'bottle', 'towel', 'glove', 'hat']
    },
    yam: {
      name: 'YAM · ABC-130K',
      description: 'YAM / ABC-130K videos will be added here. These preview clips reuse the DROID footage to show the gallery layout.',
      placeholder: true,
      clips: ['glove', 'hat', 'cups', 'bottle', 'bolts', 'towel']
    }
  }
};
