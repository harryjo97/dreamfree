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

// Selected qualitative transfer examples; source metadata stays outside the public site.
// Success and progress: the paper's task_transfer_and_video_finetuning_20ep figure (20 trials per task).
// Coffee uses updated author-provided results: base 0% / 0%, composite videos 15% / 30% (success / progress).
window.DREAMFREE.transfer = {
  "groups": [
    {
      "id": "same",
      "title": "Composite videos in pre-training",
      "description": "We add composite-task demonstrations to visual encoder pre-training, alongside atomic-task videos. Both models use the same atomic-task action labels to train the action module; no composite-task action labels are used.",
      "afterTitle": "With composite videos",
      "speed": 2,
      "tasks": [
        "microwave",
        "vegetables",
        "pantry",
        "coffee"
      ],
      "trainingRobot": "",
      "rolloutRobot": "",
      "viewNote": "",
      "trainingTitle": "Composite-task videos",
      "trainingStage": "For encoder pre-training",
      "afterStage": "Added during pre-training"
    },
    {
      "id": "cross",
      "title": "Cross-embodiment videos in pre-training",
      "description": "We add multi-view composite-task videos from dual-arm Franka to the ALOHA-AgileX encoder’s pre-training data. Both models use the same ALOHA atomic-task action labels to train the action module; no composite-task action labels are used.",
      "afterTitle": "With composite videos",
      "speed": 1,
      "tasks": [
        "franka-basket",
        "franka-cans",
        "franka-bowls"
      ],
      "trainingRobot": "Dual-arm Franka",
      "rolloutRobot": "ALOHA-AgileX",
      "viewNote": "",
      "trainingTitle": "Composite-task videos",
      "trainingStage": "For encoder pre-training",
      "afterStage": "Added during pre-training"
    },
    {
      "id": "adapt",
      "title": "Single-view video-only fine-tuning",
      "description": "Starting from a policy trained on atomic data, we fine-tune only its visual encoder using 10 head-camera demonstrations of a composite task from dual-arm Franka. The action module stays fixed, and no action labels are used for fine-tuning.",
      "afterTitle": "Fine-tuned with videos",
      "speed": 1,
      "tasks": [
        "video-adaptation"
      ],
      "trainingRobot": "Dual-arm Franka",
      "rolloutRobot": "ALOHA-AgileX",
      "viewNote": "Fine-tuning uses only the head-camera view; gray panels mark unused wrist views. Both camera settings use the same 10 demonstrations and are evaluated with all three camera views.",
      "trainingTitle": "Single-view task videos",
      "trainingStage": "For encoder fine-tuning",
      "afterStage": "Single-view · Action module fixed",
      "variants": {
        "single": {},
        "multi": {
          "title": "Multi-view video-only fine-tuning",
          "description": "Starting from the same policy trained on atomic data, we fine-tune only its visual encoder using the head and both wrist views of the same 10 dual-arm Franka demonstrations. The action module stays fixed, and no action labels are used for fine-tuning.",
          "tasks": [
            "video-finetuning-multi"
          ],
          "trainingTitle": "Multi-view task videos",
          "afterStage": "Multi-view · Action module fixed",
          "viewNote": "Fine-tuning uses the head and both wrist cameras from the same 10 demonstrations. Evaluation also uses all three camera views."
        }
      }
    }
  ],
  "tasks": [
    {
      "id": "microwave",
      "title": "Microwave thawing",
      "goal": "Open the microwave → put food inside → close the door → turn it on.",
      "training": [
        {
          "src": "assets/transfer/microwave/training.mp4"
        },
        {
          "src": "assets/transfer/microwave/training-2.mp4"
        },
        {
          "src": "assets/transfer/microwave/training-3.mp4"
        }
      ],
      "before": {
        "src": "assets/transfer/microwave/before.mp4"
      },
      "after": {
        "src": "assets/transfer/microwave/after.mp4"
      },
      "trainingNote": "Composite-task demonstration video",
      "pairNote": "Before and after use the same evaluation seed and trial.",
      "instruction": "Pick the cheese from the counter and place it in the microwave. Then turn on the microwave.",
      "successRate": {
        "before": 0,
        "after": 5
      },
      "progress": {
        "before": 8.8,
        "after": 26.3
      }
    },
    {
      "id": "vegetables",
      "title": "Arrange vegetables",
      "goal": "Move both vegetables from the sink to the cutting board.",
      "training": [
        {
          "src": "assets/transfer/vegetables/training.mp4"
        },
        {
          "src": "assets/transfer/vegetables/training-2.mp4"
        },
        {
          "src": "assets/transfer/vegetables/training-3.mp4"
        }
      ],
      "before": {
        "src": "assets/transfer/vegetables/before.mp4"
      },
      "after": {
        "src": "assets/transfer/vegetables/after.mp4"
      },
      "trainingNote": "Composite-task demonstration video",
      "pairNote": "Before and after use the same evaluation seed and trial.",
      "instruction": "Pick the vegetables from the sink and place them on the cutting board.",
      "successRate": {
        "before": 0,
        "after": 40
      },
      "progress": {
        "before": 12.5,
        "after": 57.5
      }
    },
    {
      "id": "pantry",
      "title": "Restock the pantry",
      "goal": "Place the cans on their designated sides of the cabinet.",
      "training": [
        {
          "src": "assets/transfer/pantry/training.mp4"
        },
        {
          "src": "assets/transfer/pantry/training-2.mp4"
        },
        {
          "src": "assets/transfer/pantry/training-3.mp4"
        }
      ],
      "before": {
        "src": "assets/transfer/pantry/before.mp4"
      },
      "after": {
        "src": "assets/transfer/pantry/after.mp4"
      },
      "trainingNote": "Composite-task demonstration video",
      "pairNote": "Before and after use the same evaluation seed and trial.",
      "instruction": "Pick the cans from the counter and place them in their designated side in the cabinet.",
      "successRate": {
        "before": 0,
        "after": 20
      },
      "progress": {
        "before": 2.5,
        "after": 40
      }
    },
    {
      "id": "coffee",
      "title": "Prepare coffee",
      "goal": "Take a mug from the cabinet → place it under the dispenser → start the coffee machine.",
      "training": [
        {
          "src": "assets/transfer/coffee/training.mp4"
        },
        {
          "src": "assets/transfer/coffee/training-2.mp4"
        },
        {
          "src": "assets/transfer/coffee/training-3.mp4"
        }
      ],
      "before": {
        "src": "assets/transfer/coffee/before.mp4"
      },
      "after": {
        "src": "assets/transfer/coffee/after.mp4"
      },
      "trainingNote": "Composite-task demonstration video",
      "pairNote": "Before and after use the same evaluation seed and trial.",
      "instruction": "Pick the mug from the cabinet, place it under the coffee machine dispenser, and press the start button.",
      "successRate": {
        "before": 0,
        "after": 15
      },
      "progress": {
        "before": 0,
        "after": 30
      },
      "evaluationNote": "Success and progress summarize evaluation trials; progress gives partial credit for completed subgoals. Videos are selected examples."
    },
    {
      "id": "franka-basket",
      "title": "Place a can and lift the basket",
      "goal": "Place the can in the basket → lift the basket.",
      "training": [
        {
          "src": "assets/transfer/franka-basket/training-multi-view-001100.mp4"
        },
        {
          "src": "assets/transfer/franka-basket/training-multi-view-001125.mp4"
        }
      ],
      "before": {
        "src": "assets/transfer/franka-basket/before.mp4"
      },
      "after": {
        "src": "assets/transfer/franka-basket/after.mp4"
      },
      "trainingNote": "Composite-task demonstration from another embodiment",
      "pairNote": "Before and after use the same environment seed and task instruction.",
      "instruction": "Place the can in the basket, then lift the basket.",
      "successRate": {
        "before": 0,
        "after": 5
      },
      "progress": {
        "before": 0,
        "after": 15
      }
    },
    {
      "id": "franka-cans",
      "title": "Place both cans in a box",
      "goal": "Pick up both cans and place them in the plastic box.",
      "training": [
        {
          "src": "assets/transfer/franka-cans/training-multi-view-001150.mp4"
        },
        {
          "src": "assets/transfer/franka-cans/training-multi-view-001175.mp4"
        }
      ],
      "before": {
        "src": "assets/transfer/franka-cans/before.mp4"
      },
      "after": {
        "src": "assets/transfer/franka-cans/after.mp4"
      },
      "trainingNote": "Composite-task demonstration from another embodiment",
      "pairNote": "Before and after use the same environment seed and task instruction.",
      "instruction": "Place both cans in the plastic box.",
      "successRate": {
        "before": 0,
        "after": 25
      },
      "progress": {
        "before": 20,
        "after": 55
      }
    },
    {
      "id": "franka-bowls",
      "title": "Stack three bowls",
      "goal": "Stack the three bowls in sequence.",
      "training": [
        {
          "src": "assets/transfer/franka-bowls/training-multi-view-002100.mp4"
        },
        {
          "src": "assets/transfer/franka-bowls/training-multi-view-002125.mp4"
        }
      ],
      "before": {
        "src": "assets/transfer/franka-bowls/before.mp4"
      },
      "after": {
        "src": "assets/transfer/franka-bowls/after.mp4"
      },
      "trainingNote": "Composite-task demonstration from another embodiment",
      "pairNote": "Before and after use the same environment seed and task instruction.",
      "instruction": "Stack the three bowls.",
      "successRate": {
        "before": 20,
        "after": 60
      },
      "progress": {
        "before": 46.7,
        "after": 78.3
      }
    },
    {
      "id": "video-adaptation",
      "title": "Place a can and lift the basket",
      "goal": "Place the can in the basket → lift the basket.",
      "training": [
        {
          "src": "assets/transfer/video-adaptation/training-single-view-001104.mp4"
        },
        {
          "src": "assets/transfer/video-adaptation/training-single-view-001109.mp4"
        },
        {
          "src": "assets/transfer/video-adaptation/training-single-view-001111.mp4"
        }
      ],
      "before": {
        "src": "assets/transfer/video-adaptation/before.mp4"
      },
      "after": {
        "src": "assets/transfer/video-adaptation/after.mp4"
      },
      "trainingNote": "Single-view composite-task demonstration for fine-tuning",
      "pairNote": "Before and after use the same environment seed and task instruction.",
      "instruction": "Place the can in the basket, then lift the basket.",
      "successRate": {
        "before": 0,
        "after": 5
      },
      "progress": {
        "before": 0,
        "after": 15
      }
    },
    {
      "id": "video-finetuning-multi",
      "title": "Place a can and lift the basket",
      "goal": "Place the can in the basket → lift the basket.",
      "training": [
        {
          "src": "assets/transfer/video-finetuning-multi/training-multi-view-001104.mp4"
        },
        {
          "src": "assets/transfer/video-finetuning-multi/training-multi-view-001109.mp4"
        },
        {
          "src": "assets/transfer/video-finetuning-multi/training-multi-view-001111.mp4"
        }
      ],
      "before": {
        "src": "assets/transfer/video-finetuning-multi/before.mp4"
      },
      "after": {
        "src": "assets/transfer/video-finetuning-multi/after.mp4"
      },
      "trainingNote": "Multi-view composite-task demonstration for fine-tuning",
      "pairNote": "Selected clips share the same seed and instruction. The multi-view result uses native inference; the baseline and single-view clips use FP8. These clips are qualitative examples, not a controlled comparison of camera settings.",
      "instruction": "Place the can in the basket, then lift the basket.",
      "successRate": {
        "before": 0,
        "after": 10
      },
      "progress": {
        "before": 0,
        "after": 22.5
      }
    }
  ],
  "evaluationTrials": 20
};
