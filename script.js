"use strict";

const vbvrTasks = [
  ["StableSort", "Stable Sort"], ["PipePuzzle", "Pipe Puzzle"],
  ["ConcentricRing", "Concentric Ring"], ["SlidePuzzle", "Slide Puzzle"],
  ["OutlineMatch", "Outline Match"], ["StarMatch", "Star Match"],
  ["GridShift", "Grid Shift"], ["DominoChain", "Domino Chain"],
  ["BallEater", "Ball Eater"], ["BalancingVessels", "Balancing Vessels"]
];
const pair = (name, ar, proar) => ({ name, ar, proar });
const galleries = [
  {id: "vbvr", title: "VBVR", subtitle: "Perceptual and spatial reasoning · 10 tasks", ratio: "1 / 1",
    pages: Array.from({length: 5}, (_, page) => vbvrTasks.slice(page * 2, page * 2 + 2).map(([file, name]) => pair(name, `${file}1`, `${file}2`)))},
  {id: "videorlvr", title: "VideoRLVR", subtitle: "Abstract reasoning · 3 tasks", ratio: "832 / 480",
    pages: [1, 2].map(sample => [["maze", "Maze"], ["flowfree", "FlowFree"], ["sokoban", "Sokoban"]].map(([file, name]) => pair(name, `${file}${sample}a`, `${file}${sample}b`)))},
  {id: "worldarena", title: "WorldArena", subtitle: "Embodied video simulation · 50 tasks from RoboTwin 2.0", ratio: "4 / 3",
    pages: [[pair("Place Bread Skillet", "embodied_bread1", "embodied_bread2"), pair("Blocks Ranking Size", "embodied_block1", "embodied_block2"), pair("Place Dual Shoes", "embodied_shoes1", "embodied_shoes2")]]}
];
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const controllers = [];

class ComparisonGallery {
  constructor(config, index) {
    this.config = config;
    this.page = 0;
    this.generation = 0;
    this.playing = false;
    this.manuallyPaused = false;
    this.visible = false;
    this.element = document.createElement("section");
    this.element.className = "gallery";
    this.element.dataset.kind = config.id;
    this.element.id = config.id;
    this.element.style.setProperty("--video-ratio", config.ratio);
    this.element.setAttribute("aria-label", `${config.title} comparisons`);
    this.element.innerHTML = `
      <div class="gallery-heading"><div><h3><span class="gallery-index">0${index + 1}</span>${config.title}</h3><p>${config.subtitle}</p></div></div>
      <div class="gallery-body">
        <div class="column-labels"><span>AR</span><span>ProAR (ours)</span></div>
        <div class="gallery-rows"></div>
        <div class="gallery-toolbar"><div class="playback-controls"><button class="button play-toggle" type="button">Play all</button><button class="button replay" type="button">↺ Replay</button></div>
        ${config.pages.length > 1 ? '<div class="pagination"><button class="button previous" type="button" aria-label="Previous page">‹</button><span class="page-label" aria-live="polite"></span><button class="button next" type="button" aria-label="Next page">›</button></div>' : ''}</div>
        <p class="gallery-status" role="status"></p>
      </div>`;
    document.querySelector("#galleries").append(this.element);
    this.toggle = this.element.querySelector(".play-toggle");
    this.toggle.addEventListener("click", () => {
      if (this.playing) { this.manuallyPaused = true; this.pause(); }
      else { this.manuallyPaused = false; this.play(); }
    });
    this.element.querySelector(".replay").addEventListener("click", () => { this.manuallyPaused = false; this.play(true); });
    this.element.querySelector(".previous")?.addEventListener("click", () => this.changePage(-1));
    this.element.querySelector(".next")?.addEventListener("click", () => this.changePage(1));
    this.render();
  }

  render() {
    this.pause();
    this.element.querySelector(".gallery-status").textContent = "";
    const rows = this.element.querySelector(".gallery-rows");
    rows.replaceChildren();
    for (const task of this.config.pages[this.page]) {
      const row = document.createElement("div");
      row.className = "comparison-row";
      const title = document.createElement("p");
      title.className = "task-name";
      title.textContent = task.name;
      row.append(title);
      const pairElement = document.createElement("div");
      pairElement.className = "video-pair";
      for (const [method, file] of [["AR", task.ar], ["ProAR", task.proar]]) {
        const video = document.createElement("video");
        video.src = `videos/${file}.mp4`;
        video.poster = `images/posters/${file}.jpg`;
        video.muted = true;
        video.playsInline = true;
        video.preload = "none";
        video.setAttribute("aria-label", `${task.name}, ${method}, sample ${this.page + 1}`);
        video.addEventListener("ended", () => {
          if (this.playing && this.videos.every(v => v.ended)) {
            this.playing = false;
            this.toggle.textContent = "Play all";
            this.loopTimer = setTimeout(() => {
              if (this.visible && !this.manuallyPaused && !reducedMotion.matches) this.play(true);
            }, 1800);
          }
        });
        pairElement.append(video);
      }
      row.append(pairElement);
      rows.append(row);
    }
    this.videos = [...rows.querySelectorAll("video")];
    if (this.config.pages.length > 1) {
      this.element.querySelector(".page-label").textContent = `${this.page + 1} / ${this.config.pages.length}`;
      this.element.querySelector(".previous").disabled = this.page === 0;
      this.element.querySelector(".next").disabled = this.page === this.config.pages.length - 1;
    }
  }

  changePage(delta) {
    const next = this.page + delta;
    if (next < 0 || next >= this.config.pages.length) return;
    this.page = next;
    this.render();
    if (!this.manuallyPaused && !reducedMotion.matches) this.play(true);
  }

  pause() {
    this.generation++;
    clearTimeout(this.loopTimer);
    clearInterval(this.syncTimer);
    this.videos?.forEach(video => video.pause());
    this.playing = false;
    this.toggle.textContent = "Play all";
  }

  async play(restart = false) {
    this.pause();
    const token = this.generation;
    const videos = [...this.videos];
    const status = this.element.querySelector(".gallery-status");
    this.playing = true;
    this.toggle.textContent = "Pause all";
    status.textContent = "Loading comparison…";
    try {
      await Promise.all(videos.map(video => new Promise((resolve, reject) => {
        if (video.readyState >= 3) return resolve();
        const done = () => { cleanup(); resolve(); };
        const failed = () => { cleanup(); reject(new Error("Video unavailable")); };
        const timeout = setTimeout(failed, 30000);
        const cleanup = () => { clearTimeout(timeout); video.removeEventListener("canplay", done); video.removeEventListener("error", failed); };
        video.addEventListener("canplay", done);
        video.addEventListener("error", failed);
        video.preload = "auto";
        video.load();
      })));
      if (token !== this.generation) return;
      if (restart || videos.every(video => video.ended)) videos.forEach(video => { video.currentTime = 0; });
      await Promise.all(videos.filter(video => !video.ended).map(video => video.play()));
      if (token !== this.generation) return;
      status.textContent = "";
      // Use the longest clip as the clock; shorter clips hold at their end.
      const leader = videos.reduce((a, b) => a.duration >= b.duration ? a : b);
      this.syncTimer = setInterval(() => {
        if (!this.playing) return;
        for (const video of videos) {
          if (video === leader) continue;
          const target = Math.min(leader.currentTime, video.duration);
          if (Math.abs(video.currentTime - target) > .16) video.currentTime = target;
        }
      }, 250);
    } catch {
      if (token !== this.generation) return;
      this.pause();
      status.textContent = "Playback did not start. Select Play all to try again.";
    }
  }
}

galleries.forEach((config, index) => controllers.push(new ComparisonGallery(config, index)));
const observer = new IntersectionObserver(entries => {
  for (const entry of entries) {
    const gallery = controllers.find(item => item.element === entry.target);
    gallery.visible = entry.isIntersecting;
    if (!gallery.visible) gallery.pause();
    else if (!gallery.manuallyPaused && !reducedMotion.matches && !document.hidden) gallery.play();
  }
}, {threshold: 0.12});
controllers.forEach(gallery => observer.observe(gallery.element));

const gt = document.querySelector(".gt-video");
const gtObserver = new IntersectionObserver(entries => {
  for (const entry of entries) {
    if (entry.isIntersecting && !reducedMotion.matches && !document.hidden) gt.play().catch(() => {});
    else gt.pause();
  }
}, {threshold: .4});
gtObserver.observe(gt);
document.addEventListener("visibilitychange", () => {
  if (document.hidden) { controllers.forEach(gallery => gallery.pause()); gt.pause(); }
  else controllers.filter(gallery => gallery.visible && !gallery.manuallyPaused && !reducedMotion.matches).forEach(gallery => gallery.play());
});

document.querySelector("#copy-citation").addEventListener("click", async event => {
  const button = event.currentTarget;
  try {
    await navigator.clipboard.writeText(document.querySelector("#bibtex").textContent);
    button.textContent = "Copied ✓";
    document.querySelector("#copy-status").textContent = "Citation copied.";
    setTimeout(() => { button.textContent = "Copy BibTeX"; }, 2000);
  } catch {
    button.textContent = "Select text to copy";
    const range = document.createRange();
    range.selectNodeContents(document.querySelector("#bibtex"));
    window.getSelection().removeAllRanges();
    window.getSelection().addRange(range);
  }
});
