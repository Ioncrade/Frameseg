import RotoscopeStudio from "@/components/rotoscope-studio";

export default function Home() {
  return (
    <main className="app-frame">
      <header className="topbar">
        <a className="brand" href="#" aria-label="FrameSeg home">
          <span className="brand-mark" aria-hidden>
            <i />
            <i />
          </span>
          <span>FrameSeg</span>
        </a>
        <div className="topbar-meta">
          <span className="model-status"><i /> SAM2.1 ready</span>
          <span className="version-tag">Studio alpha</span>
        </div>
      </header>

      <section className="intro">
        <div>
          <p className="eyebrow">AI-assisted rotoscoping</p>
          <h1>Isolate a subject.<br />Keep every edge.</h1>
        </div>
        <p className="intro-copy">
          Choose an object once, fine-tune its mask, and track it through up to
          1,200 frames. Export a production-ready video with transparency.
        </p>
      </section>

      <RotoscopeStudio />

      <footer className="site-footer">
        <span>FrameSeg processes footage on your connected inference server.</span>
        <span>H.264 / H.265 · MP4 / MOV · Alpha MOV output</span>
      </footer>
    </main>
  );
}
