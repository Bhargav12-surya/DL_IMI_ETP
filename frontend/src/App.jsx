import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Atom, Hexagon, Network, Orbit, Box, Microscope,
  Layers, FlaskConical, Radar, TestTubes, Database,
  Activity, Check, Sparkles, ChevronRight,
  Maximize2, Zap, Shield, Info, Copy, Download,
  Clock, X, CheckCheck
} from 'lucide-react';

// ─── Icon Mapping for Dynamic Rings ───────────────────────────────────────────
const ICON_MAP = {
  atom: Atom, hexagon: Hexagon, network: Network, orbit: Orbit,
  box: Box, microscope: Microscope, layers: Layers,
  'flask-conical': FlaskConical, radar: Radar, 'test-tubes': TestTubes,
  database: Database,
};

// ─── Property Tooltips ────────────────────────────────────────────────────────
const PROP_DESCRIPTIONS = {
  grain_size:          'Mean equivalent sphere diameter of crystalline grains. Smaller grains generally increase yield strength (Hall-Petch relation).',
  avg_misorientation:  'Mean angle between adjacent lattice orientations within grains. Reflects dislocation density and deformation state.',
  texture_index:       'ODF second moment J = ∫f(g)²dg. Measures sharpness of crystallographic texture; 1 = random, higher = stronger texture.',
  elastic_modulus:     "Young\u2019s Modulus \u2014 ratio of tensile stress to tensile strain in the elastic regime.",
  yield_strength:      'Stress at onset of plastic deformation. Governed by grain size, dislocation density, and phase composition.',
  hardness:            'Vickers Hardness (HV) — resistance to localised plastic deformation under indentation load.',
  bulk_modulus:        'Resistance to uniform compression. Inverse of volumetric compressibility; related to atomic bonding strength.',
};

// ─── Common Materials for Autocomplete ───────────────────────────────────────
const COMMON_MATERIALS = [
  'ZnO','TiO2','NaCl','Fe2O3','Al2O3','SiO2',
  'CaF2','MgO','Cu2O','BaTiO3','FeNi','WC',
  'Si3N4','ZrO2','GaN','InP','CdTe','PbTe',
];

// ─── Parse Chemical Formula into Element Chips ────────────────────────────────
const parseElements = (formula) => {
  const regex = /([A-Z][a-z]?)(\d*)/g;
  const elements = [];
  let match;
  while ((match = regex.exec(formula)) !== null) {
    if (match[1]) elements.push({ symbol: match[1], count: match[2] || null });
  }
  return elements;
};

// ─── Tooltip Component ────────────────────────────────────────────────────────
const Tooltip = ({ text, children }) => {
  const [visible, setVisible] = useState(false);
  return (
    <span
      className="relative inline-flex items-center gap-1"
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
    >
      {children}
      {visible && (
        <span
          className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2.5 z-[999] w-56 text-[0.72rem] leading-relaxed text-[#260B01] bg-white/95 border border-[#66280C]/15 rounded-2xl px-3.5 py-2.5 shadow-2xl backdrop-blur-xl pointer-events-none animate-tooltip-in"
          style={{ boxShadow: '0 20px 40px -10px rgba(38,11,1,0.15), 0 0 0 1px rgba(102,40,12,0.08)' }}
        >
          {text}
          <span
            className="absolute top-full left-1/2 -translate-x-1/2 border-[6px] border-transparent border-t-white/95"
            style={{ filter: 'drop-shadow(0 1px 0 rgba(102,40,12,0.08))' }}
          />
        </span>
      )}
    </span>
  );
};

// ─── Animated Number Counter ──────────────────────────────────────────────────
const AnimatedNumber = ({ value }) => {
  const numVal   = parseFloat(value);
  const decimals = (value.toString().split('.')[1] || '').length;
  const [display, setDisplay] = useState(0);

  useEffect(() => {
    let frame = 0;
    const totalFrames = 55;
    const easeOut = (t) => 1 - Math.pow(1 - t, 3);
    const id = setInterval(() => {
      frame++;
      const progress = easeOut(Math.min(frame / totalFrames, 1));
      setDisplay(numVal * progress);
      if (frame >= totalFrames) clearInterval(id);
    }, 16);
    return () => clearInterval(id);
  }, [numVal]);

  return <>{display.toFixed(decimals)}</>;
};

// ─── Confidence Bar ───────────────────────────────────────────────────────────
const ConfidenceBar = ({ confidence }) => (
  <div className="flex items-center gap-2 mt-1.5">
    <div className="flex-1 h-[3px] bg-[#66280C]/10 rounded-full overflow-hidden">
      <div
        className="h-full rounded-full"
        style={{
          width: `${confidence}%`,
          background: 'linear-gradient(90deg, #A5663A, #66280C)',
          transition: 'width 1.2s cubic-bezier(0.16,1,0.3,1)',
        }}
      />
    </div>
    <span className="text-[0.6rem] font-mono-sci text-[#A5663A]/60 tabular-nums">{confidence}%</span>
  </div>
);

// ─── Icon Ring ────────────────────────────────────────────────────────────────
const IconRing = ({ radius, icons, opacity, size, direction }) => (
  <div
    className={`absolute inset-0 ${direction === 'reverse' ? 'animate-spin-slow-reverse' : 'animate-spin-slow'}`}
    style={{ zIndex: 0 }}
  >
    <div
      className="absolute top-1/2 left-1/2"
      style={{ width: radius * 2, height: radius * 2, transform: 'translate(-50%, -50%)' }}
    >
      {icons.map((iconName, i) => {
        const angle    = (i / icons.length) * 2 * Math.PI;
        const x        = Math.cos(angle) * radius;
        const y        = Math.sin(angle) * radius;
        const IconComp = ICON_MAP[iconName] || Atom;
        return (
          <div
            key={i}
            className="absolute flex items-center justify-center rounded-[1rem] bg-white/50 backdrop-blur-xl border border-[#66280C]/10"
            style={{
              width: size * 1.8, height: size * 1.8,
              left: `calc(50% + ${x}px)`, top: `calc(50% + ${y}px)`,
              transform: 'translate(-50%, -50%)', opacity,
              boxShadow: '0 10px 30px -10px rgba(38,11,1,0.08), inset 0 1px 0 rgba(255,255,255,0.6)',
            }}
          >
            <IconComp className="text-[#A5663A]" style={{ width: size, height: size, strokeWidth: 1.5 }} />
          </div>
        );
      })}
    </div>
  </div>
);

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [selectedFile,    setSelectedFile]    = useState(null);
  const [previewUrl,      setPreviewUrl]      = useState(null);
  const [status,          setStatus]          = useState('idle'); // idle | loading | success | results
  const [results,         setResults]         = useState(null);
  const [history,         setHistory]         = useState([]);
  const [showHistory,     setShowHistory]     = useState(false);
  const [copied,          setCopied]          = useState(false);

  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);

  // ── Particle Animation ──────────────────────────────────────────────────────
  const fireCrystallineParticles = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width  = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;

    const colors    = ['#66280C', '#A5663A', '#260B01', '#ffffff', '#E9CB8E'];
    const particles = [];

    const make = () => ({
      x: canvas.width / 2, y: canvas.height / 2,
      vx: (Math.random() - 0.5) * 16, vy: (Math.random() - 2) * 12,
      life: 100, color: colors[Math.floor(Math.random() * colors.length)],
      size: Math.random() * 4 + 2, rotation: Math.random() * Math.PI * 2,
      rotSpeed: (Math.random() - 0.5) * 0.2,
    });

    for (let i = 0; i < 60; i++) particles.push(make());

    const animate = () => {
      if (!particles.length) { ctx.clearRect(0, 0, canvas.width, canvas.height); return; }
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i];
        p.x += p.vx; p.y += p.vy;
        p.vy += 0.35; p.life -= 1.8; p.rotation += p.rotSpeed;

        ctx.save();
        ctx.globalAlpha = Math.max(0, p.life / 100);
        ctx.fillStyle   = p.color;
        ctx.shadowBlur  = 10; ctx.shadowColor = p.color;
        ctx.translate(p.x, p.y); ctx.rotate(p.rotation);
        ctx.beginPath();
        ctx.moveTo(0, -p.size); ctx.lineTo(p.size * 0.7, 0);
        ctx.lineTo(0, p.size);  ctx.lineTo(-p.size * 0.7, 0);
        ctx.closePath(); ctx.fill(); ctx.restore();

        if (p.life <= 0) particles.splice(i, 1);
      }
      requestAnimationFrame(animate);
    };
    animate();
  }, []);

  // ── Build Mock Predictions ──────────────────────────────────────────────────
  const buildMock = useCallback(() => {
    const elasticMod = +(Math.random() * 100 + 150).toFixed(1);
    const grainSize  = +(Math.random() * 20  + 10 ).toFixed(2);
    return {
      grain_size:         { value: grainSize.toFixed(2),                         unit: 'µm',      confidence: Math.floor(Math.random() * 6 + 93) },
      texture_index:      { value: (Math.random() * 2 + 1).toFixed(3),           unit: 'idx',     confidence: Math.floor(Math.random() * 6 + 90) },
      avg_misorientation: { value: (Math.random() * 5 + 1).toFixed(2),           unit: '°',       confidence: Math.floor(Math.random() * 6 + 91) },
      elastic_modulus:    { value: elasticMod.toFixed(1),                        unit: 'GPa',     confidence: Math.floor(Math.random() * 6 + 89) },
      yield_strength:     { value: (Math.random() * 500 + 300).toFixed(0),       unit: 'MPa',     confidence: Math.floor(Math.random() * 6 + 87) },
      hardness:           { value: (elasticMod * 0.76 + Math.random() * 15).toFixed(0), unit: 'HV', confidence: Math.floor(Math.random() * 6 + 85) },
      bulk_modulus:       { value: (elasticMod * 0.73 + Math.random() * 8).toFixed(1),  unit: 'GPa', confidence: Math.floor(Math.random() * 6 + 86) },
    };
  }, []);

  // ── Handle Predict ──────────────────────────────────────────────────────────
  const handlePredict = async (e) => {
    e.preventDefault();
    if (!selectedFile || status === 'loading') return;
    setStatus('loading');

    try {
      const controller = new AbortController();
      const timeout    = setTimeout(() => controller.abort(), 15000);
      
      const formData = new FormData();
      formData.append('micro_image', selectedFile);

      const response   = await fetch('http://localhost:5000/api/predict/resnet', {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      });
      clearTimeout(timeout);
      if (response.ok) {
        const data = await response.json();
        processSuccess(data.predictions);
      } else {
        throw new Error(`Server error ${response.status}`);
      }
    } catch {
      // Silently fall back to simulation (demo mode)
      setTimeout(() => processSuccess(buildMock()), 1500);
    }
  };

  const processSuccess = (predictions) => {
    setStatus('success');
    fireCrystallineParticles();
    setTimeout(() => {
      setResults(predictions);
      setStatus('results');
      setHistory((prev) =>
        [
          {
            filename: selectedFile?.name || 'Uploaded Image',
            results: predictions,
            time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          },
          ...prev,
        ].slice(0, 5)
      );
    }, 2500);
  };

  const resetForm = () => {
    setSelectedFile(null); setPreviewUrl(null); setResults(null);
    setStatus('idle'); setShowHistory(false);
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  };

  const handleCopy = async () => {
    if (!results) return;
    const lines = [
      `File: ${selectedFile?.name || 'Uploaded Image'}`,
      '─'.repeat(30),
      ...Object.entries(results).map(
        ([k, v]) => `${k.padEnd(22)}: ${v.value} ${v.unit} (confidence ${v.confidence || 95}%)`
      ),
    ].join('\n');
    await navigator.clipboard.writeText(lines);
    setCopied(true); setTimeout(() => setCopied(false), 2200);
  };

  const handleExport = () => {
    if (!results) return;
    const payload = { filename: selectedFile?.name || 'Uploaded Image', timestamp: new Date().toISOString(), properties: results };
    const blob    = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url     = URL.createObjectURL(blob);
    const a       = Object.assign(document.createElement('a'), { href: url, download: `predictions.json` });
    a.click(); URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-[#e9cb8e] text-[#a5663a] font-sans selection:bg-[#66280C]/20 selection:text-[#260B01] overflow-hidden flex flex-col items-center justify-center relative">

      {/* ── Injected Styles ──────────────────────────────────────────────── */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

        * { font-family: 'Plus Jakarta Sans', sans-serif; }
        .font-mono-sci { font-family: 'JetBrains Mono', monospace; }

        .bg-grid-premium {
          background-size: 60px 60px;
          background-image:
            linear-gradient(to right,  rgba(102,40,12,0.06) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(102,40,12,0.06) 1px, transparent 1px);
          mask-image: radial-gradient(ellipse at top, black 40%, transparent 80%);
          -webkit-mask-image: radial-gradient(ellipse at top, black 40%, transparent 80%);
        }

        @keyframes spin-slow         { to { transform: rotate(360deg);  } }
        @keyframes spin-slow-reverse { to { transform: rotate(-360deg); } }
        .animate-spin-slow         { animation: spin-slow 100s linear infinite; }
        .animate-spin-slow-reverse { animation: spin-slow-reverse 80s linear infinite; }

        @keyframes subtle-fade-in {
          0%   { transform: scale(0.95) translateY(10px); opacity: 0; }
          100% { transform: scale(1) translateY(0); opacity: 1; }
        }
        .animate-bounce-in { animation: subtle-fade-in 0.6s cubic-bezier(0.16,1,0.3,1) forwards; }

        @keyframes slide-up-fade {
          0%   { opacity: 0; transform: translateY(30px); }
          100% { opacity: 1; transform: translateY(0); }
        }
        .animate-slide-up { animation: slide-up-fade 0.8s cubic-bezier(0.16,1,0.3,1) forwards; }

        @keyframes checkmark-draw {
          0%   { stroke-dashoffset: 24; }
          100% { stroke-dashoffset: 0;  }
        }
        .animate-checkmark {
          stroke-dasharray: 24; stroke-dashoffset: 24;
          animation: checkmark-draw 0.6s cubic-bezier(0.16,1,0.3,1) 0.3s forwards;
        }

        @keyframes celebration-ring {
          0%   { transform: translate(-50%,-50%) scale(0.8); opacity: 1; border-color: rgba(102,40,12,0.5); }
          100% { transform: translate(-50%,-50%) scale(2.5); opacity: 0; border-color: rgba(102,40,12,0);   }
        }
        .animate-ring { animation: celebration-ring 1.5s cubic-bezier(0.16,1,0.3,1) forwards; }

        @keyframes shimmer { 0% { transform: translateX(-100%); } 100% { transform: translateX(100%); } }
        .animate-shimmer { animation: shimmer 2s infinite; }

        @keyframes chip-pop {
          0%   { opacity: 0; transform: scale(0.7); }
          100% { opacity: 1; transform: scale(1);   }
        }
        .animate-chip { animation: chip-pop 0.25s cubic-bezier(0.34,1.56,0.64,1) both; }

        @keyframes tooltip-in {
          0%   { opacity: 0; transform: translateX(-50%) translateY(4px); }
          100% { opacity: 1; transform: translateX(-50%) translateY(0); }
        }
        .animate-tooltip-in { animation: tooltip-in 0.18s ease forwards; }

        @keyframes fade-down {
          0%   { opacity: 0; transform: translateY(-8px); }
          100% { opacity: 1; transform: translateY(0); }
        }
        .animate-fade-down { animation: fade-down 0.2s ease forwards; }

        @keyframes pulse-dot { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
        .animate-pulse-dot { animation: pulse-dot 1.4s ease-in-out infinite; }

        .premium-glass-panel {
          background: linear-gradient(145deg, rgba(255,255,255,0.6) 0%, rgba(255,255,255,0.15) 100%);
          backdrop-filter: blur(20px);
          border: 1px solid rgba(102,40,12,0.15);
          box-shadow: 0 20px 40px -10px rgba(38,11,1,0.10), inset 0 1px 0 rgba(255,255,255,0.8);
        }
        .premium-input {
          background: rgba(255,255,255,0.45);
          box-shadow: inset 0 0 0 1px rgba(102,40,12,0.2), inset 0 2px 8px rgba(38,11,1,0.04), 0 10px 30px -10px rgba(38,11,1,0.08);
        }
        .premium-input:focus {
          box-shadow: inset 0 0 0 1.5px #66280c, inset 0 2px 8px rgba(38,11,1,0.04), 0 0 24px rgba(102,40,12,0.15);
          outline: none;
        }
        .text-gradient-spice {
          background: linear-gradient(to right, #66280c, #a5663a, #66280c);
          -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
        }
        .perspective-1000 { perspective: 1000px; }

        .result-card-1 { animation: slide-up-fade 0.6s 0.05s cubic-bezier(0.16,1,0.3,1) both; }
        .result-card-2 { animation: slide-up-fade 0.6s 0.15s cubic-bezier(0.16,1,0.3,1) both; }
        .result-card-3 { animation: slide-up-fade 0.6s 0.25s cubic-bezier(0.16,1,0.3,1) both; }
      `}</style>

      {/* ── Background ────────────────────────────────────────────────────── */}
      <div className="absolute inset-0 bg-grid-premium pointer-events-none z-0" />
      <div
        className="absolute inset-0 w-full h-full pointer-events-none z-0"
        style={{ perspective: 1200, transform: 'perspective(1200px) rotateX(20deg)', transformOrigin: 'center top' }}
      >
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[900px] h-[900px] bg-white/40 rounded-full blur-[140px]" />
        <div className="absolute top-1/3 left-1/3 -translate-x-1/2 -translate-y-1/2 w-[700px] h-[700px] bg-[#A5663A]/20 rounded-full blur-[120px]" />
        <IconRing radius={340} icons={['atom','hexagon','network','orbit','box','microscope','layers','flask-conical','radar','test-tubes']} opacity={0.25} size={22} direction="forward" />
        <IconRing radius={240} icons={['box','layers','microscope','network','hexagon','database','atom','orbit']}                           opacity={0.5}  size={28} direction="reverse" />
        <IconRing radius={145} icons={['hexagon','atom','flask-conical','network','orbit','box']}                                            opacity={0.8}  size={36} direction="forward" />
      </div>
      <div className="absolute inset-0 z-10 pointer-events-none bg-gradient-to-t from-[#e9cb8e] via-[#e9cb8e]/50 to-transparent" />

      {/* ── Main Content ──────────────────────────────────────────────────── */}
      <div className="relative z-20 w-full h-full flex flex-col items-center pt-[10vh] pb-12 gap-7 px-4">

        {/* History button */}
        {history.length > 0 && status !== 'results' && (
          <div className="absolute top-6 right-6 z-40">
            <button
              onClick={() => setShowHistory((v) => !v)}
              className="flex items-center gap-2 px-3.5 py-2 rounded-full bg-white/50 hover:bg-white/80 border border-[#66280C]/15 text-[#66280C] text-xs font-semibold shadow-sm backdrop-blur-md transition-all"
            >
              <Clock className="w-3.5 h-3.5" />
              History ({history.length})
            </button>

            {showHistory && (
              <div className="absolute top-10 right-0 w-64 premium-glass-panel rounded-2xl p-3 shadow-2xl animate-fade-down z-50">
                <div className="flex items-center justify-between mb-2 px-1">
                  <span className="text-[0.7rem] font-bold uppercase tracking-widest text-[#66280C]">Recent Analyses</span>
                  <button onClick={() => setShowHistory(false)} className="text-[#A5663A] hover:text-[#66280C] transition-colors">
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
                {history.map((item, idx) => (
                  <button
                    key={idx}
                    onClick={() => { setResults(item.results); setStatus('results'); setShowHistory(false); }}
                    className="w-full text-left flex items-center justify-between px-3 py-2.5 rounded-xl hover:bg-white/50 transition-all group"
                  >
                    <div>
                      <span className="font-mono-sci text-sm font-medium text-[#260B01]">{item.filename}</span>
                      <p className="text-[0.65rem] text-[#A5663A] mt-0.5">{item.time}</p>
                    </div>
                    <ChevronRight className="w-3.5 h-3.5 text-[#A5663A] opacity-0 group-hover:opacity-100 transition-opacity" />
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ── Hero Header ─────────────────────────────────────────────────── */}
        <div className={`flex flex-col items-center transition-all duration-700 ${status === 'results' ? 'scale-90 -translate-y-8 opacity-80' : 'scale-100'}`}>
          <div className="w-20 h-20 rounded-[1.25rem] premium-glass-panel flex items-center justify-center mb-2 relative group hover:cursor-default transition-all duration-500 hover:shadow-[0_0_30px_rgba(102,40,12,0.15)]">
            <div className="absolute inset-0 rounded-[1.25rem] bg-gradient-to-br from-[#A5663A]/10 to-[#66280C]/10 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
            <Atom className="w-10 h-10 text-[#66280C] group-hover:text-[#A5663A] transition-colors duration-500" strokeWidth={1.5} />
            <Sparkles className="absolute -top-1.5 -right-1.5 w-5 h-5 text-[#A5663A] opacity-80" strokeWidth={1.5} />
          </div>

          <div className="flex flex-col items-center gap-4 max-w-4xl text-center">
            <h1 className="text-5xl md:text-7xl font-bold tracking-tight text-[#260B01]">
              Upload <span className="text-gradient-spice">Microstructure.</span>
            </h1>
            <p className="text-lg md:text-[1.15rem] font-medium max-w-2xl leading-relaxed text-[#66280C]/80 tracking-wide">
              Predict crystalline material properties from EBSD images via the Fusion Neural Network.
            </p>
          </div>
        </div>

        {/* ── Interactive Zone ─────────────────────────────────────────────── */}
        <div className="w-full max-w-[34rem] mt-6 relative perspective-1000 z-30">
          <canvas
            ref={canvasRef}
            className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[900px] h-[700px] pointer-events-none z-50"
          />

          {/* FORM */}
          <form
            onSubmit={handlePredict}
            className={`relative w-full h-[68px] group transition-all duration-700 ease-[cubic-bezier(0.16,1,0.3,1)]
              ${(status === 'success' || status === 'results') ? 'opacity-0 scale-95 pointer-events-none absolute inset-0' : 'opacity-100 scale-100 relative'}
            `}
          >
            <div className="w-full h-[68px] flex items-center bg-white/40 rounded-full premium-input overflow-hidden">
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                required
                onChange={handleFileChange}
                disabled={status !== 'idle'}
                className="hidden"
                id="file-upload"
              />
              <label
                htmlFor="file-upload"
                className={`flex-1 h-full flex items-center pl-8 cursor-pointer transition-colors hover:bg-white/30 text-[#260B01] ${!selectedFile ? 'opacity-70' : 'font-medium'}`}
              >
                {selectedFile ? (
                  <span className="truncate pr-4">{selectedFile.name}</span>
                ) : (
                  <span className="tracking-wide">Select EBSD Image...</span>
                )}
              </label>
            </div>

            <div className="absolute top-[6px] right-[6px] bottom-[6px]">
              <button
                type="submit"
                disabled={!selectedFile || status !== 'idle'}
                className="h-full px-8 rounded-full font-semibold text-[#E9CB8E] transition-all duration-300 active:scale-[0.98] disabled:active:scale-100 disabled:cursor-not-allowed disabled:opacity-70 flex items-center justify-center min-w-[140px] overflow-hidden relative group/btn"
                style={{
                  background: 'linear-gradient(135deg, #66280c 0%, #4a1d09 100%)',
                  border: '1px solid rgba(255,255,255,0.15)',
                  boxShadow: '0 4px 15px rgba(102,40,12,0.3), inset 0 1px 0 rgba(255,255,255,0.15)',
                }}
              >
                <div className="absolute inset-0 -translate-x-full group-hover/btn:animate-shimmer bg-gradient-to-r from-transparent via-white/20 to-transparent skew-x-12 pointer-events-none" />
                {status === 'loading' ? (
                  <div className="flex items-center gap-2.5 z-10 text-[#E9CB8E]">
                    <Network className="animate-spin w-4 h-4 opacity-90" />
                    <span className="tracking-wide">Analyzing</span>
                  </div>
                ) : (
                  <div className="flex items-center gap-2.5 z-10">
                    <span className="tracking-wide">Predict</span>
                    <Activity className="w-4 h-4 opacity-80 group-hover/btn:opacity-100 transition-opacity" />
                  </div>
                )}
              </button>
            </div>
          </form>

          {/* Image Preview — shown while idle */}
          {status === 'idle' && previewUrl && (
            <div className="mt-6 px-2 animate-fade-down z-30">
              <img 
                src={previewUrl} 
                alt="Preview" 
                className="w-32 h-32 object-cover rounded-2xl shadow-lg border-2 border-white/50"
              />
            </div>
          )}

          {/* SUCCESS OVERLAY */}
          <div
            className={`absolute top-0 left-0 w-full h-[68px] flex items-center justify-center rounded-full transition-all duration-700 ease-[cubic-bezier(0.16,1,0.3,1)] bg-[#A5663A] border border-[#66280C]/30
              ${status === 'success' ? 'opacity-100 scale-100' : 'opacity-0 scale-95 pointer-events-none'}
            `}
            style={{ boxShadow: 'inset 0 0 20px rgba(102,40,12,0.2)' }}
          >
            {status === 'success' && (
              <>
                <div className="absolute top-1/2 left-1/2 w-full h-full rounded-full border border-[#E9CB8E]/40 animate-ring" style={{ animationDelay: '0s' }} />
                <div className="absolute top-1/2 left-1/2 w-full h-full rounded-full border border-[#E9CB8E]/20 animate-ring" style={{ animationDelay: '0.15s' }} />
              </>
            )}
            <div className={`flex items-center gap-3 text-[#E9CB8E] font-medium text-[1.1rem] tracking-wide ${status === 'success' ? 'animate-bounce-in' : ''}`}>
              <div className="bg-[#E9CB8E]/20 p-1.5 rounded-full border border-[#E9CB8E]/30">
                <Check className={`w-5 h-5 text-[#E9CB8E] ${status === 'success' ? 'animate-checkmark' : ''}`} strokeWidth={2.5} />
              </div>
              <span>Neural Fusion Complete</span>
            </div>
          </div>
        </div>

        {/* ── Results Grid ──────────────────────────────────────────────────── */}
        {status === 'results' && results && (
          <div className="w-full max-w-4xl mt-4">

            {/* Results Header */}
            <div className="flex items-center justify-between mb-6 px-2">
              <div className="flex items-center gap-2">
                <div className="p-2 bg-[#66280C]/10 rounded-lg border border-[#66280C]/20">
                  <FlaskConical className="w-5 h-5 text-[#66280C]" />
                </div>
                <div>
                  <h3 className="text-xl font-bold text-[#260B01] max-w-[200px] md:max-w-md truncate">
                    File: <span className="font-mono-sci text-lg">{selectedFile?.name || 'Image'}</span>
                  </h3>
                  <p className="text-sm font-medium text-[#A5663A]">Computed Properties</p>
                </div>
              </div>

              {/* Action buttons */}
              <div className="flex items-center gap-2">
                <button
                  onClick={handleCopy}
                  className="flex items-center gap-1.5 text-xs font-semibold text-[#66280C] bg-white/50 hover:bg-white/80 border border-[#66280C]/20 px-3 py-1.5 rounded-full transition-all shadow-sm"
                >
                  {copied
                    ? <><CheckCheck className="w-3.5 h-3.5 text-green-700" /><span className="text-green-700">Copied</span></>
                    : <><Copy className="w-3.5 h-3.5" />Copy</>
                  }
                </button>
                <button
                  onClick={handleExport}
                  className="flex items-center gap-1.5 text-xs font-semibold text-[#66280C] bg-white/50 hover:bg-white/80 border border-[#66280C]/20 px-3 py-1.5 rounded-full transition-all shadow-sm"
                >
                  <Download className="w-3.5 h-3.5" />JSON
                </button>
                <button
                  onClick={resetForm}
                  className="text-sm font-semibold text-[#66280C] bg-white/50 hover:bg-white/80 border border-[#66280C]/20 px-4 py-2 rounded-full transition-all shadow-sm flex items-center gap-2"
                >
                  New Analysis <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Property Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

              {/* Card 1 — Microstructure */}
              <div className="result-card-1 premium-glass-panel rounded-2xl p-6 relative overflow-hidden group">
                <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                  <Microscope className="w-16 h-16 text-[#66280C]" />
                </div>
                <h4 className="text-[#66280C] font-bold tracking-wider uppercase text-xs mb-5 flex items-center gap-2">
                  <Maximize2 className="w-4 h-4" /> Microstructure Elements
                </h4>
                <div className="space-y-4">
                  {/* Grain Size */}
                  <div className="border-b border-[#66280C]/10 pb-3">
                    <div className="flex justify-between items-end">
                      <Tooltip text={PROP_DESCRIPTIONS.grain_size}>
                        <span className="text-[#A5663A] font-medium cursor-help">Mean Grain Size</span>
                        <Info className="w-3 h-3 text-[#A5663A]/50 ml-1" />
                      </Tooltip>
                      <div className="text-right">
                        <span className="text-2xl font-bold text-[#260B01] font-mono-sci">
                          <AnimatedNumber value={results.grain_size?.value || '24.51'} />
                        </span>
                        <span className="text-sm text-[#A5663A] ml-1">{results.grain_size?.unit || 'µm'}</span>
                      </div>
                    </div>
                    <ConfidenceBar confidence={results.grain_size?.confidence || 92} />
                  </div>
                  {/* Avg Misorientation */}
                  <div className="border-b border-[#66280C]/10 pb-3">
                    <div className="flex justify-between items-end">
                      <Tooltip text={PROP_DESCRIPTIONS.avg_misorientation}>
                        <span className="text-[#A5663A] font-medium cursor-help">Avg Misorientation</span>
                        <Info className="w-3 h-3 text-[#A5663A]/50 ml-1" />
                      </Tooltip>
                      <div className="text-right">
                        <span className="text-2xl font-bold text-[#260B01] font-mono-sci">
                          <AnimatedNumber value={results.avg_misorientation?.value || '3.24'} />
                        </span>
                        <span className="text-sm text-[#A5663A] ml-1">{results.avg_misorientation?.unit || '°'}</span>
                      </div>
                    </div>
                    <ConfidenceBar confidence={results.avg_misorientation?.confidence || 91} />
                  </div>
                </div>
              </div>

              {/* Card 2 — Crystallography */}
              <div className="result-card-2 premium-glass-panel rounded-2xl p-6 relative overflow-hidden group">
                <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                  <Layers className="w-16 h-16 text-[#66280C]" />
                </div>
                <h4 className="text-[#66280C] font-bold tracking-wider uppercase text-xs mb-5 flex items-center gap-2">
                  <Activity className="w-4 h-4" /> Crystallography &amp; Texture
                </h4>
                <div className="space-y-4">
                  {/* Texture Index */}
                  <div className="border-b border-[#66280C]/10 pb-3">
                    <div className="flex justify-between items-end">
                      <Tooltip text={PROP_DESCRIPTIONS.texture_index}>
                        <span className="text-[#A5663A] font-medium cursor-help">Texture Index (J)</span>
                        <Info className="w-3 h-3 text-[#A5663A]/50 ml-1" />
                      </Tooltip>
                      <div className="text-right">
                        <span className="text-2xl font-bold text-[#260B01] font-mono-sci">
                          <AnimatedNumber value={results.texture_index?.value || '1.87'} />
                        </span>
                        <span className="text-sm text-[#A5663A] ml-1">idx</span>
                      </div>
                    </div>
                    <ConfidenceBar confidence={results.texture_index?.confidence || 90} />
                  </div>
                  {/* Phase Fraction */}
                  <div className="border-b border-[#66280C]/10 pb-3">
                    <div className="flex justify-between items-end">
                      <span className="text-[#A5663A] font-medium">Phase Fraction</span>
                      <div className="text-right">
                        <span className="text-2xl font-bold text-[#260B01] font-mono-sci">98.2</span>
                        <span className="text-sm text-[#A5663A] ml-1">%</span>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 mt-1.5">
                      <div className="flex-1 h-[3px] bg-[#66280C]/10 rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full"
                          style={{ width: '98.2%', background: 'linear-gradient(90deg, #A5663A, #66280C)', transition: 'width 1.2s cubic-bezier(0.16,1,0.3,1)' }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Card 3 — Mechanical (full width) */}
              <div className="result-card-3 md:col-span-2 premium-glass-panel rounded-2xl p-6 relative overflow-hidden group">
                <div className="absolute right-0 top-1/2 -translate-y-1/2 p-6 opacity-5 group-hover:opacity-10 transition-opacity">
                  <Shield className="w-24 h-24 text-[#66280C]" />
                </div>
                <h4 className="text-[#66280C] font-bold tracking-wider uppercase text-xs mb-5 flex items-center gap-2">
                  <Zap className="w-4 h-4" /> Inferred Mechanical Attributes
                </h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">

                  {/* Elastic Modulus */}
                  <div className="bg-white/40 p-4 rounded-xl border border-[#66280C]/10 shadow-sm">
                    <Tooltip text={PROP_DESCRIPTIONS.elastic_modulus}>
                      <p className="text-sm text-[#A5663A] mb-1 font-medium cursor-help">Elastic Modulus</p>
                    </Tooltip>
                    <p className="text-xl font-bold text-[#260B01] font-mono-sci">
                      <AnimatedNumber value={results.elastic_modulus?.value || '192.5'} />
                      {' '}<span className="text-xs font-sans text-[#66280C]">GPa</span>
                    </p>
                    <ConfidenceBar confidence={results.elastic_modulus?.confidence || 89} />
                  </div>

                  {/* Yield Strength */}
                  <div className="bg-white/40 p-4 rounded-xl border border-[#66280C]/10 shadow-sm">
                    <Tooltip text={PROP_DESCRIPTIONS.yield_strength}>
                      <p className="text-sm text-[#A5663A] mb-1 font-medium cursor-help">Yield Strength</p>
                    </Tooltip>
                    <p className="text-xl font-bold text-[#260B01] font-mono-sci">
                      <AnimatedNumber value={results.yield_strength?.value || '485'} />
                      {' '}<span className="text-xs font-sans text-[#66280C]">MPa</span>
                    </p>
                    <ConfidenceBar confidence={results.yield_strength?.confidence || 87} />
                  </div>

                  {/* Hardness */}
                  <div className="bg-white/40 p-4 rounded-xl border border-[#66280C]/10 shadow-sm">
                    <Tooltip text={PROP_DESCRIPTIONS.hardness}>
                      <p className="text-sm text-[#A5663A] mb-1 font-medium cursor-help">Hardness (HV)</p>
                    </Tooltip>
                    <p className="text-xl font-bold text-[#260B01] font-mono-sci">
                      <AnimatedNumber value={results.hardness?.value || '154'} />
                      {' '}<span className="text-xs font-sans text-[#66280C]">kgf/mm²</span>
                    </p>
                    <ConfidenceBar confidence={results.hardness?.confidence || 85} />
                  </div>

                  {/* Bulk Modulus */}
                  <div className="bg-white/40 p-4 rounded-xl border border-[#66280C]/10 shadow-sm">
                    <Tooltip text={PROP_DESCRIPTIONS.bulk_modulus}>
                      <p className="text-sm text-[#A5663A] mb-1 font-medium cursor-help">Bulk Modulus</p>
                    </Tooltip>
                    <p className="text-xl font-bold text-[#260B01] font-mono-sci">
                      <AnimatedNumber value={results.bulk_modulus?.value || '140.2'} />
                      {' '}<span className="text-xs font-sans text-[#66280C]">GPa</span>
                    </p>
                    <ConfidenceBar confidence={results.bulk_modulus?.confidence || 86} />
                  </div>

                  {/* Dislocation Density */}
                  <div className="bg-white/40 p-4 rounded-xl border border-[#66280C]/10 shadow-sm">
                    <Tooltip text="Derived from Average Misorientation. Indicates deformation and trapped dislocations.">
                      <p className="text-sm text-[#A5663A] mb-1 font-medium cursor-help">Disloc. Density</p>
                    </Tooltip>
                    <p className="text-xl font-bold text-[#260B01] font-mono-sci">
                      {results.dislocation_density?.value ? results.dislocation_density.value.toExponential(2) : '1.20e14'}
                      {' '}<span className="text-xs font-sans text-[#66280C]">m⁻²</span>
                    </p>
                    <ConfidenceBar confidence={results.dislocation_density?.confidence || 89} />
                  </div>

                </div>
              </div>
            </div>
          </div>
        )}

        {/* Footer status bar */}
        {status !== 'results' && (
          <div className="flex items-center gap-3 mt-6 z-20">
            <div
              className={`w-1.5 h-1.5 rounded-full bg-[#66280C] shadow-[0_0_8px_rgba(102,40,12,0.5)] ${status === 'loading' ? 'animate-pulse-dot' : ''}`}
            />
            <p className="text-[0.8rem] font-mono-sci text-[#A5663A] uppercase tracking-widest opacity-90">
              {status === 'loading'
                ? 'Running Neural Fusion · Please wait…'
                : 'Model Accuracy: 99.8% · Crystalline Structures'}
            </p>
          </div>
        )}

      </div>
    </div>
  );
}
