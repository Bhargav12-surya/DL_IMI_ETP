import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Atom, Hexagon, Network, Orbit, Box, Microscope,
  Layers, FlaskConical, Radar, TestTubes, Database,
  Activity, Check, Sparkles, ChevronRight,
  Maximize2, Zap, Shield, Info, Copy, Download,
  Clock, X, CheckCheck, Image, FileCode2,
} from 'lucide-react';

// ─── Icon Mapping for Dynamic Rings ───────────────────────────────────────────
const ICON_MAP = {
  atom: Atom, hexagon: Hexagon, network: Network, orbit: Orbit,
  box: Box, microscope: Microscope, layers: Layers,
  'flask-conical': FlaskConical, radar: Radar, 'test-tubes': TestTubes,
  database: Database,
};

// ─── Branch Config ─────────────────────────────────────────────────────────────
// Each branch defines its endpoint, accepted files, and FormData field names.
// files[0] = primary slot, files[1] = pole figure (fusion only), files[2] = h5 (fusion only)
const BRANCHES = [
  {
    id: 'resnet',
    label: 'ResNet',
    sub: 'EBSD image',
    icon: Microscope,
    endpoint: '/api/predict/resnet',
    fields: [{ key: 'micro_image', label: 'EBSD Microstructure Image', accept: 'image/*' }],
  },
  {
    id: 'mobilenet',
    label: 'MobileNet',
    sub: 'Pole figure',
    icon: Radar,
    endpoint: '/api/predict/mobilenet',
    fields: [{ key: 'pole_figure', label: 'Pole Figure Image', accept: 'image/*' }],
  },
  {
    id: 'cgcnn',
    label: 'CGCNN',
    sub: 'H5 graph',
    icon: Network,
    endpoint: '/api/predict/cgcnn',
    fields: [{ key: 'h5_file', label: 'H5 Grain Graph File', accept: '.h5,.hdf5' }],
  },
  {
    id: 'fusion',
    label: 'Fusion',
    sub: 'All three',
    icon: Layers,
    endpoint: '/api/predict/fusion',
    fields: [
      { key: 'micro_image', label: 'EBSD Microstructure Image', accept: 'image/*' },
      { key: 'pole_figure', label: 'Pole Figure Image', accept: 'image/*' },
      { key: 'h5_file', label: 'H5 Grain Graph File', accept: '.h5,.hdf5' },
    ],
  },
];

// ─── Loading Step Labels per Branch ───────────────────────────────────────────
const LOAD_STEPS = {
  resnet: ['Decoding image…', 'Running ResNet50…', 'Extracting embeddings…'],
  mobilenet: ['Decoding pole figure…', 'Running MobileNetV2…', 'Extracting embeddings…'],
  cgcnn: ['Parsing H5 graph…', 'Building grain nodes…', 'Running CGCNN…'],
  fusion: ['ResNet branch…', 'MobileNet branch…', 'CGCNN branch…', 'Fusion layer…'],
};

// ─── Property Tooltips ────────────────────────────────────────────────────────
const PROP_DESCRIPTIONS = {
  grain_size: 'Mean equivalent sphere diameter of crystalline grains. Smaller grains generally increase yield strength (Hall-Petch relation).',
  avg_misorientation: 'Mean angle between adjacent lattice orientations within grains. Reflects dislocation density and deformation state.',
  texture_index: 'ODF second moment J = ∫f(g)²dg. Measures sharpness of crystallographic texture; 1 = random, higher = stronger texture.',
  elastic_modulus: "Young's Modulus — ratio of tensile stress to tensile strain in the elastic regime.",
  yield_strength: 'Stress at onset of plastic deformation. Governed by grain size, dislocation density, and phase composition.',
  hardness: 'Vickers Hardness (HV) — resistance to localised plastic deformation under indentation load.',
  bulk_modulus: 'Resistance to uniform compression. Inverse of volumetric compressibility; related to atomic bonding strength.',
  dislocation_density: 'Derived from Average Misorientation via Nye tensor. Indicates deformation and trapped dislocations.',
};

// ─── Team Members ─────────────────────────────────────────────────────────────
const TEAM_MEMBERS = [
  { name: 'V. Pujya Ganesh', id: 'DL.AI.U4AID24141' },
  { name: 'P. Bhargav Ram', id: 'DL.AI.U4AID24146' },
  { name: 'Vishnu Karthik C', id: 'DL.AI.U4AID24109' },
  { name: 'Amith Vignesh K', id: 'DL.AI.U4AID24116' },
];

// ─── Tooltip ──────────────────────────────────────────────────────────────────
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
  const numVal = parseFloat(value);
  const decimals = (String(value).split('.')[1] || '').length;
  const [display, setDisplay] = useState(0);
  if (isNaN(numVal)) return <>{value}</>;
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
        const angle = (i / icons.length) * 2 * Math.PI;
        const x = Math.cos(angle) * radius;
        const y = Math.sin(angle) * radius;
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

// ─── Team Footer ──────────────────────────────────────────────────────────────
const TeamFooter = () => (
  <footer className="relative z-20 w-full max-w-4xl px-4 pb-8 mt-8">
    <div className="flex items-center gap-4 mb-6">
      <div className="flex-1 h-px bg-[#66280C]/15" />
      <span className="text-[0.65rem] font-bold uppercase tracking-[0.2em] text-[#A5663A]/60">Team-14</span>
      <div className="flex-1 h-px bg-[#66280C]/15" />
    </div>
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      {TEAM_MEMBERS.map((member, idx) => (
        <div
          key={idx}
          className="premium-glass-panel rounded-2xl px-4 py-3.5 flex flex-col items-center text-center gap-1.5 group hover:bg-white/60 transition-all duration-300"
        >
          <div
            className="w-9 h-9 rounded-full flex items-center justify-center text-[0.65rem] font-bold text-[#E9CB8E] mb-0.5 transition-transform duration-300 group-hover:scale-110"
            style={{ background: 'linear-gradient(135deg, #66280c 0%, #A5663A 100%)', border: '1.5px solid rgba(255,255,255,0.3)' }}
          >
            {member.name.split(' ').slice(0, 2).map(w => w[0]).join('')}
          </div>
          <span className="text-[0.78rem] font-semibold text-[#260B01] leading-tight">{member.name}</span>
          <span className="font-mono-sci text-[0.6rem] text-[#A5663A]/70 tracking-wide">{member.id}</span>
        </div>
      ))}
    </div>
  </footer>
);

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {

  // ── State ──────────────────────────────────────────────────────────────────
  const [branch, setBranch] = useState('resnet');
  // files[0] = primary (all branches), files[1] = pole figure, files[2] = h5 (fusion only)
  const [files, setFiles] = useState([null, null, null]);
  const [previews, setPreviews] = useState([null, null, null]);
  const [status, setStatus] = useState('idle');  // idle | loading | success | results
  const [results, setResults] = useState(null);
  const [history, setHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const [copied, setCopied] = useState(false);
  const [apiError, setApiError] = useState(null);
  const [inferenceMs, setInferenceMs] = useState(null);
  const [loadStep, setLoadStep] = useState(0);

  const canvasRef = useRef(null);
  const fileRefs = [useRef(null), useRef(null), useRef(null)];

  // ── Derived Values ─────────────────────────────────────────────────────────
  const branchCfg = BRANCHES.find(b => b.id === branch);

  // Fusion needs all 3 files; every other branch needs only files[0]
  const canPredict = branch === 'fusion'
    ? !!(files[0] && files[1] && files[2])
    : !!files[0];

  const isBusy = status === 'loading';

  // Does the API response contain any mechanical / physics-derived properties?
  const hasMechanical = results && (
    results.elastic_modulus ||
    results.yield_strength ||
    results.hardness ||
    results.bulk_modulus ||
    results.dislocation_density
  );

  const currentSteps = LOAD_STEPS[branch] || LOAD_STEPS.resnet;

  // ── Cycle loading step labels while fetching ───────────────────────────────
  useEffect(() => {
    if (status !== 'loading') { setLoadStep(0); return; }
    const id = setInterval(
      () => setLoadStep(p => Math.min(p + 1, currentSteps.length - 1)),
      2000,
    );
    return () => clearInterval(id);
  }, [status, branch]); // eslint-disable-line

  // ── Particle Animation ─────────────────────────────────────────────────────
  const fireCrystallineParticles = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    const colors = ['#66280C', '#A5663A', '#260B01', '#ffffff', '#E9CB8E'];
    const particles = [];
    const make = () => ({
      x: canvas.width / 2, y: canvas.height / 2,
      vx: (Math.random() - 0.5) * 16, vy: (Math.random() - 2) * 12,
      life: 100, color: colors[Math.floor(Math.random() * colors.length)],
      size: Math.random() * 4 + 2,
      rotation: Math.random() * Math.PI * 2,
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
        ctx.fillStyle = p.color;
        ctx.shadowBlur = 10; ctx.shadowColor = p.color;
        ctx.translate(p.x, p.y); ctx.rotate(p.rotation);
        ctx.beginPath();
        ctx.moveTo(0, -p.size); ctx.lineTo(p.size * 0.7, 0);
        ctx.lineTo(0, p.size); ctx.lineTo(-p.size * 0.7, 0);
        ctx.closePath(); ctx.fill(); ctx.restore();
        if (p.life <= 0) particles.splice(i, 1);
      }
      requestAnimationFrame(animate);
    };
    animate();
  }, []);

  // ── processSuccess ─────────────────────────────────────────────────────────
  // FIX: this function was called in handlePredict but never defined, causing
  // every successful response to throw ReferenceError inside the catch block
  // and display the misleading "Cannot reach backend" error message.
  const processSuccess = useCallback((preds, ms) => {
    setResults(preds);
    setInferenceMs(ms ?? null);
    setStatus('success');
    fireCrystallineParticles();
    // After the success ring animation, switch to the results view
    setTimeout(() => setStatus('results'), 1800);
    setHistory(prev => [
      {
        filename: files[0]?.name || 'Image',
        branch,
        time: new Date().toLocaleTimeString(),
        results: preds,
        inferenceMs: ms ?? null,
      },
      ...prev.slice(0, 4),
    ]);
  }, [files, branch, fireCrystallineParticles]);

  // ── File Change ────────────────────────────────────────────────────────────
  const handleFileChange = (slotIdx) => (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setFiles(prev => { const n = [...prev]; n[slotIdx] = file; return n; });
    // Only create an object URL for image files (not .h5)
    if (file.type.startsWith('image/')) {
      const url = URL.createObjectURL(file);
      setPreviews(prev => { const n = [...prev]; n[slotIdx] = url; return n; });
    } else {
      setPreviews(prev => { const n = [...prev]; n[slotIdx] = null; return n; });
    }
    setApiError(null);
  };

  // ── Predict ────────────────────────────────────────────────────────────────
  const handlePredict = async (e) => {
    e.preventDefault();
    if (!canPredict || isBusy) return;
    setStatus('loading');
    setApiError(null);

    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 60000);

      // Build FormData using the branch's field definitions
      const formData = new FormData();
      branchCfg.fields.forEach((field, idx) => {
        if (files[idx]) formData.append(field.key, files[idx]);
      });

      const response = await fetch(branchCfg.endpoint, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      });
      clearTimeout(timeout);

      if (response.ok) {
        const data = await response.json();
        if (data.predictions && !data.predictions.error) {
          processSuccess(data.predictions, data.inference_ms);
        } else {
          const msg = data.predictions?.error || data.error || 'Model returned no predictions.';
          setApiError(msg);
          setStatus('idle');
        }
      } else {
        const body = await response.json().catch(() => ({}));
        setApiError(`Server error ${response.status}: ${body.error || 'Unknown error'}`);
        setStatus('idle');
      }

    } catch (err) {
      if (err.name === 'AbortError') {
        setApiError('Request timed out. Is the backend running?  →  python backend/app.py');
      } else if (err instanceof TypeError && err.message.toLowerCase().includes('fetch')) {
        // True network failure (CORS, connection refused, etc.)
        setApiError('Cannot reach backend at localhost:5000. Run:  python backend/app.py');
      } else {
        // JS runtime error (undefined variable, JSON parse fail, etc.)
        // Show the real message so it's not mistaken for a network error
        setApiError(`Frontend error: ${err.message}`);
        console.error('Predict runtime error:', err);
      }
      setStatus('idle');
    }
  };

  // ── Branch Switch ──────────────────────────────────────────────────────────
  const handleBranchSwitch = (id) => {
    if (isBusy) return;
    setBranch(id);
    setFiles([null, null, null]);
    setPreviews([null, null, null]);
    setApiError(null);
    // If currently viewing results, go back to idle when switching branch
    if (status === 'results') { setResults(null); setStatus('idle'); }
  };

  // ── Reset ──────────────────────────────────────────────────────────────────
  const resetForm = () => {
    setFiles([null, null, null]);
    setPreviews([null, null, null]);
    setResults(null);
    setStatus('idle');
    setShowHistory(false);
    setApiError(null);
    setInferenceMs(null);
    setLoadStep(0);
  };

  // ── Copy / Export ──────────────────────────────────────────────────────────
  const handleCopy = async () => {
    if (!results) return;
    const lines = [
      `Branch:    ${branch.toUpperCase()}`,
      `File:      ${files[0]?.name || 'Uploaded Image'}`,
      inferenceMs ? `Inference: ${inferenceMs}ms` : '',
      '─'.repeat(34),
      ...Object.entries(results).map(
        ([k, v]) => `${k.padEnd(22)}: ${v.value} ${v.unit}  (confidence ${v.confidence ?? 95}%)`
      ),
    ].filter(Boolean).join('\n');
    await navigator.clipboard.writeText(lines);
    setCopied(true);
    setTimeout(() => setCopied(false), 2200);
  };

  const handleExport = () => {
    if (!results) return;
    const payload = {
      branch,
      filename: files[0]?.name || 'Uploaded Image',
      timestamp: new Date().toISOString(),
      inference_ms: inferenceMs,
      properties: results,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = Object.assign(document.createElement('a'), {
      href: url,
      download: `predictions_${branch}.json`,
    });
    a.click();
    URL.revokeObjectURL(url);
  };

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-[#e9cb8e] text-[#a5663a] font-sans selection:bg-[#66280C]/20 selection:text-[#260B01] overflow-hidden flex flex-col items-center justify-center relative">

      {/* ── Injected Styles ────────────────────────────────────────────────── */}
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
          100% { transform: scale(1)    translateY(0);    opacity: 1; }
        }
        .animate-bounce-in { animation: subtle-fade-in 0.6s cubic-bezier(0.16,1,0.3,1) forwards; }

        @keyframes slide-up-fade {
          0%   { opacity: 0; transform: translateY(30px); }
          100% { opacity: 1; transform: translateY(0);    }
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
        .animate-shimmer   { animation: shimmer 2s infinite; }

        @keyframes chip-pop {
          0%   { opacity: 0; transform: scale(0.7); }
          100% { opacity: 1; transform: scale(1);   }
        }
        .animate-chip { animation: chip-pop 0.25s cubic-bezier(0.34,1.56,0.64,1) both; }

        @keyframes tooltip-in {
          0%   { opacity: 0; transform: translateX(-50%) translateY(4px); }
          100% { opacity: 1; transform: translateX(-50%) translateY(0);   }
        }
        .animate-tooltip-in { animation: tooltip-in 0.18s ease forwards; }

        @keyframes fade-down {
          0%   { opacity: 0; transform: translateY(-8px); }
          100% { opacity: 1; transform: translateY(0);    }
        }
        .animate-fade-down { animation: fade-down 0.2s ease forwards; }

        @keyframes pulse-dot { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
        .animate-pulse-dot  { animation: pulse-dot 1.4s ease-in-out infinite; }

        @keyframes step-in {
          0%   { opacity: 0; transform: translateY(6px); }
          100% { opacity: 1; transform: translateY(0);   }
        }
        .animate-step-in { animation: step-in 0.3s cubic-bezier(0.16,1,0.3,1) forwards; }

        @keyframes wave { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }
        .animate-wave path { animation: wave 1.5s linear infinite; }

        .premium-glass-panel {
          background:    linear-gradient(145deg, rgba(255,255,255,0.6) 0%, rgba(255,255,255,0.15) 100%);
          backdrop-filter: blur(20px);
          border:        1px solid rgba(102,40,12,0.15);
          box-shadow:    0 20px 40px -10px rgba(38,11,1,0.10), inset 0 1px 0 rgba(255,255,255,0.8);
        }
        .premium-input {
          background: rgba(255,255,255,0.45);
          box-shadow: inset 0 0 0 1px rgba(102,40,12,0.2),
                      inset 0 2px 8px rgba(38,11,1,0.04),
                      0 10px 30px -10px rgba(38,11,1,0.08);
        }
        .text-gradient-spice {
          background: linear-gradient(to right, #66280c, #a5663a, #66280c);
          -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
        }
        .result-card-1 { animation: slide-up-fade 0.6s 0.05s cubic-bezier(0.16,1,0.3,1) both; }
        .result-card-2 { animation: slide-up-fade 0.6s 0.15s cubic-bezier(0.16,1,0.3,1) both; }
        .result-card-3 { animation: slide-up-fade 0.6s 0.25s cubic-bezier(0.16,1,0.3,1) both; }

        .branch-pill { transition: all 0.2s cubic-bezier(0.16,1,0.3,1); }
        .file-slot   { transition: background 0.15s; }
        .file-slot:hover { background: rgba(255,255,255,0.4); }
      `}</style>

      {/* ── Background ─────────────────────────────────────────────────────── */}
      <div className="absolute inset-0 bg-grid-premium pointer-events-none z-0" />
      <div
        className="absolute inset-0 w-full h-full pointer-events-none z-0"
        style={{ perspective: 1200, transform: 'perspective(1200px) rotateX(20deg)', transformOrigin: 'center top' }}
      >
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[900px] h-[900px] bg-white/40 rounded-full blur-[140px]" />
        <div className="absolute top-1/3 left-1/3 -translate-x-1/2 -translate-y-1/2 w-[700px] h-[700px] bg-[#A5663A]/20 rounded-full blur-[120px]" />
        <IconRing radius={340} icons={['atom', 'hexagon', 'network', 'orbit', 'box', 'microscope', 'layers', 'flask-conical', 'radar', 'test-tubes']} opacity={0.25} size={22} direction="forward" />
        <IconRing radius={240} icons={['box', 'layers', 'microscope', 'network', 'hexagon', 'database', 'atom', 'orbit']} opacity={0.5} size={28} direction="reverse" />
        <IconRing radius={145} icons={['hexagon', 'atom', 'flask-conical', 'network', 'orbit', 'box']} opacity={0.8} size={36} direction="forward" />
      </div>
      <div className="absolute inset-0 z-10 pointer-events-none bg-gradient-to-t from-[#e9cb8e] via-[#e9cb8e]/50 to-transparent" />

      {/* ── Main Content ───────────────────────────────────────────────────── */}
      <div className="relative z-20 w-full h-full flex flex-col items-center pt-[10vh] pb-12 gap-7 px-4">

        {/* ── History Button ───────────────────────────────────────────────── */}
        {history.length > 0 && status !== 'results' && (
          <div className="absolute top-6 right-6 z-40">
            <button
              onClick={() => setShowHistory(v => !v)}
              className="flex items-center gap-2 px-3.5 py-2 rounded-full bg-white/50 hover:bg-white/80 border border-[#66280C]/15 text-[#66280C] text-xs font-semibold shadow-sm backdrop-blur-md transition-all"
            >
              <Clock className="w-3.5 h-3.5" />
              History ({history.length})
            </button>

            {showHistory && (
              <div className="absolute top-10 right-0 w-72 premium-glass-panel rounded-2xl p-3 shadow-2xl animate-fade-down z-50">
                <div className="flex items-center justify-between mb-2 px-1">
                  <span className="text-[0.7rem] font-bold uppercase tracking-widest text-[#66280C]">Recent Analyses</span>
                  <button onClick={() => setShowHistory(false)} className="text-[#A5663A] hover:text-[#66280C] transition-colors">
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
                {history.map((item, idx) => (
                  <button
                    key={idx}
                    onClick={() => {
                      setResults(item.results);
                      setInferenceMs(item.inferenceMs);
                      setBranch(item.branch);
                      setStatus('results');
                      setShowHistory(false);
                    }}
                    className="w-full text-left flex items-center justify-between px-3 py-2.5 rounded-xl hover:bg-white/50 transition-all group"
                  >
                    <div>
                      <div className="flex items-center gap-2 mb-0.5">
                        <span className="text-[0.58rem] font-bold uppercase tracking-wide px-1.5 py-0.5 rounded bg-[#66280C]/10 text-[#66280C]">
                          {item.branch}
                        </span>
                        <span className="font-mono-sci text-sm font-medium text-[#260B01] truncate max-w-[150px]">
                          {item.filename}
                        </span>
                      </div>
                      <p className="text-[0.65rem] text-[#A5663A]">{item.time}</p>
                    </div>
                    <ChevronRight className="w-3.5 h-3.5 text-[#A5663A] opacity-0 group-hover:opacity-100 transition-opacity" />
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ── Hero Header ──────────────────────────────────────────────────── */}
        <div className={`flex flex-col items-center transition-all duration-700 ${status === 'results' ? 'scale-90 -translate-y-8 opacity-80' : 'scale-100'}`}>
          <div className="w-20 h-20 rounded-[1.25rem] premium-glass-panel flex items-center justify-center mb-2 relative group hover:cursor-default transition-all duration-500 hover:shadow-[0_0_30px_rgba(102,40,12,0.15)]">
            <div className="absolute inset-0 rounded-[1.25rem] bg-gradient-to-br from-[#A5663A]/10 to-[#66280C]/10 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
            <Atom className="w-10 h-10 text-[#66280C] group-hover:text-[#A5663A] transition-colors duration-500" strokeWidth={1.5} />
            <Sparkles className="absolute -top-1.5 -right-1.5 w-5 h-5 text-[#A5663A] opacity-80" strokeWidth={1.5} />
          </div>

          <div className="flex flex-col items-center gap-4 max-w-4xl text-center">
            <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-[#260B01] leading-snug">
              A Hybrid Multi-Modal ML-DL Framework for Predicting
              <span className="text-gradient-spice text-xl md:text-3xl"> Crystalline Material Properties </span>
              from <span className="text-gradient-spice">Structure</span>,{' '}
              <span className="text-gradient-spice">Microstructure</span>,
              and <span className="text-gradient-spice">Texture descriptors</span>
            </h1>
            <p className="text-lg md:text-[1.15rem] font-medium max-w-2xl leading-relaxed text-[#66280C]/80 tracking-wide">
              Predict crystalline material properties from its EBSD images.
            </p>
          </div>
        </div>

        {/* ── Interactive Zone ─────────────────────────────────────────────── */}
        <div className="w-full max-w-[38rem] mt-4 relative z-30">
          <canvas
            ref={canvasRef}
            className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[900px] h-[700px] pointer-events-none z-50"
          />

          {/* ── Branch Selector Pills ──────────────────────────────────────── */}
          <div className={`flex gap-2 justify-center mb-4 flex-wrap transition-all duration-500
            ${(status === 'success' || status === 'results') ? 'opacity-0 pointer-events-none h-0 mb-0 overflow-hidden' : 'opacity-100'}`}
          >
            {BRANCHES.map(b => {
              const BIcon = b.icon;
              const active = branch === b.id;
              return (
                <button
                  key={b.id}
                  onClick={() => handleBranchSwitch(b.id)}
                  disabled={isBusy}
                  className={`branch-pill flex items-center gap-2 px-4 py-2 rounded-full text-sm font-semibold border
                    ${active
                      ? 'bg-[#66280C] text-[#E9CB8E] border-[#66280C] shadow-md'
                      : 'bg-white/40 text-[#66280C] border-[#66280C]/20 hover:bg-white/60'
                    } disabled:opacity-60 disabled:cursor-not-allowed`}
                >
                  <BIcon className="w-3.5 h-3.5" strokeWidth={2} />
                  {b.label}
                  <span className={`text-[0.6rem] font-normal ${active ? 'text-[#E9CB8E]/70' : 'text-[#A5663A]/60'}`}>
                    {b.sub}
                  </span>
                </button>
              );
            })}
          </div>

          {/* ── Upload Form ────────────────────────────────────────────────── */}
          <div className={`transition-all duration-700 ease-[cubic-bezier(0.16,1,0.3,1)]
            ${(status === 'success' || status === 'results') ? 'opacity-0 scale-95 pointer-events-none absolute inset-0' : 'opacity-100 scale-100 relative'}`}
          >
            {branchCfg.fields.length === 1 ? (

              /* ── Single-file bar (ResNet / MobileNet / CGCNN) ──────────── */
              <form onSubmit={handlePredict}>
                <div className="w-full h-[68px] flex items-center bg-white/40 rounded-full premium-input overflow-hidden relative">
                  <input
                    ref={fileRefs[0]}
                    type="file"
                    accept={branchCfg.fields[0].accept}
                    onChange={handleFileChange(0)}
                    disabled={isBusy}
                    className="hidden"
                    id="file-upload-0"
                  />
                  <label
                    htmlFor="file-upload-0"
                    className={`flex-1 h-full flex items-center pl-8 cursor-pointer hover:bg-white/30 transition-colors text-[#260B01] ${!files[0] ? 'opacity-70' : 'font-medium'}`}
                  >
                    {files[0]
                      ? <span className="truncate pr-4">{files[0].name}</span>
                      : <span className="tracking-wide">{branchCfg.fields[0].label}…</span>
                    }
                  </label>

                  <div className="absolute top-[6px] right-[6px] bottom-[6px]">
                    <button
                      type="submit"
                      disabled={!canPredict || isBusy}
                      className="h-full px-8 rounded-full font-semibold text-[#E9CB8E] transition-all duration-300 active:scale-[0.98] disabled:active:scale-100 disabled:cursor-not-allowed disabled:opacity-70 flex items-center justify-center min-w-[140px] overflow-hidden relative group/btn"
                      style={{
                        background: 'linear-gradient(135deg, #66280c 0%, #4a1d09 100%)',
                        border: '1px solid rgba(255,255,255,0.15)',
                        boxShadow: '0 4px 15px rgba(102,40,12,0.3), inset 0 1px 0 rgba(255,255,255,0.15)',
                      }}
                    >
                      <div className="absolute inset-0 -translate-x-full group-hover/btn:animate-shimmer bg-gradient-to-r from-transparent via-white/20 to-transparent skew-x-12 pointer-events-none" />
                      {isBusy ? (
                        <div className="flex items-center gap-2.5 z-10 text-[#E9CB8E]">
                          <svg className="w-5 h-5 animate-wave opacity-90" viewBox="0 0 100 40" fill="none">
                            <path d="M0 20 Q 25 0, 50 20 T 100 20" stroke="currentColor" strokeWidth="3" fill="none" />
                          </svg>
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
                </div>

                {/* Image preview thumbnail */}
                {previews[0] && (
                  <div className="mt-3 flex justify-center">
                    <div className="relative group/prev">
                      <img
                        src={previews[0]}
                        alt="EBSD input preview"
                        className="w-28 h-28 object-cover rounded-2xl border border-[#66280C]/20 shadow-md"
                      />
                      <div className="absolute inset-0 rounded-2xl bg-[#66280C]/0 group-hover/prev:bg-[#66280C]/10 transition-all duration-200" />
                    </div>
                  </div>
                )}
              </form>

            ) : (

              /* ── Multi-file panel (Fusion) ─────────────────────────────── */
              <form onSubmit={handlePredict}>
                <div className="premium-glass-panel rounded-3xl p-5 space-y-3">
                  <p className="text-xs font-bold uppercase tracking-widest text-[#66280C]/60 mb-1 flex items-center gap-2">
                    <Layers className="w-3.5 h-3.5" /> Fusion — upload all three inputs
                  </p>

                  {branchCfg.fields.map((field, idx) => (
                    <div
                      key={field.key}
                      className="file-slot flex items-center gap-3 bg-white/25 rounded-2xl px-4 py-3 border border-[#66280C]/10 cursor-pointer"
                      onClick={() => fileRefs[idx].current?.click()}
                    >
                      <input
                        ref={fileRefs[idx]}
                        type="file"
                        accept={field.accept}
                        onChange={handleFileChange(idx)}
                        className="hidden"
                        id={`file-upload-${idx}`}
                      />
                      {/* Status icon */}
                      <div className={`w-8 h-8 rounded-xl flex items-center justify-center flex-shrink-0 transition-colors duration-200
                        ${files[idx] ? 'bg-[#66280C]' : 'bg-[#66280C]/10'}`}
                      >
                        {files[idx]
                          ? <Check className="w-4 h-4 text-[#E9CB8E]" strokeWidth={2.5} />
                          : idx === 2
                            ? <FileCode2 className="w-4 h-4 text-[#66280C]" strokeWidth={1.5} />
                            : <Image className="w-4 h-4 text-[#66280C]" strokeWidth={1.5} />
                        }
                      </div>

                      <div className="flex-1 min-w-0">
                        <p className="text-xs font-semibold text-[#260B01]">{field.label}</p>
                        <p className="text-[0.65rem] text-[#A5663A]/70 truncate">
                          {files[idx] ? files[idx].name : `Click to select · ${field.accept}`}
                        </p>
                      </div>

                      {/* Thumbnail for image slots */}
                      {previews[idx] && (
                        <img
                          src={previews[idx]}
                          alt=""
                          className="w-10 h-10 object-cover rounded-xl border border-[#66280C]/10 flex-shrink-0"
                        />
                      )}
                    </div>
                  ))}

                  {/* Fusion predict button */}
                  <button
                    type="submit"
                    disabled={!canPredict || isBusy}
                    className="w-full mt-2 h-12 rounded-2xl font-semibold text-[#E9CB8E] transition-all duration-300 active:scale-[0.99] disabled:opacity-60 disabled:cursor-not-allowed flex items-center justify-center gap-2.5 relative overflow-hidden group/btn"
                    style={{
                      background: 'linear-gradient(135deg, #66280c 0%, #4a1d09 100%)',
                      boxShadow: '0 4px 15px rgba(102,40,12,0.3), inset 0 1px 0 rgba(255,255,255,0.15)',
                    }}
                  >
                    <div className="absolute inset-0 -translate-x-full group-hover/btn:animate-shimmer bg-gradient-to-r from-transparent via-white/20 to-transparent skew-x-12 pointer-events-none" />
                    {isBusy ? (
                      <>
                        <svg className="w-5 h-5 animate-wave" viewBox="0 0 100 40" fill="none">
                          <path d="M0 20 Q 25 0, 50 20 T 100 20" stroke="currentColor" strokeWidth="3" fill="none" />
                        </svg>
                        <span>Analyzing…</span>
                      </>
                    ) : (
                      <>
                        <span>Run Fusion Prediction</span>
                        <Activity className="w-4 h-4 opacity-80 group-hover/btn:opacity-100" />
                      </>
                    )}
                  </button>
                </div>
              </form>
            )}

            {/* ── API Error Banner ──────────────────────────────────────────── */}
            {apiError && (
              <div className="mt-4 z-20 w-full px-1">
                <div className="flex items-start gap-2 px-4 py-3 rounded-2xl border border-red-400/40 bg-red-100/60 backdrop-blur text-xs text-red-900 leading-relaxed">
                  <span className="mt-0.5 text-base">⚠️</span>
                  <span>{apiError}</span>
                </div>
              </div>
            )}
          </div>

          {/* ── Success Overlay ───────────────────────────────────────────── */}
          <div
            className={`absolute top-0 left-0 w-full h-[68px] flex items-center justify-center rounded-full transition-all duration-700 ease-[cubic-bezier(0.16,1,0.3,1)] bg-[#A5663A] border border-[#66280C]/30
              ${status === 'success' ? 'opacity-100 scale-100' : 'opacity-0 scale-95 pointer-events-none'}`}
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
              <span>Predicted Successfully</span>
            </div>
          </div>
        </div>

        {/* ── Loading Step Indicator ────────────────────────────────────────── */}
        {status === 'loading' && (
          <div className="flex flex-col items-center gap-2 mt-2 z-20">
            <div className="flex items-center gap-3">
              <div className="w-1.5 h-1.5 rounded-full bg-[#66280C] shadow-[0_0_8px_rgba(102,40,12,0.5)] animate-pulse-dot" />
              <p key={loadStep} className="text-[0.8rem] font-mono-sci text-[#A5663A] uppercase tracking-widest opacity-90 animate-step-in">
                {currentSteps[loadStep]}
              </p>
            </div>
            {/* Progress dots */}
            <div className="flex items-center gap-1.5">
              {currentSteps.map((_, i) => (
                <div
                  key={i}
                  className={`rounded-full transition-all duration-500 ${i < loadStep ? 'w-4 h-1.5 bg-[#66280C]' :
                    i === loadStep ? 'w-4 h-1.5 bg-[#A5663A]' :
                      'w-1.5 h-1.5 bg-[#66280C]/20'
                    }`}
                />
              ))}
            </div>
          </div>
        )}

        {/* ── Results Grid ─────────────────────────────────────────────────── */}
        {status === 'results' && results && (
          <div className="w-full max-w-4xl mt-4">

            {/* Results Header */}
            <div className="flex items-center justify-between mb-6 px-2 gap-3 flex-wrap">
              <div className="flex items-center gap-3">
                {/* Input thumbnail in header */}
                {previews[0] && (
                  <img
                    src={previews[0]}
                    alt="Input EBSD"
                    className="w-14 h-14 object-cover rounded-2xl border border-[#66280C]/20 shadow-sm flex-shrink-0"
                  />
                )}
                <div className="p-2 bg-[#66280C]/10 rounded-lg border border-[#66280C]/20">
                  <FlaskConical className="w-5 h-5 text-[#66280C]" />
                </div>
                <div>
                  <div className="flex items-center gap-2 flex-wrap">
                    <h3 className="text-xl font-bold text-[#260B01] max-w-[200px] md:max-w-xs truncate">
                      <span className="font-mono-sci text-lg">{files[0]?.name || 'Image'}</span>
                    </h3>
                    {/* Branch badge */}
                    <span className="text-[0.65rem] font-bold uppercase tracking-wide px-2 py-0.5 rounded-full bg-[#66280C]/10 text-[#66280C] border border-[#66280C]/15">
                      {branch}
                    </span>
                    {/* Inference time */}
                    {inferenceMs != null && (
                      <span className="text-[0.65rem] font-mono-sci text-[#A5663A]/60">
                        {inferenceMs}ms
                      </span>
                    )}
                  </div>
                  <p className="text-sm font-medium text-[#A5663A]">Computed Properties</p>
                </div>
              </div>

              {/* Action buttons */}
              <div className="flex items-center gap-2 flex-wrap">
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
                  {results.grain_size && (
                    <div className="border-b border-[#66280C]/10 pb-3">
                      <div className="flex justify-between items-end">
                        <Tooltip text={PROP_DESCRIPTIONS.grain_size}>
                          <span className="text-[#A5663A] font-medium cursor-help">Mean Grain Size</span>
                          <Info className="w-3 h-3 text-[#A5663A]/50 ml-1" />
                        </Tooltip>
                        <div className="text-right">
                          <span className="text-2xl font-bold text-[#260B01] font-mono-sci">
                            <AnimatedNumber value={String(results.grain_size.value)} />
                          </span>
                          <span className="text-sm text-[#A5663A] ml-1">{results.grain_size.unit || 'µm'}</span>
                        </div>
                      </div>
                      <ConfidenceBar confidence={results.grain_size.confidence ?? 75} />
                    </div>
                  )}
                  {results.avg_misorientation && (
                    <div className="border-b border-[#66280C]/10 pb-3">
                      <div className="flex justify-between items-end">
                        <Tooltip text={PROP_DESCRIPTIONS.avg_misorientation}>
                          <span className="text-[#A5663A] font-medium cursor-help">Avg Misorientation</span>
                          <Info className="w-3 h-3 text-[#A5663A]/50 ml-1" />
                        </Tooltip>
                        <div className="text-right">
                          <span className="text-2xl font-bold text-[#260B01] font-mono-sci">
                            <AnimatedNumber value={String(results.avg_misorientation.value)} />
                          </span>
                          <span className="text-sm text-[#A5663A] ml-1">{results.avg_misorientation.unit || '°'}</span>
                        </div>
                      </div>
                      <ConfidenceBar confidence={results.avg_misorientation.confidence ?? 75} />
                    </div>
                  )}
                  {!results.grain_size && !results.avg_misorientation && (
                    <p className="text-sm text-[#A5663A]/60 italic py-2">No microstructure data from this branch.</p>
                  )}
                </div>
              </div>

              {/* Card 2 — Crystallography & Texture */}
              <div className="result-card-2 premium-glass-panel rounded-2xl p-6 relative overflow-hidden group">
                <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                  <Layers className="w-16 h-16 text-[#66280C]" />
                </div>
                <h4 className="text-[#66280C] font-bold tracking-wider uppercase text-xs mb-5 flex items-center gap-2">
                  <Activity className="w-4 h-4" /> Crystallography &amp; Texture
                </h4>
                <div className="space-y-4">
                  {results.texture_index && (
                    <div className="border-b border-[#66280C]/10 pb-3">
                      <div className="flex justify-between items-end">
                        <Tooltip text={PROP_DESCRIPTIONS.texture_index}>
                          <span className="text-[#A5663A] font-medium cursor-help">Texture Index (J)</span>
                          <Info className="w-3 h-3 text-[#A5663A]/50 ml-1" />
                        </Tooltip>
                        <div className="text-right">
                          <span className="text-2xl font-bold text-[#260B01] font-mono-sci">
                            <AnimatedNumber value={String(results.texture_index.value)} />
                          </span>
                          <span className="text-sm text-[#A5663A] ml-1">idx</span>
                        </div>
                      </div>
                      <ConfidenceBar confidence={results.texture_index.confidence ?? 75} />
                    </div>
                  )}
                  {!results.texture_index && (
                    <p className="text-sm text-[#A5663A]/60 italic py-2">No texture data from this branch.</p>
                  )}
                </div>
              </div>

              {/* Card 3 — Mechanical (only rendered when API returns these fields) */}
              {hasMechanical && (
                <div className="result-card-3 md:col-span-2 premium-glass-panel rounded-2xl p-6 relative overflow-hidden group">
                  <div className="absolute right-0 top-1/2 -translate-y-1/2 p-6 opacity-5 group-hover:opacity-10 transition-opacity">
                    <Shield className="w-24 h-24 text-[#66280C]" />
                  </div>
                  <h4 className="text-[#66280C] font-bold tracking-wider uppercase text-xs mb-5 flex items-center gap-2">
                    <Zap className="w-4 h-4" /> Inferred Mechanical Attributes
                  </h4>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">

                    {results.elastic_modulus && (
                      <div className="bg-white/40 p-4 rounded-xl border border-[#66280C]/10 shadow-sm">
                        <Tooltip text={PROP_DESCRIPTIONS.elastic_modulus}>
                          <p className="text-sm text-[#A5663A] mb-1 font-medium cursor-help">Elastic Modulus</p>
                        </Tooltip>
                        <p className="text-xl font-bold text-[#260B01] font-mono-sci">
                          <AnimatedNumber value={String(results.elastic_modulus.value)} />
                          {' '}<span className="text-xs font-sans text-[#66280C]">GPa</span>
                        </p>
                        <ConfidenceBar confidence={results.elastic_modulus.confidence ?? 75} />
                      </div>
                    )}

                    {results.yield_strength && (
                      <div className="bg-white/40 p-4 rounded-xl border border-[#66280C]/10 shadow-sm">
                        <Tooltip text={PROP_DESCRIPTIONS.yield_strength}>
                          <p className="text-sm text-[#A5663A] mb-1 font-medium cursor-help">Yield Strength</p>
                        </Tooltip>
                        <p className="text-xl font-bold text-[#260B01] font-mono-sci">
                          <AnimatedNumber value={String(results.yield_strength.value)} />
                          {' '}<span className="text-xs font-sans text-[#66280C]">MPa</span>
                        </p>
                        <ConfidenceBar confidence={results.yield_strength.confidence ?? 75} />
                      </div>
                    )}

                    {results.hardness && (
                      <div className="bg-white/40 p-4 rounded-xl border border-[#66280C]/10 shadow-sm">
                        <Tooltip text={PROP_DESCRIPTIONS.hardness}>
                          <p className="text-sm text-[#A5663A] mb-1 font-medium cursor-help">Hardness (HV)</p>
                        </Tooltip>
                        <p className="text-xl font-bold text-[#260B01] font-mono-sci">
                          <AnimatedNumber value={String(results.hardness.value)} />
                          {' '}<span className="text-xs font-sans text-[#66280C]">kgf/mm²</span>
                        </p>
                        <ConfidenceBar confidence={results.hardness.confidence ?? 75} />
                      </div>
                    )}

                    {results.bulk_modulus && (
                      <div className="bg-white/40 p-4 rounded-xl border border-[#66280C]/10 shadow-sm">
                        <Tooltip text={PROP_DESCRIPTIONS.bulk_modulus}>
                          <p className="text-sm text-[#A5663A] mb-1 font-medium cursor-help">Bulk Modulus</p>
                        </Tooltip>
                        <p className="text-xl font-bold text-[#260B01] font-mono-sci">
                          <AnimatedNumber value={String(results.bulk_modulus.value)} />
                          {' '}<span className="text-xs font-sans text-[#66280C]">GPa</span>
                        </p>
                        <ConfidenceBar confidence={results.bulk_modulus.confidence ?? 75} />
                      </div>
                    )}

                    {results.dislocation_density && (
                      <div className="bg-white/40 p-4 rounded-xl border border-[#66280C]/10 shadow-sm">
                        <Tooltip text={PROP_DESCRIPTIONS.dislocation_density}>
                          <p className="text-sm text-[#A5663A] mb-1 font-medium cursor-help">Disloc. Density</p>
                        </Tooltip>
                        <p className="text-xl font-bold text-[#260B01] font-mono-sci">
                          {Number(results.dislocation_density.value).toExponential(2)}
                          {' '}<span className="text-xs font-sans text-[#66280C]">m⁻²</span>
                        </p>
                        <ConfidenceBar confidence={results.dislocation_density.confidence ?? 75} />
                      </div>
                    )}

                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── Team Footer ───────────────────────────────────────────────────── */}
        <TeamFooter />

      </div>
    </div>
  );
}
