import { useEffect, useRef, useState } from "react";

/**
 * Signature element: an oscilloscope-style trace representing the
 * detector's live anomaly score stream. Calm baseline jitter; spikes
 * when `pulse` (0-1 intensity) is supplied, e.g. right after a
 * simulated attack, echoing the paper's framing of detection as a
 * continuous signal that explanation turns into actionable meaning.
 */
export default function Waveform({ height = 64, pulse = 0, color = "var(--signal-cyan)" }) {
  const [points, setPoints] = useState(() => Array.from({ length: 80 }, () => 0.5));
  const pulseRef = useRef(pulse);

  useEffect(() => {
    pulseRef.current = pulse;
  }, [pulse]);

  useEffect(() => {
    const id = setInterval(() => {
      setPoints((prev) => {
        const next = prev.slice(1);
        const base = 0.5 + (Math.random() - 0.5) * 0.12;
        const spike = pulseRef.current > 0
          ? pulseRef.current * (0.35 * Math.sin(Date.now() / 60) + 0.35)
          : 0;
        next.push(Math.max(0.02, Math.min(0.98, base + spike)));
        return next;
      });
    }, 90);
    return () => clearInterval(id);
  }, []);

  const w = 800;
  const h = height;
  const step = w / (points.length - 1);
  const path = points
    .map((p, i) => `${i === 0 ? "M" : "L"}${(i * step).toFixed(1)},${(h - p * h).toFixed(1)}`)
    .join(" ");

  return (
    <svg viewBox={`0 0 ${w} ${h}`} width="100%" height={h} preserveAspectRatio="none" style={{ display: "block" }}>
      <defs>
        <linearGradient id="wf-fade" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stopColor={color} stopOpacity="0" />
          <stop offset="0.15" stopColor={color} stopOpacity="0.9" />
          <stop offset="1" stopColor={color} stopOpacity="0.9" />
        </linearGradient>
      </defs>
      {[0.25, 0.5, 0.75].map((g) => (
        <line key={g} x1="0" x2={w} y1={h * g} y2={h * g} stroke="var(--hairline)" strokeWidth="1" />
      ))}
      <path d={path} fill="none" stroke="url(#wf-fade)" strokeWidth="1.6" />
    </svg>
  );
}
