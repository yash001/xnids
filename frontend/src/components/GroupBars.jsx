import "./GroupBars.css";

const GROUP_COLORS = {
  packet_shape: "#4fd6d2",
  timing: "#ffb020",
  tcp_flags: "#ff5468",
  protocol: "#9b8cff",
  port_entropy: "#5ec8ff",
  destination_spread: "#ffd166",
  http_behavior: "#ff8fab",
  link_layer: "#7bd389",
};

/**
 * Visualizes the explainer's group-level importance (sparse group lasso,
 * paper Sec 3.4): each feature group becomes one segmented bar, with
 * width proportional to its aggregated importance. This is the
 * "signature element" for the Attack Screen, directly representing the
 * paper's core technical contribution rather than a generic bar chart.
 */
export default function GroupBars({ groupImportance }) {
  if (!groupImportance) return null;
  const entries = Object.entries(groupImportance).sort((a, b) => b[1] - a[1]);
  const max = Math.max(...entries.map(([, v]) => v), 0.0001);

  return (
    <div className="group-bars">
      {entries.map(([group, value]) => (
        <div className="group-bars__row" key={group}>
          <div className="group-bars__label mono">{group.replace(/_/g, " ")}</div>
          <div className="group-bars__track">
            <div
              className="group-bars__fill"
              style={{
                width: `${(value / max) * 100}%`,
                background: GROUP_COLORS[group] || "var(--signal-cyan)",
              }}
            />
          </div>
          <div className="group-bars__value mono">{(value * 100).toFixed(1)}%</div>
        </div>
      ))}
    </div>
  );
}
