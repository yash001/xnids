import "./ui.css";

export function Panel({ title, eyebrow, action, children, className = "" }) {
  return (
    <div className={`panel ${className}`}>
      {(title || eyebrow || action) && (
        <div className="panel__head">
          <div>
            {eyebrow && <div className="eyebrow">{eyebrow}</div>}
            {title && <div className="panel__title">{title}</div>}
          </div>
          {action && <div className="panel__action">{action}</div>}
        </div>
      )}
      <div className="panel__body">{children}</div>
    </div>
  );
}

export function StatCard({ label, value, sub, tone = "default" }) {
  return (
    <div className={`stat-card tone-${tone}`}>
      <div className="eyebrow">{label}</div>
      <div className="stat-card__value">{value}</div>
      {sub && <div className="stat-card__sub">{sub}</div>}
    </div>
  );
}

export function Badge({ tone = "default", children }) {
  return <span className={`badge tone-${tone}`}>{children}</span>;
}

export function Button({ children, variant = "primary", ...props }) {
  return (
    <button className={`btn btn--${variant}`} {...props}>
      {children}
    </button>
  );
}

export function Select({ value, onChange, options, ...props }) {
  return (
    <select className="select" value={value} onChange={onChange} {...props}>
      {options.map((o) => (
        <option key={o.value} value={o.value}>{o.label}</option>
      ))}
    </select>
  );
}
