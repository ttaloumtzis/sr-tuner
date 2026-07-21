import { CSSProperties } from "react";

interface IconProps {
  size?: number;
  color?: string;
  style?: CSSProperties;
  strokeWidth?: number;
}

const base = (size: number) => ({
  width: size,
  height: size,
  viewBox: "0 0 24 24",
  fill: "none" as const,
});

export function IconCheck({ size = 12, color = "currentColor", style, strokeWidth = 2.5 }: IconProps) {
  return (
    <svg {...base(size)} style={style}>
      <path d="M20 6L9 17l-5-5" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function IconChevron({ size = 12, color = "currentColor", style, strokeWidth = 2 }: IconProps) {
  return (
    <svg {...base(size)} style={style}>
      <path d="M6 9l6 6 6-6" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function IconInfo({ size = 12, color = "currentColor", style, strokeWidth = 2 }: IconProps) {
  return (
    <svg {...base(size)} style={style}>
      <circle cx="12" cy="12" r="9" stroke={color} strokeWidth={strokeWidth} />
      <path d="M12 11v5.5" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" />
      <circle cx="12" cy="7.75" r="1" fill={color} />
    </svg>
  );
}

export function IconAlert({ size = 12, color = "currentColor", style, strokeWidth = 2 }: IconProps) {
  return (
    <svg {...base(size)} style={style}>
      <path d="M12 3.5l9.5 16.5H2.5L12 3.5z" stroke={color} strokeWidth={strokeWidth} strokeLinejoin="round" />
      <path d="M12 10v4.5" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" />
      <circle cx="12" cy="17.25" r="1" fill={color} />
    </svg>
  );
}

export function IconCpu({ size = 12, color = "currentColor", style, strokeWidth = 1.8 }: IconProps) {
  return (
    <svg {...base(size)} style={style}>
      <rect x="7" y="7" width="10" height="10" rx="1.5" stroke={color} strokeWidth={strokeWidth} />
      <rect x="10" y="10" width="4" height="4" stroke={color} strokeWidth={strokeWidth} />
      <path d="M9 3v2.5M15 3v2.5M9 18.5V21M15 18.5V21M3 9h2.5M3 15h2.5M18.5 9H21M18.5 15H21" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" />
    </svg>
  );
}

export function IconDatabase({ size = 12, color = "currentColor", style, strokeWidth = 1.8 }: IconProps) {
  return (
    <svg {...base(size)} style={style}>
      <ellipse cx="12" cy="6" rx="7.5" ry="3" stroke={color} strokeWidth={strokeWidth} />
      <path d="M4.5 6v12c0 1.66 3.36 3 7.5 3s7.5-1.34 7.5-3V6" stroke={color} strokeWidth={strokeWidth} />
      <path d="M4.5 12c0 1.66 3.36 3 7.5 3s7.5-1.34 7.5-3" stroke={color} strokeWidth={strokeWidth} />
    </svg>
  );
}

export function IconSliders({ size = 12, color = "currentColor", style, strokeWidth = 1.8 }: IconProps) {
  return (
    <svg {...base(size)} style={style}>
      <path d="M4 7h9M17 7h3M4 17h3M11 17h9" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" />
      <circle cx="13" cy="7" r="2.3" stroke={color} strokeWidth={strokeWidth} />
      <circle cx="9" cy="17" r="2.3" stroke={color} strokeWidth={strokeWidth} />
    </svg>
  );
}

export function IconSettings({ size = 12, color = "currentColor", style, strokeWidth = 1.8 }: IconProps) {
  return (
    <svg {...base(size)} style={style}>
      <circle cx="12" cy="12" r="3" stroke={color} strokeWidth={strokeWidth} />
      <path
        d="M19.4 13.5a1.7 1.7 0 000-3l-.9-.2a6.9 6.9 0 00-.7-1.7l.5-.8a1.7 1.7 0 00-2.4-2.4l-.8.5a6.9 6.9 0 00-1.7-.7l-.2-.9a1.7 1.7 0 00-3 0l-.2.9a6.9 6.9 0 00-1.7.7l-.8-.5a1.7 1.7 0 00-2.4 2.4l.5.8a6.9 6.9 0 00-.7 1.7l-.9.2a1.7 1.7 0 000 3l.9.2a6.9 6.9 0 00.7 1.7l-.5.8a1.7 1.7 0 002.4 2.4l.8-.5a6.9 6.9 0 001.7.7l.2.9a1.7 1.7 0 003 0l.2-.9a6.9 6.9 0 001.7-.7l.8.5a1.7 1.7 0 002.4-2.4l-.5-.8a6.9 6.9 0 00.7-1.7l.9-.2z"
        stroke={color} strokeWidth={strokeWidth * 0.85} strokeLinejoin="round"
      />
    </svg>
  );
}

export function IconRocket({ size = 14, color = "currentColor", style, strokeWidth = 1.8 }: IconProps) {
  return (
    <svg {...base(size)} style={style}>
      <path d="M12 2.5c3 1.7 5 5 5 9 0 2-1 4-1 4l-4 4-4-4s-1-2-1-4c0-4 2-7.3 5-9z" stroke={color} strokeWidth={strokeWidth} strokeLinejoin="round" />
      <circle cx="12" cy="9.5" r="1.8" stroke={color} strokeWidth={strokeWidth} />
      <path d="M8.5 16.5L6 19.5M15.5 16.5L18 19.5M9.5 19.5l-1 2M14.5 19.5l1 2" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" />
    </svg>
  );
}

export function IconLayers({ size = 12, color = "currentColor", style, strokeWidth = 1.8 }: IconProps) {
  return (
    <svg {...base(size)} style={style}>
      <path d="M12 3l9 4.5-9 4.5-9-4.5L12 3z" stroke={color} strokeWidth={strokeWidth} strokeLinejoin="round" />
      <path d="M3 12l9 4.5 9-4.5M3 16.5L12 21l9-4.5" stroke={color} strokeWidth={strokeWidth} strokeLinejoin="round" />
    </svg>
  );
}

export function IconRewind({ size = 11, color = "currentColor", style, strokeWidth = 1.8 }: IconProps) {
  return (
    <svg {...base(size)} style={style}>
      <path d="M20 6.5v11L12 12l8-5.5zM12 6.5v11L4 12l8-5.5z" stroke={color} strokeWidth={strokeWidth} strokeLinejoin="round" />
    </svg>
  );
}
