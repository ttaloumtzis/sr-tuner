type TagColor = "green" | "blue" | "amber" | "red" | "purple" | "cyan";

const COLOR_MAP: Record<TagColor, string> = {
  green:  "var(--green)",
  blue:   "var(--blue)",
  amber:  "var(--amber)",
  red:    "var(--red)",
  purple: "var(--purple)",
  cyan:   "var(--cyan)",
};

interface TagProps {
  color?: TagColor;
  children: React.ReactNode;
}

export function Tag({ color = "green", children }: TagProps) {
  const c = COLOR_MAP[color] ?? COLOR_MAP.green;
  return (
    <span
      style={{
        background: c + "22",
        color: c,
        border: `1px solid ${c}44`,
        borderRadius: "var(--radius-sm)",
        padding: "1px 6px",
        fontFamily: "var(--font-mono)",
        fontSize: 10,
        whiteSpace: "nowrap",
      }}
    >
      {children}
    </span>
  );
}
