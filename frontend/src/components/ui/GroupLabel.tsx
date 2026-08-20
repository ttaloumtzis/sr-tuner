import type { ReactNode } from "react";

export function GroupLabel({ children }: { children: ReactNode }) {
  return (
    <div style={{
      fontSize: 9.5, color: "var(--dim)", textTransform: "uppercase", letterSpacing: "0.08em",
      fontFamily: "var(--font-sans)", fontWeight: 600, marginBottom: 5,
    }}>
      {children}
    </div>
  );
}