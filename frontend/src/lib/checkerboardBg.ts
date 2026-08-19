import type { CSSProperties } from "react";

// Shared cross-hatch background for image preview stages (inference compare,
// validation lightbox). Lets transparency and dark patches read correctly
// against the flat app background.
export const CHECKERBOARD_BG: CSSProperties = {
  backgroundImage:
    "repeating-linear-gradient(45deg, var(--bg2) 0, var(--bg2) 4px, transparent 0, transparent 50%), " +
    "repeating-linear-gradient(-45deg, var(--bg2) 0, var(--bg2) 4px, transparent 0, transparent 50%)",
  backgroundSize: "12px 12px",
  backgroundColor: "var(--bg1)",
};
