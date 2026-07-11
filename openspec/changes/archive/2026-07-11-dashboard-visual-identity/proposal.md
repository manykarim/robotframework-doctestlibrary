# Proposal: dashboard-visual-identity

## Why

The dashboard works but looks like an unstyled prototype: ad-hoc hex colors sprinkled across the stylesheet, no dark mode (reviewers comparing bright document pages benefit from dark chrome), and dead-end blank screens when no data exists yet (fresh install shows an empty table; the mask editor without a document shows only toolbars).

## What Changes

- **Design tokens**: all colors/radii/shadows move to CSS custom properties on `:root`; component rules reference tokens only. One place defines the visual language.
- **Dark mode**: a `data-theme="dark"` token override; initial theme from `prefers-color-scheme`, user toggle in the top bar persisted in `localStorage`. Document renderings stay on light surfaces (pages are white paper); only the chrome switches.
- **Empty states**: a fresh dashboard greets with an onboarding card (how to ingest: path, folder upload, CLI); the mask editor without a document points at Browse/Upload instead of rendering dead toolbars.
- **Consistency/a11y**: visible `:focus-visible` outlines; consistent radius/shadow scale.

## Capabilities

### Modified Capabilities

- `dashboard-review`: visual identity requirements (tokens, theme, empty states).

## Impact

`frontend/src/styles.css` (token layer + dark overrides), `frontend/src/App.tsx` (theme toggle, runs empty state), `frontend/src/MaskEditor.tsx` (empty state). e2e journey J14. No backend changes.
