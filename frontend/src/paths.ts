// Server paths are native: a Windows host returns backslash-separated paths.
// Build derived paths with the separator the path already uses.

function lastSeparatorIndex(path: string): number {
  return Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
}

/** `name` inside directory `dir`, e.g. a file picked in the browser. */
export function joinPath(dir: string, name: string): string {
  const cut = lastSeparatorIndex(dir);
  const separator = cut >= 0 ? dir[cut] : "/";
  return dir.endsWith(separator) ? `${dir}${name}` : `${dir}${separator}${name}`;
}

/** `name` in the same directory as `file`, e.g. masks.json next to an upload. */
export function siblingPath(file: string, name: string): string {
  const cut = lastSeparatorIndex(file);
  return cut >= 0 ? `${file.slice(0, cut + 1)}${name}` : name;
}
