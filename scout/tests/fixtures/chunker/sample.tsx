// Fixture: top-level declarations of a TSX module.
import React from "react";

type Props = { title: string; items: string[]; onPick: (item: string) => void; selected?: string };

export function List({ title, items, onPick, selected }: Props) {
  // Render the title and one clickable row per item, highlighting the selected one.
  return (
    <section>
      <h2>{title}</h2>
      <ul>
        {items.map((item) => (
          <li key={item} onClick={() => onPick(item)} style={{ fontWeight: item === selected ? 700 : 400 }}>
            {item}
          </li>
        ))}
      </ul>
    </section>
  );
}

export class Boundary extends React.Component<{ children: React.ReactNode }, { failed: boolean }> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  render() {
    return this.state.failed ? <p>Something went wrong while rendering this part.</p> : this.props.children;
  }
}
