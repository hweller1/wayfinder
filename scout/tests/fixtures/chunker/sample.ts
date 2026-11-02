// Fixture: top-level declarations of a TypeScript module.
export interface Shape {
  area(): number;
  name: string;
  // Perimeter is optional because some shapes in this fixture cannot compute it.
  perimeter?(): number;
}

export type Dimension = { width: number; height: number; label: string; unit: "cm" | "mm" | "in" };

export class Rect implements Shape {
  name = "rect";
  constructor(private dim: Dimension) {}

  area(): number {
    return this.dim.width * this.dim.height;
  }

  perimeter(): number {
    return 2 * (this.dim.width + this.dim.height);
  }
}

export function largest(shapes: Shape[]): Shape | undefined {
  // Pick the shape with the greatest area; an empty list has no largest shape.
  return shapes.reduce<Shape | undefined>((best, s) => (!best || s.area() > best.area() ? s : best), undefined);
}
