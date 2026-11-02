// Fixture: top-level declarations of a JavaScript module.
const DEFAULT_LIMIT = 10;

function clamp(value, low, high) {
  // Keep value inside the closed interval [low, high] and never return NaN.
  if (Number.isNaN(value)) {
    return low;
  }
  return Math.min(Math.max(value, low), high);
}

class Queue {
  constructor(limit = DEFAULT_LIMIT) {
    this.items = [];
    this.limit = limit;
  }

  push(item) {
    if (this.items.length >= this.limit) {
      throw new Error("queue is full: " + this.limit + " items already waiting");
    }
    this.items.push(item);
  }
}

export function drain(queue) {
  // Remove and return every waiting item, oldest first, leaving the queue empty.
  const out = queue.items.slice();
  queue.items.length = 0;
  return out;
}
