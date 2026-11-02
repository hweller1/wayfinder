package fixture;

import java.util.ArrayList;
import java.util.List;

/** Collects log lines and hands them back in insertion order, bounded by a fixed capacity. */
public class Journal {
    private final List<String> lines = new ArrayList<>();
    private final int capacity;

    public Journal(int capacity) {
        this.capacity = capacity;
    }

    public void append(String line) {
        if (lines.size() == capacity) {
            lines.remove(0);
        }
        lines.add(line);
    }

    public List<String> snapshot() {
        return new ArrayList<>(lines);
    }
}

/** Marker for anything that can be flushed to durable storage after a batch of writes. */
interface Flushable {
    void flush() throws java.io.IOException;

    default boolean isDirty() {
        return true;
    }
}
