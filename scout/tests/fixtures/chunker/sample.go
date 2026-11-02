package fixture

import "fmt"

// Counter tracks how many times each key was seen and reports totals on demand.
type Counter struct {
	counts map[string]int
	total  int
}

// NewCounter builds an empty counter that is ready to use right away.
func NewCounter() *Counter {
	return &Counter{counts: make(map[string]int)}
}

// Add records one more sighting of key and keeps the running total in step.
func (c *Counter) Add(key string) {
	c.counts[key]++
	c.total++
}

// Report prints every key with its count followed by the grand total line.
func (c *Counter) Report() {
	for key, n := range c.counts {
		fmt.Printf("%s=%d\n", key, n)
	}
	fmt.Printf("total=%d\n", c.total)
}
