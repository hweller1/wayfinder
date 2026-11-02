package main

import (
	"fmt"

	"example.com/repo/internal/session"
)

func main() {
	s := session.getUserSession("demo")
	fmt.Println(s)
}
