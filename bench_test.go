package main

import (
	"fmt"
	"os"
	"testing"

	"uk.ac.bris.cs/gameoflife/constants"
	"uk.ac.bris.cs/gameoflife/gol"
)

const benchLength = constants.BenchLength

func BenchmarkStudentVersion(b *testing.B) {
	os.Stdout = nil
	p := gol.Params{
		Turns:       benchLength,
		Threads:     1, // arbituary value as not dictating amount of threads in the benchmark
		ImageWidth:  constants.GridSize,
		ImageHeight: constants.GridSize,
	}
	name := fmt.Sprintf("%dx%dx%d-%d", p.ImageWidth, p.ImageHeight, p.Turns, p.Threads)
	b.Run(name, func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			events := make(chan gol.Event)
			go gol.Run(p, events, nil)
			for range events {
			}
		}
	})
}
