package tests

import (
	"csa/conway/gol"
	"fmt"
	"os"
	"testing"
)

const benchLength = 100

func BenchmarkGol(b *testing.B) {
    // tests := []gol.Params{
	// 	{ImageWidth: 16, ImageHeight: 16},
	// 	{ImageWidth: 64, ImageHeight: 64},
	// 	{ImageWidth: 512, ImageHeight: 512},
	// }
    threads := []int{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}
    for _, thread := range threads{
        os.Stdout = nil // Disable all program output apart from benchmark results
        p := gol.Params{
            Turns:       benchLength,
            Threads:     thread,
            ImageWidth:  512,
            ImageHeight: 512,
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
}