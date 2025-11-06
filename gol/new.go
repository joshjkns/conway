package gol

/*
#cgo CFLAGS: -I.
#cgo LDFLAGS:
#include "simd.h"
*/
import "C"

import (
	"strconv"

	"uk.ac.bris.cs/gameoflife/util"
)

type distributorChannels struct {
	events     chan<- Event
	ioCommand  chan<- ioCommand
	ioIdle     <-chan bool
	ioFilename chan<- string
	ioOutput   chan<- uint8
	ioInput    <-chan uint8
	keyPresses <-chan rune
}

func pgmImage(p *Params, world *[][]byte, c *distributorChannels, turns *int) {
	(*c).ioCommand <- ioOutput
	filename := strconv.Itoa((*p).ImageWidth) + "x" + strconv.Itoa((*p).ImageHeight) + "x" + strconv.Itoa(*turns)
	(*c).ioFilename <- filename

	for y := 0; y < (*p).ImageHeight; y++ {
		for x := 0; x < (*p).ImageWidth; x++ {
			(*c).ioOutput <- (*world)[y][x]
		}
	}

	(*c).ioCommand <- ioCheckIdle
	<-(*c).ioIdle
	(*c).events <- ImageOutputComplete{Filename: filename, CompletedTurns: *turns}
}

func getAliveCells(world *[][]byte, p *Params) []util.Cell {
	var alive []util.Cell
	for y := 0; y < (*p).ImageHeight; y++ {
		for x := 0; x < (*p).ImageWidth; x++ {
			if (*world)[y][x] == 255 {
				cell := util.Cell{X: x, Y: y}
				alive = append(alive, cell)
			}
		}
	}
	return alive
}

func createWorld(width, height int) [][]byte {
	newWorld := make([][]byte, height)
	for i := range newWorld {
		newWorld[i] = make([]byte, width)
	}
	return newWorld
}

// distributor divides the work between workers and interacts with other goroutines.
func distributor(p Params, c distributorChannels) {
	c.ioCommand <- ioInput // give us the world in bytes
	c.ioFilename <- strconv.Itoa(p.ImageWidth) + "x" + strconv.Itoa(p.ImageHeight)
	world := createWorld(p.ImageWidth, p.ImageHeight)

	for y := 0; y < p.ImageHeight; y++ {
		for x := 0; x < p.ImageWidth; x++ {
			world[y][x] = <-c.ioInput
		}
	}

	c.ioCommand <- ioCheckIdle
	<-c.ioIdle

	// do C stuff here
	flat :

	c.events <- FinalTurnComplete{CompletedTurns: p.Turns, Alive: getAliveCells(&world, &p)}

	c.ioCommand <- ioCheckIdle
	<-c.ioIdle

	pgmImage(&p, &world, &c, &p.Turns)

	c.events <- StateChange{p.Turns, Quitting}

	close(c.events)
}
