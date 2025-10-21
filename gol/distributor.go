package gol

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
}

func createWorld(p *Params) [][]byte {
	newWorld := make([][]byte, (*p).ImageHeight)
	for i := range newWorld {
		newWorld[i] = make([]byte, (*p).ImageWidth)
	}
	return newWorld
}

func incrementGol(world *[][]byte, p *Params) [][]byte {
	newWorld := createWorld(p)
	for x := 0; x < (*p).ImageWidth; x++ {
		for y := 0; y < (*p).ImageHeight; y++ {
			liveNeighbours := countLiveNeighbours(world, x, y, p)
			if (*world)[x][y] == 255 { // current cell is alive
				if liveNeighbours < 2 || liveNeighbours > 3 {
					newWorld[x][y] = 0
				} else {
					newWorld[x][y] = 255
				}
			} else { // current cell is dead
				if liveNeighbours == 3 {
					newWorld[x][y] = 255
				} else {
					newWorld[x][y] = 0
				}
			}
		}
	}
	return newWorld
}

func constrainValue(value int, constraint int) int {
	if value < 0 {
		value += constraint
	} else if value >= constraint {
		value -= constraint
	}
	return value
}

func countLiveNeighbours(world *[][]byte, x int, y int, p *Params) int {
	count := 0
	offsets := [][]int{{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}}
	for _, offset := range offsets {
		xNeighbour := x + offset[0]
		yNeighbour := y + offset[1]
		xNeighbour = constrainValue(xNeighbour, (*p).ImageWidth)
		yNeighbour = constrainValue(yNeighbour, (*p).ImageHeight)

		if (*world)[xNeighbour][yNeighbour] == 255 { // alive
			count += 1
		}
	}
	return count
}

func getAliveCells(world *[][]byte, p *Params) []util.Cell {
	var alive []util.Cell
	for x := 0; x < (*p).ImageWidth; x++ {
		for y := 0; y < (*p).ImageHeight; y++ {
			if (*world)[x][y] == 255 {
				cell := util.Cell{X: x, Y: y}
				alive = append(alive, cell)
			}
		}
	}
	return alive
}

// distributor divides the work between workers and interacts with other goroutines.
func distributor(p Params, c distributorChannels) {
	c.ioCommand <- ioInput // give us the world in bytes
	c.ioFilename <- strconv.Itoa(p.ImageWidth) + "x" + strconv.Itoa(p.ImageHeight)
	world := createWorld(&p)
	for y := 0; y < p.ImageHeight; y++ {
		for x := 0; x < p.ImageWidth; x++ {
			world[x][y] = <-c.ioInput
		}
	}
	c.ioCommand <- ioCheckIdle
	<-c.ioIdle

	// TODO: Create a 2D slice to store the world.
	numberOfTurns := p.Turns

	turn := 0
	c.events <- StateChange{CompletedTurns: turn, NewState: Executing}
	turn++ // turn is now 1

	// TODO: Execute all turns of the Game of Life.
	for i := 1; i <= numberOfTurns; i++ {
		world = incrementGol(&world, &p)
		c.events <- StateChange{CompletedTurns: i, NewState: Executing}
		turn++
	}

	// TODO: Report the final state using FinalTurnCompleteEvent.
	c.events <- FinalTurnComplete{CompletedTurns: turn, Alive: getAliveCells(&world, &p)}

	// Make sure that the Io has finished any output before exiting.
	c.ioCommand <- ioCheckIdle
	<-c.ioIdle

	c.events <- StateChange{turn, Quitting}

	// Close the channel to stop the SDL goroutine gracefully. Removing may cause deadlock.
	close(c.events)
}
