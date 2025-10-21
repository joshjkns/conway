package gol

import (
	"strconv"
	"sync"
	"uk.ac.bris.cs/gameoflife/util"
)

var world [][]byte
var result [][]byte

type distributorChannels struct {
	events     chan<- Event
	ioCommand  chan<- ioCommand
	ioIdle     <-chan bool
	ioFilename chan<- string
	ioOutput   chan<- uint8
	ioInput    <-chan uint8
}

func incrementGol(p *Params, row int) {
	for x := 0; x < (*p).ImageWidth; x++ {
		y := row
		liveNeighbours := countLiveNeighbours(x, y, p)
		if world[x][y] == 255 { // current cell is alive
			if liveNeighbours < 2 || liveNeighbours > 3 {
				result[x][y] = 0
			} else {
				result[x][y] = 255
			}
		} else { // current cell is dead
			if liveNeighbours == 3 {
				result[x][y] = 255
			} else {
				result[x][y] = 0
			}
		}
	}
}

func createWorld(width, height int) [][]byte {
	newWorld := make([][]byte, height)
	for i := range newWorld {
		newWorld[i] = make([]byte, width)
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

func countLiveNeighbours(x int, y int, p *Params) int {
	count := 0
	offsets := [][]int{{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}}
	for _, offset := range offsets {
		xNeighbour := x + offset[0]
		yNeighbour := y + offset[1]
		xNeighbour = constrainValue(xNeighbour, (*p).ImageWidth)
		yNeighbour = constrainValue(yNeighbour, (*p).ImageHeight)

		if world[xNeighbour][yNeighbour] == 255 { // alive
			count += 1
		}
	}
	return count
}

func getAliveCells(p *Params) []util.Cell {
	var alive []util.Cell
	for x := 0; x < (*p).ImageWidth; x++ {
		for y := 0; y < (*p).ImageHeight; y++ {
			if world[x][y] == 255 {
				cell := util.Cell{X: x, Y: y}
				alive = append(alive, cell)
			}
		}
	}
	return alive
}

func worker(p *Params, jobs <-chan int, wg *sync.WaitGroup) {
	for j := range jobs {
		incrementGol(p, j)
		wg.Done()
	}
}

// distributor divides the work between workers and interacts with other goroutines.
func distributor(p Params, c distributorChannels) {
	c.ioCommand <- ioInput // give us the world in bytes
	c.ioFilename <- strconv.Itoa(p.ImageWidth) + "x" + strconv.Itoa(p.ImageHeight)

	world = createWorld(p.ImageWidth, p.ImageHeight)
	result = createWorld(p.ImageWidth, p.ImageHeight)
	for y := 0; y < p.ImageHeight; y++ {
		for x := 0; x < p.ImageWidth; x++ {
			world[x][y] = <-c.ioInput
		}
	}

	c.ioCommand <- ioCheckIdle
	<-c.ioIdle

	numberOfTurns := p.Turns

	turn := 0
	c.events <- StateChange{CompletedTurns: turn, NewState: Executing}
	turn++ // turn is now 1

	jobs := make(chan int, p.ImageHeight)
	var wg sync.WaitGroup

	// TODO: Report the final state using FinalTurnCompleteEvent.
	for i := 0; i < p.Threads; i++ {
		go worker(&p, jobs, &wg)
	}

	for i := 1; i <= numberOfTurns; i++ {
		for j := 0; j < p.ImageHeight; j++ {
			wg.Add(1)
			jobs <- j
		}
		wg.Wait()
		for y := 0; y < p.ImageHeight; y++ {
			for x := 0; x < p.ImageWidth; x++ {
				world[x][y] = result[x][y]
			}
		}
		c.events <- StateChange{CompletedTurns: i, NewState: Executing}
		turn++
	}
	close(jobs)
	c.events <- FinalTurnComplete{CompletedTurns: turn, Alive: getAliveCells(&p)}

	// Make sure that the Io has finished any output before exiting.
	c.ioCommand <- ioCheckIdle
	<-c.ioIdle

	c.events <- StateChange{turn, Quitting}

	// Close the channel to stop the SDL goroutine gracefully. Removing may cause deadlock.
	close(c.events)
}
