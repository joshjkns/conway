package gol

import (
	"strconv"
	"sync"
	"time"

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

func incrementGol(world, result *[][]byte, p *Params, row int) {
	for x := 0; x < (*p).ImageWidth; x++ {
		y := row
		liveNeighbours := countLiveNeighbours(world, x, y, p)
		if (*world)[x][y] == 255 { // current cell is alive
			if liveNeighbours < 2 || liveNeighbours > 3 {
				(*result)[x][y] = 0
			} else {
				(*result)[x][y] = 255
			}
		} else { // current cell is dead
			if liveNeighbours == 3 {
				(*result)[x][y] = 255
			} else {
				(*result)[x][y] = 0
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

func worker(world, result *[][]byte, p *Params, jobs <-chan int, wg *sync.WaitGroup) {
	for j := range jobs {
		incrementGol(world, result, p, j)
		wg.Done()
	}
}

func startTicker(ticker *time.Ticker, events chan<- Event, tickerStop <-chan bool, world *[][]byte, p *Params, turns *int) {
	for {
		select {
		case <-ticker.C:
			events <- AliveCellsCount{CellsCount: len(getAliveCells(world, p)), CompletedTurns: *turns}
		case <-tickerStop:
			return
		}
	}
}

// distributor divides the work between workers and interacts with other goroutines.
func distributor(p Params, c distributorChannels) {
	c.ioCommand <- ioInput // give us the world in bytes
	c.ioFilename <- strconv.Itoa(p.ImageWidth) + "x" + strconv.Itoa(p.ImageHeight)

	world := createWorld(p.ImageWidth, p.ImageHeight)
	result := createWorld(p.ImageWidth, p.ImageHeight)
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

	for i := 0; i < p.Threads; i++ {
		go worker(&world, &result, &p, jobs, &wg)
	}

	// create a ticker to track time
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()
	tickerStop := make(chan bool)

	go func() {
		for {
			select {
			case <-ticker.C:
				c.events <- AliveCellsCount{CellsCount: len(getAliveCells(&world, &p)), CompletedTurns: turn}
			case <-tickerStop:
				return
			}
		}
	}()

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
	tickerStop <- true
	c.events <- FinalTurnComplete{CompletedTurns: turn, Alive: getAliveCells(&world, &p)}

	c.ioCommand <- ioCheckIdle
	<-c.ioIdle

	c.events <- StateChange{turn, Quitting}

	c.ioCommand <- ioOutput
	c.ioFilename <- strconv.Itoa(p.ImageWidth) + "x" + strconv.Itoa(p.ImageHeight) + "x" + strconv.Itoa(p.Turns)

	for y := 0; y < p.ImageHeight; y++ {
		for x := 0; x < p.ImageWidth; x++ {
			c.ioOutput <- world[x][y]
		}
	}

	c.ioCommand <- ioCheckIdle
	<-c.ioIdle

	close(c.events)
}
