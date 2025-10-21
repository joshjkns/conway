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
	keyPresses <-chan rune
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

func worker(world, result *[][]byte, p *Params, jobs <-chan int, wg *sync.WaitGroup) {
	for j := range jobs {
		incrementGol(world, result, p, j)
		wg.Done()
	}
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

// distributor divides the work between workers and interacts with other goroutines.
func distributor(p Params, c distributorChannels) {
	c.ioCommand <- ioInput // give us the world in bytes
	c.ioFilename <- strconv.Itoa(p.ImageWidth) + "x" + strconv.Itoa(p.ImageHeight)
	world := createWorld(p.ImageWidth, p.ImageHeight)
	result := createWorld(p.ImageWidth, p.ImageHeight)
	var flipped []util.Cell

	for y := 0; y < p.ImageHeight; y++ {
		for x := 0; x < p.ImageWidth; x++ {
			world[y][x] = <-c.ioInput
			if world[y][x] == 255 {
				flipped = append(flipped, util.Cell{X: x, Y: y})
			}
		}
	}

	c.ioCommand <- ioCheckIdle
	<-c.ioIdle

	turn := 0
	c.events <- CellsFlipped{Cells: flipped, CompletedTurns: turn}
	c.events <- StateChange{CompletedTurns: turn, NewState: Executing}

	jobs := make(chan int, p.ImageHeight)
	var wg sync.WaitGroup

	for i := 0; i < p.Threads; i++ {
		go worker(&world, &result, &p, jobs, &wg)
	}

	//create a ticker to track time
	ticker := time.NewTicker(2 * time.Second)
	//tickerStop := make(chan bool)

	for i := 1; i <= p.Turns; i++ {
		select {
		case <-ticker.C:
			c.events <- AliveCellsCount{CellsCount: len(getAliveCells(&world, &p)), CompletedTurns: turn}
		case kp := <-c.keyPresses:
			switch kp {
			case 's':
				{
					pgmImage(&p, &world, &c, &turn)
				}
			case 'q':
				{
					c.events <- FinalTurnComplete{CompletedTurns: turn, Alive: getAliveCells(&world, &p)}
					pgmImage(&p, &world, &c, &turn)
					c.events <- StateChange{CompletedTurns: turn, NewState: Quitting}
					return
				}
			case 'p':
				{
					c.events <- StateChange{CompletedTurns: turn, NewState: Paused}
					for {
						kp := <-c.keyPresses
						if kp == 'p' {
							break
						}
						if kp == 's' {
							pgmImage(&p, &world, &c, &turn)
						}
						if kp == 'q' {
							c.events <- FinalTurnComplete{CompletedTurns: turn, Alive: getAliveCells(&world, &p)}
							pgmImage(&p, &world, &c, &turn)
							c.events <- StateChange{CompletedTurns: turn, NewState: Quitting}
							return
						}
					}
					c.events <- StateChange{CompletedTurns: turn, NewState: Executing}
				}
			}
		default:
			for j := 0; j < p.ImageHeight; j++ {
				wg.Add(1)
				jobs <- j
			}

			wg.Wait()
			var tempFlipped []util.Cell
			for y := 0; y < p.ImageHeight; y++ {
				for x := 0; x < p.ImageWidth; x++ {
					if world[y][x] != result[y][x] {
						tempFlipped = append(tempFlipped, util.Cell{X: x, Y: y})
					}
					world[y][x] = result[y][x]
				}
			}
			c.events <- CellsFlipped{Cells: tempFlipped, CompletedTurns: i}
			c.events <- TurnComplete{CompletedTurns: i}
			turn++

		}
	}
	close(jobs)
	defer ticker.Stop()

	c.events <- FinalTurnComplete{CompletedTurns: turn, Alive: getAliveCells(&world, &p)}

	c.ioCommand <- ioCheckIdle
	<-c.ioIdle

	pgmImage(&p, &world, &c, &turn)

	c.events <- StateChange{turn, Quitting}

	close(c.events)
}
