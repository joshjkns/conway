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

type Pair struct {
	startRow int
	endRow   int
}

var offsets = [8][2]int{
	{-1, -1}, {-1, 0}, {-1, 1},
	{0, -1}, {0, 1},
	{1, -1}, {1, 0}, {1, 1},
}

var mu sync.RWMutex

func incrementGol(world, result *[][]byte, p *Params, startRow, endRow int, flipped *[]util.Cell, flippedMu *sync.Mutex) {
	var tempFlipped []util.Cell
	for y := startRow + 1; y <= endRow + 1; y++ {
		for x := 1; x <= (*p).ImageWidth; x++ {
			var newState byte
			oldState := (*world)[y][x]
			liveNeighbours := countLiveNeighbours(*world, x, y)
			if oldState == 255 { // current cell is alive
				if liveNeighbours < 2 || liveNeighbours > 3 {
					newState = 0
				} else {
					newState= 255
				}
			} else { // current cell is dead
				if liveNeighbours == 3 {
					newState = 255
				} else {
					newState = 0
				}
			}
			(*result)[y][x] = newState

			if newState != oldState {
				tempFlipped = append(tempFlipped, util.Cell{X: x-1, Y: y-1})
			}
		}
	}
	flippedMu.Lock()
	*flipped = append(*flipped, tempFlipped...)
	flippedMu.Unlock()
}

func createWorld(width, height int) [][]byte {
	newWorld := make([][]byte, height)
	for i := range newWorld {
		newWorld[i] = make([]byte, width)
	}
	return newWorld
}

// func constrainValue(value int, constraint int) int {
// 	if value < 0 {
// 		value += constraint
// 	} else if value >= constraint {
// 		value -= constraint
// 	}
// 	return value
// }

func addHalo(world [][]byte, width, height int) {
    // top and bottom
    copy(world[0], world[height])
    copy(world[height+1], world[1])
    
    // left and right
    for y := 0; y <= height+1; y++ {
        world[y][0] = world[y][width]
        world[y][width+1] = world[y][1]
    }
    
    // corners
    world[0][0] = world[height][width]
    world[0][width+1] = world[height][1]
    world[height+1][0] = world[1][width]
    world[height+1][width+1] = world[1][1]
}

func countLiveNeighbours(world [][]byte, x int, y int) int {
	count := 0

	for _, offset := range offsets {
		xNeighbour := x + offset[0]
    yNeighbour := y + offset[1]
		// xNeighbour = constrainValue(xNeighbour, (*p).ImageWidth)
		// yNeighbour = constrainValue(yNeighbour, (*p).ImageHeight)

		if (world)[yNeighbour][xNeighbour] == 255 { // alive
			count += 1
		}
	}
	return count
}

func getAliveCells(world *[][]byte, p *Params) []util.Cell {
	mu.RLock()
	defer mu.RUnlock()
	var alive []util.Cell
	for y := 1; y <= (*p).ImageHeight; y++ {
		for x := 1; x <= (*p).ImageWidth; x++ {
			if (*world)[y][x] == 255 {
				cell := util.Cell{X: x-1, Y: y-1}
				alive = append(alive, cell)
			}
		}
	}
	return alive
}

func worker(world, result *[][]byte, p *Params, jobs <-chan Pair, wg *sync.WaitGroup, flipped *[]util.Cell, flippedMu *sync.Mutex) {
	for j := range jobs {
			incrementGol(world, result, p, j.startRow, j.endRow, flipped, flippedMu)
			wg.Done()
	}
}
func pgmImage(p *Params, world *[][]byte, c *distributorChannels, turns *int) {
	mu.RLock()
	defer mu.RUnlock()
	(*c).ioCommand <- ioOutput
	filename := strconv.Itoa((*p).ImageWidth) + "x" + strconv.Itoa((*p).ImageHeight) + "x" + strconv.Itoa(*turns)
	(*c).ioFilename <- filename

	for y := 1; y <= (*p).ImageHeight; y++ {
		for x := 1; x <= (*p).ImageWidth; x++ {
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
	// +2 for the halos l+r, t+b
	world := createWorld(p.ImageWidth + 2, p.ImageHeight + 2)
	result := createWorld(p.ImageWidth + 2, p.ImageHeight + 2) 
	var flipped []util.Cell

	for y := 1; y <= p.ImageHeight; y++ {
		for x := 1; x <= p.ImageWidth; x++ {
			world[y][x] = <-c.ioInput
			if world[y][x] == 255 {
				flipped = append(flipped, util.Cell{X: x-1, Y: y-1})
			}
		}
	}

	c.ioCommand <- ioCheckIdle
	<-c.ioIdle

	turn := 0
	c.events <- CellsFlipped{Cells: flipped, CompletedTurns: turn}
	c.events <- StateChange{CompletedTurns: turn, NewState: Executing}

	jobs := make(chan Pair, p.Threads)
	var wg sync.WaitGroup

	var tempFlipped []util.Cell
	var flippedMu sync.Mutex
	for i := 0; i < p.Threads; i++ {
		go worker(&world, &result, &p, jobs, &wg, &tempFlipped, &flippedMu)
	}

	//create a ticker to track time
	ticker := time.NewTicker(2 * time.Second)
	done := make(chan bool)
	paused := make(chan bool)
	resumed := make(chan bool)
	quit := make(chan bool)

	go func() {
		for {
			select {
					case <- done:
						return
					case <-ticker.C:
						mu.RLock()
						currentTurn := turn
						mu.RUnlock()
						c.events <- AliveCellsCount{CellsCount: len(getAliveCells(&world, &p)), CompletedTurns: currentTurn}
					case kp := <-c.keyPresses:
						switch kp {
						case 's':
							{
								mu.RLock()
								currentTurn := turn
								mu.RUnlock()
								pgmImage(&p, &world, &c, &currentTurn)
							}
						case 'q':
							{
								mu.RLock()
								currentTurn := turn
								mu.RUnlock()
								c.events <- FinalTurnComplete{CompletedTurns: currentTurn, Alive: getAliveCells(&world, &p)}
								pgmImage(&p, &world, &c, &currentTurn)
								c.events <- StateChange{CompletedTurns: currentTurn, NewState: Quitting}
								quit <- true
        				return
							}
						case 'p':
							{
								mu.RLock()
								currentTurn := turn
								mu.RUnlock()
								c.events <- StateChange{CompletedTurns: currentTurn, NewState: Paused}
								paused <- true
								currentTurn++
								for {
									kp := <-c.keyPresses
									if kp == 'p' {
										c.events <- StateChange{CompletedTurns: currentTurn, NewState: Executing}
										resumed <- true
										break
									}
									if kp == 's' {
										pgmImage(&p, &world, &c, &currentTurn)
									}
									if kp == 'q' {
										c.events <- FinalTurnComplete{CompletedTurns: currentTurn, Alive: getAliveCells(&world, &p)}
										pgmImage(&p, &world, &c, &currentTurn)
										c.events <- StateChange{CompletedTurns: currentTurn, NewState: Quitting}
										quit <- true
										return
									}
								}
							}
						default:
						}
					}
			}
	}()

	for i := 1; i <= p.Turns; i++ {
			select {
			case <-paused:
				select {
				case <- resumed:
					break
				case <- quit:
					return
				}
			case <- quit:
					return
			default:
			}
			addHalo(world, p.ImageWidth, p.ImageHeight)

			chunkHeight := p.ImageHeight / p.Threads
			for j := 0; j < p.Threads; j++ {
				startRow := j * chunkHeight
				endRow := startRow + chunkHeight - 1
				if j == p.Threads-1 {
					endRow = p.ImageHeight - 1
				}
				wg.Add(1)
				jobs <- Pair{startRow, endRow}
			}

			wg.Wait()

			mu.Lock()
			world, result = result, world
			turn++
			mu.Unlock()

			flippedMu.Lock()
			c.events <- CellsFlipped{Cells: tempFlipped, CompletedTurns: i}
			tempFlipped = make([]util.Cell, 0)
			flippedMu.Unlock()
			
			c.events <- TurnComplete{CompletedTurns: i}
	}

	close(jobs)
	defer ticker.Stop()
	close(done)

	c.events <- FinalTurnComplete{CompletedTurns: turn, Alive: getAliveCells(&world, &p)}

	c.ioCommand <- ioCheckIdle
	<-c.ioIdle

	pgmImage(&p, &world, &c, &turn)

	c.events <- StateChange{turn, Quitting}

	close(c.events)
}
