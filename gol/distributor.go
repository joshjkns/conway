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

type Set struct {
    mu   sync.RWMutex
    data map[util.Cell]struct{}
}

func NewSet() *Set {
    return &Set{
        data: make(map[util.Cell]struct{}),
    }
}

func (s *Set) Add(cell util.Cell) {
    s.mu.Lock()
    defer s.mu.Unlock()
    s.data[cell] = struct{}{}
}

func (s *Set) Remove(cell util.Cell) {
    s.mu.Lock()
    defer s.mu.Unlock()
    delete(s.data, cell)
}

func (s *Set) Contains(cell util.Cell) bool {
    s.mu.RLock()
    defer s.mu.RUnlock()
    _, exists := s.data[cell]
    return exists
}

func (s *Set) Size() int {
    s.mu.RLock()
    defer s.mu.RUnlock()
    return len(s.data)
}

func (s *Set) ToList() []util.Cell {
    s.mu.RLock()
    defer s.mu.RUnlock()
    
    list := make([]util.Cell, 0, len(s.data))
    for cell := range s.data {
        list = append(list, cell)
    }
    return list
}

var offsets = [8][2]int{
	{-1, -1}, {-1, 0}, {-1, 1},
	{0, -1}, {0, 1},
	{1, -1}, {1, 0}, {1, 1},
}

func incrementGol(world, result *Set, width, height, startRow, endRow int) {
	for cell := range world.data {
		aliveNeighbours := newCountAliveNeighbours(world, cell)
		if aliveNeighbours == 2 || aliveNeighbours == 3 {
			result.Add(cell)
		}
		for _, offset := range offsets {
			new_x := cell.X + offset[0]
			new_y := cell.Y + offset[1]
			new_x = constrainValue(new_x, width)
			new_y = constrainValue(new_y, height)
			new_cell := util.Cell{X: new_x, Y:new_y}
			neighbourAliveNeighbours := newCountAliveNeighbours(world, new_cell)
			if neighbourAliveNeighbours == 2 || neighbourAliveNeighbours == 3 {
				result.Add(cell)
			}
		}
	}
}

func newCountAliveNeighbours(world *Set, cell util.Cell) int {
	total := 0
	for _, offset := range offsets {
		new_x := cell.X + offset[0]
		new_y := cell.Y + offset[1]
		new_cell := util.Cell{X: new_x, Y: new_y}
		if world.Contains(new_cell) {
			total += 1 
		}
	}
	return total
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

func countLiveNeighbours(world [][]byte, x int, y int, p *Params) int {
	count := 0

	for _, offset := range offsets {
		xNeighbour := x + offset[0]
		yNeighbour := y + offset[1]
		xNeighbour = constrainValue(xNeighbour, (*p).ImageWidth)
		yNeighbour = constrainValue(yNeighbour, (*p).ImageHeight)

		if (world)[yNeighbour][xNeighbour] == 255 { // alive
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

func worker(world, result *Set, p *Params, jobs <-chan Pair, wg *sync.WaitGroup) {
	for j := range jobs {
		func() {
			incrementGol(world, result, p.ImageWidth, p.ImageHeight, j.startRow, j.endRow)
			defer wg.Done()
		}()

	}
}

func pgmImage(p *Params, world *Set, c *distributorChannels, turns *int) {
	(*c).ioCommand <- ioOutput
	filename := strconv.Itoa((*p).ImageWidth) + "x" + strconv.Itoa((*p).ImageHeight) + "x" + strconv.Itoa(*turns)
	(*c).ioFilename <- filename

	for y := 0; y < (*p).ImageHeight; y++ {
		for x := 0; x < (*p).ImageWidth; x++ {
			// (*c).ioOutput <- (*world)[y][x]
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
	// world := createWorld(p.ImageWidth, p.ImageHeight)
	// result := createWorld(p.ImageWidth, p.ImageHeight)
	flipped := NewSet()
	result := NewSet()

	for y := 0; y < p.ImageHeight; y++ {
		for x := 0; x < p.ImageWidth; x++ {
			b := <-c.ioInput
			if b == 255 {
				flipped.Add(util.Cell{X:x, Y:y})
			}
		}
	}

	c.ioCommand <- ioCheckIdle
	<-c.ioIdle

	turn := 0
	c.events <- CellsFlipped{Cells: flipped.ToList(), CompletedTurns: turn}
	c.events <- StateChange{CompletedTurns: turn, NewState: Executing}

	jobs := make(chan Pair, p.ImageHeight)
	var wg sync.WaitGroup

	for i := 0; i < p.Threads; i++ {
		go worker(flipped, result, &p, jobs, &wg)
	}

	//create a ticker to track time
	ticker := time.NewTicker(2 * time.Second)
	//tickerStop := make(chan bool)

	for i := 1; i <= p.Turns; i++ {
		select {
		case <-ticker.C:
			c.events <- AliveCellsCount{CellsCount: flipped.Size(), CompletedTurns: turn}
		case kp := <-c.keyPresses:
			switch kp {
			case 's':
				{
					// pgmImage(&p, &result, &c, &turn)
				}
			case 'q':
				{
					c.events <- FinalTurnComplete{CompletedTurns: turn, Alive: flipped.ToList()}
					// pgmImage(&p, &world, &c, &turn)
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
							// pgmImage(&p, &world, &c, &turn)
						}
						if kp == 'q' {
							c.events <- FinalTurnComplete{CompletedTurns: turn, Alive: flipped.ToList()}
							// pgmImage(&p, &world, &c, &turn)
							c.events <- StateChange{CompletedTurns: turn, NewState: Quitting}
							return
						}
					}
					c.events <- StateChange{CompletedTurns: turn, NewState: Executing}
				}
			}
		default:
			chunkHeight := flipped.Size() / p.Threads
			for j := 0; j < p.Threads; j++ {
				startRow := j * chunkHeight
				endRow := startRow + chunkHeight - 1
				if j == p.Threads-1 {
					endRow = p.ImageHeight - 1
				}
				wg.Add(1)
				jobs <- Pair{startRow, endRow}
			}

			c.events <- CellsFlipped{Cells: flipped.ToList(), CompletedTurns: i}
			c.events <- TurnComplete{CompletedTurns: i}

			turn++
		}
	}
	close(jobs)
	defer ticker.Stop()

	c.events <- FinalTurnComplete{CompletedTurns: turn, Alive: flipped.ToList()}

	c.ioCommand <- ioCheckIdle
	<-c.ioIdle

	// pgmImage(&p, &world, &c, &turn)

	c.events <- StateChange{turn, Quitting}

	close(c.events)
}
