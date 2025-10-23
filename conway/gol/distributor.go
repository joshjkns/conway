package gol

import (
	"fmt"
	"net/rpc"
	"strconv"
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

type Input struct {
	World  [][]byte
	Height int
	Width  int
	Turns  int
}

type Output struct {
	World [][]byte
	Turns int
}

var paused bool

func createWorld(p *Params) [][]byte {
	newWorld := make([][]byte, (*p).ImageHeight)
	for i := range newWorld {
		newWorld[i] = make([]byte, (*p).ImageWidth)
	}
	return newWorld
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
	world := createWorld(&p)
	for y := 0; y < p.ImageHeight; y++ {
		for x := 0; x < p.ImageWidth; x++ {
			world[y][x] = <-c.ioInput
		}
	}
	c.ioCommand <- ioCheckIdle
	<-c.ioIdle

	turn := 0
	c.events <- StateChange{CompletedTurns: turn, NewState: Executing}
	turn++ // turn is now 1

	client, err := rpc.Dial("tcp", "localhost:8030")
	defer client.Close()
	if err != nil {
		panic(err)
	}
	var id = time.Now().String()
	err = client.Call("DistributingComp.Register", id, nil)
	if err != nil {
		panic(err)
	}

	args := Input{world, p.ImageHeight, p.ImageWidth, p.Turns}
	var reply Output

	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()
	done := make(chan *rpc.Call, 1)
	go client.Go("DistributingComp.Process", args, &reply, done)

	for {
		select {
		case <-ticker.C:
			var empty Input
			var alive Output
			if !paused {
				err := client.Call("DistributingComp.GetCurrentState", empty, &alive)
				if err != nil {
					panic(err)
				}
				aliveCells := getAliveCells(&alive.World, &p)
				c.events <- AliveCellsCount{CompletedTurns: alive.Turns, CellsCount: len(aliveCells)}
			}
		case kp := <-c.keyPresses:
			switch kp {
			case 'q':
				var empty Input
				var alive Output
				err := client.Call("DistributingComp.GetCurrentState", empty, &alive)
				err = client.Call("DistributingComp.Unregister", id, nil)
				if err != nil {
					panic(err)
				}
				c.events <- StateChange{CompletedTurns: alive.Turns, NewState: Quitting}
				return
			case 's':
				var empty Input
				var alive Output
				err := client.Call("DistributingComp.GetCurrentState", empty, &alive)
				if err != nil {
					panic(err)
				}
				pgmImage(&p, &alive.World, &c, &alive.Turns)
			case 'k':
				var empty Input
				var alive Output
				err := client.Call("DistributingComp.GetCurrentState", empty, &alive)
				if err != nil {
					panic(err)
				}
				c.events <- FinalTurnComplete{CompletedTurns: alive.Turns, Alive: getAliveCells(&alive.World, &p)}
				pgmImage(&p, &alive.World, &c, &alive.Turns)
				c.events <- StateChange{CompletedTurns: alive.Turns, NewState: Quitting}
				_ = client.Call("DistributingComp.QuitProgram", empty, &alive)
				return
			case 'p':
				var empty Input
				var alive Output

				if paused {
					err = client.Call("DistributingComp.GetCurrentState", empty, &alive)
					_ = client.Call("DistributingComp.TogglePaused", false, &alive)
					c.events <- StateChange{CompletedTurns: alive.Turns, NewState: Executing}
					paused = false
				} else {
					err = client.Call("DistributingComp.TogglePaused", true, &alive)
					err = client.Call("DistributingComp.GetCurrentState", empty, &alive)

					c.events <- StateChange{CompletedTurns: alive.Turns, NewState: Paused}
					fmt.Println(alive.Turns)
					paused = true
				}
			}
		case <-done:
			if err != nil {
				panic(err)
			}

			c.events <- FinalTurnComplete{CompletedTurns: reply.Turns, Alive: getAliveCells(&reply.World, &p)}

			pgmImage(&p, &reply.World, &c, &reply.Turns)

			c.ioCommand <- ioCheckIdle
			<-c.ioIdle
			c.events <- StateChange{reply.Turns, Quitting}

			close(c.events)
			return
		}
	}
}
