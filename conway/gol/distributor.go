package gol

import (
	"csa/conway/util"
	"csa/stubs"
	"fmt"
	"net"
	"net/rpc"
	"strconv"
	"time"
)

type Distributor struct{
	channels distributorChannels
}

type distributorChannels struct {
	events     chan<- Event
	ioCommand  chan<- ioCommand
	ioIdle     <-chan bool
	ioFilename chan<- string
	ioOutput   chan<- uint8
	ioInput    <-chan uint8
	keyPresses <-chan rune
}

var channels distributorChannels

func initialRead(world* [][]byte, flipped* []util.Cell, c distributorChannels, w, h int) {
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			(*world)[y][x] = <-c.ioInput
			if (*world)[y][x] == 255 {
				*flipped = append(*flipped, util.Cell{X: x, Y: y})
			}
		}
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

func (d *Distributor) Flip(args stubs.CellsFlippedData, reply *stubs.Response) (err error) {
	reply.Resp = true
	channels.events <- CellsFlipped{Cells: args.CellsFlipped, CompletedTurns: args.CompletedTurns}
	channels.events <- TurnComplete{CompletedTurns: args.CompletedTurns}
	return nil
}

func distributor(p Params, c distributorChannels) {
	var paused bool
	// Create the world and flipped cell slice
	world := stubs.CreateWorld(p.ImageWidth, p.ImageHeight)
	var flipped []util.Cell
	
	// Signal to the channels to send the data across c.ioInput
	c.ioCommand <- ioInput
	c.ioFilename <- strconv.Itoa(p.ImageWidth) + "x" + strconv.Itoa(p.ImageHeight)

	// Read in the world
	initialRead(&world, &flipped, c, p.ImageWidth, p.ImageHeight)

	// Check it is idle (done with sending)
	c.ioCommand <- ioCheckIdle
	<-c.ioIdle

	// args
	ipPort := "localhost:8020"

	// listen on port
	listener, err := net.Listen("tcp", ipPort)
	if err != nil {
		panic(err)
	}
	fmt.Println("[Distributor] - Listening on port ", ipPort)
	defer listener.Close()
	
	go rpc.Accept(listener)

	// making new Distributor
	d := &Distributor{}
	rpc.Register(d)

	channels = c

	// dial the broker
	broker, err := rpc.Dial("tcp", "localhost:8029")
	if err != nil {
		panic(err)
	}
	// fmt.Println("[Distributor] - Dialed broker successfully.")
	defer broker.Close()
	// response is if its been used before - true is yes there is a state, false is no there isnt a state
	var stateResponse stubs.Response
	broker.Call("Broker.RegisterDistributor", &stubs.DistributorInfo{Address: ipPort}, &stateResponse)

	// begin logic
	if !stateResponse.Resp {
		c.events <- CellsFlipped{Cells: flipped, CompletedTurns: 0}
	} else {
		var initialReply stubs.WorldInfo
		broker.Call("Broker.Consolidate", true, &initialReply)
		// turn := initialReply.CurrentTurns
		c.events <- CellsFlipped{Cells: stubs.GetAliveCells(&initialReply.World), CompletedTurns: initialReply.CurrentTurns}
		if initialReply.Paused {
			paused = true
			// c.events <- StateChange{CompletedTurns: turn, NewState: Paused}
		} else {
			paused = false
			// c.events <- StateChange{CompletedTurns: turn, NewState: Executing}
		}
	}
	c.events <- StateChange{CompletedTurns: 0, NewState: Executing}

	// rpc call args
	args := stubs.WorldInfo{World: world, Width: p.ImageWidth, Height: p.ImageHeight, Turns: p.Turns, CurrentTurns: 0, Threads: p.Threads}
	reply := stubs.WorldInfo{}

	// create ticker
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	// create done channel and call the rpc via a goroutine
	done := make(chan *rpc.Call, 1)
	go broker.Go("Broker.GameOfLife", args, &reply, done)

	for {
		select {
		case <- ticker.C:
			var tickerReply stubs.WorldInfo
			if !paused {
				broker.Call("Broker.Consolidate", true, &tickerReply)
				aliveCells := stubs.GetAliveCells(&tickerReply.World)
				c.events <- AliveCellsCount{CompletedTurns: tickerReply.CurrentTurns, CellsCount: len(aliveCells)}
			}
		case kp := <- c.keyPresses:
			switch kp {
			case 's': 
				var saveReply stubs.WorldInfo
				broker.Call("Broker.Consolidate", true, &saveReply)
				pgmImage(&p, &saveReply.World, &c, &saveReply.CurrentTurns)
			case 'k':
				var killReply stubs.WorldInfo
				broker.Call("Broker.Consolidate", true, &killReply)
				c.events <- FinalTurnComplete{CompletedTurns: killReply.Turns, Alive: stubs.GetAliveCells(&killReply.World)}
				pgmImage(&p, &killReply.World, &c, &killReply.CurrentTurns)
				c.events <- StateChange{CompletedTurns: killReply.CurrentTurns, NewState: Quitting}
				broker.Call("Broker.QuitProgram", true, &stubs.Response{})
				return
			case 'q':
				var quitReply stubs.WorldInfo
				broker.Call("Broker.Consolidate", true, &quitReply)
				pgmImage(&p, &quitReply.World, &c, &quitReply.CurrentTurns)
				c.events <- StateChange{CompletedTurns: quitReply.CurrentTurns, NewState: Quitting}
				broker.Call("Broker.UnregisterDistributor", true, &stubs.Response{})
				return
			case 'p':
				var pausedReply stubs.WorldInfo
				var isPaused stubs.Response
				broker.Call("Broker.Consolidate", true, &pausedReply)
				broker.Call("Broker.TogglePause", true, &isPaused)
				if isPaused.Resp { // paused
					paused = false
					c.events <- StateChange{CompletedTurns: pausedReply.CurrentTurns, NewState: Executing}
				} else {
					paused = true
					c.events <- StateChange{CompletedTurns: pausedReply.CurrentTurns, NewState: Paused}
				}
			}
		case <- done:
			c.events <- FinalTurnComplete{CompletedTurns: reply.Turns, Alive: stubs.GetAliveCells(&reply.World)}

			// save pgm image
			pgmImage(&p, &reply.World, &c, &reply.Turns)

			c.ioCommand <- ioCheckIdle
			<-c.ioIdle
			c.events <- StateChange{reply.Turns, Quitting}
			close(c.events)
			return
		}
	}
}