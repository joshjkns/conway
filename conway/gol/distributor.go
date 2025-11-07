package gol

import (
	"csa/conway/util"
	"csa/stubs"
	"net/rpc"
	"strconv"
	"time"
)

type Distributor struct{}

type distributorChannels struct {
	events     chan<- Event
	ioCommand  chan<- ioCommand
	ioIdle     <-chan bool
	ioFilename chan<- string
	ioOutput   chan<- uint8
	ioInput    <-chan uint8
	keyPresses <-chan rune
}

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

func distributor(p Params, c distributorChannels) {
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

	// // args
	// port := flag.String("port", ":8020", "Port to listen on.")
	// flag.Parse()

	// // listen on port
	// listener, err := net.Listen("tcp", *port)
	// if err != nil {
	// 	panic(err)
	// }
	// fmt.Println("[Distributor] - Listening on port ", *port)
	// defer listener.Close()
	// go rpc.Accept(listener)

	// // making new Distributor
	// d := &Distributor{}
	// rpc.Register(d)

	// dial the broker
	broker, err := rpc.Dial("tcp", "localhost:8029")
	if err != nil {
		panic(err)
	}
	// fmt.Println("[Distributor] - Dialed broker successfully.")
	defer broker.Close()

	// begin logic
	c.events <- CellsFlipped{Cells: flipped, CompletedTurns: 0}
	c.events <- StateChange{CompletedTurns: 0, NewState: Executing}

	// rpc call args
	args := stubs.WorldInfo{World: world, Width: p.ImageWidth, Height: p.ImageHeight, Turns: p.Turns, CurrentTurns: 1}
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
			var tickerReply []util.Cell
			broker.Call("Broker.Consoldidate", true, &tickerReply)

		case kp := <- c.keyPresses:
			switch kp {
			}
		case <- done:
			c.events <- FinalTurnComplete{CompletedTurns: p.Turns, Alive: stubs.GetAliveCells(&reply.World)}

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