package main

import (
	"csa/conway/util"
	"csa/stubs"
	"flag"
	"fmt"
	"net"
	"net/rpc"
	"sync"
)

type Broker struct{
	workers map[stubs.Data]*rpc.Client
	currentID	int
	currentWorld [][]byte
	currentTurns int
	mu      sync.Mutex
  cond    *sync.Cond
}

func (b *Broker) Register(args stubs.WorkerInfo, reply *stubs.Confirmation) (err error) {
	address := "localhost" + args.Port
	worker, err := rpc.Dial("tcp", address)
	if err != nil {
		return err
	}
	data := stubs.Data{Address: address, ID: b.currentID}
	b.workers[data] = worker

	reply.ID = b.currentID
	fmt.Println("[Broker] - Registered worker ID: ", reply.ID, "on port ", args.Port)
	b.currentID += 1
	return nil
}

func (b *Broker) GameOfLife(args, reply *stubs.WorldInfo) (err error) {
	workers := len(b.workers)
	
	// making arrays to sync the workers
	var channels = make([]chan *rpc.Call, workers)
	var argArray = make([]stubs.ChunkInfo, workers)
	var replyArray = make([]stubs.ChunkInfo, workers)

	// set the world
	var world = args.World
	b.currentWorld = stubs.CopyWorld(&world, args.Width, args.Height)

	// split the data
	rowsPerWorker := args.Height / workers
	
	// add all addresses(data) to an array
	var dataArray = make([]stubs.Data, workers)
	for data := range b.workers {
		dataArray[data.ID] = data
	}

	// create all params for the call
	for data := range b.workers {
		i := data.ID

		channels[i] = make(chan *rpc.Call, 1)

		startY := i * rowsPerWorker
		endY := startY + rowsPerWorker - 1
		if i == workers - 1 {
			endY = args.Height - 1
		}
		fmt.Println("STARTY ", startY, " ENDY ", endY, "HEIGHT ", (endY - startY + 1), " ROWS PER WORKER ", rowsPerWorker, " ARGS HEIGHT ", args.Height)

		// make neighbours array and add left and right neighbours
		leftNeighbour := stubs.ConstrainValue(data.ID - 1, workers)
		rightNeighbour := stubs.ConstrainValue(data.ID + 1, workers)
		neighbours := stubs.NeighbourPair{LeftNeighbour: dataArray[leftNeighbour], RightNeighbour: dataArray[rightNeighbour]}

		argArray[i] = stubs.ChunkInfo{Chunk: stubs.CreateChunk(world, args.Width, (endY - startY + 1), startY, endY), StartRow: startY, EndRow: endY, Neighbours: neighbours, Turns: args.Turns}
		replyArray[i] = stubs.ChunkInfo{}
	}

	// call function on all workers (they work together and do ALL moves before returning)
	// - have to send neighbours addresses or rpc client stuff so they can call each other.
	for data, worker := range b.workers {
		go worker.Go("Worker.GameOfLife", argArray[data.ID], &replyArray[data.ID], channels[data.ID])
	}

	// check all channels are done
	for _, call := range channels {
		ch := call
		<-ch
	}

	// put all chunks back together
	for data := range b.workers {
		reply := replyArray[data.ID]
		startRow := reply.StartRow
		endRow := reply.EndRow
		chunkWorld := reply.Chunk

		for i := startRow; i <= endRow; i++ {
			for j := 0; j < args.Width; j++ {
				world[i][j] = chunkWorld[i-startRow][j]
				b.currentWorld[i][j] = world[i][j]
			}
		}
	}
	reply.World = world
	reply.Turns = args.Turns
	b.currentTurns = 1
	b.currentWorld = nil
	return nil
}

// need to stop all workers, get all their return states and then getalivecells of it and reply with it
func (b *Broker) Consoldidate(args bool, reply *[]util.Cell) (err error) {
	return nil
}

func main() {
	// making new broker
	b := &Broker{workers: make(map[stubs.Data]*rpc.Client), currentID: 0}
	b.cond = sync.NewCond(&b.mu)
	rpc.Register(b)

	// args
	port := flag.String("port", ":8029", "Port to listen on")
	flag.Parse()

	// listen
	listener, err := net.Listen("tcp", *port)
	if err != nil {
		panic(err)
	}
	fmt.Println("[Broker] - Listening on port ", *port)
	defer listener.Close()

	for {
		rpc.Accept(listener)
	}
}
