package main

import (
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
	distributor *rpc.Client
	paused bool
	quit bool
	pausedMu sync.Mutex
  pausedCond *sync.Cond
	disconnect bool
}

func (b *Broker) RegisterDistributor(args stubs.DistributorInfo, reply *stubs.Response) (err error) {
	if b.currentWorld != nil {
		reply.Resp = true
	} else {
		reply.Resp = false
	}
	b.distributor, _ = rpc.Dial("tcp", args.Address)
	b.disconnect = false
	return nil
}

func (b *Broker) UnregisterDistributor(args bool, reply *stubs.Response) (err error) {
	b.distributor = nil
	b.disconnect = true
	return nil
}

func (b *Broker) Register(args stubs.WorkerInfo, reply *stubs.Confirmation) (err error) {
	address := args.Address
	worker, err := rpc.Dial("tcp", address)
	if err != nil {
		return err
	}
	data := stubs.Data{Address: address, ID: b.currentID}
	b.workers[data] = worker

	reply.ID = b.currentID
	fmt.Println("[Broker] - Registered worker ID: ", reply.ID, "on port ", args.Address)
	b.currentID += 1
	return nil
}

func (b *Broker) TogglePause(args bool, reply *stubs.Response) (err error) {
	reply.Resp = b.paused
	b.paused = !b.paused
	b.pausedCond.Broadcast()
	return nil
}

func (b *Broker) Consolidate(args bool, reply *stubs.WorldInfo) (err error) {
	reply.CurrentTurns = b.currentTurns - 1
	reply.World = b.currentWorld
	reply.Paused = b.paused
	return nil
}

func (b *Broker) QuitProgram(args bool, reply *stubs.Response) error {
	b.quit = true
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

	oldTurns := b.currentTurns

	for k := oldTurns; k <= args.Turns; k++ {
		if b.quit || b.disconnect {
			b.currentWorld = world
			return nil
		}

		b.pausedMu.Lock()
		if b.paused {
			for b.paused {
				b.pausedCond.Wait()
			}

			if b.quit || b.disconnect {
				b.currentWorld = world
				b.paused = false
				return nil
			}
		}
		b.pausedMu.Unlock()

		// create all params for the call
		for data := range b.workers {
			i := data.ID

			channels[i] = make(chan *rpc.Call, 1)

			startY := i * rowsPerWorker
			endY := startY + rowsPerWorker - 1
			if i == workers - 1 {
				endY = args.Height - 1
			}

			chunk := world[startY:endY+1]
			haloedChunk := stubs.AddHalos(b.currentWorld, chunk, args.Width, (endY - startY + 1), startY, endY)

			argArray[i] = stubs.ChunkInfo{Chunk: haloedChunk, StartRow: startY, EndRow: endY, Threads: args.Threads}
			replyArray[i] = stubs.ChunkInfo{}
		}

		// call function on all workers (they work together and do ALL moves before returning)
		// - have to send neighbours addresses or rpc client stuff so they can call each other.
		for data, worker := range b.workers {
			worker.Go("Worker.GameOfLife", argArray[data.ID], &replyArray[data.ID], channels[data.ID])
		}

		// check all channels are done
		for _, call := range channels {
			ch := call
			<-ch
		}

		// put all chunks back together
		// var flipped []util.Cell

		for data := range b.workers {
			reply := replyArray[data.ID]
			startRow := reply.StartRow
			endRow := reply.EndRow
			chunkWorld := reply.Chunk
			
			for i := startRow; i <= endRow; i++ {
				for j := 0; j < args.Width; j++ {
					world[i][j] = chunkWorld[i-startRow][j]
					// if world[i][j] != b.currentWorld[i][j] {
					// 	flipped = append(flipped, util.Cell{X:j, Y:i})
					// }
					b.currentWorld[i][j] = world[i][j]
				}
			}
		}
		// var cellsFlippedData stubs.CellsFlippedData
		// cellsFlippedData.CellsFlipped = flipped
		// cellsFlippedData.CompletedTurns = b.currentTurns
		// var flipResponse stubs.Response
		// if !b.disconnect {
		// 		b.distributor.Call("Distributor.Flip", cellsFlippedData, &flipResponse)
		// }
		b.currentTurns += 1
	}
	
	reply.World = world
	reply.Turns = args.Turns
	b.currentTurns = 1
	b.currentWorld = nil
	return nil
}

func main() {
	// making new broker
	b := &Broker{workers: make(map[stubs.Data]*rpc.Client), currentID: 0, currentTurns: 1}
	b.cond = sync.NewCond(&b.mu)
	b.pausedCond = sync.NewCond(&b.pausedMu)

	// due to firewall issue
	// b.workers[stubs.Data{ID: 1, Address: "34.237.53.207:8031"}], err = rpc.Dial("tcp", "34.237.53.207:8031")
	// if err != nil {
	// 	panic(err)
	// }
	// b.workers[stubs.Data{ID: 3, Address: "3.236.240.172:8032"}], _ = rpc.Dial("tcp", "3.236.240.172:8032")
	// b.workers[stubs.Data{ID: 4, Address: "35.174.61.28:8033"}], _ = rpc.Dial("tcp", "35.174.61.28:8033")
	rpc.Register(b)

	// args
	port := flag.String("port", "localhost:8035", "(PRIVATE) Port to listen on")
	// distributorIP := flag.String("distributor", "localhost:8029", "Distributor's IP")

	flag.Parse()

	// listen
	listener, err := net.Listen("tcp", *port)
	if err != nil {
		panic(err)
	}
	fmt.Println("[Broker] - Listening on port ", *port)
	defer listener.Close()

	go rpc.Accept(listener)

	for {
		if b.quit {
			var reply stubs.Response
			for _, worker := range b.workers {
				worker.Call("Worker.Quit", true, &reply)
			}
			return
		}
	}
}
