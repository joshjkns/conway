package main

import (
	"fmt"
	"net"
	"net/rpc"
	"sync"
)

type BrokerComp struct {
	workers      map[Data]*rpc.Client
	mu           sync.Mutex
	cond         sync.Cond
	currentWorld [][]byte
	turns        int
	paused       bool
	quit         bool
}

type Data struct {
	Address string
	Id      int
}

type Input struct {
	World  [][]byte
	Width  int
	Height int
	Turns  int
}

type WorkerInput struct {
	Read   [][]byte
	Width  int
	StartY int
	EndY   int
}

type WorkerOutput struct {
	Write    [][]byte
	StartRow int
	EndRow   int
}

type Output struct {
	World [][]byte
	Turns int
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

func addRow(world [][]byte, row, left, right int) []byte {
	var rowStore []byte
	rowStore = append(rowStore, world[row][left])
	rowStore = append(rowStore, world[row]...)
	rowStore = append(rowStore, world[row][right])
	return rowStore
}

func createChunk(world [][]byte, width, height, startY, endY int) [][]byte {
	var outputWorld [][]byte
	topBlock := constrainValue(startY-1, height)
	bottomBlock := constrainValue(endY+1, height)
	leftBlock := width - 1
	rightBlock := 0
	//fmt.Println(width, height, startY, endY)
	outputWorld = append(outputWorld, addRow(world, topBlock, leftBlock, rightBlock))
	for i := startY; i <= endY; i++ {
		outputWorld = append(outputWorld, addRow(world, i, leftBlock, rightBlock))
	}
	outputWorld = append(outputWorld, addRow(world, bottomBlock, leftBlock, rightBlock))
	fmt.Println(width, height, startY, endY, len(outputWorld), len(outputWorld[0]))
	return outputWorld
}

func (b *BrokerComp) TogglePaused(args bool, reply *Output) error {
	// toggle paused in every worker
	b.paused = args
	b.cond.Broadcast()
	return nil
}

func (b *BrokerComp) GetCurrentState(args *Input, reply *Output) error {
	// get state of all individual workers, reconstruct and return to the distributor
	reply.World = b.currentWorld
	reply.Turns = b.turns
	return nil
}

func (b *BrokerComp) QuitProgram(args *Input, reply *Output) error {
	b.quit = true
	return nil
}

func (b *BrokerComp) Process(args *Input, reply *Output) error {
	// Assign work to the workers and rpc call it with client.Go(DistributingComp.Process)
	workers := len(b.workers)
	var channels = make([]chan *rpc.Call, workers) // making done channels
	var argArray = make([]WorkerInput, workers)
	var replyArray = make([]WorkerOutput, workers)
	world := args.World

	for k := 1; k <= args.Turns; k++ {
		if b.quit {
			return nil
		}

		b.mu.Lock()
		if b.paused {
			for b.paused {
				b.cond.Wait()
			}
		}
		b.mu.Unlock()

		rowsPerWorker := args.Height / workers
		count := 0

		for data := range b.workers {
			i := data.Id
			channels[i] = make(chan *rpc.Call, 10)

			StartY := i * rowsPerWorker
			EndY := StartY + rowsPerWorker - 1
			if count == workers-1 {
				EndY = args.Height - 1
			}

			fmt.Println(StartY, EndY)
			argArray[i] = WorkerInput{Read: createChunk(world, args.Width, args.Height, StartY, EndY), Width: args.Width, StartY: StartY, EndY: EndY}
			replyArray[i] = WorkerOutput{}
			count++
		}

		for data, worker := range b.workers {
			go worker.Go("WorkerComp.GameOfLife", argArray[data.Id], &replyArray[data.Id], channels[data.Id])
		}

		// check all channels are done
		for _, call := range channels {
			ch := call
			<-ch
		}

		for data, _ := range b.workers {
			out := replyArray[data.Id]
			startRow := out.StartRow
			endRow := out.EndRow
			chunkWorld := out.Write
			fmt.Println("length:  ", len(chunkWorld), len(chunkWorld[0]))
			fmt.Println(startRow, endRow)
			for i := startRow; i <= endRow; i++ {
				//fmt.Println(i)
				for j := 0; j < len(chunkWorld[i-startRow]); j++ {
					world[i][j] = chunkWorld[i-startRow][j]
				}
			}
		}
		b.currentWorld = world
		b.turns = k
	}
	reply.World = world // done all turns
	reply.Turns = b.turns
	b.turns = 0
	b.currentWorld = nil
	return nil
}

func (b *BrokerComp) Register(args *Data, reply *Output) error {
	client, err := rpc.Dial("tcp", args.Address)
	if err != nil {
		return err
	}

	data := Data{Address: args.Address, Id: args.Id}

	b.mu.Lock()
	b.workers[*args] = client
	b.mu.Unlock()

	fmt.Println("registered worker", data.Address, data.Id)
	return nil
}

func (b *BrokerComp) Unregister(args *Data, reply *Output) error {
	// unregister the distributor into the broker
	b.mu.Lock()
	defer b.mu.Unlock()

	client, ok := b.workers[*args]
	if ok {
		client.Close()
		delete(b.workers, *args)
		fmt.Println("unregistered worker", args.Address)
	} else {
		fmt.Println("no worker found", args.Address)
	}
	return nil
}

func main() {
	b := &BrokerComp{workers: make(map[Data]*rpc.Client)}
	b.cond = *sync.NewCond(&b.mu)

	err := rpc.Register(b)
	if err != nil {
		panic(err)
	}

	listener, err := net.Listen("tcp", ":8031")
	if err != nil {
		panic(err)
	}
	defer listener.Close()

	fmt.Println("[Broker] RPC connected on port 8031")
	go rpc.Accept(listener)
	for {
		if b.quit {
			fmt.Println(b.workers)
			// send rpc to all workers saying quit
			//for _, worker := range b.workers {
			//	var reply Output
			//	worker.Call("WorkerComp.QuitWorker", true, &reply)
			//	if err != nil {
			//		print("ERROR: ", err)
			//	}
			//}
			return
		}
	}
}
