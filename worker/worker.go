package main

import (
	"csa/stubs"
	"flag"
	"fmt"
	"net"
	"net/rpc"
	"sync"
)

type Worker struct{
	id int
	broker *rpc.Client
	chunk [][]byte
	width int
	height int
	startRow int
	endRow int
	mu sync.RWMutex
	quit bool
}

var offsets = [][]int{{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}}

func countLiveNeighbours(world *[][]byte, x, y, width, height int) int {
	count := 0

	for _, offset := range offsets {
		xNeighbour := x + offset[0]
		yNeighbour := y + offset[1]
		xNeighbour = stubs.ConstrainValue(xNeighbour, width)
		yNeighbour = stubs.ConstrainValue(yNeighbour, height)
		if (*world)[yNeighbour][xNeighbour] == 255 { // alive
			count += 1
		}
	}
	return count
}

func increment(chunk *[][]byte, width, height int) [][]byte {
	newWorld := stubs.CreateWorld(width, height) // actual part we are updating
	for y := 1; y < len(*chunk)-1; y++ { // only checks actual part
		for x := 0; x < width; x++ {
			liveNeighbours := countLiveNeighbours(chunk, x, y, width, len(*chunk))
			if (*chunk)[y][x] == 255 { // current cell is alive
				if liveNeighbours < 2 || liveNeighbours > 3 {
					newWorld[y-1][x] = 0
				} else {
					newWorld[y-1][x] = 255
				}
			} else { // current cell is dead
				if liveNeighbours == 3 {
					newWorld[y-1][x] = 255
				} else {
					newWorld[y-1][x] = 0
				}
			}
		}
	}
	return newWorld
}

func (w *Worker) Quit(args bool, reply *stubs.Response) (err error) {
	w.quit = true
	return nil
}

func (w *Worker) GameOfLife(args stubs.ChunkInfo, reply *stubs.ChunkInfo) (err error) {
	w.chunk = stubs.CopyWorld(&args.Chunk, len(args.Chunk[0]), len(args.Chunk))
	w.width = len(w.chunk[0])
	w.height = len((w.chunk))
	w.startRow = args.StartRow
	w.endRow = args.EndRow

	current := stubs.CopyWorld(&args.Chunk, w.width, w.height)

	w.mu.Lock()
	w.chunk = stubs.CopyWorld(&current, w.width, w.height)
	w.mu.Unlock()

	newChunk := increment(&current, w.width, len(current))

	current = newChunk
	
	reply.Chunk = current
	reply.StartRow = w.startRow
	reply.EndRow = w.endRow
	return nil
}

func main() {
	// making new Worker
	w := &Worker{}
	rpc.Register(w)

	// args
	ip := flag.String("ip", "localhost:8030", "IP to listen on.")
	flag.Parse()

	// listen
	listener, err := net.Listen("tcp", *ip)
	if err != nil {
		panic(err)
	}
	defer listener.Close()
	fmt.Println("[Worker] - Listening on port ", *ip)

	// dial the broker
	broker, err := rpc.Dial("tcp", "localhost:8029")
	if err != nil {
		panic(err)
	}
	w.broker = broker
	fmt.Println("[Worker] - Dialed broker successfully.")

	// define args and reply for the registration
	args := stubs.WorkerInfo{Address: *ip}
	reply := stubs.Confirmation{}

	// register the worker with the broker (ids will be sorted on the broker side)
	broker.Call("Broker.Register", args, &reply)

	// set the workers id locally
	w.id = reply.ID

	go rpc.Accept(listener)

	for {
		if w.quit {
			return
		}
	}
}