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
	neighbours stubs.NeighbourPair
	mu sync.RWMutex
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


func addHalo(chunk [][]byte, width, height int, pos stubs.Position, address string, id int) [][]byte {
	neighbour, err := rpc.Dial("tcp", address)
	if err != nil {
			panic(err)
	}
	defer neighbour.Close()

	var halo stubs.Halo
	if err := neighbour.Call("Worker.SendHalo", pos, &halo); err != nil {
			panic(err)
	}

	// create a new world with one extra row for the new halo
	res := stubs.CreateWorld(width, height+1)
	switch pos {
	case stubs.Top:
		copy(res[0], halo.Row)
		copy(res[1:], chunk)
	case stubs.Bottom:
		copy(res[:height], chunk)
		copy(res[height], halo.Row)
	}

	// if id == 1 && pos == stubs.Top {
	// 	fmt.Println("HERE: ",halo.Row)
	// } else {
	// 	fmt.Println("HERE: ", chunk[0])
	// }

	return res
}

func (w *Worker) SendHalo(pos stubs.Position, reply *stubs.Halo) error {
	w.mu.RLock()
	defer w.mu.RUnlock()

	if pos == stubs.Top {
		reply.Row = make([]byte, len(w.chunk[0]))
		copy(reply.Row, w.chunk[len(w.chunk)-1])
	} else {
		reply.Row = make([]byte, len(w.chunk[0]))
		copy(reply.Row, w.chunk[0])
	}
	return nil
}

// func syncWithNeighbour(args stubs.Data) {
// 	fmt.Println("SYNC")
// 	rightNeighbour, err := rpc.Dial("tcp", args.Address)
// 	if err != nil {
// 		panic(err)
// 	}
// 	defer rightNeighbour.Close()
// 	var resp stubs.Response
// 	rightNeighbour.Call("Worker.Ready", true, &resp)
// }

// func (w *Worker) Ready(args bool, reply *stubs.Response) (err error) {
// 	fmt.Println("READY")
// 	reply.Resp = true
// 	return nil
// }

func (w *Worker) GameOfLife(args stubs.ChunkInfo, reply *stubs.ChunkInfo) (err error) {
	w.chunk = stubs.CopyWorld(&args.Chunk, len(args.Chunk[0]), len(args.Chunk))
	w.width = len(w.chunk[0])
	w.height = len((w.chunk))
	w.startRow = args.StartRow
	w.endRow = args.EndRow
	w.neighbours = args.Neighbours

	current := stubs.CopyWorld(&args.Chunk, w.width, w.height)

	w.mu.Lock()
	w.chunk = stubs.CopyWorld(&current, w.width, w.height)
	w.mu.Unlock()

	// add halos from neighbours
	for i := 1; i <= args.Turns; i++ {
    // create a copy of the base chunk each turn

    // get top and bottom halos each turn
    withTop := addHalo(current, w.width, len(current), stubs.Top, w.neighbours.LeftNeighbour.Address, w.id)
    withBoth := addHalo(withTop, w.width, len(withTop), stubs.Bottom, w.neighbours.RightNeighbour.Address, w.id)

    newChunk := increment(&withBoth, w.width, len(current))

    current = newChunk

		var resp stubs.Response
    w.broker.Call("Broker.WaitForEveryone", stubs.WaitArgs{Turn: i, Chunk: stubs.ChunkInfo{Chunk: current}, Width: w.width, ID: w.id}, &resp)
		fmt.Println("DONE WAITING")

		w.mu.Lock()
		w.chunk = stubs.CopyWorld(&current, w.width, len(current))
		w.mu.Unlock()
	}
	
	reply.Chunk = current
	reply.StartRow = w.startRow
	reply.EndRow = w.endRow
	w.chunk = nil
	return nil
}

func main() {
	// making new Worker
	w := &Worker{}
	rpc.Register(w)

	// args
	port := flag.String("port", ":8030", "Port to listen on.")
	flag.Parse()

	// listen
	listener, err := net.Listen("tcp", *port)
	if err != nil {
		panic(err)
	}
	defer listener.Close()
	fmt.Println("[Worker] - Listening on port ", *port)

	// dial the broker
	broker, err := rpc.Dial("tcp", "localhost:8029")
	if err != nil {
		panic(err)
	}
	w.broker = broker
	fmt.Println("[Worker] - Dialed broker successfully.")

	// define args and reply for the registration
	args := stubs.WorkerInfo{Port: *port}
	reply := stubs.Confirmation{}

	// register the worker with the broker (ids will be sorted on the broker side)
	broker.Call("Broker.Register", args, &reply)

	// set the workers id locally
	w.id = reply.ID

	go rpc.Accept(listener)

	select{}
}