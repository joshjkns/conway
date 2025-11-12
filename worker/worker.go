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

func increment(chunk, result *[][]byte, width, height, startRow, endRow int) {
	for y := startRow; y <= endRow; y++ { // only checks actual part
		for x := 0; x < width; x++ {
			liveNeighbours := countLiveNeighbours(chunk, x, y, width, len(*chunk))
			if (*chunk)[y][x] == 255 { // current cell is alive
				if liveNeighbours < 2 || liveNeighbours > 3 {
					(*result)[y-1][x] = 0
				} else {
					(*result)[y-1][x] = 255
				}
			} else { // current cell is dead
				if liveNeighbours == 3 {
					(*result)[y-1][x] = 255
				} else {
					(*result)[y-1][x] = 0
				}
			}
		}
	}
}

func (w *Worker) Quit(args bool, reply *stubs.Response) (err error) {
	w.quit = true
	return nil
}

func worker(current, result *[][]byte, width, height int, jobs <-chan stubs.Pair, wg *sync.WaitGroup) {
	for j := range jobs {
			increment(current, result, width, height, j.StartRow, j.EndRow)
			wg.Done()
	}
}

func (w *Worker) GameOfLife(args stubs.ChunkInfo, reply *stubs.ChunkInfo) (err error) {
	w.width = len(args.Chunk[0])
	w.height = len((args.Chunk))
	w.startRow = args.StartRow
	w.endRow = args.EndRow

	res := stubs.CreateWorld(w.width, w.height - 2)

	jobs := make(chan stubs.Pair, w.height)
	var wg sync.WaitGroup

	for i := 0; i < args.Threads; i++ {
		go worker(&args.Chunk, &res, w.width, w.height, jobs, &wg)
	}

	chunkHeight := (w.height - 2) / args.Threads
	for j := 0; j < args.Threads; j++ {
		startRow := j * chunkHeight + 1
		endRow := startRow + chunkHeight - 1
		if j == args.Threads-1 {
			endRow = w.height - 2
		}
		wg.Add(1)
		jobs <- stubs.Pair{StartRow: startRow, EndRow: endRow}
	}

	wg.Wait()
	
	reply.Chunk = res
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