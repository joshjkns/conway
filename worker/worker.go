package main

import (
	"flag"
	"fmt"
	"net"
	"net/rpc"
)

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

type Data struct {
	Address string
	Id      int
}
type WorkerComp struct {
	quit bool
}

var offsets = [][]int{{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}}

//func stripHalo(world [][]byte) [][]byte {
//	height := len(world)
//	width := len(world[0])
//	out := make([][]byte, height-2)
//	for y := 1; y < height-1; y++ {
//		out = append(out, world[y][1:width-1])
//	}
//	return out
//}

func createWorldWorker(width, height int) [][]byte {
	newWorld := make([][]byte, height)
	for i := range newWorld {
		newWorld[i] = make([]byte, width)
	}
	return newWorld
}

func incrementGolWorker(section *[][]byte, width, height, startX, startY int) [][]byte {
	//fmt.Println(width, height, startX, startY, len(*section), len((*section)[0]))
	newWorld := createWorldWorker(width, height) // actual part we are updating
	//fmt.Println(len(*section), len((*section)[0]), width, height)
	for y := 1; y < len(*section)-1; y++ { // only checks actual part
		for x := 1; x < len((*section)[0])-1; x++ {
			liveNeighbours := countLiveNeighboursWorker(section, x, y, width, height)
			if (*section)[y][x] == 255 { // current cell is alive
				if liveNeighbours < 2 || liveNeighbours > 3 {
					newWorld[y-1][x-1] = 0
				} else {
					newWorld[y-1][x-1] = 255
				}
			} else { // current cell is dead
				if liveNeighbours == 3 {
					newWorld[y-1][x-1] = 255
				} else {
					newWorld[y-1][x-1] = 0
				}
			}
		}
	}
	return newWorld
}

func constrainValueWorker(value int, constraint int) int {
	if value < 0 {
		value += constraint
	} else if value >= constraint {
		value -= constraint
	}
	return value
}

func countLiveNeighboursWorker(world *[][]byte, x int, y int, width, height int) int {
	count := 0

	for _, offset := range offsets {
		xNeighbour := x + offset[0]
		yNeighbour := y + offset[1]
		//xNeighbour = constrainValueWorker(xNeighbour, width)
		//yNeighbour = constrainValueWorker(yNeighbour, height)
		//fmt.Println(x, y, width, height, xNeighbour, yNeighbour)
		if (*world)[yNeighbour][xNeighbour] == 255 { // alive
			count += 1
		}
	}
	return count
}

func (w *WorkerComp) GameOfLife(args *WorkerInput, reply *WorkerOutput) error {
	width := args.Width
	startRow := args.StartY
	endRow := args.EndY
	height := (endRow + 1) - startRow
	worldInput := (*args).Read

	newWorld := incrementGolWorker(&worldInput, width, height, 0, startRow)
	fmt.Println(len(newWorld))
	reply.Write = newWorld
	reply.StartRow = startRow
	reply.EndRow = endRow
	return nil
}

func (w *WorkerComp) QuitWorker(args bool, repl *WorkerOutput) error {
	w.quit = true
	return nil
}

func main() {
	w := new(WorkerComp)
	err := rpc.Register(w)
	if err != nil {
		panic(err)
	}

	var portAddr = flag.String("port", ":8030", "port to listen on")
	var id = flag.Int("id", 0, "id of worker")
	flag.Parse()

	listener, err := net.Listen("tcp", *portAddr)
	if err != nil {
		panic(err)
	}
	defer listener.Close()
	fmt.Println("[Worker] RPC connected on port", *portAddr)

	client, _ := rpc.Dial("tcp", "localhost:8031")
	inputData := &Data{Address: "localhost" + *portAddr, Id: *id}
	outputData := new(WorkerOutput)
	client.Call("BrokerComp.Register", inputData, outputData)
	go rpc.Accept(listener)
	for {
		if w.quit {
			return
		}
	}
}
