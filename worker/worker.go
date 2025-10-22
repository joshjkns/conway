package main

import (
	"fmt"
	"net"
	"net/rpc"
	"sync"
)

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

type DistributingComp struct{}

var offsets = [][]int{{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}}

var currentWorld [][]byte
var currentTurns int
var mu sync.Mutex

func createWorldWorker(width, height int) [][]byte {
	newWorld := make([][]byte, height)
	for i := range newWorld {
		newWorld[i] = make([]byte, width)
	}
	return newWorld
}

func incrementGolWorker(world *[][]byte, width, height int) [][]byte {
	newWorld := createWorldWorker(width, height)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			liveNeighbours := countLiveNeighboursWorker(world, x, y, width, height)
			if (*world)[y][x] == 255 { // current cell is alive
				if liveNeighbours < 2 || liveNeighbours > 3 {
					newWorld[y][x] = 0
				} else {
					newWorld[y][x] = 255
				}
			} else { // current cell is dead
				if liveNeighbours == 3 {
					newWorld[y][x] = 255
				} else {
					newWorld[y][x] = 0
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
		xNeighbour = constrainValueWorker(xNeighbour, width)
		yNeighbour = constrainValueWorker(yNeighbour, height)

		if (*world)[yNeighbour][xNeighbour] == 255 { // alive
			count += 1
		}
	}
	return count
}

func (d *DistributingComp) GetCurrentState(args *Input, reply *Output) error {
	mu.Lock()
	defer mu.Unlock()
	reply.World = currentWorld
	reply.Turns = currentTurns
	fmt.Println("done")
	return nil
}

func (d *DistributingComp) Process(args *Input, reply *Output) error {
	// TODO: Create a 2D slice to store the world.
	width := args.Width
	height := args.Height
	worldInput := (*args).World
	numberOfTurns := args.Turns

	// TODO: Execute all turns of the Game of Life.
	for i := 1; i <= numberOfTurns; i++ {
		worldInput = incrementGolWorker(&worldInput, width, height)
		mu.Lock()
		currentWorld = worldInput
		currentTurns++
		mu.Unlock()
	}
	reply.World = worldInput
	reply.Turns = numberOfTurns
	return nil
}

func main() {
	err := rpc.Register(&DistributingComp{})
	if err != nil {
		panic(err)
	}

	listener, err := net.Listen("tcp", ":8030")
	if err != nil {
		panic(err)
	}

	fmt.Println("RPC connected on port 8030")
	defer listener.Close()
	rpc.Accept(listener)
}
