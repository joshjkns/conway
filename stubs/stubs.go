package stubs

import (
	"csa/conway/util"
)

type Position int

const (
	Top Position = iota
	Bottom
)

type WaitArgs struct {
	ID int
	Turn int
}

type Halo struct {
	Row []byte
}

type Data struct {
	Address string
	ID      int
}

type WorkerInfo struct {
	Port string
}

type Confirmation struct {
	ID int
}

type Response struct {
	Resp bool
}

type WorldInfo struct {
	World [][]byte
	Width int
	Height int
	Turns int
	CurrentTurns int
}

type NeighbourPair struct {
	LeftNeighbour Data
	RightNeighbour Data
}

type ChunkInfo struct {
	Chunk [][]byte
	StartRow int
	EndRow int
	Neighbours NeighbourPair
	Turns int
}

func CreateWorld(w int, h int) [][]byte {
	newWorld := make([][]byte, h)
	for i := range newWorld {
		newWorld[i] = make([]byte, w)
	}
	return newWorld
}

func GetAliveCells(world *[][]byte) []util.Cell {
	var alive []util.Cell
	for y := 0; y < len(*world); y++ {
		for x := 0; x < (len((*world)[0])); x++ {
			if (*world)[y][x] == 255 {
				cell := util.Cell{X: x, Y: y}
				alive = append(alive, cell)
			}
		}
	}
	return alive
}

func CopyWorld(world *[][]byte, w, h int) [][]byte {
	newWorld := CreateWorld(w,h)
	for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				newWorld[y][x] = (*world)[y][x]
			}
		}
	return newWorld
}

func ConstrainValue(value int, constraint int) int {
	if value < 0 {
		value += constraint
	} else if value >= constraint {
		value -= constraint
	}
	return value
}

func CreateChunk(world [][]byte, width, height, startY, endY int) [][]byte {
	res := CreateWorld(width, height)

	for y := startY; y <= endY; y++ {
		for x := 0; x < width; x++ {
			res[y-startY][x] = world[y][x]
		}
	}
	return res
}