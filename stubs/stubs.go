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
	Chunk ChunkInfo
	Width int
}

type Halo struct {
	Row []bool
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
	World [][]bool
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
	Chunk [][]bool
	StartRow int
	EndRow int
	Neighbours NeighbourPair
	Turns int
}

func CreateWorld(w int, h int) [][]bool {
	newWorld := make([][]bool, h)
	for i := range newWorld {
		newWorld[i] = make([]bool, w)
	}
	return newWorld
}

func GetAliveCells(world *[][]bool) []util.Cell {
	var alive []util.Cell
	for y := 0; y < len(*world); y++ {
		for x := 0; x < (len((*world)[0])); x++ {
			if (*world)[y][x] {
				cell := util.Cell{X: x, Y: y}
				alive = append(alive, cell)
			}
		}
	}
	return alive
}

func CopyWorld(world *[][]bool, w, h int) [][]bool {
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

func CreateChunk(world [][]bool, width, height, startY, endY int) [][]bool {
	res := CreateWorld(width, height)

	for y := startY; y <= endY; y++ {
		for x := 0; x < width; x++ {
			res[y-startY][x] = world[y][x]
		}
	}
	return res
}