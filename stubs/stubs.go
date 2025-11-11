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
	Address string
}

type DistributorInfo struct {
	Address string
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
	Paused bool
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

type CellsFlippedData struct {
	CompletedTurns int
	CellsFlipped []util.Cell
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

func AddHalos(world, chunk [][]byte, width, height, startY, endY int) [][]byte {
	res := CreateWorld(width, height + 2)

	topIndex := ConstrainValue(startY - 1, height)
	bottomIndex := ConstrainValue(endY + 1, height)

	copy(res[0], world[topIndex])

	for i := 0; i < height; i++ {
		copy(res[i+1], chunk[i])
	}

	copy(res[height+1], world[bottomIndex])

	return res
}