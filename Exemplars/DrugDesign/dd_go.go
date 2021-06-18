package main

import (
	"flag"
	"fmt"
	"math/rand"
	"runtime"
	"sort"
	"time"
)

var (
	nCPU      = flag.Int("cpu", 1, "number of CPUs to use")
	nLigands  = flag.Int("n", 120, "number of ligands")
	maxLigand = flag.Int("len", 7, "maximum length of a ligand")
	protein   = flag.String("protein",
		"the cat in the hat wore the hat to the cat hat party",
		"protein to use")
)

// Pair represents a ligand and its score.
type Pair struct {
	score  int
	ligand []byte
}

type PairArray []Pair

func main() {
	flag.Parse()
	runtime.GOMAXPROCS(*nCPU) // Set number of OS threads to use.

	// Generate tasks phase: enqueues random ligands into the channel.
	ligands := make(chan []byte, 1024)
	go func() {
		for i := 0; i < *nLigands; i++ {
			len := rand.Intn(*maxLigand) + 1
			l := make([]byte, len)
			for j := 0; j < len; j++ {
				l[j] = byte('a' + rand.Intn(26))
			}
			ligands <- l
		}
		// Close the channel
		// so that the map phase will be able decide when to stop.
		close(ligands)
	}()

	// time the 'work' portion of the code after ligands 
	// are artificially made
	start := time.Now()

	// Map phase: consume tasks from the ligands channels,
	// compute the score and send results into the pairs channel.
	pairs := make(chan Pair, 1024)
	for i := 0; i < *nCPU; i++ {
		go func() {
			p := []byte(*protein)
			for l := range ligands {
				pairs <- Pair{score(l, p), l}
			}
		}()
	}

	// Reduce phase.
	// 1. Collect results from the pairs channel
	// into the sorted array.
	sorted := make(PairArray, *nLigands)
	for i := 0; i < *nLigands; i++ {
		sorted[i] = <-pairs
	}
	// 2. Sort the results based on scores.
	sort.Sort(&sorted)
	// 3. Reduce the sorted array into the results array:
	// merge ligands with equal scores.
	var results PairArray
	for i := 0; i < len(sorted); {
		s := sorted[i].score
		var l []byte
		for ; i < len(sorted) && s == sorted[i].score; i++ {
			l = append(append(l, sorted[i].ligand...), ' ')
		}
		results = append(results, Pair{s, l})
	}

	duration := time.Since(start)
	// Formatted string, such as "2h3m0.5s" or "4.503μs"
	fmt.Println(duration)

	// Output ligands with the highest score.
	fmt.Printf("maximal score is %d, achieved by ligands %s\n",
		results[0].score, string(results[0].ligand))
}

// score calculates the score for protein str1 and ligand str2.
func score(str1, str2 []byte) int {
	if len(str1) == 0 || len(str2) == 0 {
		return 0
	}
	// Both argument strings are non-empty.
	if str1[0] == str2[0] {
		return 1 + score(str1[1:], str2[1:])
	}
	// First characters do not match.
	s1 := score(str1, str2[1:])
	s2 := score(str1[1:], str2)
	if s1 > s2 {
		return s1
	}
	return s2
}

// Implement sort.Interface for PairArray (required by sort.Sort()).
func (a PairArray) Len() int {
	return len(a)
}
func (a PairArray) Less(i, j int) bool {
	return a[i].score > a[j].score
}
func (a PairArray) Swap(i, j int) {
	a[i], a[j] = a[j], a[i]
}
