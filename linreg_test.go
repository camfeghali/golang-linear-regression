package linreg

import (
	"fmt"
	"testing"
	"time"
	"math/rand"
	"github.com/stretchr/testify/assert"
)

var v1 []float64 = randFloats(1.0, 100.0, 1000000)
var v2 []float64 = randFloats(1.0, 100.0, 1000000)

func TestDot(t *testing.T) {
		assert := assert.New(t)

		start := time.Now()
		result := dot(v1, v2)
 		elapsed := time.Since(start)
    fmt.Printf("dot took %s", elapsed)

		assert.Equal(v1[2]*v2[2], result[2], "");
		assert.Equal(v1[4]*v2[4], result[4], "");
}

func randFloats(min, max float64, n int) []float64 {
    res := make([]float64, n)
    for i := range res {
        res[i] = min + rand.Float64() * (max - min)
    }
    return res
}

func TestSlowDot(t *testing.T) {
		assert := assert.New(t)

		start := time.Now()
		result := slowDot(v1, v2)
 		elapsed := time.Since(start)
    fmt.Printf("Slow dot took %s", elapsed)

		assert.Equal(v1[2]*v2[2], result[2], "");
		assert.Equal(v1[4]*v2[4], result[4], "");
}

func TestSquare(t *testing.T) {
		assert := assert.New(t)

		num1 := 3
		num2 := 3.0

		assert.Equal(9, square(num1), "squaring a int returns a int.");
		assert.Equal(9.0, square(num2), "squaring a float returns a float.");
}