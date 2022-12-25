package linreg

import (
	"fmt"
	"math/rand"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestGradientDescent(t *testing.T) {
	assert := assert.New(t)

	var x_train = []float64{1.0, 2.0}
	var y_train = []float64{300.0, 500.0}
	w_init := 0.0
	b_init := 0.0
	iterations := 10000
	tmp_alpha := 0.01

	w_final, b_final, _, _ := single_var_gradient_descent(x_train, y_train, w_init, b_init, tmp_alpha, iterations)

	assert.Equal(199.99285075131766, w_final, "");
	assert.Equal(100.011567727362, b_final, "");

}

func TestComputeGradient (t *testing.T) {
	assert := assert.New(t)
	// x: [1. 2.], y: [300. 500.], w: 0, b: 0, dj_dw: -650.0, dj_db: -400.0
	var x_train = []float64{1.0, 2.0}
	var y_train = []float64{300.0, 500.0}

	w := 0.0
	b := 0.0

	dj_dw, dj_db := compute_gradient_single_var(x_train, y_train, w, b)
	assert.Equal(-650.0, dj_dw, "");
	assert.Equal(-400.0, dj_db, "");

	dj_dw, dj_db = compute_gradient_single_var(x_train, y_train, 55.849817910123896, 34.345440679260854)
	assert.Equal(-458.857294205799, dj_dw, "");
	assert.Equal(-281.8798324555533, dj_db, "");

	dj_dw, dj_db = compute_gradient_single_var(x_train, y_train, 191.23105305470705, 114.13601208201743)
	assert.Equal(-0.718349240206237, dj_dw, "");
	assert.Equal(0.9825916640780008, dj_db, "");

		dj_dw, dj_db = compute_gradient_single_var(x_train, y_train, 199.99284553220375, 100.01157617206572)
	assert.Equal(-0.0005219113920702512, dj_dw, "");
	assert.Equal(0.0008444703713337276, dj_db, "");
}

func TestCostFunction(t *testing.T) {
	assert := assert.New(t)

	var x_train = []float64{1.0, 2.0}
	var y_train = []float64{300.0, 500.0}

	w := 190.0
	b := 100.0

	cost := compute_cost_single_var(x_train, y_train, w, b)
	assert.Equal(125.0, cost, "");
}

func TestMultiVarCostFunction(t *testing.T) {
	assert := assert.New(t)

	x_train := [][]float64 {{2104.0, 5.0, 1.0, 45.0}, {1416.0, 3.0, 2.0, 40.0}, {852.0, 2.0, 1.0, 35.0}}
	y_train := []float64 {460.0, 232.0, 178.0}

	b := 785.1811367994083
	w := []float64 { 0.39133535, 18.75376741, -53.36032453, -26.42131618 }

	cost := compute_cost_multi_var(x_train, y_train, w, b)
	assert.Equal(1.5578904428966628e-12, cost, "");
}

func TestCalcStdDev(t *testing.T) {
	assert := assert.New(t)

	var sample1 = []float64{66.0, 30.0, 40.0, 64.0}
	mean1 := calcMean(sample1)
	result1 := calcStdDev(sample1, mean1)

	var sample2 = []float64{51.0, 21.0, 79.0, 49.0}
	mean2 := calcMean(sample2)
	result2 := calcStdDev(sample2, mean2)

	assert.Equal(17.813852287849848, result1, "");
	assert.Equal(23.692474191889147, result2, "");
}

func TestCalcMean(t *testing.T) {
	var sample = []float64{66.0, 30.0, 40.0, 64.0}

	assert := assert.New(t)
	result := calcMean(sample)
	assert.Equal(50.0, result, "");
}

func TestDot(t *testing.T) {
	var v1 []float64 = []float64 {2, 4, 6}
	var v2 []float64 = []float64 {2, 4, 6}

	assert := assert.New(t)

	start := time.Now()
	result := dotProduct(v1, v2)
	elapsed := time.Since(start)
	fmt.Printf("dot took %s", elapsed)

	assert.Equal(56.0, result, "");
}

func randFloats(min, max float64, n int) []float64 {
    res := make([]float64, n)
    for i := range res {
        res[i] = min + rand.Float64() * (max - min)
    }
    return res
}

func TestSquare(t *testing.T) {
		assert := assert.New(t)

		num1 := 3
		num2 := 3.0

		assert.Equal(9, square(num1), "squaring a int returns a int.");
		assert.Equal(9.0, square(num2), "squaring a float returns a float.");
}
