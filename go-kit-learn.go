package main

import (
	"fmt"
	"math"
)

func main() {
	fmt.Println(sigmoid(6))
}

func sigmoid(z float64) float64 {
	g := 1/(1 + math.Exp(-z))
	return g
}

// Single variable linear regression
func f_wb(i float64, w float64, b float64) float64 {
	return (w * i) + b
}

// Mean Square Error Cost Function
func compute_cost(x []float64, y []float64, w float64, b float64) float64 {
    //	Computes the cost function for linear regression.
    //	Args:
    //  	x: Data, m examples 
    //  	y: target values
    //  	w,b: model parameters  
    //	Returns
    //		total_cost (float): The cost of using w,b as the parameters for linear regression
    //    to fit the data points in x and y
		var m float64 = float64(len(x))
		cost_sum := 0.0
		
		for idx, _ := range x {
			f_wb := w * x[idx] + b 
			cost := square(f_wb - y[idx])
			cost_sum = cost_sum + cost
		}
		total_cost := (1/(2*m)) * cost_sum
		return total_cost
}

func square[T int | float64](num T) T {
	return num * num
}