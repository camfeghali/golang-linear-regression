package main

import (
	"fmt"
	"math"
)

type ComputeCostFunction func(x []float64, y []float64, w float64, b float64) float64 
type ComputeGradientFunction func(x []float64, y []float64, w float64, b float64) (float64, float64)

func main() {

	features := []float64{1.0, 2.0}
	targets := []float64{300.0, 500.0}

	
	// w_final, b_final, J_hist, p_hist := gradient_descent(features, targets, 0, 0, 0.001, 1000, compute_cost, compute_gradient)
	
	// fmt.Printf("w_final: %f\n", w_final)
	// fmt.Printf("b_final: %f\n", b_final)
	// fmt.Printf("J_hist: %f\n", J_hist)
	// fmt.Printf("p_hist: %f\n", p_hist)

	gradient_descent(features, targets, 0, 0, 0.001, 1000, compute_cost, compute_gradient)

}

func sigmoid(z float64) float64 {
	g := 1/(1 + math.Exp(-z))
	return g
}

// Single variable linear regression
func f_wb(i float64, w float64, b float64) float64 {
	return (w * i) + b
}

func gradient_descent(x []float64, y []float64, w_init float64, b_init float64, alpha float64, num_iterations int, cost_function ComputeCostFunction, gradient_function ComputeGradientFunction) (float64, float64, []float64, [][]float64){
    // Performs gradient descent to fit w,b. Updates w,b by taking 
    // num_iters gradient steps with learning rate alpha
    
    // Args:
    //   x					: Data, m examples 
    //   y					: target values
    //   w_in,b_in			: initial values of model parameters  
    //   alpha				: Learning rate
    //   num_iters			: number of iterations to run gradient descent
    //   cost_function		: function to call to produce cost
    //   gradient_function	: function to call to produce gradient
      
    // Returns:
    //   w					: Updated value of parameter after running gradient descent
    //   b 					: Updated value of parameter after running gradient descent
    //   J_history (List)	: History of cost values
    //   p_history (list)	: History of parameters [w,b] 

	J_history := make([]float64, 0)
	parameter_history := make([][]float64, 0)
	b := b_init
	w := w_init

	for idx := range x {
		dj_dw, dj_db := gradient_function(x, y, b, w)

		b = b - alpha*dj_db
		w = w - alpha*dj_dw

		if idx < 1000 {
			J_history = append(J_history, cost_function(x, y, w, b))
			parameter_history = append(parameter_history, []float64{w, b})

			fmt.Printf("J_history=%p, parameter_history=%p, b=%f, w=%f\n", J_history, parameter_history, b, w)

			// if (idx % num_iterations/10) == 0 {
			// 	fmt.Printf("J_history=%p, parameter_history=%p, b=%f, w=%f", J_history, parameter_history, b, w)
			// }

		}
	}
	return w, b, J_history, parameter_history
}

// Mean Square Error Cost Function
func compute_cost(x []float64, y []float64, w float64, b float64) float64 {
    //	Computes the cost function for linear regression.
    //	Args:
    //  	x: Data, m examples 
    //  	y: target values
    //  	w,b: model parameters  
    //	Returns
    //		total_cost: The cost of using w,b as the parameters for linear regression
    //    to fit the data points in x and y
		m := float64(len(x))
		cost_sum := 0.0
		
		for idx, _ := range x {
			f_wb := w * x[idx] + b 
			cost := square(f_wb - y[idx])
			cost_sum = cost_sum + cost
		}
		total_cost := (1/(2*m)) * cost_sum
		return total_cost
}

func compute_gradient(x []float64, y []float64, w float64, b float64) (float64, float64) {
    // Computes the gradient for linear regression 
    // Args:
    //   x: Data, m examples 
    //   y: target values
    //   w,b: model parameters  
    // Returns
    //   dj_dw: The gradient of the cost w.r.t. the parameters w
    //   dj_db: The gradient of the cost w.r.t. the parameter b     

	m := float64(len(x))
	dj_w := 0.0
	dj_b := 0.0

	for idx := range x {
		f_wb := w * x[idx] + b
		dj_w_i := (f_wb - y[idx]) * x[idx]
		dj_b_i := f_wb - y[idx]

		dj_w = dj_w + dj_w_i
		dj_b = dj_b + dj_b_i
	}
	dj_w = dj_w / m
	dj_b = dj_b / m

	return dj_w, dj_b
}

func square[T int | float64](num T) T {
	return num * num
}
