package main

import (
	linerregression "GoLinerRegression/linerRegression"
	"fmt"
)

func main() {
	sTrainDataX := make([][]float64, 4)
	sTrainDataY := make([]float64, 4)
	sTestDataX := make([]float64, 2)
	sTestDataY := make([]float64, 1)

	sTrainDataX[0] = []float64{1.0, 1.0}
	sTrainDataX[1] = []float64{2.0, 2.0}
	sTrainDataX[2] = []float64{3.0, 3.0}
	sTrainDataX[3] = []float64{4.0, 4.0}
	sTrainDataY = []float64{5.3, 8.0, 9.0, 11.7}
	sTestDataX = []float64{5.0, 5.0}
	sTestDataY = []float64{13.4}

	lr := new(linerregression.LinerRegressionModel)
	lr.W = make([]float64, len(sTrainDataX))
	//初始化
	for i := range lr.W {
		lr.W[i] = 1.0
	}
	lr.B = 1.0
	lr.Train(10000, 0.01, sTrainDataX, sTrainDataY)
	for i := 0; i < 10000; i++ {
		lr.GredientDescent(0.01, sTrainDataX, sTrainDataY)
	}
	//lr.Train(100, 0.1, sTrainDataX, sTrainDataY)
	fmt.Println(lr.W, lr.B)
	y := lr.FX(sTestDataX)
	fmt.Println(y, sTestDataY)

}
