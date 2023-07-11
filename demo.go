package main

import (
	linerregression "GoLinerRegression/linerRegression"
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"strconv"
)

func main() {
	file, err := os.OpenFile("boston_house_prices.csv", os.O_RDONLY, 0)
	if err != nil {
		fmt.Println(err)
	}
	defer file.Close()

	reader := csv.NewReader(file) //具有解码功能，能够将csv文件解码成二维数组
	//fmt.Println(string(data))
	data, _ := reader.ReadAll()
	//fmt.Println(len(data))
	//fmt.Println(len(data[0]))
	//划分训练集以及测试集，划分比例为8:2
	//trainingDataX := data[1:405][:14]
	//trainingDataY := data[:][14:]
	//testDataX := data[405:][:14]
	//fmt.Println(trainingDataX)
	//fmt.Println("---------------")
	//fmt.Println(len(data), len(data[0]))
	//划分训练集和测试集。划分比例8:2
	trainData := data[1:405]
	testData := data[405:]
	//fmt.Println(len(trainData))
	//fmt.Println("----------------------------")
	//fmt.Println(len(testData))
	//需要将其定义为切片
	// var trainDataX [404][13]float64
	// var trainDataY [404]float64
	// var testDataX [102][13]float64
	// var testDataY [102]float64

	sTrainDataX := make([][]float64, 404)
	sTrainDataY := make([]float64, 404)
	sTestDataX := make([][]float64, 102)
	sTestDataY := make([]float64, 102)
	//range sTestDataX 返回索引和值
	for i := range sTrainDataX {
		sTrainDataX[i] = make([]float64, 13)
	}
	for i := range sTestDataX {
		sTestDataX[i] = make([]float64, 13)
	}

	for i := 0; i < 404; i++ {
		for j := 0; j < 13; j++ {
			sTrainDataX[i][j], _ = strconv.ParseFloat(trainData[i][j], 64)
		}
		sTrainDataY[i], _ = strconv.ParseFloat(trainData[i][13], 64)
		//fmt.Println(trainDataX[i])
		//fmt.Println(trainDataY[i])
	}

	for i := 0; i < 102; i++ {
		for j := 0; j < 13; j++ {
			sTestDataX[i][j], _ = strconv.ParseFloat(testData[i][j], 64)
		}
		sTestDataY[i], _ = strconv.ParseFloat(testData[i][13], 64)
		//fmt.Println(testDataX[i])
		//fmt.Println(testDataY[i])
	}
	//初始化一个线性回归模型
	lr := new(linerregression.LinerRegressionModel)
	lr.Train(20000, 0.000001, sTrainDataX, sTrainDataY)
	//lr.Train(100, 0.1, sTrainDataX, sTrainDataY)
	fmt.Println(lr.W, lr.B)
	//fmt.Println(len(sTestDataX[0]))
	var predictV float64
	for i := 0; i < len(sTestDataX); i++ {
		predictV = lr.FX(sTestDataX[i])
		fmt.Println(predictV, sTestDataY[i], math.Abs(predictV-sTestDataY[i]))
	}
	// pre := lr.FX(sTestDataX[0])
	// fmt.Println(pre, sTestDataY[0])
}
