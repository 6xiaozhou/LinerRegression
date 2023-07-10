package main

import (
	"encoding/csv"
	"fmt"
	"os"
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
	trainData := data[1:405]
	testData := data[405:]
	fmt.Println(trainData)
	fmt.Println("----------------------------")
	fmt.Println(testData)

}
