package linerregression

//线性回归模型，包含权重以及误差
type LinerRegressionModel struct {
	W []float64
	B float64
}

//训练方法，传入参数为训练轮次、学习率、训练数据
func (m *LinerRegressionModel) Train(batch int, learningRate float64, X [][]float64, Y []float64) {
	//fmt.Println("aa")
	d := len(X[0])
	//d := len(m.W)
	if d == 0 {
		//fmt.Print("a")
		return
	}
	m.W = make([]float64, d)
	for i := range m.W {
		m.W[i] = 1.0
	}
	m.B = 1.0
	for i := 0; i < batch; i++ {
		m.GredientDescent(learningRate, X, Y)
	}

}

//权重一次更新   learningR学习率，X为自变量数值，Y为因变量数值
func (m *LinerRegressionModel) GredientDescent(learningR float64, X [][]float64, Y []float64) {
	//fmt.Println(m.W)
	//var sliceMake []int = make([]int, 5, 10) // 分配了cap则 >= len如何传入动态数组长度
	var gredientW []float64 = make([]float64, len(m.W))
	var gredientB float64
	//2/n
	twoPerN := float64(2) / float64(len(X))

	//计算本次迭代的梯度
	//i表示变量x的组数,j表示变量x的维度
	for i := 0; i < len(X); i++ {
		for j := 0; j < len(X[0]); j++ {
			gredientW[j] = gredientW[j] + (X[i][j] * (m.FX(X[i]) - Y[i]))
		}
		gredientB = gredientB + m.FX(X[i]) - Y[i]
	}
	for i := range gredientW {
		gredientW[i] = gredientW[i] * twoPerN
	}
	gredientB = gredientB * twoPerN
	//根据梯度更新权重w以及误差b
	for i := range m.W {
		m.W[i] = m.W[i] - learningR*gredientW[i]
		//fmt.Print("a")
	}
	m.B = m.B - learningR*gredientB
	//fmt.Println(m.W, m.B)
}

//计算fX，w1*x1+w2*x2+...+wd*xd+b
func (m *LinerRegressionModel) FX(X []float64) float64 {
	var res float64

	for i := 0; i < len(X); i++ {
		res = res + X[i]*m.W[i]
	}
	res = res + m.B
	return res
}
