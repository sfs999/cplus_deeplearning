#include <iostream>
#include <vector>

using namespace std;

// 激活函数，返回 0 或 1
int activate(double x) {
    return x > 0 ? 1 : 0;
}
  
// 感知器
class Perceptron {
public:
    // 构造函数，初始化权重和偏置项，类对象的成员数据
    Perceptron(int num_features, double learning_rate = 0.1): 
        weights_(num_features + 1), learning_rate_(learning_rate) {}//初始化偏置+权重（1+num_features）和学习率

    // 预测
    int Predict(const vector<double>& input) {
        double prediction = weights_[0];//y=b
        for (int i = 0; i < input.size(); ++i) {
            prediction += weights_[i + 1] * input[i];//y=b+x*w
        }
        return activate(prediction);
    }

    // 训练
    void Train(const vector<vector<double>>& inputs,const vector<int>& labels, const int iteration=1) {
        /*inputs：二维矩阵，其中每一行为一个样本
          labels：一维标量，表示标签
          const表示源数据不可修改
        */
        int num_samples = inputs.size();//样本数量
        int num_features = inputs[0].size();//样本大小

        // 初始化权重和偏置项，全部置0
        for (int i = 0; i < num_features + 1; ++i) {
            //+1为weights_中最后一维，偏置值
            weights_[i] = 0;
        }

        // 训练循环
        for (int k=0; k< iteration;++k){
            for (int i = 0; i < num_samples; ++i) {//样本循环一遍
                // 预测值
                double prediction = Predict(inputs[i]);//调用Predict函数对单个样本进行推理计算
                // 计算误差
                int error = labels[i] - activate(prediction);//单个样本的真实值与预测值之差
                // 更新权重和偏置项
                weights_[0] += learning_rate_ * error;//根据误差与学习率对偏置项进行更新
                for (int j = 0; j < num_features; ++j) {
                    weights_[j + 1] += learning_rate_ * error * inputs[i][j];//根据误差与学习率对权重项进行更新
                }
            }
            //cout << "第: " << k+1 << "次迭代 " << endl;
        }
    }

private:
    vector<double> weights_;  // 偏置项和权重
    double learning_rate_;    // 学习率
};

int main() {
    vector<vector<double>> inputs = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
    vector<int> labels = { 0, 0, 0, 1 };//与门
    //单层感知器无法解决异或等非线性可分问题
    Perceptron perceptron(2);//定义了一个感知器， 表示样本长度为2，以此来确定w的个数
    perceptron.Train(inputs, labels,50);

    for (const auto& input : inputs) {
        cout << "Input: " << input[0] << ", " << input[1] << "   ";
        cout << "Output: " << perceptron.Predict(input) << endl;
    }
}