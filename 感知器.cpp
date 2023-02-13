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
    // 构造函数，初始化权重和偏置项
    Perceptron(int num_features, double learning_rate = 0.1): 
        weights_(num_features + 1), learning_rate_(learning_rate) {}

    // 训练
    void Train(const vector<vector<double>>& inputs,
        const vector<int>& labels) {
        int num_samples = inputs.size();
        int num_features = inputs[0].size();

        // 初始化权重和偏置项
        for (int i = 0; i < num_features + 1; ++i) {
            weights_[i] = 0;
        }

        // 训练循环
        for (int i = 0; i < num_samples; ++i) {
            // 预测值
            double prediction = Predict(inputs[i]);
            // 计算误差
            int error = labels[i] - activate(prediction);
            // 更新权重和偏置项
            weights_[0] += learning_rate_ * error;
            for (int j = 0; j < num_features; ++j) {
                weights_[j + 1] += learning_rate_ * error * inputs[i][j];
            }
        }
    }

    // 预测
    int Predict(const vector<double>& input) {
        double prediction = weights_[0];
        for (int i = 0; i < input.size(); ++i) {
            prediction += weights_[i + 1] * input[i];
        }
        return activate(prediction);
    }

private:
    vector<double> weights_;  // 权重和偏置项
    double learning_rate_;    // 学习率
};

int main() {
    vector<vector<double>> inputs = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
    vector<int> labels = { 0, 1, 1, 0 };

    Perceptron perceptron(2);
    perceptron.Train(inputs, labels);

    for (const auto& input : inputs) {
        cout << "Input: " << input[0] << ", " << input[1] << endl;
        cout << "Output: " << perceptron.Predict(input);
    }
}