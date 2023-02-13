#include <iostream>
#include <vector>

using namespace std;

// ����������� 0 �� 1
int activate(double x) {
    return x > 0 ? 1 : 0;
}
  
// ��֪��
class Perceptron {
public:
    // ���캯������ʼ��Ȩ�غ�ƫ����
    Perceptron(int num_features, double learning_rate = 0.1): 
        weights_(num_features + 1), learning_rate_(learning_rate) {}

    // ѵ��
    void Train(const vector<vector<double>>& inputs,
        const vector<int>& labels) {
        int num_samples = inputs.size();
        int num_features = inputs[0].size();

        // ��ʼ��Ȩ�غ�ƫ����
        for (int i = 0; i < num_features + 1; ++i) {
            weights_[i] = 0;
        }

        // ѵ��ѭ��
        for (int i = 0; i < num_samples; ++i) {
            // Ԥ��ֵ
            double prediction = Predict(inputs[i]);
            // �������
            int error = labels[i] - activate(prediction);
            // ����Ȩ�غ�ƫ����
            weights_[0] += learning_rate_ * error;
            for (int j = 0; j < num_features; ++j) {
                weights_[j + 1] += learning_rate_ * error * inputs[i][j];
            }
        }
    }

    // Ԥ��
    int Predict(const vector<double>& input) {
        double prediction = weights_[0];
        for (int i = 0; i < input.size(); ++i) {
            prediction += weights_[i + 1] * input[i];
        }
        return activate(prediction);
    }

private:
    vector<double> weights_;  // Ȩ�غ�ƫ����
    double learning_rate_;    // ѧϰ��
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