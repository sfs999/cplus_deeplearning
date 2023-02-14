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
    // ���캯������ʼ��Ȩ�غ�ƫ��������ĳ�Ա����
    Perceptron(int num_features, double learning_rate = 0.1): 
        weights_(num_features + 1), learning_rate_(learning_rate) {}//��ʼ��ƫ��+Ȩ�أ�1+num_features����ѧϰ��

    // Ԥ��
    int Predict(const vector<double>& input) {
        double prediction = weights_[0];//y=b
        for (int i = 0; i < input.size(); ++i) {
            prediction += weights_[i + 1] * input[i];//y=b+x*w
        }
        return activate(prediction);
    }

    // ѵ��
    void Train(const vector<vector<double>>& inputs,const vector<int>& labels, const int iteration=1) {
        /*inputs����ά��������ÿһ��Ϊһ������
          labels��һά��������ʾ��ǩ
          const��ʾԴ���ݲ����޸�
        */
        int num_samples = inputs.size();//��������
        int num_features = inputs[0].size();//������С

        // ��ʼ��Ȩ�غ�ƫ���ȫ����0
        for (int i = 0; i < num_features + 1; ++i) {
            //+1Ϊweights_�����һά��ƫ��ֵ
            weights_[i] = 0;
        }

        // ѵ��ѭ��
        for (int k=0; k< iteration;++k){
            for (int i = 0; i < num_samples; ++i) {//����ѭ��һ��
                // Ԥ��ֵ
                double prediction = Predict(inputs[i]);//����Predict�����Ե������������������
                // �������
                int error = labels[i] - activate(prediction);//������������ʵֵ��Ԥ��ֵ֮��
                // ����Ȩ�غ�ƫ����
                weights_[0] += learning_rate_ * error;//���������ѧϰ�ʶ�ƫ������и���
                for (int j = 0; j < num_features; ++j) {
                    weights_[j + 1] += learning_rate_ * error * inputs[i][j];//���������ѧϰ�ʶ�Ȩ������и���
                }
            }
            //cout << "��: " << k+1 << "�ε��� " << endl;
        }
    }

private:
    vector<double> weights_;  // ƫ�����Ȩ��
    double learning_rate_;    // ѧϰ��
};

int main() {
    vector<vector<double>> inputs = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
    vector<int> labels = { 0, 0, 0, 1 };//����
    //�����֪���޷�������ȷ����Կɷ�����
    Perceptron perceptron(2);//������һ����֪���� ��ʾ��������Ϊ2���Դ���ȷ��w�ĸ���
    perceptron.Train(inputs, labels,50);

    for (const auto& input : inputs) {
        cout << "Input: " << input[0] << ", " << input[1] << "   ";
        cout << "Output: " << perceptron.Predict(input) << endl;
    }
}