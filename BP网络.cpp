#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
using namespace std;

//定义激活函数
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

//定义激活函数的导数
double sigmoid_der(double x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

//定义神经元类
class Neuron {
public:
    //构造函数，初始化输入、输出、权重、误差
    Neuron(int input_num) {
        this->input_num = input_num;
        this->output = 0.0;
        this->error = 0.0;
        this->weights.resize(input_num);
        //随机初始化权重
        srand(time(NULL));
        for (int i = 0; i < input_num; i++) {
            this->weights[i] = (rand() % 100) / 100.0;
        }
    }

    //计算神经元的输出，输入为前一层的输出向量
    void calc_output(vector<double> inputs) {
        this->inputs = inputs;
        double sum = 0.0;
        for (int i = 0; i < input_num; i++) {
            sum += inputs[i] * weights[i];
        }
        this->output = sigmoid(sum);
    }

    //计算神经元的误差，输入为后一层的误差向量和权重矩阵
    void calc_error(vector<double> errors, vector<vector<double>> weights) {
        double sum = 0.0;
        for (int i = 0; i < errors.size(); i++) {
            sum += errors[i] * weights[i][this->index];
        }
        this->error = sum * sigmoid_der(this->output);
    }

    //更新神经元的权重，输入为学习率
    void update_weights(double lr) {
        for (int i = 0; i < input_num; i++) {
            this->weights[i] += lr * this->error * this->inputs[i];
        }
    }

    //获取神经元的输出
    double get_output() {
        return this->output;
    }

    //获取神经元的误差
    double get_error() {
        return this->error;
    }

    //获取神经元的权重
    vector<double> get_weights() {
        return this->weights;
    }

    //设置神经元的索引，用于计算误差
    void set_index(int index) {
        this->index = index;
    }

private:
    int input_num; //输入个数
    int index; //索引
    double output; //输出
    double error; //误差
    vector<double> inputs; //输入向量
    vector<double> weights; //权重向量
};

//定义神经网络层类
class Layer {
public:
    //构造函数，初始化神经元个数、前后层的连接
    Layer(int neuron_num, int input_num) {
        this->neuron_num = neuron_num;
        this->input_num = input_num;
        this->neurons.resize(neuron_num);
        //创建神经元对象，并设置索引
        for (int i = 0; i < neuron_num; i++) {
            this->neurons[i] = new Neuron(input_num);
            this->neurons[i]->set_index(i);
        }
    }

    //计算神经网络层的输出，输入为前一层的输出向量
    void calc_output(vector<double> inputs) {
        this->inputs = inputs;
        this->outputs.resize(neuron_num);
        for (int i = 0; i < neuron_num; i++) {
            this->neurons[i]->calc_output(inputs);
            this->outputs[i] = this->neurons[i]->get_output();
        }
    }

    //计算神经网络层的误差，输入为后一层的误差向量和权重矩阵
    void calc_error(vector<double> errors, vector<vector<double>> weights) {
        this->errors.resize(neuron_num);
        for (int i = 0; i < neuron_num; i++) {
            this->neurons[i]->calc_error(errors, weights);
            this->errors[i] = this->neurons[i]->get_error();
        }
    }

    //更新神经网络层的权重，输入为学习率
    void update_weights(double lr) {
        for (int i = 0; i < neuron_num; i++) {
            this->neurons[i]->update_weights(lr);
        }
    }

    //获取神经网络层的输出
    vector<double> get_outputs() {
        return this->outputs;
    }

    //获取神经网络层的误差
    vector<double> get_errors() {
        return this->errors;
    }

    //获取神经网络层的权重矩阵
    vector<vector<double>> get_weights() {
        vector<vector<double>> weights;
        weights.resize(neuron_num);
        for (int i = 0; i < neuron_num; i++) {
            weights[i] = this->neurons[i]->get_weights();
        }
        return weights;
    }

private:
    int neuron_num; //神经元个数
    int input_num; //输入个数
    vector<double> inputs; //输入向量
    vector<double> outputs; //输出向量
    vector<double> errors; //误差向量
    vector<Neuron*> neurons; //神经元对象
};

//定义BP神经网络类
class BPNetwork {
public:
    //构造函数，初始化输入层、隐含层、输出层的对象
    BPNetwork(int input_num, int hidden_num, int output_num) {
        this->input_num = input_num;
        this->hidden_num = hidden_num;
        this->output_num = output_num;
        this->input_layer = new Layer(input_num, 1);
        this->hidden_layer = new Layer(hidden_num, input_num);
        this->output_layer = new Layer(output_num, hidden_num);
    }

    //初始化网络的参数，输入为学习率、迭代次数、误差阈值
    void init_params(double lr, int epoch, double epsilon) {
        this->lr = lr;
        this->epoch = epoch;
        this->epsilon = epsilon;
    }

    //训练网络，输入为训练数据集
    void train(vector<vector<double>> train_data) {
        int data_num = train_data.size();
        for (int i = 0; i < epoch; i++) {
            double error_sum = 0.0;
            for (int j = 0; j < data_num; j++) {
                //获取输入向量和期望输出向量
                vector<double> inputs = train_data[j];
                vector<double> targets(inputs.begin() + input_num, inputs.end());
                inputs.resize(input_num);
                //前向传播
                input_layer->calc_output(inputs);
                hidden_layer->calc_output(input_layer->get_outputs());
                output_layer->calc_output(hidden_layer->get_outputs());
                //反向传播
                output_layer->calc_error(targets, output_layer->get_weights());
                hidden_layer->calc_error(output_layer->get_errors(), output_layer->get_weights());
                input_layer->calc_error(hidden_layer->get_errors(), hidden_layer->get_weights());
                //更新权重
                output_layer->update_weights(lr);
                hidden_layer->update_weights(lr);
                input_layer->update_weights(lr);
                //计算误差
                error_sum += calc_error(targets, output_layer->get_outputs());
            }
            //输出误差
            cout << "Epoch " << i + 1 << ": Error = " << error_sum / data_num << endl;
            //判断是否达到误差阈值
            if (error_sum / data_num < epsilon) {
                cout << "Training finished." << endl;
                break;
            }
        }
    }

    //预测网络，输入为测试数据集
    void predict(vector<vector<double>> test_data) {
        int data_num = test_data.size();
        for (int i = 0; i < data_num; i++) {
            //获取输入向量和期望输出向量
            vector<double> inputs = test_data[i];
            vector<double> targets(inputs.begin() + input_num, inputs.end());
            inputs.resize(input_num);
            //前向传播
            input_layer->calc_output(inputs);
            hidden_layer->calc_output(input_layer->get_outputs());
            output_layer->calc_output(hidden_layer->get_outputs());
            //输出预测结果
            cout << "Input: ";
            for (int j = 0; j < input_num; j++) {
                cout << inputs[j] << " ";
            }
            cout << endl;
            cout << "Target: ";
            for (int j = 0; j < output_num; j++) {
                cout << targets[j] << " ";
            }
            cout << endl;
            cout << "Output: ";
            for (int j = 0; j < output_num; j++) {
                cout << output_layer->get_outputs()[j] << " ";
            }
            cout << endl;
            cout << "--------------------------" << endl;
        }
    }

private:
    int input_num; //输入个数
    int hidden_num; //隐含层神经元个数
    int output_num; //输出个数
    double lr; //学习率
    int epoch; //迭代次数
    double epsilon; //误差阈值
    Layer* input_layer; //输入层对象
    Layer* hidden_layer; //隐含层对象
    Layer* output_layer; //输出层对象

    //计算误差，输入为期望输出向量和实际输出向量
    double calc_error(vector<double> targets, vector<double> outputs) {
        double error = 0.0;
        for (int i = 0; i < output_num; i++) {
            error += pow(targets[i] - outputs[i], 2);
        }
        return error / 2.0;
    }
};

//编写主函数
int main() {
    //创建BP神经网络对象，设置输入个数为2，隐含层神经元个数为4，输出个数为1
    BPNetwork bp(2, 4, 1);
    //初始化网络的参数，设置学习率为0.1，迭代次数为1000，误差阈值为0.01
    bp.init_params(0.1, 1000, 0.01);
    //创建训练数据集，每个数据包含输入向量和期望输出向量，这里用异或运算作为示例
    vector<vector<double>> train_data = { {0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0} };
    //训练网络
    bp.train(train_data);
    //创建测试数据集，与训练数据集相同
    vector<vector<double>> test_data = train_data;
    //预测网络
    bp.predict(test_data);
    return 0;
}