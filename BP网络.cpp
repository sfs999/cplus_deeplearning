#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
using namespace std;

//���弤���
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

//���弤����ĵ���
double sigmoid_der(double x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

//������Ԫ��
class Neuron {
public:
    //���캯������ʼ�����롢�����Ȩ�ء����
    Neuron(int input_num) {
        this->input_num = input_num;
        this->output = 0.0;
        this->error = 0.0;
        this->weights.resize(input_num);
        //�����ʼ��Ȩ��
        srand(time(NULL));
        for (int i = 0; i < input_num; i++) {
            this->weights[i] = (rand() % 100) / 100.0;
        }
    }

    //������Ԫ�����������Ϊǰһ����������
    void calc_output(vector<double> inputs) {
        this->inputs = inputs;
        double sum = 0.0;
        for (int i = 0; i < input_num; i++) {
            sum += inputs[i] * weights[i];
        }
        this->output = sigmoid(sum);
    }

    //������Ԫ��������Ϊ��һ������������Ȩ�ؾ���
    void calc_error(vector<double> errors, vector<vector<double>> weights) {
        double sum = 0.0;
        for (int i = 0; i < errors.size(); i++) {
            sum += errors[i] * weights[i][this->index];
        }
        this->error = sum * sigmoid_der(this->output);
    }

    //������Ԫ��Ȩ�أ�����Ϊѧϰ��
    void update_weights(double lr) {
        for (int i = 0; i < input_num; i++) {
            this->weights[i] += lr * this->error * this->inputs[i];
        }
    }

    //��ȡ��Ԫ�����
    double get_output() {
        return this->output;
    }

    //��ȡ��Ԫ�����
    double get_error() {
        return this->error;
    }

    //��ȡ��Ԫ��Ȩ��
    vector<double> get_weights() {
        return this->weights;
    }

    //������Ԫ�����������ڼ������
    void set_index(int index) {
        this->index = index;
    }

private:
    int input_num; //�������
    int index; //����
    double output; //���
    double error; //���
    vector<double> inputs; //��������
    vector<double> weights; //Ȩ������
};

//�������������
class Layer {
public:
    //���캯������ʼ����Ԫ������ǰ��������
    Layer(int neuron_num, int input_num) {
        this->neuron_num = neuron_num;
        this->input_num = input_num;
        this->neurons.resize(neuron_num);
        //������Ԫ���󣬲���������
        for (int i = 0; i < neuron_num; i++) {
            this->neurons[i] = new Neuron(input_num);
            this->neurons[i]->set_index(i);
        }
    }

    //���������������������Ϊǰһ����������
    void calc_output(vector<double> inputs) {
        this->inputs = inputs;
        this->outputs.resize(neuron_num);
        for (int i = 0; i < neuron_num; i++) {
            this->neurons[i]->calc_output(inputs);
            this->outputs[i] = this->neurons[i]->get_output();
        }
    }

    //������������������Ϊ��һ������������Ȩ�ؾ���
    void calc_error(vector<double> errors, vector<vector<double>> weights) {
        this->errors.resize(neuron_num);
        for (int i = 0; i < neuron_num; i++) {
            this->neurons[i]->calc_error(errors, weights);
            this->errors[i] = this->neurons[i]->get_error();
        }
    }

    //������������Ȩ�أ�����Ϊѧϰ��
    void update_weights(double lr) {
        for (int i = 0; i < neuron_num; i++) {
            this->neurons[i]->update_weights(lr);
        }
    }

    //��ȡ�����������
    vector<double> get_outputs() {
        return this->outputs;
    }

    //��ȡ�����������
    vector<double> get_errors() {
        return this->errors;
    }

    //��ȡ��������Ȩ�ؾ���
    vector<vector<double>> get_weights() {
        vector<vector<double>> weights;
        weights.resize(neuron_num);
        for (int i = 0; i < neuron_num; i++) {
            weights[i] = this->neurons[i]->get_weights();
        }
        return weights;
    }

private:
    int neuron_num; //��Ԫ����
    int input_num; //�������
    vector<double> inputs; //��������
    vector<double> outputs; //�������
    vector<double> errors; //�������
    vector<Neuron*> neurons; //��Ԫ����
};

//����BP��������
class BPNetwork {
public:
    //���캯������ʼ������㡢�����㡢�����Ķ���
    BPNetwork(int input_num, int hidden_num, int output_num) {
        this->input_num = input_num;
        this->hidden_num = hidden_num;
        this->output_num = output_num;
        this->input_layer = new Layer(input_num, 1);
        this->hidden_layer = new Layer(hidden_num, input_num);
        this->output_layer = new Layer(output_num, hidden_num);
    }

    //��ʼ������Ĳ���������Ϊѧϰ�ʡ����������������ֵ
    void init_params(double lr, int epoch, double epsilon) {
        this->lr = lr;
        this->epoch = epoch;
        this->epsilon = epsilon;
    }

    //ѵ�����磬����Ϊѵ�����ݼ�
    void train(vector<vector<double>> train_data) {
        int data_num = train_data.size();
        for (int i = 0; i < epoch; i++) {
            double error_sum = 0.0;
            for (int j = 0; j < data_num; j++) {
                //��ȡ���������������������
                vector<double> inputs = train_data[j];
                vector<double> targets(inputs.begin() + input_num, inputs.end());
                inputs.resize(input_num);
                //ǰ�򴫲�
                input_layer->calc_output(inputs);
                hidden_layer->calc_output(input_layer->get_outputs());
                output_layer->calc_output(hidden_layer->get_outputs());
                //���򴫲�
                output_layer->calc_error(targets, output_layer->get_weights());
                hidden_layer->calc_error(output_layer->get_errors(), output_layer->get_weights());
                input_layer->calc_error(hidden_layer->get_errors(), hidden_layer->get_weights());
                //����Ȩ��
                output_layer->update_weights(lr);
                hidden_layer->update_weights(lr);
                input_layer->update_weights(lr);
                //�������
                error_sum += calc_error(targets, output_layer->get_outputs());
            }
            //������
            cout << "Epoch " << i + 1 << ": Error = " << error_sum / data_num << endl;
            //�ж��Ƿ�ﵽ�����ֵ
            if (error_sum / data_num < epsilon) {
                cout << "Training finished." << endl;
                break;
            }
        }
    }

    //Ԥ�����磬����Ϊ�������ݼ�
    void predict(vector<vector<double>> test_data) {
        int data_num = test_data.size();
        for (int i = 0; i < data_num; i++) {
            //��ȡ���������������������
            vector<double> inputs = test_data[i];
            vector<double> targets(inputs.begin() + input_num, inputs.end());
            inputs.resize(input_num);
            //ǰ�򴫲�
            input_layer->calc_output(inputs);
            hidden_layer->calc_output(input_layer->get_outputs());
            output_layer->calc_output(hidden_layer->get_outputs());
            //���Ԥ����
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
    int input_num; //�������
    int hidden_num; //��������Ԫ����
    int output_num; //�������
    double lr; //ѧϰ��
    int epoch; //��������
    double epsilon; //�����ֵ
    Layer* input_layer; //��������
    Layer* hidden_layer; //���������
    Layer* output_layer; //��������

    //����������Ϊ�������������ʵ���������
    double calc_error(vector<double> targets, vector<double> outputs) {
        double error = 0.0;
        for (int i = 0; i < output_num; i++) {
            error += pow(targets[i] - outputs[i], 2);
        }
        return error / 2.0;
    }
};

//��д������
int main() {
    //����BP��������������������Ϊ2����������Ԫ����Ϊ4���������Ϊ1
    BPNetwork bp(2, 4, 1);
    //��ʼ������Ĳ���������ѧϰ��Ϊ0.1����������Ϊ1000�������ֵΪ0.01
    bp.init_params(0.1, 1000, 0.01);
    //����ѵ�����ݼ���ÿ�����ݰ�����������������������������������������Ϊʾ��
    vector<vector<double>> train_data = { {0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0} };
    //ѵ������
    bp.train(train_data);
    //�����������ݼ�����ѵ�����ݼ���ͬ
    vector<vector<double>> test_data = train_data;
    //Ԥ������
    bp.predict(test_data);
    return 0;
}