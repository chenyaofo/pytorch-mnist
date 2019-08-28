import numpy
import torch
import pickle
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

net = torch.load("LeNet5.pth", map_location="cpu")

def predict_with_pretrain_model(sample):
    sample = torch.from_numpy(sample)
    sample = (sample - 0.1307) / 0.3081
    sample = torch.autograd.Variable(sample)
    scores = net.forward(sample)
    probilities = torch.nn.functional.softmax(scores)
    return probilities.data.tolist()[0]


@app.route('/api/mnist', methods=['POST'])
def mnist():
    input = ((255 - (numpy.array(request.json, dtype=numpy.float32))) / 255).reshape(1, 1, 28, 28)
    output = predict_with_pretrain_model(input)
    return jsonify(results=output)


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    # please use port larger than 10000
    app.run(host="0.0.0.0",port=23333)
