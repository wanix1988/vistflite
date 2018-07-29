#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import sys
#sys.path.append('..')

from flask import Flask
from flask import render_template
from tflite.Model import Model
from tflite.OperatorCode import OperatorCode
from tflite.BuiltinOperator import BuiltinOperator
from tflite.SubGraph import SubGraph

app = Flask(__name__)

def getBuiltinOperatorStringName(idx):
    allAttrs = [i for i in BuiltinOperator.__dict__.items() if not i[0].startswith('__')]
    for name, index in allAttrs:
        if index == idx:
            return name

@app.route('/')
def index():
    return render_template('analyze_tflite.html', name='tflite')

def analyze_tflite():
    print('vistflite...')
    with open(sys.argv[1], 'rb') as df:
        content = df.read()
        model = Model.GetRootAsModel(content, 0)
        print(model.Version())
        print(model.Description())
        # OperatorCodes
        for i in range(model.OperatorCodesLength()):
            op = model.OperatorCodes(i)
            if op.CustomCode():
                print('Custom OP:', op.Version(), op.CustomCode())
            else:
                print('Builtin OP:', op.Version(), op.BuiltinCode(), getBuiltinOperatorStringName(op.BuiltinCode()))
        # SubGraphs
        print('SubGraphs:')
        for i in range(model.SubgraphsLength()):
            subGraph = model.Subgraphs(i)
            print('Name:', subGraph.Name())
            print('Tensors:')
            for j in range(subGraph.TensorsLength()):
                print(subGraph.Tensors(j).Name())

if __name__ == '__main__':
    app.run(debug=True)
