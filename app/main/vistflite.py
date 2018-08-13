#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import os
import sys
import tempfile
import struct
#sys.path.append('..')

from flask import Flask
from flask import render_template
from flask import request, flash, redirect, url_for
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from tflite.Model import Model
from tflite.OperatorCode import OperatorCode
from tflite.BuiltinOperator import BuiltinOperator
from tflite.SubGraph import SubGraph
from tflite.TensorType import TensorType

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024 #1GB
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['useLocal'] = True

IN_MEMORY_SIZE = 100 * 1024 * 1024 #100MB

bootstrap = Bootstrap(app)

def getStringName(clazz, idx):
    allAttrs = [i for i in clazz.__dict__.items() if not i[0].startswith('__')]
    for name, index in allAttrs:
        if index == idx:
            return name

def __getBuiltinOperatorStringName(idx):
    return getStringName(BuiltinOperator, idx)

@app.route('/')
def index():
    return render_template(r'analyze_tflite.html', name='tflite', useLocal=app.config['useLocal'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] == 'tflite'

@app.route('/analyze/<clazz>/<type>', methods=['POST'])
def analyze(clazz, type):
    if clazz == 'tflite':
        if type == 'remote':
            return __handle_analyze_tflite(request)
        elif type == 'local':
            fname = request.form['local_tflite']
            with open(fname, 'rb') as df:
                content = df.read()
                return __analyze_tflite(fname, content)
    return 'Invalid Request', 500

def __handle_analyze_tflite(request):
    global IN_MEMORY_SIZE
    if not request.files:
        flash('TFLite file is required!!!')
        return redirect(request.url)
    f = request.files['remote_tflite']
    if f and allowed_file(f.filename):
        fname = secure_filename(f.filename)
        f.seek(0, os.SEEK_END)
        length = f.tell()
        f.seek(0, os.SEEK_SET)
        print('length:', length)
        if length < IN_MEMORY_SIZE:
            #content = struct.unpack('!c', f.stream.read())
            content = f.stream.read()
            return __analyze_tflite(fname, content)
        else:
            return f.save(os.path.join(app.config['UPLOAD_FOLDER'], fname))
    flash('Wrong Request!!!')
    return redirect(request.url)

def __getTensorTypeStringName(idx):
    return getStringName(TensorType, idx)

def __catenate_list(length, func):
    val = ''
    container = []
    for i in range(length):
        container.append(str(func(i)))
    val += ','.join(container)
    container.clear()
    return val

def __join_list(length, lst):
    return ','.join([str(lst(i)) for i in range(length)])

def __dump_operator(operator):
    return {
        'opcode': operator.OpcodeIndex(),
        'opname': __getBuiltinOperatorStringName(operator.OpcodeIndex()), 
        'custom_operations_format': operator.CustomOptionsFormat(),
        'inputs': __join_list(operator.InputsLength(), operator.Inputs),
        'outputs': __join_list(operator.OutputsLength(), operator.Outputs)
    }

def __dump_quantization_parameters(quantization):
    minVal = 'min:'
    maxVal = 'max:'
    scaleVal = 'scale:'
    zeroPointVal = 'zeroPoint:'
    minVal += __catenate_list(quantization.MinLength(), quantization.Min)
    maxVal += __catenate_list(quantization.MaxLength(), quantization.Max)
    scaleVal += __catenate_list(quantization.ScaleLength(), quantization.Scale)
    zeroPointVal += __catenate_list(quantization.ZeroPointLength(), quantization.ZeroPoint)
    return minVal, maxVal, scaleVal, zeroPointVal

def __dump_tensor(tensor):
    return {
        'name': bytes.decode(tensor.Name()),
        'shape': ','.join([str(tensor.Shape(i)) for i in range(tensor.ShapeLength())]),
        'type': __getTensorTypeStringName(tensor.Type()),
        'is_variable': 'True' if tensor.IsVariable() else 'False',
        'quantization': __dump_quantization_parameters(tensor.Quantization())
    }        

def __analyze_tflite(fname, buffer):
    print('vistflite...')
    model = Model.GetRootAsModel(buffer, 0)
    # OperatorCodes
    operatorCodes = []
    for i in range(model.OperatorCodesLength()):
        op = model.OperatorCodes(i)
        if op.CustomCode():
            print('Custom OP:', op.Version(), op.CustomCode())
            operatorCodes.append(['Custom OP', op.Version(), op.CustomCode()])
        else:
            print('Builtin OP:', op.Version(), op.BuiltinCode(), __getBuiltinOperatorStringName(op.BuiltinCode()))
            operatorCodes.append(['Builtin OP', op.Version(), op.BuiltinCode(), __getBuiltinOperatorStringName(op.BuiltinCode())])
    # SubGraphs
    print('SubGraphs:')
    subGraphs = []
    for i in range(model.SubgraphsLength()):
        subGraph = model.Subgraphs(i)
        print('Name:', subGraph.Name())
        print('Tensors:')
        tensors = []
        for j in range(subGraph.TensorsLength()):
            tensor = __dump_tensor(subGraph.Tensors(j))
            tensors.append(tensor)
        operators = []
        for j in range(subGraph.OperatorsLength()):
            operator = __dump_operator(subGraph.Operators(j))
            operators.append(operator)
        subGraphs.append({'name': subGraph.Name(), 'tensors': tensors, 'operators': operators})
        print('input length:', subGraph.InputsLength(), 'output length:', subGraph.OutputsLength(), 'operators length:', subGraph.OperatorsLength())
    return render_template('tflite_model_details.html',
                            tflite_file_name=fname,
                            model_version=model.Version(),
                            model_description=model.Description(),
                            model_operator_codes=operatorCodes,
                            model_sub_graphs=subGraphs)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404
 
@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True)
