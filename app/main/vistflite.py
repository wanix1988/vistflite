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

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024 #1GB
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

IN_MEMORY_SIZE = 100 * 1024 * 1024 #100MB

bootstrap = Bootstrap(app)

def getBuiltinOperatorStringName(idx):
    allAttrs = [i for i in BuiltinOperator.__dict__.items() if not i[0].startswith('__')]
    for name, index in allAttrs:
        if index == idx:
            return name

@app.route('/')
def index():
    return render_template(r'analyze_tflite.html', name='tflite')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] == 'tflite'

@app.route('/analyze_tflite', methods=['POST'])
def analyze_tflite():
    global IN_MEMORY_SIZE
    if not request.files:
        flash('TFLite file is required!!!')
        return redirect(request.url)
    f = request.files['tflite']
    print(dir(f))
    if f and allowed_file(f.filename):
        fname = secure_filename(f.filename)
        print(dir(f))
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

def __analyze_tflite(fname, buffer):
    print('vistflite...')
    model = Model.GetRootAsModel(buffer, 0)
    # OperatorCodes
    operatorCodes = []
    for i in range(model.OperatorCodesLength()):
        op = model.OperatorCodes(i)
        if op.CustomCode():
            print('Custom OP:', op.Version(), op.CustomCode())
            operatorCodes.append(['Custom OP:', op.Version(), op.CustomCode()])
        else:
            print('Builtin OP:', op.Version(), op.BuiltinCode(), getBuiltinOperatorStringName(op.BuiltinCode()))
            operatorCodes.append(['Builtin OP:', op.Version(), op.BuiltinCode(), getBuiltinOperatorStringName(op.BuiltinCode())])
    # SubGraphs
    print('SubGraphs:')
    for i in range(model.SubgraphsLength()):
        subGraph = model.Subgraphs(i)
        print('Name:', subGraph.Name())
        print('Tensors:')
        for j in range(subGraph.TensorsLength()):
            print(subGraph.Tensors(j).Name())
    return render_template('tflite_model_details.html',
                            tflite_file_name=fname,
                            model_version=model.Version(),
                            model_description=model.Description(),
                            model_operator_codes=operatorCodes)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404
 
@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True)
