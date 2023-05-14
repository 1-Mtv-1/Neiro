
from flask import Flask
from flask import render_template
from flask import request
from flask import redirect
from flask import session
import neiro

app = Flask(__name__)

app.secret_key = 'qwerty123!'
models = neiro.pre_init()

@app.route('/',methods = ['POST','GET'])
def redirection():
    session['mode'] = '1'
    return redirect('http://127.0.0.1:8080/home',302)

@app.route('/home',methods = ['POST','GET'])
def home():
    global res
    global models
    if request.method == 'POST':
        if request.form.get('mode'):
            session['mode'] = request.form.get('mode')

        try:
            scan_image = request.files['photo']
            photo = open('cache/image.png','wb')
            photo1 = open('static/image.jpg','wb')
            for x in scan_image:
                photo1.write(x)
                photo.write(x)
            photo.close()
            photo1.close()

            print('127.0.0.1 - - "New image" 200 -')
            session['result'] = '1'

            if session.get('mode') == '2':
                res = neiro.single_res('cache/image.png',models[0],models[1])
            if session.get('mode') == '1':
                neiro.switch_res('cache/image.png',models[0],models[1])
                res=''
                print('detected')

            return render_template('home.html', mode = session.get('mode'), result = 'True')
        except:
            session['result'] = ''
               


    return render_template('home.html', mode = session['mode'], result = 'False')

@app.route('/result',methods = ['POST','GET'])
def result():
    global res
    if session.get('result'):
        return render_template('result.html', mode = session.get('mode'), result = res)
    return redirect('http://127.0.0.1:8080/home',302)

if __name__ == "__main__" :
    app.run(host='127.0.0.1', port='8080')