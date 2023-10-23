from flask import Flask
import flask
from werkzeug.utils import secure_filename
import os
from celery import Celery, Task, shared_task
from celery.result import AsyncResult
from make_celery import detect, flask_app


uploaded = False


@flask_app.route('/', methods=['GET'])
def home():
    status = 'lol'
    return flask.render_template('index.html', status=status)

@flask_app.route('/result/<id>')
def task_result(id: str):
    result = AsyncResult(id)
    return {
        "ready": result.ready(),
        "successful": result.successful(),
        "value": result.result if result.ready() else None,
    }

@flask_app.route('/upload', methods=['GET', 'POST'])
def upload_video():
    global uploaded
    if flask.request.method == 'POST':
        if 'input_video' not in flask.request.files:
            print ('no file part')
            return flask.redirect(flask.request.url)
        
        file = flask.request.files['input_video']
        if file.filename == '':
            flask.flash('No image selected for uploading')
            return flask.redirect(flask.request.url)
        else:
            filename = secure_filename(file.filename)
            file.save(os.path.join(flask_app.config['uploads'], filename))
            print('upload_video filename: ' + filename)
            uploaded  = True
            result = detect.delay(f'./static/{filename}')
            return {'result_id': result.id}
    else:
        return flask.render_template('index.html')


if __name__ == '__main__':
    flask_app.secret_key = 'super secret key'
    flask_app.config['uploads'] = './static/'
    flask_app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    flask_app.debug = True
    flask_app.run()