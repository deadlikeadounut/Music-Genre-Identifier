# import requirements needed
from flask import Flask, flash, request, redirect, url_for, render_template
from flask import send_from_directory
from werkzeug.utils import secure_filename
from utils import get_base_url, allowed_file, and_syntax
import librosa
import os


#preprocessing function
def preprocessing(y, sr):
    trimmed, index = librosa.effects.trim(y)
    y = trimmed[0:66149]
    result = []
    print(type(y))
    chroma_y = librosa.feature.chroma_stft(y=y)
    rms_y = librosa.feature.rms(y=y)
    spectral_y = librosa.feature.spectral_centroid(y=y)
    bandwidth_y = librosa.feature.spectral_bandwidth(y=y)
    rolloff_y = librosa.feature.spectral_rolloff(y=y)
    mfccs = librosa.feature.mfcc(y=y, n_mfcc=21)
    y_harm, y_perc = librosa.effects.hpss(y)

    result = [y_perc.var(), chroma_y.mean(),  mfccs[4].mean(), y_perc.mean(), chroma_y.var(), mfccs[5].var(), mfccs[9].mean(), y_harm.mean(), librosa.beat.tempo(y, sr=sr)[0], mfccs[17].mean(), mfccs[4].var(), spectral_y.var(), rms_y.var(), mfccs[6].mean(), mfccs[3].mean(), mfccs[6].var(), mfccs[12].mean(), mfccs[20].mean(), rolloff_y.var(), mfccs[20].var(), mfccs[11].mean(), mfccs[15].mean(), bandwidth_y.mean(), mfccs[3].var()]

    #scaler = MinMaxScaler()
    #result = scaler.fit_transform(result)

    return result

# setup the webserver
# port may need to be changed if there are multiple flask servers running on same server
port = 12345
base_url = get_base_url(port)

# if the base url is not empty, then the server is running in development, and we need to specify the static folder so that the static files are served
if base_url == '/':
    app = Flask(__name__)
else:
    app = Flask(__name__, static_url_path=base_url+'static')

#@app.route('/', methods=['POST'])
@app.route(base_url, methods=['POST'])
def home_post():
    # check if the post request has the file part
    if 'file' not in request.files:
        Flask.flash('No file part')
        return Flask.redirect(Flask.request.url)

    file = Flask.request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        Flask.flash('No selected file')
        return Flask.redirect(Flask.request.url)

    if file and Flask.allowed_file(file.filename):
        print("Worked")
        filename = Flask.secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #return Flask.redirect(Flask.url_for('results', filename=filename))
        array, sr = librosa.read(filename)
        return render_template('Home.html',prediction_text=f'Sample Rate {sr}')





# set up the routes and logic for the webserver
@app.route(f'{base_url}')
def home():
    return render_template('Home.html')

#@app.route('/about')
@app.route(f'{base_url}/About')
def about():
    return render_template('About.html')
# define additional routes here
# for example:
# @app.route(f'{base_url}/team_members')
# def team_members():
#     return render_template('team_members.html') # would need to actually make this page

if __name__ == '__main__':
    # IMPORTANT: change url to the site where you are editing this file.
    website_url = 'localhost'
    
    print(f'Try to open\n\n    https://{website_url}' + base_url + '\n\n')
    app.run(host = '0.0.0.0', port=port, debug=True)


