
import base64
import os
from datetime import datetime
import time
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from tensorflow.keras.models import load_model # type: ignore

import time





app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'image'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.static_folder = 'static'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

model = load_model('l_action.h5')
actions = np.array(['morning','good','how','you','are'])
# ,'hello','sorry','night','please','I','fine'

# model = load_model('action2.h5')
# actions = np.array(['a', 'book'])
# base_colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
# desired_length = 10
# colors = [base_colors[i % len(base_colors)] for i in range(desired_length)]

seq = []
sentence = []
output_text = []
threshold = 0.5
last_capture_time = time.time()
predicted_action = None 

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              )
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )


def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([lh, rh])







@app.route('/')
def landing_page():
    return render_template('landing_page.html')

# Add a new route for the detection page
@app.route('/detection')
def detection_page():
    return render_template('predict_pose.html')

# ...


@app.route('/upload', methods=['POST'])
def upload_image():
    global seq 
    global sentence 
    global output_text 
    global threshold 
    global last_capture_time
    
    try:

        if(request.json['imageSent']):
            text = []
            seq=[]
            image_data_url = request.json['image']
            base64_data = image_data_url.split(',')[1]
            filename = f"image_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(image_path, 'wb') as f:
                f.write(base64.b64decode(base64_data))
            
            
            
            current_time = time.time()
            image = cv2.imread(image_path)
            image, results = mediapipe_detection(image, mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5))
            draw_styled_landmarks(image, results)
            keypoints = extract_keypoints(results)
            seq.append(keypoints)

            for i in range(39):
                seq.insert(0,keypoints) 
            
            print("SEQ")
            print(len(seq))
            seq = seq[:40]
            if len(seq) == 40:
                res = model.predict(np.expand_dims(seq, axis=0))[0]

                action = actions[np.argmax(res)]
                print(res)
                print(action)
                if res[np.argmax(res)] > threshold:
                

                # if action not in sentence:
                    if(action=='good'):
                            sentence.append(action)
                            text.append("শুভ")
                            output_text.append(sentence.copy())        
                    elif(action=='morning'):
                            sentence.append(action)
                            text.append("সকাল")
                            output_text.append(sentence.copy())
                    elif(action=='you'):
                            sentence.append(action)
                            text.append("কেমন")
                            output_text.append(sentence.copy())
                    elif(action=='how'):
                            sentence.append(action)
                            text.append("আপনি")
                            output_text.append(sentence.copy())
                #     elif(action=='good'):
                #             sentence.append(action)
                #             text.append("আমি ভালো আছি")
                #             output_text.append(sentence.copy())
                # #     elif(action=='night'):
                #             sentence.append(action)
                #             text.append("রাত্রি")
                #             output_text.append(sentence.copy())
                #     elif(action=='please'):
                #             sentence.append(action)
                #             text.append("দয়া করে")
                #             output_text.append(sentence.copy())
                #     elif(action=='I'):
                #             sentence.append(action)
                #             text.append("আমি")
                #             output_text.append(sentence.copy())
                #     elif(action=='fine'):
                #             sentence.append(action)
                #             text.append("ভালো আছি")
                #             output_text.append(sentence.copy())                                                
                    else:
                            sentence.append(action)
                            text.append("আছেন")
                            output_text.append(sentence.copy())
                    



                    

                    sentence_limit = 0
                    if len(output_text) > sentence_limit:
                        output_text = output_text[-sentence_limit:]

                    os.remove(image_path)
                    return jsonify({'status': 'success', 'output': text, 'imageSent': False})

            else:
                os.remove(image_path)
                return jsonify({'status': 'success', 'output': 'Not enough keypoints captured', 'imageSent': False})

        else:
            return jsonify({'status': 'success', 'output': 'Don\'t capture properly', 'imageSent': False})
    except Exception as e:
        print(e)
        return jsonify({'error': 'Error processing image'}), 500

if __name__ == '__main__':
    app.run(debug=True)

