# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 01:35:03 2021

@author: ragha
"""

import cv2
import dash
import time
import base64
import queue
import random
import threading
import numpy as np
import simplejpeg
from dash import dcc
from dash import html
import mediapipe as mp
from flask import Response
from dash.dependencies import Input, Output, State

# from imutils import face_utils
from scipy.spatial.distance import euclidean
from imutils.video import FPS, WebcamVideoStream

RIGHT_EYE_IDX = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398] 
LEFT_EYE_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

resolution = 1000
t = np.linspace(0, np.pi * 2, resolution)
x, y = np.cos(t), np.sin(t)

# y = [random.random() for i in range(resolution)]
# x = np.linspace(1, resolution, resolution)

figure = dict(data=[{'x': [], 'y': []}], layout=dict(xaxis=dict(range=[-1, 1]), yaxis=dict(range=[-1, 1])))

# figure = dict(data=[{'x': [], 'y': []}], layout=dict(yaxis=dict(range=[0, 1])))

app = dash.Dash(__name__, update_title=None)
server = app.server

app.layout = html.Div(children=[
                  html.Div(className='row',  # Define the row element
                           children=[
                           html.Div(className='three columns div-user-controls',
                                    children = [
                                        html.P('Computer Vision Application', style = {'font-size': '35px'}),
                                        html.H2('⦿ This dashboard allows users to access several computer vision applications in real-time.', style = {'font-size': '20px'}),
                                        html.H2('⦿ Users can choose a variety of approaches through the dropdown menu.', style = {'font-size': '20px'}),
                                        html.Br(),
                                        
                                        # html.H2('Choose action:', style = {"padding-top": "10px", 
                                        #                             "padding-left": "0",'font-size': '25px'
                                        #                             }),
                                        # html.Div([
                                        #     dcc.Dropdown(
                                        #                id="drop_down",
                                        #                options=[
                                        #                    {'label': 'Driver monitoring system', 'value': 'monitor'},
                                        #                    {'label': 'Sign language system', 'value': 'language'},
                                        #                    {'label': 'Facial expressing recognition', 'value': 'face'},
                                        #                ],
                                        #                style={'height':50, 'width':500},
                                        #                value='monitor',
                                        #                clearable=False)
                                        #         ]),
                                        
                                        html.Br(),
                                        html.Br(),
                                        
                                        html.H2('Web cam access:', style = {"padding-top": "0px", 
                                                                   "padding-left": "0",'font-size': '25px'
                                                                   }),
                                        
                                        html.Button(id='toggle_button', n_clicks=0, className = "graphButtons",
                                                    style = {'font-size': '15px',
                                                             'cursor': 'pointer',
                                                             'text-align': 'center',
                                                             'color': 'white',
                                                            }
                                                    ),
                                        dcc.Store(id="store_toggle", data = False),
                                        html.Br(),
                                        html.Br(),
                                        
                                  ]),  # four column Div
                                   
                           html.Div(className='nine columns div-for-charts bg-grey',  # Define the right element
                                    children = [
                                        html.P('Computer Vision', style = {'text-align':'center', "padding-top": "25px", 
                                                                        'font-size': '35px', 'color': 'red'}),
                                         
                                        html.H2('⦿ Display interface:', style = {"padding-top": "40px", 
                                                                   "padding-left": "0",'font-size': '25px'
                                                                   }),
                                        html.Div([ 
                                            html.Img(id = 'feed',
                                                      style = {'border':'1px solid',
                                                                'float': 'center',
                                                                'margin': '0px 50px',}),
                                            
                                            dcc.Graph(id='graph', figure=figure,
                                                      style = {'height':500,'width':1000, 
                                                               'border':'0.5px solid',
                                                               'float': 'left',
                                                               'margin': '0px 50px'}), 
                                            dcc.Interval(id="interval", interval=20),
                                            # dcc.Store(id='offset', data=0), 
                                            # dcc.Store(id='store', data=dict(x=x, y=y, resolution=resolution)),
                                                ]),
                                   ]),  # eight column Div
                               
                          ]) # row Div
                    ]) # main Div

# app.clientside_callback(
#     """
#     function (n_intervals, data, offset) {
#         offset = offset % data.x.length;
#         const end = Math.min((offset + 10), data.x.length);
#         return [[{x: [data.x.slice(offset, end)], y: [data.y.slice(offset, end)]}, [0], 500], end]
#     }
#     """,
#     [Output('graph', 'extendData'), 
#      Output('offset', 'data')],
#     [Input('interval', 'n_intervals')], 
#     [State('store', 'data'), 
#      State('offset', 'data')]
# )

class Detector:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.mp_hands = mp.solutions.hands
        self.hand_obj = self.mp_hands.Hands(
                                    max_num_hands=2,
                                    model_complexity=0,
                                    min_detection_confidence=0.7,
                                    min_tracking_confidence=0.7)
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_obj = self.mp_face_mesh.FaceMesh(
                                            max_num_faces=1,
                                            refine_landmarks=True,
                                            min_detection_confidence=0.7,
                                            min_tracking_confidence=0.7)

    def get_face_landmarks(self, img_RGB, draw=True):
        h, w, c = img_RGB.shape
        results = self.face_obj.process(img_RGB)
        if results.multi_face_landmarks:
            face_mesh_points = {idx : (int(lm.x*w), int(lm.y*h)) for idx, lm in enumerate(results.multi_face_landmarks[0].landmark)}
            if draw:
                [cv2.circle(img_RGB, p, 2, (0,255,0), -1) for p in face_mesh_points.values()]
            return face_mesh_points
          
    def get_hand_landmarks(self, img_BGR, draw=True):
        # img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
        results = self.hand_obj.process(img_BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if draw:
                    pass
                    # self.mp_drawing.draw_landmarks(
                    #     image = img_BGR,
                    #     landmark_list = hand_landmarks,
                    #     connections = self.mp_hands.HAND_CONNECTIONS,
                    #     landmark_drawing_spec = self.mp_drawing_styles
                    #     .get_default_hand_landmarks_style(),
                    #     connection_drawing_spec = self.mp_drawing_styles
                    #     .get_default_hand_connections_style())


class VideoCamera():
    def __init__(self):
        self.detector = Detector()
        self.web_cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.dummy_img = base64.b64encode(open('assets/play_img.jpg', 'rb').read())
        self.img_shape = cv2.imread('assets/play_img.jpg').shape
        self.right_eye_idxs = RIGHT_EYE_IDX
        self.left_eye_idxs = LEFT_EYE_IDX
    
    def disable(self):
        self.web_cam.release()
        cv2.destroyAllWindows()    
    
    def enable(self):
        frame_counter = 0
        start_time = time.time()
        success, img_BGR = self.web_cam.read()
        if img_BGR is not None:
            frame_counter += 1
            img_BGR = cv2.resize(img_BGR, (self.img_shape[1], self.img_shape[0]))
            img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
            face_landmarks = self.detector.get_face_landmarks(img_RGB, draw=False)
            
            if face_landmarks is not None:
                [cv2.circle(img_RGB, (face_landmarks[idx]), 1,(0,255,0), -1) for idx in self.right_eye_idxs]
                [cv2.circle(img_RGB, (face_landmarks[idx]), 1,(0,255,0), -1) for idx in self.left_eye_idxs]
                
                R_ear, L_ear = self.eye_aspect_ratio(img_RGB, face_landmarks)
                
                if R_ear < 0.2 and L_ear < 0.2:
                    img_RGB = self.textWithBackground(img_RGB, 'BLINK', 1.0, (20,150), textColor=(255,0,0))
            
            end_time = time.time() - start_time
            fps = frame_counter/end_time
                
            img_RGB = self.textWithBackground(img_RGB, f'FPS:{round(fps,1)}', 1.0, (20,50))
                
            jpeg = simplejpeg.encode_jpeg(img_RGB)
            return jpeg, (R_ear, L_ear)
        else:
            return self.dummy_img, (0,0)

    def textWithBackground(self, frame, text, fontScale, textPos, textThickness=1,textColor=(0,255,0), bgColor=(0,0,0), pad_x=5, pad_y=5, bgOpacity=1):
        FONT = cv2.FONT_HERSHEY_COMPLEX
        (t_w, t_h), _= cv2.getTextSize(text, FONT, fontScale, textThickness)  
        x, y = textPos
        new_img = frame.copy()
        # cv2.circle(new_img, (t_w, t_h), 1,(255,0,0), 2)
        cv2.rectangle(new_img, (x-pad_x, y+pad_y), (x+t_w+pad_x, y-t_h-pad_y), bgColor,-1)  
        new_img = cv2.addWeighted(new_img, bgOpacity, frame, 1 - bgOpacity, 0)  
        cv2.putText(new_img,text, textPos, FONT, fontScale, textColor,textThickness )  
        return new_img

    def eye_aspect_ratio(self, frame, landmarks):
        # horizontal line data 
        RHa = landmarks[self.right_eye_idxs[0]]
        RHb = landmarks[self.right_eye_idxs[7]]
       
        LHa = landmarks[self.left_eye_idxs[0]]
        LHb = landmarks[self.left_eye_idxs[7]]
       
        # Top_Vertical Points 
        RVa = landmarks[self.right_eye_idxs[12]]
        RVb = landmarks[self.right_eye_idxs[4]]
       
        LVa = landmarks[self.left_eye_idxs[12]]
        LVb = landmarks[self.left_eye_idxs[4]]
       
        # cv2.line(frame, RHa, RHb, (255,255,255))
        # cv2.line(frame, RVa, RVb, (255,255,255))
       
        # cv2.line(frame, LHa, LHb, (255,255,255))
        # cv2.line(frame, LVa, LVb, (255,255,255))
       
        R_width, R_height = euclidean(RHa, RHb), euclidean(RVa, RVb)
        L_width, L_height = euclidean(LHa, LHb), euclidean(LVa, LVb)
       
        R_ear = round(R_height/R_width, 2)
        L_ear = round(L_height/L_width, 2)
        return R_ear, L_ear

def get_dummy_img():
    encoded_instruct = base64.b64encode(open('assets/play_img.jpg', 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded_instruct.decode())

def feed_generator(camera):
    while True:
        frame, _ = camera.enable()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@server.route('/video_feed')
def video_feed():
    return Response(feed_generator(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.callback([Output("toggle_button","children"),
               Output("store_toggle","data")],
              Input("toggle_button", 'n_clicks')
              )
def toggle_stream(n_clicks):
    click = n_clicks%2
    if click == 0:
        data = False
        return ("Enable video", data)
    elif click == 1:
        data = True
        return ("Disable video", data)

@app.callback(Output("feed", "src"),
              Input('store_toggle', 'data')
              )
def web_cam_access(toggle_value):
    
    # def loop_data(q):
    #     cam = VideoCamera()
    #     while True:
    #         _, ears = cam.enable()
    #         q.put(ears)
            
    # q = queue.Queue()
    # t1 = threading.Thread(target=loop_data, name=loop_data, args=(q,))
    # t1.start()
    if toggle_value:
        # value = q.get()
        # print(value)
        return '/video_feed'
    
    if not toggle_value:
        return get_dummy_img()
    
    else:
        return get_dummy_img()
    
    
@app.callback(Output('graph', 'extendData'), 
              Input('interval', 'n_intervals')
              )
def update_data(n_intervals):
    if n_intervals is not None:
        index = n_intervals % resolution
        # tuple is (dict of new data, target trace index, number of points to keep)
        print(dict(x=[[x[index]]], y=[[y[index]]]), [0], 10)
        return dict(x=[[x[index]]], y=[[y[index]]]), [0], 10
    else:
        dash.no_update
    

# @app.callback(Output("data_visualization", "figure"),
#               Input('drop_down', 'value'),
#               )
# def dropdown_options(drop_value):
#     data_df = pd.read_csv(DATA_PATH)
#     fig_table = table_fig(data_df)
    
#     mod_df = crop_id_col(df = data_df)
#     vis_df = data_grouping(mod_df)
#     fig_graph = build_fig(vis_df)
    
#     if drop_value == 'table':
#         return fig_table
    
#     if drop_value == 'graph':
#         return fig_graph
    
#     else:
#         dash.no_update


if __name__ == '__main__':
    app.run_server(debug=False)

