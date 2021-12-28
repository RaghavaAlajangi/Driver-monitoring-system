# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 01:35:03 2021

@author: ragha
"""

import cv2
import dash
import base64
import simplejpeg
from dash import dcc
from dash import html
import mediapipe as mp
from flask import Response
from dash.dependencies import Input, Output 


app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(children=[
                  html.Div(className='row',  # Define the row element
                           children=[
                           html.Div(className='four columns div-user-controls',
                                    children = [
                                        html.P('Computer Vision Application', style = {'font-size': '35px'}),
                                        html.H2('⦿ This dashboard allows users to access several computer vision applications in real-time.', style = {'font-size': '20px'}),
                                        html.H2('⦿ Users can choose a variety of approaches through the dropdown menu.', style = {'font-size': '20px'}),
                                        html.Br(),
                                        
                                        html.H2('Choose action:', style = {"padding-top": "10px", 
                                                                    "padding-left": "0",'font-size': '25px'
                                                                    }),
                                        html.Div([
                                            dcc.Dropdown(
                                                       id="drop_down",
                                                       options=[
                                                           {'label': 'Driver monitoring system', 'value': 'monitor'},
                                                           {'label': 'Sign language system', 'value': 'language'},
                                                           {'label': 'Facial expressing recognition', 'value': 'face'},
                                                       ],
                                                       style={'height':50, 'width':650},
                                                       value='monitor',
                                                       clearable=False)
                                                ]),
                                        
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
                                        
                                        # html.Button('Disable', id='disable_button', n_clicks=0, disabled=False, 
                                        #             style = {'font-size': '15px',
                                        #                      'cursor': 'pointer',
                                        #                      'text-align': 'center',
                                        #                      'color': 'white',
                                        #                     }
                                        #             ),
                                        
                                  ]),  # four column Div
                                   
                           html.Div(className='eight columns div-for-charts bg-grey',  # Define the right element
                                    children = [
                                        html.H2('Computer Vision', style = {'text-align':'center', "padding-top": "10px", 
                                                                        'font-size': '35px', 'color': 'red'}),
                                         
                                        html.H2('Display interface:', style = {"padding-top": "80px", 
                                                                   "padding-left": "0",'font-size': '25px'
                                                                   }),
                                        html.Div([ 
                                            html.Img(id = 'feed',
                                                      style = {'border':'1px solid',
                                                                'float': 'center',
                                                                'margin': '0px 50px',})
                                                ]),
                                   ]),  # eight column Div
                               
                          ]) # row Div
                    ]) # main Div



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

    def get_face_landmarks(self, img_BGR, draw=True):
        # img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
        results = self.face_obj.process(img_BGR)
        face_keypoints = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(
                        image = img_BGR,
                        landmark_list = face_landmarks,
                        connections = self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec = None,
                        connection_drawing_spec = self.mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())
                    
                    self.mp_drawing.draw_landmarks(
                        image=img_BGR,
                        landmark_list = face_landmarks,
                        connections = self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec = None,
                        connection_drawing_spec = self.mp_drawing_styles
                        .get_default_face_mesh_contours_style())
                    
                    self.mp_drawing.draw_landmarks(
                        image = img_BGR,
                        landmark_list = face_landmarks,
                        connections = self.mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec = None,
                        connection_drawing_spec = self.mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style())
                
                for idx, lm in enumerate(face_landmarks.landmark):
                    h, w, c = img_BGR.shape
                    lm_x, lm_y, lm_z = lm.x*w, lm.y*h, lm.z
                    face_keypoints.append([idx, lm_x, lm_y, lm_z])
        return face_keypoints
          
    def get_hand_landmarks(self, img_BGR, draw=True):
        # img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
        results = self.hand_obj.process(img_BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(
                        image = img_BGR,
                        landmark_list = hand_landmarks,
                        connections = self.mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec = self.mp_drawing_styles
                        .get_default_hand_landmarks_style(),
                        connection_drawing_spec = self.mp_drawing_styles
                        .get_default_hand_connections_style())


class VideoCamera():
    def __init__(self):
        self.detector = Detector()
        self.web_cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.dummy_img = base64.b64encode(open('assets/play_img.jpg', 'rb').read())
        self.img_shape = cv2.imread('assets/play_img.jpg').shape
        self.righteye_idxs = [130, 27, 243, 23]
        self.lefteye_idxs = [463, 257, 359, 253]
        
    def enable(self):
        success, img_BGR = self.web_cam.read()
        if img_BGR is not None:
            img_BGR = cv2.resize(img_BGR, (self.img_shape[1], self.img_shape[0]))
            img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
            # self.detector.get_hand_landmarks(img_RGB)
            # face_keypoints = self.detector.get_face_landmarks(img_RGB)
            # if len(face_keypoints)!= 0:
            #     re_up = face_keypoints[self.righteye_idxs[1]]
            #     re_dw = face_keypoints[self.righteye_idxs[3]]
                
                # print('re_up', re_up)
                # print('re_dw', re_dw)
                # print(re_up[2], re_dw[2])
                # print(abs(re_up[2]-re_dw[2]))
                
            jpeg = simplejpeg.encode_jpeg(img_RGB)
            return jpeg
        else:
            return self.dummy_img
         
    def disable(self):
        self.web_cam.release()
        cv2.destroyAllWindows()
        

def get_dummy_img():
    encoded_instruct = base64.b64encode(open('assets/play_img.jpg', 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded_instruct.decode())


def feed_generator(camera):
    while True:
        frame = camera.enable()
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
    if toggle_value:
        return '/video_feed'
    
    if not toggle_value:
        return get_dummy_img()
    
    else:
        return get_dummy_img()
    

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
    app.run_server(debug=True)
