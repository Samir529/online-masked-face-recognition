import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer,  RTCConfiguration, WebRtcMode
from PIL import Image
import av
import queue
from typing import List, NamedTuple


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.set_page_config(
    page_title="Masked-Face-Recognition",
    page_icon="😷"
)
local_css('css/styles.css')

st.markdown('<h2 align="center">Real Time Masked Face Recognition</h2>', unsafe_allow_html=True)
# hide_streamlit_style = """
#     <style>
#     ul[data-testid=main-menu-list] > li:nth-of-type(2), /* Settings */
#     ul[data-testid=main-menu-list] > li:nth-of-type(3), /* Record a screencast */
#     ul[data-testid=main-menu-list] > li:nth-of-type(4), /* Report a bug */
#     ul[data-testid=main-menu-list] > li:nth-of-type(5), /* Get help */
#     ul[data-testid=main-menu-list] > li:nth-of-type(6), /* Share this app */
#     ul[data-testid=main-menu-list] > li:nth-of-type(7), /* About */
#     ul[data-testid=main-menu-list] > li:nth-of-type(8),
#     ul[data-testid=main-menu-list] > li:nth-of-type(9),
#     ul[data-testid=main-menu-list] > li:nth-of-type(10),
#     ul[data-testid=main-menu-list] > div:nth-of-type(2), /* 2nd divider */
#     ul[data-testid=main-menu-list] > div:nth-of-type(3),
#     ul[data-testid=main-menu-list] > div:nth-of-type(4),
#     ul[data-testid=main-menu-list] > div:nth-of-type(5),
#     ul[data-testid=main-menu-list] > div:nth-of-type(6)
#     {display: none;}
#     </style>
# """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True)


faces = ['Abdur Samad', 'Ahsan Ahmed', 'Asef', 'Ashik', 'Azizul Hakim', 'DDS', 'Mahmud', 'Mayaz', 'Meheraj', 'Nayeem Khan', 'Nayem', 'Risul Islam Fahim', 'Saif', 'Saki', 'Samir', 'Shahtab', 'Shimul Rahman Fahad', 'Shourov', 'Shuvo', 'Sizan']

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('masked_face_detector.h5')
    return model
#app = Flask(__name__)

@st.cache(allow_output_mutation=True)
def dnn_extract_face1(img):
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
    (height, width) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img,(300,300)),1.0,(300,300),(104.0,177.0,123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(0, detections.shape[1]):
        confidence = detections[0,0,i,2]
#         print("confidence ",confidence)
        if confidence > 0.5:
            box = detections[0,0,i,3:7] * np.array([width,height,width,height])
            (startX, startY, endX, endY) = box.astype("int")
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(img, (startX, startY), (endX, endY), (255, 255, 0), 2)
            cv2.putText(img, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            face = img[startY:endY,startX:endX]
            return face
        else:
            return None


@st.cache(allow_output_mutation=True)
class Detection(NamedTuple):
        Name: str
        Prob: float


@st.cache(allow_output_mutation=True)
class VideoProcessor(VideoProcessorBase):
    result_queue: "queue.Queue[List[Detection]]"
    def __init__(self) -> None:
        self.result_queue = queue.Queue()
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        result: List[Detection] = []
        frame = frame.to_ndarray(format="bgr24")
        frame = cv2.flip(frame,1)
        face = dnn_extract_face1(frame)
        if type(face) is np.ndarray:
            face = cv2.resize(face, (350, 350))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(face, 'RGB')
            img_array = np.array(im)
            img_array = np.expand_dims(img_array, axis=0)
            pred = model.predict(img_array)
            predition = np.squeeze(pred)
            predIndex = np.argmax(predition)

            #             name = 'None matching'
            if (predition[predIndex] > 0.95):
                text = "{:.2f}%".format(predition[predIndex] * 100)
                name = str(faces[predIndex]) + ' ' + str(text)
                cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
                result.append(Detection(Name=faces[predIndex], Prob=float(predition[predIndex])))
            else:
                cv2.putText(frame, '', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
        else:
            cv2.putText(frame, 'No face detected :(', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        #             cv2.putText(frame,'',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        self.result_queue.put(result)
        return av.VideoFrame.from_ndarray(frame, format="bgr24")

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/')
# def index():
#     return render_template('index.html')


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
# RTC_CONFIGURATION = RTCConfiguration(
#     {"iceServers": [{"urls": ["stun:stun.xten.com:3478"]}]}
# )

model = load_model()
webrtc_ctx = webrtc_streamer(key="key",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=VideoProcessor,
                media_stream_constraints = {"video": True, "audio": False},
                async_processing = True)              

st.markdown("""
            <style>
            table td:nth-child(1) {
                display: none
            }
            table th:nth-child(1) {
                display: none
            }
            table th {
                text-align: center !important;           
            }
            table td {
                text-align: center !important;
                color: lime !important;
            }
            </style>
            """, unsafe_allow_html=True)
if st.checkbox("Show the detected face", value=True):
        if webrtc_ctx.state.playing:
            labels_placeholder = st.empty()
            while True:
                if webrtc_ctx.video_processor:
                    try:
                        result = webrtc_ctx.video_processor.result_queue.get(
                            timeout=1.0
                        )
                    except queue.Empty:
                        result = None
                    labels_placeholder.table(result)
                else:
                    break
                    
#if __name__ == '__main__':
#    app.run(debug=True)

