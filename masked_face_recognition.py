import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, VideoTransformerBase, webrtc_streamer,  RTCConfiguration, WebRtcMode
from PIL import Image
import av
import queue
from typing import List, NamedTuple
from streamlit_option_menu import option_menu
import threading
from typing import Union


def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.set_page_config(
    page_title="Masked-Face-Recognition",
    page_icon="😷",
    layout="wide"
)
load_css('css/styles.css')

@st.cache(allow_output_mutation=True)
def dnn_extract_face(img):
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
    (height, width) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img,(300,300)),1.0,(300,300),(104.0,177.0,123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(0, detections.shape[1]):
        confidence = detections[0,0,i,2]
        # print("confidence ",confidence)
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
def load_model():
    model = tf.keras.models.load_model('masked_face_detector.h5')
    return model
#app = Flask(__name__)

model = load_model()

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

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
# RTC_CONFIGURATION = RTCConfiguration(
#     {"iceServers": [{"urls": ["stun:stun.xten.com:3478"]}]}
# )

choice = option_menu("Masked Face Recognition App", ["Upload Image", "Take Snapshot", "Real Time Detection", "About"],
                     icons=['file-earmark-arrow-up', 'camera', 'camera-video', 'house'],
                     menu_icon="emoji-smile", default_index=0, orientation="horizontal",
                     styles={
                            "container": {"background-color": "#002522"},
                            "icon": {"color": "orange", "font-size": "25px"},
                            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#ff2d00"},
                            "nav-link-selected": {"background-color": "#02ab21"},
                     }
)

if choice == "Upload Image":
    # st.subheader("Image")
    image_file = st.file_uploader("Choose an image...")

    if image_file is not None:
        image = Image.open(image_file)

        col1, col2 = st.columns([0.2, 0.5])
        with col1:
            st.image(image_file, width=250, caption='Uploaded Image.')
        img_array = np.array(image)
        img_array = dnn_extract_face(img_array)
        if img_array is None:
            st.warning("No face is detected.")
        else:
            img_array = Image.fromarray(img_array)
            imResize = img_array.resize((350, 350), Image.ANTIALIAS)
            # imResize.save('predict.jpg', 'JPEG', quality=90)

            predictimg = np.array(imResize)
            predictimg = predictimg / 255.0
            predictimg = np.expand_dims(predictimg, axis=0)
            predition = model.predict(predictimg)
            predition = np.squeeze(predition)
            predIndex = np.argmax(predition)
            st.markdown(""" <style> .font {
                font-size: 50px; font-family: ''; color: white;} 
                </style> """, unsafe_allow_html=True)
            with col2:
                st.image(imResize, width=200, caption='Extracted Face.')
            st.markdown('<p class="font"><b>You are %s (Accuracy %.2f%%)</b></p>' % (faces[predIndex], predition[predIndex] * 100), unsafe_allow_html=True)


if choice == "Take Snapshot":
    class VideoTransformer(VideoTransformerBase):
        frame_lock: threading.Lock
        in_image: Union[np.ndarray, None]
        out_image: Union[np.ndarray, None]

        def __init__(self) -> None:
            self.frame_lock = threading.Lock()
            self.in_image = None
            self.out_image = None

        def transform(self, frame: av.VideoFrame) -> np.ndarray:
            in_image = frame.to_ndarray(format="bgr24")
            out_image = in_image[:, ::-1, :]
            with self.frame_lock:
                self.in_image = in_image
                self.out_image = out_image
            return out_image

    ctx = webrtc_streamer(key="snapshot", video_transformer_factory=VideoTransformer, media_stream_constraints={"video": True, "audio": False})

    if ctx.video_transformer:
        if st.button("Snapshot"):
            with ctx.video_transformer.frame_lock:
                in_image = ctx.video_transformer.in_image
                out_image = ctx.video_transformer.out_image

            if in_image is not None and out_image is not None:
                image = cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB)
                # st.write("Input image:")
                # st.image(in_image, channels="BGR")
                # st.write("Output image:")
                # st.image(out_image, channels="BGR")
                col1, col2 = st.columns([0.3, 0.5])
                with col1:
                    st.image(image, width=400, caption='Snapshot Image.')
                img_array = np.array(image)
                img_array = dnn_extract_face(img_array)
                if img_array is None:
                    st.warning("No face is detected.")
                else:
                    img_array = Image.fromarray(img_array)
                    imResize = img_array.resize((350, 350), Image.ANTIALIAS)
                    # imResize.save('predict.jpg', 'JPEG', quality=90)

                    predictimg = np.array(imResize)
                    predictimg = predictimg / 255.0
                    predictimg = np.expand_dims(predictimg, axis=0)

                    predition = model.predict(predictimg)
                    predition = np.squeeze(predition)
                    predIndex = np.argmax(predition)
                    st.markdown(""" <style> .font {
                                    font-size: 50px; font-family: ''; color: white;} 
                                    </style> """, unsafe_allow_html=True)
                    with col2:
                        st.image(imResize, width=200, caption='Extracted Face.')
                    st.markdown('<p class="font"><b>You are %s (Accuracy %.2f%%)</b></p>' % (faces[predIndex], predition[predIndex] * 100), unsafe_allow_html=True)
            else:
                st.warning("No snapshot available yet. Please take a snapshot.")


if choice == "About":
    st.markdown(""" <style> .font {
    font-size: 40px; font-family: ''; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font"><b>About</b></p>', unsafe_allow_html=True)
    st.write("This is a Masked Face Recognition Application. You can detect your face by Uploading An Image or by Taking A Snapshot. You can also detect your face real time by using the Real Time Detection.")


if choice == "Real Time Detection":
    st.markdown('<h2 align="center">Real Time Masked Face Recognition</h2>', unsafe_allow_html=True)
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
            # frame = frame[:, ::-1, :]
            face = dnn_extract_face(frame)
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


    webrtc_ctx = webrtc_streamer(key="key",
                                 mode=WebRtcMode.SENDRECV,
                                 rtc_configuration=RTC_CONFIGURATION,
                                 video_processor_factory=VideoProcessor,
                                 media_stream_constraints={"video": True, "audio": False},
                                 async_processing=True)

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
                    font-size: 130% !important;           
                }
                table td {
                    text-align: center !important;
                    color: lime !important;
                    font-size: 130% !important;
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


# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/')
# def index():
#     return render_template('index.html')

                    
#if __name__ == '__main__':
#    app.run(debug=True)



