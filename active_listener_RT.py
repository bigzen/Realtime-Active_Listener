import socket
import threading
import time
import numpy as np
from BlinkController import BlinkController 
import torch as T
from head_model.RT_combi import GCN
from collections import deque
from torch_geometric.data import Data
from pylivelinkface import PyLiveLinkFace, FaceBlendShape
from transformers import Wav2Vec2Model as w2v2
from transformers import Wav2Vec2Tokenizer as fe

# Global device configuration
device = T.device('cuda' if T.cuda.is_available() else 'cpu')

#speech-foundation and graph model initialization
model_name = 'facebook/wav2vec2-base-960h'#"facebook/wav2vec2-large-xlsr-53"
fextractor = fe.from_pretrained(model_name)
Wav_model = w2v2.from_pretrained(model_name).to(device)
GCN_model = GCN(feat=768, hidden=256, classes=3).to(device)
edge_index = T.tensor([[0], [1]], dtype=T.long)

#communication socket variables
unreal_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
unreal_address = ('130.209.247.98', 11111)
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

#data queues
frame_queue = deque(maxlen=400)
smoothing_queue = deque(maxlen=1000)
raw_pose_queue = deque(maxlen=30)
final_pose_queue = deque(maxlen=20)
final_pose = np.zeros((1,3))[0]
frame_gen = []
frame_counter = 0

#global variables for live_link
left_int = 1.0
right_int = 1.0
pitch_int = 0.0
roll_int = 0.0
yaw_int = 0.0
py_face = PyLiveLinkFace()
lock = threading.Lock()

def __init__():
    # Load the pre-trained GCN model
    global GCN_model
    saved_model = T.load('iemocap_head.pth', weights_only=False)
    saved_list = list(saved_model.keys())
    for n, e in enumerate(GCN_model.state_dict().keys()):
        GCN_model.state_dict()[e] = saved_model[saved_list[n]]
    GCN_model.eval()
    print("Model loaded and set to evaluation mode.")

def gaussian_filter1d(size,sigma):
    filter_range = np.linspace(-int(size/2),int(size/2),size)
    gaussian_filter = [1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-x**2/(2*sigma**2)) for x in filter_range]
    return gaussian_filter

def process_wav(data):
    # Process raw data using wav2vec2 foundatioon model
    global fextractor, Wav_model
    input = fextractor(data, sampling_rate=16000, return_tensors='pt' )
    with T.no_grad():
        inp = input.input_values.squeeze(0)
        feature = Wav_model(inp.to(device))
        feat = feature.last_hidden_state.squeeze()
        pyg_data = Data(x=feat.unsqueeze(0), edge_index=edge_index)
        out = GCN_model((pyg_data.x.to(device), pyg_data.edge_index.to(device)))
    #feat = feature.extract_features.squeeze()
    return out

def set_blink(l, r):
    global left_int, right_int
    with lock:
        left_int = float(l)
        right_int = float(r)

def t_action():
    with lock:
        py_face.set_blendshape(FaceBlendShape.EyeBlinkLeft, left_int)
        py_face.set_blendshape(FaceBlendShape.EyeBlinkRight, right_int)
        py_face.set_blendshape(FaceBlendShape.HeadPitch, pitch_int)
        #py_face.set_blendshape(FaceBlendShape.HeadRoll, roll_int)
        py_face.set_blendshape(FaceBlendShape.HeadYaw, yaw_int)
        unreal_socket.sendall(py_face.encode())
    time.sleep(0.05)

def main():
    # Initialize components
    __init__()
    # Bind the socket to an address and port
    global udp_socket
    udp_socket.bind(("172.18.236.8", 5052))
    print("Listening for UDP data on port 50550...")

    global unreal_socket, final_pose
    unreal_socket.connect(unreal_address)

    filter = np.array(gaussian_filter1d(40,4))[:20]
    filter = np.stack((filter, filter, filter), axis=1)
    #print(filter)

    global frame_counter, frame_queue, roll_int, pitch_int, yaw_int
    blink_controller = BlinkController(set_blink, t_action, mean_interval=4.0)
    blink_controller.start()
    action_thread = threading.Thread(target=t_action, daemon=True)
    
    try:
        mean = np.zeros((1,3))[0]
        while True:
            # Receive data from the socket
            start_time = time.time()
            data, _ = udp_socket.recvfrom(1024)  # Buffer size is 1024 bytes
            data  = list(np.frombuffer(data, dtype=np.float32))
            #print(data)
            multiplier = 1
            for dat in data:
                frame_queue.append(dat)
                smoothing_queue.append(np.abs(dat))
                    #sm_val = (smoothed_dat>0.1)*smoothed_dat
            frame_counter += 2
            #print(frame_counter)
            if frame_counter >= 400:
                input = T.tensor(np.array(frame_queue, dtype=np.float32)).unsqueeze(0)
                pose = process_wav(input)
                data = pose.squeeze().detach().cpu().numpy()
                raw_pose_queue.append(data)
                new_pose = data
                
                #print(data)
                multiplier = 0
                
                # No roll for now
                #if len(raw_pose_queue) < 100:
                #    raw_pose_queue.append(data)
                #    if len(raw_pose_queue) == 1:
                #        mean = data
                #elif len(raw_pose_queue) == 100:
                #    mean = np.mean(np.array(raw_pose_queue), axis=0)
                if len(raw_pose_queue) >= 30:
                    sm_val = np.mean(np.array(smoothing_queue))
                    multiplier = sm_val if sm_val>0.03 else 0#((np.abs(np.random.normal(0, 0.2))>0.7)*0.2)
                    new_pose[1] = 0 
                    deviation = (new_pose-np.mean(np.array(raw_pose_queue), axis=0))*multiplier#*10000
                    deviation = deviation*(10**(-1-np.max(np.log10(np.abs(deviation)).round())))
                    print("deviation:", deviation, new_pose)
                    if multiplier == 0 or np.dot(deviation,deviation) < 0.05:
                        new_pose = final_pose
                    else:
                        deviation[np.abs(deviation)!=np.max(np.abs(deviation))] = 0#*= multiplier*2
                        deviation[np.abs(deviation) < 0.07] = 0.0
                        new_pose = deviation*2#*multiplier*5*10000#(new_pose - np.mean(np.array(raw_pose_queue), axis=0))*multiplier
                        #new_pose = new_pose*(10**(-2-np.log10(np.abs(new_pose)).round()))
                        #final_pose = new_pose*50

                #if len(smoothing_queue) >= 1000:
                #pose_queue.pop()
                #print(new_pose, np.log10(new_pose*1000))
                #pose_queue.append(new_pose)
                #    print(avg_pose, data)
                #    data -= avg_pose
                #if  
                #final_pose = new_pose
                final_pose_queue.append(new_pose)
                if len(final_pose_queue) >= 20:
                    new_pose = np.sum(np.multiply(np.array(final_pose_queue),filter), axis=0)
            
                #print(new_pose, multiplier)
                #final_pose = new_pose
                
                new_pose = np.sign(data)*new_pose
                #print("Final pose:", new_pose, data)
                with lock:
                    roll_int = 0#float(new_pose[1])/7
                    pitch_int = float(new_pose[0])
                    yaw_int = float(new_pose[2])
                if not action_thread.is_alive():
                    action_thread = threading.Thread(target=t_action, daemon=True)
                    action_thread.start()
                frame_counter -= 320
                end_time = time.time()
                frame_gen.append(end_time - start_time)
                
                #t = threading.Thread(target=t_action)
                #t.start()
            #print(list(map(float,data.decode('utf-8')[2:-2].split(' '))))
            #pose = process_wav(data)
            #server_address = ('172.18.236.8', 50550)
            #send_data(pose.cpu().numpy().tobytes(), server_address)# Placeholder for any threaded action if needed
            
            #print(f"Received message from {addr}: {data.decode('utf-8')}")
    except KeyboardInterrupt:
        print("\nExiting...")
        print("Max frame generation time:", np.max(frame_gen))
        print("Average frame generation FPS:", 1/np.mean(frame_gen))
        print("median frame generation time:", np.median(frame_gen))

    finally:
        udp_socket.close()
        unreal_socket.close()

if __name__ == "__main__":
    main()