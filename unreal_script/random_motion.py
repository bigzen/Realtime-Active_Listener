from pylivelinkface import PyLiveLinkFace, FaceBlendShape
import threading
import numpy as np
import socket
import random
import time
import math

left_el = FaceBlendShape.EyeBlinkLeft
left_int = 1.0
right_el = FaceBlendShape.EyeBlinkRight
right_int = 1.0
head_pitch = FaceBlendShape.HeadPitch
pitch_int = 0.0
head_roll = FaceBlendShape.HeadRoll
roll_int = 0.0
head_yaw = FaceBlendShape.HeadYaw
yaw_int = 0.0
py_face = PyLiveLinkFace()
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 
s.connect(("127.0.0.1", 11111))
lock = threading.Lock()

class BlinkController(threading.Thread):
    """
    Threaded realistic blink controller for a PyLiveLinkFace instance.
    - py_face: PyLiveLinkFace instance
    - send_callback: function(py_face) -> should transmit py_face.encode() to Unreal (optional)
    - stop_event: threading.Event used to stop the thread
    - intensity: max blink value (0.0..1.0)
    - mean_interval: average seconds between blinks
    """
    def __init__(self, send_callback=None, stop_event=None, mean_interval=4.0):
        super().__init__(daemon=True)
        self.send_callback = send_callback
        self.stop_event = stop_event or threading.Event()
        self.defaultintensity = 1.0
        self.mean_interval = float(mean_interval)
        # blendshape enum names used here - adapt if your enum uses different names
        

    def ease_cos(self, t):
        # cosine ease in/out: 0..1 -> 0..1 smooth
        return 0.5 - 0.5 * math.cos(math.pi * t)

    def do_blink(self):
        global left_int, right_int
        # randomized timings (seconds)
        close_time = random.uniform(0.06, 0.14)   # time to close
        closed_hold = random.uniform(0.02, 0.12)  # eyelid hold closed
        open_time = random.uniform(0.06, 0.14)    # time to open

        # small asynchrony between eyes
        asymmetry = random.uniform(-0.03, 0.03)
        step = 0.033
        # closing
        t = 0.0
        while t < close_time:
            if self.stop_event.is_set():
                return
            p = self.ease_cos(min(1.0, t / close_time))
            with lock:
                left_int = self.defaultintensity * p
                right_int = self.defaultintensity * max(0.0, min(1.0, p + asymmetry))
            if self.send_callback:
                self.send_callback()
            time.sleep(step)
            t += step

        # closed hold
        if self.stop_event.is_set():
            return
        with lock:
            left_int = float(self.defaultintensity)
            right_int = float(self.defaultintensity)
        if self.send_callback:
            self.send_callback()

        # small micro-blinks (double blink) chance
        if random.random() < 0.12:
            # very short re-open then close again to simulate double blink
            with lock:
                left_int = 0.0
                right_int = 0.0
            if self.send_callback:
                self.send_callback()
            time.sleep(random.uniform(0.03, 0.08))
            # re-close quickly
            with lock:
                left_int = float(self.defaultintensity)
                right_int = float(self.defaultintensity)
            if self.send_callback:
                self.send_callback()
            time.sleep(random.uniform(0.02, 0.06))

        if self.stop_event.is_set():
            return
        time.sleep(closed_hold)

        # opening (reverse of closing)
        t = 0.0
        while t < open_time:
            if self.stop_event.is_set():
                return
            p = self.ease_cos(min(1.0, 1.0 - (t / open_time)))
            with lock:
                left_int = self.defaultintensity * p
                right_int = self.defaultintensity * max(0.0, min(1.0, p + asymmetry))
            if self.send_callback:
                self.send_callback()
            time.sleep(step)
            t += step

        # ensure fully open
        with lock:
            left_int = 0.0
            right_int = 0.0
        if self.send_callback:
            self.send_callback()


    def run(self):
        while not self.stop_event.is_set():
            # interval sampled from a skewed distribution (more realistic)
            interval = max(0.8, random.gauss(self.mean_interval, 1.2))
            # small chance of shorter interval (frequent blinks)
            if random.random() < 0.08:
                interval = random.uniform(1.0, 2.5)
            # wait but wake early if stop_event set
            if self.stop_event.wait(interval):
                break
            self.do_blink()

    def stop(self):
        self.stop_event.set()

def gaussian_randfilter1d(size,sigma):
    filter_range = np.linspace(-int(size/2),int(size/2),size)
    gaussian_filter = [1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-x**2/(2*sigma**2)) for x in filter_range]
    return gaussian_filter

def set_pitch():
    global pitch_int
    a = random.choice([-1,1])
    sigma = random.randint(3,7)
    g_filter = gaussian_randfilter1d(30,sigma)
    for val in g_filter:
        with lock:
            pitch_int = val * a
        _send_pyface()
        time.sleep(0.033)

def set_yaw():
    global yaw_int
    a = random.choice([-1,1])
    sigma = random.randint(3,7)
    g_filter = gaussian_randfilter1d(30,sigma)
    for val in g_filter:
        with lock:
            yaw_int = val * a
        _send_pyface()
        time.sleep(0.033)

def _send_pyface():
    global left_int, right_int, pitch_int, yaw_int, py_face, s
    with lock:
        py_face.set_blendshape(left_el, float(left_int))
        py_face.set_blendshape(right_el, float(right_int))
        py_face.set_blendshape(head_pitch, float(pitch_int))
        py_face.set_blendshape(head_yaw, float(yaw_int))
    #py_face.set_blendshape(head_roll, float(roll_int))
    try:
        s.sendall(py_face.encode())
    except Exception:
        pass

if __name__ == '__main__':

    stop_event = threading.Event()
    # callback that sends current encoded face to the socket

    blink_ctrl = BlinkController(send_callback=_send_pyface, stop_event=stop_event, mean_interval=4.0)
    blink_ctrl.start()
    print("Blinking started.")
    t1 = None
    t2 = None
    
    try:
        while True:
            a, b = random.random()>0.995, random.random()>0.995
            if a and (t1 is None or not t1.is_alive()):
                t1 = threading.Thread(target=set_pitch, daemon=True)
                t1.start()
            if b and (t2 is None or not t2.is_alive()):
                t2 = threading.Thread(target=set_yaw, daemon=True)
                t2.start()
            time.sleep(0.01)
            

            
        
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        blink_ctrl.stop()
        blink_ctrl.join()